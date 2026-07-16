#!/usr/bin/env python3
"""Add historical and 7-day future weather sequence features to shortterm v1.

Version 1 uses observed/reanalysis weather as a proxy for the future forecast
block. Later this block can be replaced by GEFS without changing the sample
schema.
"""

from __future__ import annotations

from pathlib import Path
import argparse
import json

import pandas as pd


OUT_DIR = Path("Maize_shortterm_surrogate_v1")
COMPACT = OUT_DIR / "shortterm_surrogate_samples_v1_compact.csv"
WEATHER_OUT = OUT_DIR / "weather_sequences_by_date_v1.csv"
MERGED_OUT = OUT_DIR / "shortterm_surrogate_samples_v1_with_weather.csv"
REPORT = OUT_DIR / "shortterm_surrogate_weather_sequence_report.md"

WEATHER_COLS = ["Solar", "T-max", "T-min", "RelHum", "Precip", "WindSpeed"]


def find_weather_file(explicit: str | None) -> Path:
    if explicit:
        path = Path(explicit)
        if not path.exists():
            raise FileNotFoundError(f"Weather file not found: {path}")
        return path

    candidates = [
        Path("df_gridmet.csv"),
        Path("../df_gridmet.csv"),
        Path("weather_gridmet_out.csv"),
        Path("../weather_gridmet_out.csv"),
        Path("df_era.csv"),
        Path("weather_era_out.csv"),
    ]
    for path in candidates:
        if path.exists():
            return path
    raise FileNotFoundError("Could not find df_gridmet.csv, weather_gridmet_out.csv, df_era.csv, or weather_era_out.csv")


def normalize_weather(path: Path) -> pd.DataFrame:
    raw = pd.read_csv(path)
    df = raw.copy()
    if "DOY" not in df.columns:
        needed = {"year", "month", "day"}
        if needed.issubset(df.columns):
            dt = pd.to_datetime(df[["year", "month", "day"]])
            df["DOY"] = dt.dt.dayofyear
            df["Year"] = dt.dt.year
            df["Date"] = dt.dt.strftime("%m/%d/%Y")
        else:
            raise ValueError(f"Weather file lacks DOY and year/month/day columns: {path}")
    if "Year" not in df.columns and "year" in df.columns:
        df["Year"] = df["year"]
    for col in WEATHER_COLS:
        if col not in df.columns:
            raise ValueError(f"Weather file {path} missing required column: {col}")
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df["DOY"] = pd.to_numeric(df["DOY"], errors="coerce")
    df = df.dropna(subset=["DOY"]).copy()
    df["DOY"] = df["DOY"].astype(int)
    return df.sort_values("DOY").reset_index(drop=True)


def sequence_json(df: pd.DataFrame) -> str:
    rows = []
    for row in df.itertuples(index=False):
        rows.append(
            {
                "doy": int(row.DOY),
                "solar": round(float(row.Solar), 4),
                "tmax": round(float(getattr(row, "_2")), 4) if False else None,
            }
        )
    # itertuples mangles names with hyphens, so use records for clarity.
    records = []
    for _, row in df.iterrows():
        records.append(
            {
                "doy": int(row["DOY"]),
                "Solar": round(float(row["Solar"]), 4),
                "T-max": round(float(row["T-max"]), 4),
                "T-min": round(float(row["T-min"]), 4),
                "RelHum": round(float(row["RelHum"]), 6),
                "Precip": round(float(row["Precip"]), 6),
                "WindSpeed": round(float(row["WindSpeed"]), 4),
            }
        )
    return json.dumps(records, separators=(",", ":"))


def summarize(prefix: str, seq: pd.DataFrame) -> dict:
    out = {
        f"{prefix}_days_available": int(len(seq)),
        f"{prefix}_weather_json": sequence_json(seq),
    }
    for col in WEATHER_COLS:
        safe = col.lower().replace("-", "")
        out[f"{prefix}_{safe}_mean"] = float(seq[col].mean()) if not seq.empty else None
        out[f"{prefix}_{safe}_min"] = float(seq[col].min()) if not seq.empty else None
        out[f"{prefix}_{safe}_max"] = float(seq[col].max()) if not seq.empty else None
    out[f"{prefix}_precip_sum"] = float(seq["Precip"].sum()) if not seq.empty else None
    return out


def markdown_table(df: pd.DataFrame) -> str:
    cols = list(df.columns)
    lines = [
        "| " + " | ".join(cols) + " |",
        "| " + " | ".join(["---"] * len(cols)) + " |",
    ]
    for row in df.itertuples(index=False):
        lines.append("| " + " | ".join(str(v) for v in row) + " |")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--weather", help="Weather CSV to use, default auto-detects df_gridmet.csv")
    parser.add_argument("--history-days", type=int, default=14)
    args = parser.parse_args()

    if not COMPACT.exists():
        raise FileNotFoundError(f"Missing compact table: {COMPACT}")
    samples = pd.read_csv(COMPACT)
    dates = samples[["date_t", "decision_doy", "horizon_days"]].drop_duplicates().sort_values("decision_doy")

    weather_path = find_weather_file(args.weather)
    weather = normalize_weather(weather_path)

    rows = []
    for item in dates.itertuples(index=False):
        decision_doy = int(item.decision_doy)
        horizon_days = int(item.horizon_days)
        hist_start = decision_doy - args.history_days
        hist_end = decision_doy - 1
        future_start = decision_doy
        future_end = decision_doy + horizon_days - 1

        hist = weather[(weather["DOY"] >= hist_start) & (weather["DOY"] <= hist_end)]
        future = weather[(weather["DOY"] >= future_start) & (weather["DOY"] <= future_end)]
        rows.append(
            {
                "date_t": item.date_t,
                "decision_doy": decision_doy,
                "horizon_days": horizon_days,
                "history_days_requested": args.history_days,
                "weather_source_file": str(weather_path),
                "future_weather_source_type": "observed_proxy_pending_gefs",
                **summarize("hist", hist),
                **summarize("future", future),
            }
        )

    features = pd.DataFrame(rows)
    features.to_csv(WEATHER_OUT, index=False)

    merged = samples.merge(features, on=["date_t", "decision_doy", "horizon_days"], how="left")
    if merged["weather_source_file"].isna().any():
        raise RuntimeError("Some sample rows did not match weather features.")
    merged["history_weather_status"] = "filled_from_observed_weather_sequence"
    merged["forecast_weather_status"] = "filled_with_observed_proxy_pending_gefs"
    merged.to_csv(MERGED_OUT, index=False)

    display = features[
        [
            "date_t",
            "decision_doy",
            "hist_days_available",
            "future_days_available",
            "hist_precip_sum",
            "future_precip_sum",
            "future_tmax_mean",
            "future_tmin_mean",
        ]
    ].copy()
    report = [
        "# Short-Term Surrogate Weather Sequence Report",
        "",
        f"- Sample table: `{COMPACT}`",
        f"- Weather source: `{weather_path}`",
        f"- Output weather features: `{WEATHER_OUT}`",
        f"- Output merged table: `{MERGED_OUT}`",
        f"- History window: {args.history_days} days before decision date.",
        "- Future window: 7-day observed weather proxy. Replace with GEFS in the next version.",
        "",
        "## Window Summary",
        "",
        markdown_table(display),
        "",
    ]
    REPORT.write_text("\n".join(report), encoding="utf-8")

    print(f"Wrote {WEATHER_OUT}")
    print(f"Wrote {MERGED_OUT}")
    print(f"Wrote {REPORT}")


if __name__ == "__main__":
    main()
