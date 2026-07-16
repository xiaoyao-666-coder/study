#!/usr/bin/env python3
"""Add complete weather windows to shortterm v1 using observed + S2S proxy.

History window:
  - Prefer df_gridmet.csv where available.
  - Fill missing historical days with weather_s2s_out.csv.

Future 7-day window:
  - Use weather_s2s_out.csv as the current forecast/proxy source.

This keeps the rolling-optimization structure close to the paper while making
the v1 sample table complete for all decision dates.
"""

from __future__ import annotations

from pathlib import Path
import argparse
import json

import pandas as pd


OUT_DIR = Path("Maize_shortterm_surrogate_v1")
COMPACT = OUT_DIR / "shortterm_surrogate_samples_v1_compact.csv"
WEATHER_OUT = OUT_DIR / "weather_sequences_by_date_v2.csv"
MERGED_OUT = OUT_DIR / "shortterm_surrogate_samples_v1_with_weather_v2.csv"
REPORT = OUT_DIR / "shortterm_surrogate_weather_sequence_v2_report.md"

WEATHER_COLS = ["Solar", "T-max", "T-min", "RelHum", "Precip", "WindSpeed"]


def pick_existing(names: list[str]) -> Path:
    for name in names:
        path = Path(name)
        if path.exists():
            return path
    raise FileNotFoundError(f"Could not find any of: {names}")


def normalize_weather(path: Path, source_name: str) -> pd.DataFrame:
    raw = pd.read_csv(path)
    df = raw.copy()
    if "DOY" not in df.columns:
        needed = {"year", "month", "day"}
        if needed.issubset(df.columns):
            dt = pd.to_datetime(df[["year", "month", "day"]])
            df["DOY"] = dt.dt.dayofyear
            df["Year"] = dt.dt.year
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
    df["weather_row_source"] = source_name
    return df.sort_values("DOY").reset_index(drop=True)


def build_history(primary: pd.DataFrame, filler: pd.DataFrame, start_doy: int, end_doy: int) -> pd.DataFrame:
    rows = []
    for doy in range(start_doy, end_doy + 1):
        p = primary[primary["DOY"] == doy]
        if not p.empty:
            rows.append(p.iloc[0])
            continue
        f = filler[filler["DOY"] == doy]
        if not f.empty:
            rows.append(f.iloc[0])
    if not rows:
        return primary.iloc[0:0].copy()
    return pd.DataFrame(rows).reset_index(drop=True)


def build_window(source: pd.DataFrame, start_doy: int, end_doy: int) -> pd.DataFrame:
    rows = []
    for doy in range(start_doy, end_doy + 1):
        sub = source[source["DOY"] == doy]
        if not sub.empty:
            rows.append(sub.iloc[0])
    if not rows:
        return source.iloc[0:0].copy()
    return pd.DataFrame(rows).reset_index(drop=True)


def sequence_json(df: pd.DataFrame) -> str:
    records = []
    for _, row in df.iterrows():
        records.append(
            {
                "doy": int(row["DOY"]),
                "source": str(row["weather_row_source"]),
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
        f"{prefix}_row_sources": "/".join(sorted(seq["weather_row_source"].unique().tolist())) if not seq.empty else "",
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
    parser.add_argument("--history-weather", default=None, help="Primary historical weather CSV, default df_gridmet.csv")
    parser.add_argument("--future-weather", default=None, help="Future forecast/proxy CSV, default weather_s2s_out.csv")
    parser.add_argument("--history-days", type=int, default=14)
    args = parser.parse_args()

    if not COMPACT.exists():
        raise FileNotFoundError(f"Missing compact table: {COMPACT}")

    hist_path = Path(args.history_weather) if args.history_weather else pick_existing(["df_gridmet.csv", "weather_gridmet_out.csv"])
    future_path = Path(args.future_weather) if args.future_weather else pick_existing(["weather_s2s_out.csv", "df_gridmet.csv"])
    if not hist_path.exists():
        raise FileNotFoundError(hist_path)
    if not future_path.exists():
        raise FileNotFoundError(future_path)

    hist_primary = normalize_weather(hist_path, f"history_primary:{hist_path.name}")
    future = normalize_weather(future_path, f"future_proxy:{future_path.name}")

    samples = pd.read_csv(COMPACT)
    dates = samples[["date_t", "decision_doy", "horizon_days"]].drop_duplicates().sort_values("decision_doy")

    rows = []
    for item in dates.itertuples(index=False):
        decision_doy = int(item.decision_doy)
        horizon_days = int(item.horizon_days)
        hist_start = decision_doy - args.history_days
        hist_end = decision_doy - 1
        future_start = decision_doy
        future_end = decision_doy + horizon_days - 1

        hist = build_history(hist_primary, future, hist_start, hist_end)
        fut = build_window(future, future_start, future_end)
        rows.append(
            {
                "date_t": item.date_t,
                "decision_doy": decision_doy,
                "horizon_days": horizon_days,
                "history_days_requested": args.history_days,
                "history_weather_primary_file": str(hist_path),
                "history_weather_fill_file": str(future_path),
                "future_weather_file": str(future_path),
                "future_weather_source_type": "s2s_proxy_pending_gefs",
                **summarize("hist", hist),
                **summarize("future", fut),
            }
        )

    features = pd.DataFrame(rows)
    features.to_csv(WEATHER_OUT, index=False)

    merged = samples.merge(features, on=["date_t", "decision_doy", "horizon_days"], how="left")
    if merged["future_weather_file"].isna().any():
        raise RuntimeError("Some sample rows did not match weather features.")
    merged["history_weather_status"] = "filled_from_gridmet_with_s2s_fill"
    merged["forecast_weather_status"] = "filled_with_s2s_proxy_pending_gefs"
    merged.to_csv(MERGED_OUT, index=False)

    display = features[
        [
            "date_t",
            "decision_doy",
            "hist_days_available",
            "future_days_available",
            "hist_row_sources",
            "future_row_sources",
            "hist_precip_sum",
            "future_precip_sum",
            "future_tmax_mean",
            "future_tmin_mean",
        ]
    ].copy()
    report = [
        "# Short-Term Surrogate Weather Sequence V2 Report",
        "",
        f"- Sample table: `{COMPACT}`",
        f"- History primary weather: `{hist_path}`",
        f"- History fill weather: `{future_path}`",
        f"- Future weather proxy: `{future_path}`",
        f"- Output weather features: `{WEATHER_OUT}`",
        f"- Output merged table: `{MERGED_OUT}`",
        f"- History window: {args.history_days} days before decision date.",
        "- Future window: 7-day S2S/proxy weather. Replace with GEFS in the next version.",
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
