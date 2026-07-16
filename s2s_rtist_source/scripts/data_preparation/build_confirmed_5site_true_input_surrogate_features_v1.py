#!/usr/bin/env python3
"""Build richer true-input surrogate features from confirmed 5-site multidate runs.

This table includes candidate irrigation, pre-decision SWAP state, soil summary,
and history/future weather windows. It is still a table-building step, not model
training.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd


RUN_ROOT = Path("site_general_surrogate_eval") / "confirmed_5site_restart_generation_smoke_v1"
OUT_DIR = Path("site_general_surrogate_eval") / "confirmed_5site_true_input_surrogate_features_v1"
DEFAULT_SITES = ["P1", "P15", "P2", "P3", "P4"]
DEFAULT_HISTORY_DAYS = 14
YEAR_LENGTH = 366

SITE_META = {
    "P1": {"code_site_id": "N1", "longitude": -98.224144, "latitude": 42.015928},
    "P2": {"code_site_id": "N2", "longitude": -88.415, "latitude": 40.595},
    "P3": {"code_site_id": "N3", "longitude": -96.877, "latitude": 46.321},
    "P4": {"code_site_id": "N4", "longitude": -94.6686, "latitude": 42.6816},
    "P15": {"code_site_id": "coord_12", "longitude": -112.265, "latitude": 41.735},
}

CRP_COLUMNS = [
    "Date", "Daynr", "Daycrp", "DVS", "TSUM", "LAIpot", "LAI", "Height", "CrpFac",
    "RootdPot", "Rootd", "PWLV", "WLV", "PWST", "WST", "PWRT", "WRT", "CPWDM",
    "CWDM", "CPWSO", "CWSO", "PGRASSDM", "GRASSDM", "PMOWDM", "MOWDM",
    "PGRAZDM", "GRAZDM", "DWLVCROP", "DWLVSOIL", "DWST", "DWRT", "DWSO",
    "HarLosOrm",
]
WEATHER_COLS = ["Solar", "T-max", "T-min", "RelHum", "Precip", "WindSpeed"]


def latest_run_dir() -> Path:
    candidates = [p for p in RUN_ROOT.iterdir() if p.is_dir()]
    if not candidates:
        raise FileNotFoundError(f"No run directories found under {RUN_ROOT}")
    return max(candidates, key=lambda p: p.stat().st_mtime)


def safe_label(date_t: str) -> str:
    return date_t.replace("-", "").lower()


def site_workspace(run_dir: Path, site: str) -> Path:
    return run_dir / site


def read_site_csv(run_dir: Path, site: str) -> pd.DataFrame:
    path = site_workspace(run_dir, site) / "site_restart_generation_smoke.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing site CSV: {path}")
    df = pd.read_csv(path)
    if "site" not in df.columns:
        df.insert(0, "site", site)
    return df


def bool_series(series: pd.Series) -> pd.Series:
    if series.dtype == bool:
        return series
    return series.astype(str).str.lower().isin(["true", "1", "yes"])


def sample_id(site: str, date_text: str, ir: float) -> str:
    date_token = date_text.replace("-", "").replace("/", "").replace(" ", "")
    ir_token = f"{int(ir):02d}" if float(ir).is_integer() else str(ir).replace(".", "p")
    return f"{site}_{date_token}_ir{ir_token}"


def site_date_id(site: str, date_text: str) -> str:
    date_token = date_text.replace("-", "").replace("/", "").replace(" ", "")
    return f"{site}_{date_token}"


def candidate_sequence(ir: float, horizon_days: int) -> str:
    return json.dumps([float(ir)] + [0.0] * max(horizon_days - 1, 0), separators=(",", ":"))


def read_last_crp(path: Path) -> dict:
    rows = []
    with path.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            if not line.startswith("2024-"):
                continue
            values = [v.strip() for v in line.rstrip("\n").split(",")]
            if len(values) == len(CRP_COLUMNS):
                rows.append(values)
    if not rows:
        raise RuntimeError(f"No crop rows found in {path}")
    df = pd.DataFrame(rows, columns=CRP_COLUMNS)
    for col in ["Daynr", "DVS", "LAI", "Rootd", "CWDM", "CWSO"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    row = df.dropna(subset=["Daynr", "DVS", "LAI", "Rootd", "CWDM", "CWSO"]).iloc[-1]
    return {
        "state_daynr": int(row["Daynr"]),
        "state_dvs": float(row["DVS"]),
        "state_lai": float(row["LAI"]),
        "state_rootd": float(row["Rootd"]),
        "state_cwdm": float(row["CWDM"]),
        "state_cwso": float(row["CWSO"]),
    }


def parse_pressure_heads(path: Path) -> pd.DataFrame:
    rows = []
    in_table = False
    with path.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            stripped = line.strip()
            if stripped == "z_h,h":
                in_table = True
                continue
            if not in_table:
                continue
            if not stripped:
                if rows:
                    break
                continue
            parts = [p.strip() for p in stripped.split(",")]
            if len(parts) != 2:
                if rows:
                    break
                continue
            try:
                rows.append({"z_cm": float(parts[0]), "h_cm": float(parts[1])})
            except ValueError:
                if rows:
                    break
    if not rows:
        raise RuntimeError(f"No pressure-head rows found in {path}")
    return pd.DataFrame(rows)


def summarize_soil(df: pd.DataFrame) -> dict:
    out = {
        "soil_layer_count": int(len(df)),
        "soil_depth_min_cm": float(df["z_cm"].min()),
        "soil_depth_max_cm": float(df["z_cm"].max()),
    }
    bands = {
        "0_30": (-30.0, 0.0),
        "30_60": (-60.0, -30.0),
        "60_100": (-100.0, -60.0),
        "0_100": (-100.0, 0.0),
    }
    for name, (lo, hi) in bands.items():
        sub = df[(df["z_cm"] >= lo) & (df["z_cm"] < hi)]
        out[f"soil_h_mean_{name}_cm"] = float(sub["h_cm"].mean()) if not sub.empty else np.nan
        out[f"soil_h_min_{name}_cm"] = float(sub["h_cm"].min()) if not sub.empty else np.nan
        out[f"soil_h_max_{name}_cm"] = float(sub["h_cm"].max()) if not sub.empty else np.nan
    return out


def normalize_weather(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "DOY" not in df.columns:
        if {"year", "month", "day"}.issubset(df.columns):
            dt = pd.to_datetime(df[["year", "month", "day"]])
            df["DOY"] = dt.dt.dayofyear
            df["Year"] = dt.dt.year
        else:
            raise ValueError(f"Weather file lacks DOY/year-month-day columns: {path}")
    for col in WEATHER_COLS:
        if col not in df.columns:
            raise ValueError(f"Weather file {path} missing {col}")
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df["DOY"] = pd.to_numeric(df["DOY"], errors="coerce").astype(int)
    return df.sort_values("DOY").reset_index(drop=True)


def build_window(source: pd.DataFrame, start_doy: int, end_doy: int) -> pd.DataFrame:
    rows = []
    for doy in range(start_doy, end_doy + 1):
        sub = source[source["DOY"] == doy]
        if not sub.empty:
            rows.append(sub.iloc[0])
    return pd.DataFrame(rows) if rows else source.iloc[0:0].copy()


def sequence_json(df: pd.DataFrame) -> str:
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


def summarize_weather(prefix: str, seq: pd.DataFrame) -> dict:
    out = {
        f"{prefix}_days_available": int(len(seq)),
        f"{prefix}_weather_json": sequence_json(seq),
    }
    for col in WEATHER_COLS:
        safe = col.lower().replace("-", "")
        out[f"{prefix}_{safe}_mean"] = float(seq[col].mean()) if not seq.empty else np.nan
        out[f"{prefix}_{safe}_min"] = float(seq[col].min()) if not seq.empty else np.nan
        out[f"{prefix}_{safe}_max"] = float(seq[col].max()) if not seq.empty else np.nan
    out[f"{prefix}_precip_sum"] = float(seq["Precip"].sum()) if not seq.empty else np.nan
    return out


def build_state_table(run_dir: Path, dates: pd.DataFrame, site: str) -> pd.DataFrame:
    rows = []
    ws = site_workspace(run_dir, site)
    for item in dates.itertuples(index=False):
        label = safe_label(str(item.date_t))
        crp_candidates = [
            ws / f"result_pre_{label}.crp",
            ws / f"restart_initial_{label}.crp",
        ]
        end_candidates = [
            ws / f"result_pre_{label}.end",
            ws / f"restart_initial_{label}.end",
        ]
        crp = next((p for p in crp_candidates if p.exists()), None)
        end = next((p for p in end_candidates if p.exists()), None)
        if end is None:
            raise FileNotFoundError(
                f"Missing pre-decision soil-state file for {site} {item.date_t}: "
                + ", ".join(str(p) for p in end_candidates)
            )
        if crp is not None:
            crop_state = read_last_crp(crp)
            state_source = str(crp)
            state_status = "filled_from_saved_pre_decision_crp"
        else:
            crop_state = {
                "state_daynr": np.nan,
                "state_dvs": np.nan,
                "state_lai": np.nan,
                "state_rootd": np.nan,
                "state_cwdm": np.nan,
                "state_cwso": np.nan,
            }
            state_source = ""
            state_status = "missing_pre_decision_crp"
        soil = summarize_soil(parse_pressure_heads(end))
        rows.append(
            {
                "site_id": site,
                "date_t": item.date_t,
                "decision_doy": int(item.decision_doy),
                "pre_end_doy": int(item.decision_doy) - 1,
                "state_source": state_source,
                "state_source_status": state_status,
                "soil_state_source": str(end),
                **crop_state,
                **soil,
            }
        )
    return pd.DataFrame(rows)


def build_weather_table(run_dir: Path, dates: pd.DataFrame, site: str, history_days: int) -> pd.DataFrame:
    ws = site_workspace(run_dir, site)
    weather = normalize_weather(ws / "df_gridmet.csv")
    rows = []
    for item in dates.itertuples(index=False):
        decision_doy = int(item.decision_doy)
        horizon_days = int(item.horizon_days)
        hist = build_window(weather, decision_doy - history_days, decision_doy - 1)
        fut = build_window(weather, decision_doy, decision_doy + horizon_days - 1)
        rows.append(
            {
                "site_id": site,
                "date_t": item.date_t,
                "decision_doy": decision_doy,
                "horizon_days": horizon_days,
                "history_days_requested": history_days,
                "weather_source": str(ws / "df_gridmet.csv"),
                **summarize_weather("hist", hist),
                **summarize_weather("future", fut),
            }
        )
    return pd.DataFrame(rows)


def target_signature(group: pd.DataFrame, digits: int) -> str:
    ordered = group.sort_values("candidate_ir")
    target_col = "target_7d" if "target_7d" in ordered.columns else "target_value"
    return "|".join(
        f"{float(row['candidate_ir']):g}:{round(float(row[target_col]), digits):g}"
        for _, row in ordered.iterrows()
    )


def markdown_table(df: pd.DataFrame) -> str:
    if df.empty:
        return "_No rows._"
    cols = list(df.columns)
    rows = ["| " + " | ".join(cols) + " |", "| " + " | ".join(["---"] * len(cols)) + " |"]
    for row in df.itertuples(index=False):
        rows.append("| " + " | ".join(str(v) for v in row) + " |")
    return "\n".join(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", default=None, help="Path to confirmed multidate restart smoke run. Defaults to latest.")
    parser.add_argument("--sites", nargs="+", default=DEFAULT_SITES)
    parser.add_argument("--history-days", type=int, default=DEFAULT_HISTORY_DAYS)
    parser.add_argument("--output-dir", default=str(OUT_DIR))
    parser.add_argument("--round-digits", type=int, default=4)
    parser.add_argument("--tolerance", type=float, default=1e-9)
    args = parser.parse_args()

    run_dir = Path(args.run_dir) if args.run_dir else latest_run_dir()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    sample_parts = []
    state_parts = []
    weather_parts = []
    site_date_rows = []

    for site in args.sites:
        raw = read_site_csv(run_dir, site)
        for col in ["decision_doy", "horizon_end_doy", "ir", "end_daynr", "dvs", "lai", "rootd", "cwdm_value", "cwso_value", "target_value", "best_ir_for_date", "best_target_for_date"]:
            if col in raw.columns:
                raw[col] = pd.to_numeric(raw[col], errors="coerce")
        raw["is_best_ir"] = bool_series(raw["is_best_ir"])

        site_dates = raw[["date_t", "decision_doy", "horizon_end_doy"]].drop_duplicates().sort_values("decision_doy")
        site_dates["horizon_days"] = site_dates["horizon_end_doy"] - site_dates["decision_doy"]

        state = build_state_table(run_dir, site_dates[["date_t", "decision_doy"]], site)
        weather = build_weather_table(run_dir, site_dates, site, args.history_days)
        state_parts.append(state)
        weather_parts.append(weather)

        out = raw.copy()
        meta = SITE_META[site]
        out["site_id"] = site
        out["paper_site_id"] = site
        out["site_date_id"] = [site_date_id(site, str(r.date_t)) for r in out.itertuples(index=False)]
        out["code_site_id"] = meta["code_site_id"]
        out["longitude"] = meta["longitude"]
        out["latitude"] = meta["latitude"]
        out["sample_id"] = [sample_id(site, str(r.date_t), float(r.ir)) for r in out.itertuples(index=False)]
        out["horizon_days"] = out["horizon_end_doy"] - out["decision_doy"]
        out["decision_doy_sin"] = out["decision_doy"].map(lambda x: math.sin(2 * math.pi * x / YEAR_LENGTH))
        out["decision_doy_cos"] = out["decision_doy"].map(lambda x: math.cos(2 * math.pi * x / YEAR_LENGTH))
        out["candidate_ir"] = out["ir"].astype(float)
        out["candidate_ir_sq"] = out["candidate_ir"] ** 2
        out["is_zero_ir"] = (out["candidate_ir"] == 0.0).astype(int)
        out["candidate_ir_sequence"] = [
            candidate_sequence(float(r.candidate_ir), int(r.horizon_days)) for r in out.itertuples(index=False)
        ]
        out["dvs_7d"] = out["dvs"].astype(float)
        out["lai_7d"] = out["lai"].astype(float)
        out["rootd_7d"] = out["rootd"].astype(float)
        out["cwdm_7d"] = out["cwdm_value"].astype(float)
        out["cwso_7d"] = out["cwso_value"].astype(float)
        out["target_7d"] = out["target_value"].astype(float)
        no_ir = out[out["candidate_ir"] == 0.0][["site_date_id", "target_value"]].rename(
            columns={"target_value": "no_irrigation_target_7d"}
        )
        out = out.merge(no_ir, on="site_date_id", how="left")
        out["net_gain_7d"] = out["target_7d"] - out["no_irrigation_target_7d"]
        out["target_regret"] = out["best_target_for_date"] - out["target_7d"]
        out["best_ir_gap"] = out["candidate_ir"] - out["best_ir_for_date"]
        sample_parts.append(out)

        site_date = (
            out.groupby(["site_date_id", "site_id", "date_t"])
            .agg(
                decision_doy=("decision_doy", "first"),
                horizon_days=("horizon_days", "first"),
                n_candidates=("candidate_ir", "count"),
                best_ir_for_date=("best_ir_for_date", "first"),
                best_target_for_date=("best_target_for_date", "first"),
                min_target_7d=("target_value", "min"),
                max_target_7d=("target_value", "max"),
                target_range_7d=("target_value", lambda s: float(s.max() - s.min())),
                no_irrigation_target_7d=("no_irrigation_target_7d", "first"),
                state_dvs=("dvs", "first"),
                state_lai=("lai", "first"),
                state_rootd=("rootd", "first"),
                state_cwdm=("cwdm_value", "first"),
                state_cwso=("cwso_value", "first"),
            )
            .reset_index()
        )
        site_date["target_collapse"] = (
            (site_date["best_target_for_date"] <= args.tolerance)
            & (site_date["min_target_7d"] <= args.tolerance)
            & (site_date["target_range_7d"] > args.tolerance)
        )
        site_date["target_curve_signature"] = site_date["site_date_id"].map(
            lambda sid: target_signature(out[out["site_date_id"] == sid], args.round_digits)
        )
        site_date_rows.append(site_date)

    samples = pd.concat(sample_parts, ignore_index=True)
    state_df = pd.concat(state_parts, ignore_index=True)
    weather_df = pd.concat(weather_parts, ignore_index=True)
    site_date = pd.concat(site_date_rows, ignore_index=True)

    dup_counts = (
        site_date.groupby(["date_t", "target_curve_signature"])
        .size()
        .reset_index(name="same_date_duplicate_target_signature_count")
    )
    site_date = site_date.merge(dup_counts, on=["date_t", "target_curve_signature"], how="left")
    site_date["same_date_duplicate_target_curve"] = site_date["same_date_duplicate_target_signature_count"] > 1

    samples = samples.merge(
        state_df,
        on=["site_id", "date_t", "decision_doy"],
        how="left",
        suffixes=("", "_state"),
    )
    samples = samples.merge(
        weather_df,
        on=["site_id", "date_t", "decision_doy", "horizon_days"],
        how="left",
        suffixes=("", "_weather"),
    )
    samples = samples.merge(
        site_date[
            [
                "site_date_id",
                "target_collapse",
                "same_date_duplicate_target_curve",
                "same_date_duplicate_target_signature_count",
                "target_curve_signature",
            ]
        ],
        on="site_date_id",
        how="left",
    )
    samples["training_weight_suggested"] = samples["target_collapse"].map(lambda v: 0.5 if bool(v) else 1.0)
    samples["feature_ready_status"] = "rich_true_input_features_built"

    labels = (
        samples[samples["is_best_ir"]][
            [
                "sample_id",
                "site_date_id",
                "site_id",
                "date_t",
                "decision_doy",
                "horizon_days",
                "best_ir_for_date",
                "best_target_for_date",
                "target_collapse",
                "same_date_duplicate_target_curve",
            ]
        ]
        .sort_values(["site_id", "decision_doy"])
        .reset_index(drop=True)
    )

    samples_path = out_dir / "confirmed_5site_true_input_surrogate_features_samples_v1.csv"
    labels_path = out_dir / "confirmed_5site_true_input_surrogate_features_labels_v1.csv"
    site_date_path = out_dir / "confirmed_5site_true_input_surrogate_features_site_date_v1.csv"
    state_path = out_dir / "confirmed_5site_true_input_surrogate_features_state_v1.csv"
    weather_path = out_dir / "confirmed_5site_true_input_surrogate_features_weather_v1.csv"
    report_path = out_dir / "confirmed_5site_true_input_surrogate_features_v1.md"

    samples.to_csv(samples_path, index=False)
    labels.to_csv(labels_path, index=False)
    site_date.to_csv(site_date_path, index=False)
    state_df.to_csv(state_path, index=False)
    weather_df.to_csv(weather_path, index=False)

    summary = pd.DataFrame(
        [
            {
                "run_dir": str(run_dir),
                "samples_rows": len(samples),
                "label_rows": len(labels),
                "n_sites": samples["site_id"].nunique(),
                "n_dates": samples["date_t"].nunique(),
                "n_site_dates": samples["site_date_id"].nunique(),
                "n_ir_values": samples["candidate_ir"].nunique(),
                "target_collapse_site_dates": int(site_date["target_collapse"].sum()),
                "same_date_duplicate_target_site_dates": int(site_date["same_date_duplicate_target_curve"].sum()),
            }
        ]
    )

    lines = [
        "# Confirmed 5-Site True-Input Surrogate Features V1",
        "",
        "## Summary",
        "",
        markdown_table(summary),
        "",
        "## Best Labels",
        "",
        markdown_table(labels),
        "",
        "## Site-Date Summary",
        "",
        markdown_table(site_date),
        "",
        "## Outputs",
        "",
        f"- `{samples_path}`",
        f"- `{labels_path}`",
        f"- `{site_date_path}`",
        f"- `{state_path}`",
        f"- `{weather_path}`",
    ]
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print("Confirmed 5-site true-input surrogate features v1")
    print(f"run_dir: {run_dir}")
    print(f"samples: {samples_path}")
    print(f"labels: {labels_path}")
    print(f"site_date: {site_date_path}")
    print(f"state: {state_path}")
    print(f"weather: {weather_path}")
    print(f"report: {report_path}")
    print(summary.to_string(index=False))
    print("")
    print(labels.to_string(index=False))


if __name__ == "__main__":
    main()
