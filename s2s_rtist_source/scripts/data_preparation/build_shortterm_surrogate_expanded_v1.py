#!/usr/bin/env python3
"""Build expanded short-term surrogate table from expanded restart dataset."""

from __future__ import annotations

from pathlib import Path
import argparse
import json
import re

import numpy as np
import pandas as pd


CRP_COLUMNS = [
    "Date", "Daynr", "Daycrp", "DVS", "TSUM", "LAIpot", "LAI", "Height", "CrpFac",
    "RootdPot", "Rootd", "PWLV", "WLV", "PWST", "WST", "PWRT", "WRT", "CPWDM",
    "CWDM", "CPWSO", "CWSO", "PGRASSDM", "GRASSDM", "PMOWDM", "MOWDM",
    "PGRAZDM", "GRAZDM", "DWLVCROP", "DWLVSOIL", "DWST", "DWRT", "DWSO",
    "HarLosOrm",
]
WEATHER_COLS = ["Solar", "T-max", "T-min", "RelHum", "Precip", "WindSpeed"]


def safe_label(date_t: str) -> str:
    return date_t.replace("-", "").lower()


def sample_id(date_text: str, ir: float, site_id: str) -> str:
    token = date_text.replace("-", "").replace("/", "").replace(" ", "")
    ir_token = f"{int(ir):02d}" if float(ir).is_integer() else str(ir).replace(".", "p")
    return f"{site_id}_{token}_ir{ir_token}"


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


def normalize_weather(path: Path, source_name: str) -> pd.DataFrame:
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
    return pd.DataFrame(rows) if rows else primary.iloc[0:0].copy()


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
        records.append({
            "doy": int(row["DOY"]),
            "source": str(row["weather_row_source"]),
            "Solar": round(float(row["Solar"]), 4),
            "T-max": round(float(row["T-max"]), 4),
            "T-min": round(float(row["T-min"]), 4),
            "RelHum": round(float(row["RelHum"]), 6),
            "Precip": round(float(row["Precip"]), 6),
            "WindSpeed": round(float(row["WindSpeed"]), 4),
        })
    return json.dumps(records, separators=(",", ":"))


def summarize_weather(prefix: str, seq: pd.DataFrame) -> dict:
    out = {
        f"{prefix}_days_available": int(len(seq)),
        f"{prefix}_weather_json": sequence_json(seq),
        f"{prefix}_row_sources": "/".join(sorted(seq["weather_row_source"].unique().tolist())) if not seq.empty else "",
    }
    for col in WEATHER_COLS:
        safe = col.lower().replace("-", "")
        out[f"{prefix}_{safe}_mean"] = float(seq[col].mean()) if not seq.empty else np.nan
        out[f"{prefix}_{safe}_min"] = float(seq[col].min()) if not seq.empty else np.nan
        out[f"{prefix}_{safe}_max"] = float(seq[col].max()) if not seq.empty else np.nan
    out[f"{prefix}_precip_sum"] = float(seq["Precip"].sum()) if not seq.empty else np.nan
    return out


def build_samples(restart: pd.DataFrame, site_id: str) -> pd.DataFrame:
    work = restart.copy()
    work["horizon_days"] = work["horizon_end_doy"] - work["decision_doy"]
    no_ir = (
        work[work["ir"].astype(float) == 0.0][["date_t", "target_value"]]
        .rename(columns={"target_value": "no_irrigation_target_7d"})
    )
    work = work.merge(no_ir, on="date_t", how="left")
    out = pd.DataFrame()
    out["sample_id"] = [sample_id(str(r.date_t), float(r.ir), site_id) for r in work.itertuples(index=False)]
    out["site_id"] = site_id
    out["date_t"] = work["date_t"]
    out["decision_doy"] = work["decision_doy"].astype(int)
    out["horizon_days"] = work["horizon_days"].astype(int)
    out["candidate_ir"] = work["ir"].astype(float)
    out["candidate_ir_sequence"] = [candidate_sequence(float(r.ir), int(r.horizon_days)) for r in work.itertuples(index=False)]
    out["end_daynr"] = work["end_daynr"].astype(int)
    out["dvs_7d"] = work["dvs"]
    out["lai_7d"] = work["lai"]
    out["rootd_7d"] = work["rootd"]
    out["cwdm_7d"] = work["cwdm_value"]
    out["cwso_7d"] = work["cwso_value"]
    out["target_7d"] = work["target_value"]
    out["no_irrigation_target_7d"] = work["no_irrigation_target_7d"]
    out["net_gain_7d"] = out["target_7d"] - out["no_irrigation_target_7d"]
    out["best_ir_for_date"] = work["best_ir_for_date"]
    out["best_target_for_date"] = work["best_target_for_date"]
    out["is_best_ir"] = work["is_best_ir"].astype(bool)
    out["target_regret"] = out["best_target_for_date"] - out["target_7d"]
    return out


def build_state_table(dates: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for item in dates.itertuples(index=False):
        label = safe_label(str(item.date_t))
        crp = Path(f"result_pre_{label}.crp")
        end = Path(f"result_pre_{label}.end")
        if not crp.exists() or not end.exists():
            raise FileNotFoundError(f"Missing pre-decision files for {item.date_t}: {crp}, {end}")
        soil = summarize_soil(parse_pressure_heads(end))
        rows.append({
            "date_t": item.date_t,
            "decision_doy": int(item.decision_doy),
            "pre_end_doy": int(item.decision_doy) - 1,
            "state_source": str(crp),
            "soil_state_source": str(end),
            **read_last_crp(crp),
            **soil,
        })
    return pd.DataFrame(rows)


def build_weather_table(dates: pd.DataFrame, history_weather: Path, future_weather: Path, history_days: int) -> pd.DataFrame:
    hist_primary = normalize_weather(history_weather, f"history_primary:{history_weather.name}")
    future = normalize_weather(future_weather, f"future_proxy:{future_weather.name}")
    rows = []
    for item in dates.itertuples(index=False):
        decision_doy = int(item.decision_doy)
        horizon_days = int(item.horizon_days)
        hist = build_history(hist_primary, future, decision_doy - history_days, decision_doy - 1)
        fut = build_window(future, decision_doy, decision_doy + horizon_days - 1)
        rows.append({
            "date_t": item.date_t,
            "decision_doy": decision_doy,
            "horizon_days": horizon_days,
            "history_days_requested": history_days,
            "history_weather_primary_file": str(history_weather),
            "history_weather_fill_file": str(future_weather),
            "future_weather_file": str(future_weather),
            "future_weather_source_type": "s2s_proxy_pending_gefs",
            **summarize_weather("hist", hist),
            **summarize_weather("future", fut),
        })
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="restart_decision_dataset_expanded.csv")
    parser.add_argument("--output-dir", default="Maize_shortterm_surrogate_expanded_v1")
    parser.add_argument("--site-id", default="maize_test_site")
    parser.add_argument("--history-weather", default="df_gridmet.csv")
    parser.add_argument("--future-weather", default="weather_s2s_out.csv")
    parser.add_argument("--history-days", type=int, default=14)
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(exist_ok=True)
    restart = pd.read_csv(args.input)
    samples = build_samples(restart, args.site_id)
    dates = samples[["date_t", "decision_doy", "horizon_days"]].drop_duplicates().sort_values("decision_doy")
    state = build_state_table(dates[["date_t", "decision_doy"]].drop_duplicates())
    weather = build_weather_table(dates, Path(args.history_weather), Path(args.future_weather), args.history_days)

    full = samples.merge(state, on=["date_t", "decision_doy"], how="left")
    full = full.merge(weather, on=["date_t", "decision_doy", "horizon_days"], how="left")
    full["current_state_status"] = "filled_from_true_pre_decision_state"
    full["history_weather_status"] = "filled_from_gridmet_with_s2s_fill"
    full["forecast_weather_status"] = "filled_with_s2s_proxy_pending_gefs"

    samples_path = out_dir / "shortterm_surrogate_expanded_samples_v1.csv"
    state_path = out_dir / "current_state_soil_by_date_expanded_v1.csv"
    weather_path = out_dir / "weather_sequences_by_date_expanded_v1.csv"
    best_path = out_dir / "shortterm_surrogate_expanded_labels_v1.csv"
    full.to_csv(samples_path, index=False)
    state.to_csv(state_path, index=False)
    weather.to_csv(weather_path, index=False)
    full.loc[full["is_best_ir"], ["site_id", "date_t", "decision_doy", "horizon_days", "best_ir_for_date", "best_target_for_date"]].to_csv(best_path, index=False)

    print(f"wrote {samples_path}", flush=True)
    print(f"wrote {state_path}", flush=True)
    print(f"wrote {weather_path}", flush=True)
    print(f"wrote {best_path}", flush=True)
    print(f"rows: {len(full)}", flush=True)
    print(f"decision dates: {full['date_t'].nunique()}", flush=True)
    print("best irrigation by date:", flush=True)
    print(full.loc[full["is_best_ir"], ["date_t", "best_ir_for_date", "best_target_for_date"]].drop_duplicates().to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
