#!/usr/bin/env python3
"""Build 12-site continuous-irrigation surrogate features from SWAP outputs.

This is the table-building step after the 10k 12-site SWAP generation. It does
not train a model. It creates candidate-level samples for a site-general
continuous-irrigation surrogate:

- static site attributes from site_feature_screening_12_code_sites.csv
- pre-decision crop and soil state from saved restart files
- historical and future gridMET weather windows
- candidate irrigation amount, site-specific feasible max, and ir_fraction
- 7-day SWAP target and decision labels
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

from build_confirmed_5site_true_input_surrogate_features_v1 import (
    YEAR_LENGTH,
    bool_series,
    build_window,
    candidate_sequence,
    markdown_table,
    normalize_weather,
    parse_pressure_heads,
    read_last_crp,
    safe_label,
    summarize_soil,
    summarize_weather,
    target_signature,
)


OUT_DIR = Path("site_general_surrogate_eval")
DEFAULT_SITE_FEATURE_CSV = OUT_DIR / "site_feature_screening_12_code_sites.csv"
DEFAULT_RUN_DIR = (
    OUT_DIR
    / "continuous_ir_12site_restart_generation_v1"
    / "continuous_ir_12site_10k_sitecap27p5_v1"
)
DEFAULT_OUTPUT_DIR = OUT_DIR / "continuous_ir_12site_10k_surrogate_features_v1"
DEFAULT_HISTORY_DAYS = 14


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", default=str(DEFAULT_RUN_DIR))
    parser.add_argument("--site-feature-csv", default=str(DEFAULT_SITE_FEATURE_CSV))
    parser.add_argument("--sites", nargs="+", help="Optional subset of site ids.")
    parser.add_argument("--history-days", type=int, default=DEFAULT_HISTORY_DAYS)
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--round-digits", type=int, default=4)
    parser.add_argument("--tolerance", type=float, default=1e-9)
    return parser.parse_args()


def sample_id(site: str, date_text: str, ir: float) -> str:
    date_token = date_text.replace("-", "").replace("/", "").replace(" ", "").lower()
    ir_token = f"{float(ir):.6g}".replace("-", "m").replace(".", "p")
    return f"{site}_{date_token}_ir{ir_token}"


def site_date_id(site: str, date_text: str) -> str:
    date_token = date_text.replace("-", "").replace("/", "").replace(" ", "").lower()
    return f"{site}_{date_token}"


def read_site_features(path: Path, requested: list[str] | None) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing site feature CSV: {path}")
    df = pd.read_csv(path)
    if "site" not in df.columns:
        raise ValueError(f"{path} must contain a site column")
    if requested:
        requested_set = set(requested)
        df = df[df["site"].astype(str).isin(requested_set)].copy()
        missing = sorted(requested_set.difference(set(df["site"].astype(str))))
        if missing:
            raise ValueError(f"Requested sites not found in feature CSV: {missing}")
    return df.reset_index(drop=True)


def read_site_csv(run_dir: Path, site: str) -> pd.DataFrame:
    path = run_dir / site / "site_restart_generation_smoke.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing site SWAP output: {path}")
    df = pd.read_csv(path)
    if "site" not in df.columns:
        df.insert(0, "site", site)
    return df


def site_workspace(run_dir: Path, site: str) -> Path:
    return run_dir / site


def build_state_table(run_dir: Path, site_dates: pd.DataFrame, site: str) -> pd.DataFrame:
    rows = []
    ws = site_workspace(run_dir, site)
    for item in site_dates.itertuples(index=False):
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


def build_weather_table(run_dir: Path, site_dates: pd.DataFrame, site: str, history_days: int) -> pd.DataFrame:
    ws = site_workspace(run_dir, site)
    weather = normalize_weather(ws / "df_gridmet.csv")
    rows = []
    for item in site_dates.itertuples(index=False):
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


def static_feature_dict(row: pd.Series) -> dict[str, object]:
    out: dict[str, object] = {
        "site_id": str(row["site"]),
        "code_site_id": str(row["site"]).replace("code_", ""),
        "longitude": float(row["lon"]),
        "latitude": float(row["lat"]),
    }
    for col, value in row.items():
        if col == "site":
            continue
        safe = col.replace("-", "_")
        out[f"static_{safe}"] = value
    return out


def add_static_columns(samples: pd.DataFrame, static_df: pd.DataFrame) -> pd.DataFrame:
    static_rows = [static_feature_dict(row) for _, row in static_df.iterrows()]
    static = pd.DataFrame(static_rows)
    return samples.merge(static, on="site_id", how="left")


def build_samples_for_site(
    *,
    run_dir: Path,
    site: str,
    site_feature_row: pd.Series,
    history_days: int,
    tolerance: float,
    round_digits: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    raw = read_site_csv(run_dir, site)
    numeric_cols = [
        "decision_doy",
        "horizon_end_doy",
        "ir",
        "end_daynr",
        "dvs",
        "lai",
        "rootd",
        "cwdm_value",
        "cwso_value",
        "target_value",
        "best_ir_for_date",
        "best_target_for_date",
    ]
    for col in numeric_cols:
        if col in raw.columns:
            raw[col] = pd.to_numeric(raw[col], errors="coerce")
    raw["is_best_ir"] = bool_series(raw["is_best_ir"])

    site_dates = raw[["date_t", "decision_doy", "horizon_end_doy"]].drop_duplicates().sort_values("decision_doy")
    site_dates["horizon_days"] = site_dates["horizon_end_doy"] - site_dates["decision_doy"]

    state = build_state_table(run_dir, site_dates[["date_t", "decision_doy"]], site)
    weather = build_weather_table(run_dir, site_dates, site, history_days)

    out = raw.copy()
    out["site_id"] = site
    out["site_date_id"] = [site_date_id(site, str(r.date_t)) for r in out.itertuples(index=False)]
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

    site_ir_max = float(out["candidate_ir"].max())
    out["site_ir_min"] = float(out["candidate_ir"].min())
    out["site_ir_max"] = site_ir_max
    out["candidate_ir_fraction"] = out["candidate_ir"] / site_ir_max if site_ir_max > 0 else 0.0
    out["candidate_ir_fraction_sq"] = out["candidate_ir_fraction"] ** 2

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
    out["best_ir_fraction_for_date"] = out["best_ir_for_date"] / site_ir_max if site_ir_max > 0 else 0.0
    out["best_ir_fraction_gap"] = out["candidate_ir_fraction"] - out["best_ir_fraction_for_date"]

    site_date = (
        out.groupby(["site_date_id", "site_id", "date_t"])
        .agg(
            decision_doy=("decision_doy", "first"),
            horizon_days=("horizon_days", "first"),
            n_candidates=("candidate_ir", "count"),
            site_ir_max=("site_ir_max", "first"),
            best_ir_for_date=("best_ir_for_date", "first"),
            best_ir_fraction_for_date=("best_ir_fraction_for_date", "first"),
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
        (site_date["best_target_for_date"] <= tolerance)
        & (site_date["min_target_7d"] <= tolerance)
        & (site_date["target_range_7d"] > tolerance)
    )
    site_date["target_curve_signature"] = site_date["site_date_id"].map(
        lambda sid: target_signature(out[out["site_date_id"] == sid], round_digits)
    )
    site_date["site_feature_source"] = str(site_feature_row.get("site", site))
    return out, state, weather, site_date


def write_outputs(
    *,
    out_dir: Path,
    run_dir: Path,
    samples: pd.DataFrame,
    labels: pd.DataFrame,
    site_date: pd.DataFrame,
    state_df: pd.DataFrame,
    weather_df: pd.DataFrame,
) -> None:
    samples_path = out_dir / "continuous_ir_12site_surrogate_features_samples_v1.csv"
    labels_path = out_dir / "continuous_ir_12site_surrogate_features_labels_v1.csv"
    site_date_path = out_dir / "continuous_ir_12site_surrogate_features_site_date_v1.csv"
    state_path = out_dir / "continuous_ir_12site_surrogate_features_state_v1.csv"
    weather_path = out_dir / "continuous_ir_12site_surrogate_features_weather_v1.csv"
    report_path = out_dir / "continuous_ir_12site_surrogate_features_v1.md"

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
                "n_site_dates": samples["site_date_id"].nunique(),
                "n_dates": samples["date_t"].nunique(),
                "min_candidates_per_site_date": int(site_date["n_candidates"].min()),
                "max_candidates_per_site_date": int(site_date["n_candidates"].max()),
                "target_collapse_site_dates": int(site_date["target_collapse"].sum()),
                "same_date_duplicate_target_site_dates": int(site_date["same_date_duplicate_target_curve"].sum()),
            }
        ]
    )
    by_site = (
        samples.groupby("site_id")
        .agg(
            samples=("sample_id", "count"),
            site_dates=("site_date_id", "nunique"),
            ir_min=("candidate_ir", "min"),
            ir_max=("candidate_ir", "max"),
            best_ir_mean=("best_ir_for_date", "mean"),
        )
        .reset_index()
    )

    lines = [
        "# Continuous Irrigation 12-Site Surrogate Features V1",
        "",
        "## Summary",
        "",
        markdown_table(summary),
        "",
        "## By Site",
        "",
        markdown_table(by_site),
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

    print("Continuous irrigation 12-site surrogate features v1")
    print(f"run_dir: {run_dir}")
    print(f"samples: {samples_path}")
    print(f"labels: {labels_path}")
    print(f"site_date: {site_date_path}")
    print(f"state: {state_path}")
    print(f"weather: {weather_path}")
    print(f"report: {report_path}")
    print(summary.to_string(index=False))
    print("")
    print(by_site.to_string(index=False))


def main() -> None:
    args = parse_args()
    run_dir = Path(args.run_dir)
    site_feature_csv = Path(args.site_feature_csv)
    out_dir = Path(args.output_dir)
    if not run_dir.exists():
        raise FileNotFoundError(f"Missing run dir: {run_dir}")
    out_dir.mkdir(parents=True, exist_ok=True)

    site_features = read_site_features(site_feature_csv, args.sites)
    sample_parts = []
    state_parts = []
    weather_parts = []
    site_date_parts = []

    for _, site_row in site_features.iterrows():
        site = str(site_row["site"])
        samples, state, weather, site_date = build_samples_for_site(
            run_dir=run_dir,
            site=site,
            site_feature_row=site_row,
            history_days=args.history_days,
            tolerance=args.tolerance,
            round_digits=args.round_digits,
        )
        sample_parts.append(samples)
        state_parts.append(state)
        weather_parts.append(weather)
        site_date_parts.append(site_date)

    samples = pd.concat(sample_parts, ignore_index=True)
    state_df = pd.concat(state_parts, ignore_index=True)
    weather_df = pd.concat(weather_parts, ignore_index=True)
    site_date = pd.concat(site_date_parts, ignore_index=True)

    samples = add_static_columns(samples, site_features)

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
    samples["feature_ready_status"] = "continuous_ir_12site_true_input_features_built"

    labels = (
        samples[samples["is_best_ir"]][
            [
                "sample_id",
                "site_date_id",
                "site_id",
                "date_t",
                "decision_doy",
                "horizon_days",
                "site_ir_max",
                "best_ir_for_date",
                "best_ir_fraction_for_date",
                "best_target_for_date",
                "target_collapse",
                "same_date_duplicate_target_curve",
            ]
        ]
        .sort_values(["site_id", "decision_doy"])
        .reset_index(drop=True)
    )

    write_outputs(
        out_dir=out_dir,
        run_dir=run_dir,
        samples=samples,
        labels=labels,
        site_date=site_date,
        state_df=state_df,
        weather_df=weather_df,
    )


if __name__ == "__main__":
    main()
