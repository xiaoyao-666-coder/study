#!/usr/bin/env python3
"""Finalize frozen GEFS weather for a historical year using prior-year history."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd

from scripts.data_preparation.apply_gefs_exact_schedule_frozen_nonprecip_weather_v1 import (
    run as run_nonprecipitation,
)
from scripts.data_preparation.build_gefs_2015_2019_frozen_all_variable_weather_v1 import (
    run as run_all_variable_integration,
)
from scripts.data_preparation.extract_era5_nonprecip_reference_for_gefs_v1 import (
    CANONICAL_FIELDS,
    run as run_era5_reference,
)


HISTORY_KEYS = ["decision_date", "site_id", "local_date", "lead_day"]


def normalize_dates(frame: pd.DataFrame) -> pd.DataFrame:
    output = frame.copy()
    for column in ("decision_date", "local_date"):
        output[column] = pd.to_datetime(output[column]).dt.strftime("%Y-%m-%d")
    return output


def prepare_strict_prior_year_history(
    history_frames: Sequence[pd.DataFrame],
    target_weather: pd.DataFrame,
    *,
    minimum_samples: int,
) -> tuple[pd.DataFrame, dict[str, object]]:
    if not history_frames:
        raise ValueError("at least one prior-year history file is required")
    target = normalize_dates(target_weather)
    target_years = pd.to_datetime(target["decision_date"]).dt.year.unique()
    if len(target_years) != 1:
        raise ValueError("target weather must contain exactly one target year")
    target_year = int(target_years[0])

    parts = []
    for frame in history_frames:
        history = normalize_dates(frame)
        if "gefs_member" in history.columns:
            history = history.loc[history["gefs_member"].eq("c00")].copy()
        history["gefs_member"] = "c00"
        parts.append(history)
    combined = pd.concat(parts, ignore_index=True)

    required = set(HISTORY_KEYS) | set(CANONICAL_FIELDS) | {"gefs_member"}
    missing = sorted(required - set(combined.columns))
    if missing:
        raise ValueError(f"prior-year history is missing columns: {missing}")
    if combined.empty or combined[HISTORY_KEYS].duplicated().any():
        raise ValueError("prior-year c00 history must contain unique nonempty keys")

    history_years = pd.to_datetime(combined["decision_date"]).dt.year.astype(int)
    if (history_years >= target_year).any():
        raise ValueError("history contains target-year or future decision dates")
    values = combined[list(CANONICAL_FIELDS)].to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise ValueError("prior-year history contains nonfinite values")
    if (combined["temperature_min_c"] > combined["temperature_max_c"]).any():
        raise ValueError("prior-year history has Tmin above Tmax")
    if (
        combined[
            ["actual_vapor_pressure_kpa", "wind_speed_m_s", "solar_kj_m2_day"]
        ]
        < 0.0
    ).any().any():
        raise ValueError("prior-year history contains negative physical values")

    target_groups = target[["site_id", "lead_day"]].drop_duplicates()
    counts = (
        combined.groupby(["site_id", "lead_day"], as_index=False)
        .size()
        .rename(columns={"size": "fit_sample_count"})
    )
    coverage = target_groups.merge(
        counts, on=["site_id", "lead_day"], how="left", validate="one_to_one"
    )
    if coverage["fit_sample_count"].isna().any():
        raise ValueError("prior-year history does not cover all target site-lead groups")
    if int(coverage["fit_sample_count"].min()) < minimum_samples:
        raise ValueError("prior-year history has fewer than the minimum fit samples")

    columns = [*HISTORY_KEYS, "gefs_member", *CANONICAL_FIELDS]
    combined = combined[columns].sort_values(HISTORY_KEYS).reset_index(drop=True)
    audit = {
        "status": "exact_schedule_strict_prior_year_c00_history_passed",
        "mandatory_gate_passed": True,
        "target_year": target_year,
        "history_rows": int(len(combined)),
        "history_cycle_count": int(combined["decision_date"].nunique()),
        "history_first_year": int(history_years.min()),
        "history_last_year": int(history_years.max()),
        "site_count": int(combined["site_id"].nunique()),
        "lead_day_count": int(combined["lead_day"].nunique()),
        "minimum_fit_samples_per_site_lead": int(
            coverage["fit_sample_count"].min()
        ),
        "maximum_fit_samples_per_site_lead": int(
            coverage["fit_sample_count"].max()
        ),
        "contains_target_or_future_year": False,
        "era5_reference_extracted": False,
        "weather_correction_applied": False,
        "swap_simulation_performed": False,
        "label_generation_performed": False,
        "surrogate_training_performed": False,
        "tta_performed": False,
    }
    return combined, audit


def run(args: argparse.Namespace) -> dict[str, Path]:
    raw = pd.read_csv(args.target_weather)
    history_frames = [pd.read_csv(path) for path in args.history_weather]
    history, history_audit = prepare_strict_prior_year_history(
        history_frames,
        raw,
        minimum_samples=args.minimum_samples,
    )
    target_year = int(history_audit["target_year"])

    args.output_root.mkdir(parents=True, exist_ok=False)
    history_dir = args.output_root / "01_causal_fit_history"
    era5_dir = args.output_root / "02_era5_reference"
    nonprecip_dir = args.output_root / "03_frozen_nonprecipitation"
    all_weather_dir = args.output_root / "04_frozen_all_variable_weather"
    history_dir.mkdir()
    history_path = history_dir / f"gefs_exact_schedule_{target_year}_prior_year_fit_c00_v1.csv"
    history_audit_path = history_dir / f"gefs_exact_schedule_{target_year}_prior_year_fit_c00_audit_v1.json"
    history.to_csv(history_path, index=False)
    history_audit_path.write_text(
        json.dumps(history_audit, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    era5_outputs = run_era5_reference(
        argparse.Namespace(
            gefs_weather=history_path,
            era5_root=args.era5_root,
            output_dir=era5_dir,
        )
    )
    nonprecip_outputs = run_nonprecipitation(
        argparse.Namespace(
            target_weather=args.target_weather,
            history_gefs_c00=history_path,
            history_era5=era5_outputs["reference"],
            nonprecip_policy=args.nonprecip_policy,
            output_dir=nonprecip_dir,
            minimum_samples=args.minimum_samples,
        )
    )
    all_weather_outputs = run_all_variable_integration(
        argparse.Namespace(
            raw_weather=args.target_weather,
            nonprecip_branches=nonprecip_outputs["branches"],
            nonprecip_policy=args.nonprecip_policy,
            precipitation_cv_factors=args.precipitation_cv_factors,
            precipitation_2019_factors=args.precipitation_2019_factors,
            output_dir=all_weather_dir,
        )
    )
    final_audit = json.loads(all_weather_outputs["audit"].read_text(encoding="utf-8"))
    pipeline_passed = bool(final_audit.get("mandatory_structural_gate_passed"))
    pipeline_audit = {
        "status": (
            f"exact_schedule_{target_year}_frozen_all_variable_weather_pipeline_passed"
            if pipeline_passed
            else f"exact_schedule_{target_year}_frozen_all_variable_weather_pipeline_failed"
        ),
        "mandatory_pipeline_gate_passed": pipeline_passed,
        "target_year": target_year,
        "history_last_year": int(history_audit["history_last_year"]),
        "strict_prior_year_history": bool(
            int(history_audit["history_last_year"]) < target_year
        ),
        "final_member_rows": int(final_audit.get("member_rows", 0)),
        "final_cycle_count": int(final_audit.get("cycle_count", 0)),
        "final_site_count": int(final_audit.get("site_count", 0)),
        "final_lead_day_count": int(final_audit.get("lead_day_count", 0)),
        "target_or_future_era5_used_for_fit": False,
        "weather_correction_applied": pipeline_passed,
        "swap_simulation_performed": False,
        "label_generation_performed": False,
        "surrogate_training_performed": False,
        "tta_performed": False,
        "next_gate": (
            f"review_{target_year}_corrected_weather_before_server_packaging"
            if pipeline_passed
            else "repair_failed_weather_stage"
        ),
    }
    pipeline_audit_path = args.output_root / f"gefs_exact_schedule_{target_year}_frozen_weather_pipeline_audit_v1.json"
    pipeline_audit_path.write_text(
        json.dumps(pipeline_audit, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    if not pipeline_passed:
        raise RuntimeError(
            f"{target_year} frozen weather pipeline failed; see {pipeline_audit_path}"
        )
    return {
        "history": history_path,
        "history_audit": history_audit_path,
        "era5_reference": era5_outputs["reference"],
        "nonprecipitation": nonprecip_outputs["branches"],
        "all_variable_weather": all_weather_outputs["weather"],
        "all_variable_audit": all_weather_outputs["audit"],
        "pipeline_audit": pipeline_audit_path,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target-weather", type=Path, required=True)
    parser.add_argument("--history-weather", type=Path, nargs="+", required=True)
    parser.add_argument("--era5-root", type=Path, required=True)
    parser.add_argument("--nonprecip-policy", type=Path, required=True)
    parser.add_argument("--precipitation-cv-factors", type=Path, required=True)
    parser.add_argument("--precipitation-2019-factors", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--minimum-samples", type=int, default=8)
    return parser.parse_args()


if __name__ == "__main__":
    generated = run(parse_args())
    print(json.dumps({key: str(value) for key, value in generated.items()}, indent=2))
