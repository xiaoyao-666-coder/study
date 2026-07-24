#!/usr/bin/env python3
"""Finalize frozen causal all-variable GEFS weather for the 2015 schedule."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

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


def combine_causal_fit_gefs_history(
    calibration_c00: pd.DataFrame,
    target_weather: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, object]]:
    calibration = normalize_dates(calibration_c00)
    target = normalize_dates(target_weather)
    if "gefs_member" in calibration.columns:
        calibration = calibration.loc[calibration["gefs_member"].eq("c00")].copy()
    target = target.loc[target["gefs_member"].eq("c00")].copy()
    if calibration.empty or target.empty:
        raise ValueError("calibration and target c00 histories must both be nonempty")
    required = set(HISTORY_KEYS) | set(CANONICAL_FIELDS)
    for name, frame in (("calibration", calibration), ("target", target)):
        missing = sorted(required - set(frame.columns))
        if missing:
            raise ValueError(f"{name} c00 history is missing columns: {missing}")
        if frame[HISTORY_KEYS].duplicated().any():
            raise ValueError(f"{name} c00 history contains duplicate keys")

    first_target = pd.Timestamp(target["decision_date"].min())
    calibration_decision = pd.to_datetime(calibration["decision_date"])
    completed_before_first_target = (
        calibration_decision + pd.Timedelta(days=6) < first_target
    )
    eligible_preseason_cycles = sorted(
        calibration.loc[
            completed_before_first_target, "decision_date"
        ].unique()
    )
    if len(eligible_preseason_cycles) < 8:
        raise ValueError("fewer than eight completed c00 cycles precede the first target")

    columns = [*HISTORY_KEYS, *CANONICAL_FIELDS]
    if "gefs_member" in calibration.columns or "gefs_member" in target.columns:
        columns.insert(2, "gefs_member")
        calibration["gefs_member"] = "c00"
        target["gefs_member"] = "c00"
    combined = pd.concat(
        [calibration[columns], target[columns]], ignore_index=True
    ).sort_values(HISTORY_KEYS).reset_index(drop=True)
    if combined[HISTORY_KEYS].duplicated().any():
        raise ValueError("calibration and target c00 histories overlap on sample keys")
    values = combined[list(CANONICAL_FIELDS)].to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise ValueError("combined c00 fit history contains nonfinite values")
    if (combined["temperature_min_c"] > combined["temperature_max_c"]).any():
        raise ValueError("combined c00 fit history has Tmin above Tmax")
    if (
        combined[
            ["actual_vapor_pressure_kpa", "wind_speed_m_s", "solar_kj_m2_day"]
        ]
        < 0.0
    ).any().any():
        raise ValueError("combined c00 fit history has negative physical values")

    audit = {
        "status": "exact_schedule_2015_causal_fit_gefs_history_passed",
        "mandatory_gate_passed": True,
        "calibration_rows": int(len(calibration)),
        "target_c00_rows": int(len(target)),
        "combined_rows": int(len(combined)),
        "calibration_cycle_count": int(calibration["decision_date"].nunique()),
        "target_cycle_count": int(target["decision_date"].nunique()),
        "first_target_date": first_target.strftime("%Y-%m-%d"),
        "completed_preseason_cycle_count": len(eligible_preseason_cycles),
        "completed_preseason_cycles": eligible_preseason_cycles,
        "contains_non_c00_member": bool(
            "gefs_member" in combined.columns
            and not combined["gefs_member"].eq("c00").all()
        ),
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
    calibration = pd.read_csv(args.calibration_gefs_c00)
    combined, history_audit = combine_causal_fit_gefs_history(calibration, raw)

    args.output_root.mkdir(parents=True, exist_ok=False)
    history_dir = args.output_root / "01_causal_fit_history"
    era5_dir = args.output_root / "02_era5_reference"
    nonprecip_dir = args.output_root / "03_frozen_nonprecipitation"
    all_weather_dir = args.output_root / "04_frozen_all_variable_weather"
    history_dir.mkdir()
    combined_path = history_dir / "gefs_exact_schedule_2015_causal_fit_c00_v1.csv"
    history_audit_path = history_dir / "gefs_exact_schedule_2015_causal_fit_c00_audit_v1.json"
    combined.to_csv(combined_path, index=False)
    history_audit_path.write_text(
        json.dumps(history_audit, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    era5_outputs = run_era5_reference(
        argparse.Namespace(
            gefs_weather=combined_path,
            era5_root=args.era5_root,
            output_dir=era5_dir,
        )
    )
    nonprecip_outputs = run_nonprecipitation(
        argparse.Namespace(
            target_weather=args.target_weather,
            history_gefs_c00=combined_path,
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
            "exact_schedule_2015_frozen_all_variable_weather_pipeline_passed"
            if pipeline_passed
            else "exact_schedule_2015_frozen_all_variable_weather_pipeline_failed"
        ),
        "mandatory_pipeline_gate_passed": pipeline_passed,
        "target_year": 2015,
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
            "review_2015_corrected_weather_before_server_packaging"
            if pipeline_passed
            else "repair_failed_weather_stage"
        ),
    }
    pipeline_audit_path = args.output_root / "gefs_exact_schedule_2015_frozen_weather_pipeline_audit_v1.json"
    pipeline_audit_path.write_text(
        json.dumps(pipeline_audit, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    if not pipeline_passed:
        raise RuntimeError(f"2015 frozen weather pipeline failed; see {pipeline_audit_path}")
    return {
        "history": combined_path,
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
    parser.add_argument("--calibration-gefs-c00", type=Path, required=True)
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
