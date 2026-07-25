#!/usr/bin/env python3
"""Build the leakage-safe surrogate table from frozen formal GEFS labels."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from s2s_rtist.pipelines.season_decision_schedule import read_crop_trajectory


IRRIGATION_GRID_MM = [0.0, 10.0, 15.0, 20.0, 25.0, 30.0, 40.0, 60.0]
EXPECTED_GEFS_MEMBERS = ("c00", "p01", "p02", "p03", "p04")
WEATHER_VARIABLES = [
    "precipitation_mm",
    "temperature_min_c",
    "temperature_max_c",
    "actual_vapor_pressure_kpa",
    "wind_speed_m_s",
    "solar_kj_m2_day",
]
CHECKPOINT_FEATURES = {
    "DVS": "predecision_dvs",
    "LAI": "predecision_lai",
    "Rootd": "predecision_crop_root_depth_cm",
    "CWDM": "predecision_cwdm_kg_ha",
    "CWSO": "predecision_cwso_kg_ha",
}
TARGET_COLUMNS = [
    "target_net_gain_7d",
    "target_aet_7d_mm",
    *[f"target_soil_vwc_0_100cm_day{day:02d}" for day in range(1, 8)],
]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def crop_state_from_row(row: pd.Series) -> dict[str, Any]:
    numeric = pd.to_numeric(row[list(CHECKPOINT_FEATURES)], errors="coerce")
    if numeric.isna().any() or not np.isfinite(numeric.to_numpy()).all():
        raise ValueError("crop state contains missing or nonfinite feature values")
    crop_date = row["Date"] if "Date" in row.index else row.name
    return {
        "checkpoint_crop_date": pd.Timestamp(crop_date).strftime("%Y-%m-%d"),
        **{
            output: float(numeric[source])
            for source, output in CHECKPOINT_FEATURES.items()
        },
    }


def parse_checkpoint_crop(
    path: Path, checkpoint_date: str | None = None
) -> dict[str, Any]:
    frame = read_crop_trajectory(path)
    if checkpoint_date is None:
        row = frame.iloc[-1]
    else:
        expected = pd.Timestamp(checkpoint_date)
        selected = frame.loc[frame["Date"].eq(expected)]
        if len(selected) != 1:
            raise ValueError(
                f"crop trajectory must contain one row for {expected.date()}: {path}"
            )
        row = selected.iloc[0]
    return crop_state_from_row(row)


def collect_checkpoint_states(
    plan: pd.DataFrame, formal_generation_dir: Path
) -> pd.DataFrame:
    required = {"target_year", "site_id", "decision_date", "checkpoint_date"}
    if missing := sorted(required - set(plan.columns)):
        raise ValueError(f"formal plan is missing checkpoint fields: {missing}")
    normalized = plan.copy()
    normalized["target_year"] = pd.to_numeric(
        normalized["target_year"], errors="raise"
    ).astype(int)
    normalized["site_id"] = normalized["site_id"].astype(str)
    for column in ("decision_date", "checkpoint_date"):
        normalized[column] = pd.to_datetime(
            normalized[column], errors="raise"
        ).dt.strftime("%Y-%m-%d")

    rows: list[dict[str, Any]] = []
    for (year, site), unit_plan in normalized.groupby(["target_year", "site_id"]):
        unit_root = formal_generation_dir / "units" / f"Y{year}" / site
        setup_audit_path = unit_root / "setup_audit_v1.json"
        setup_audit = json.loads(setup_audit_path.read_text(encoding="utf-8"))
        if not bool(setup_audit.get("mandatory_gate_passed")):
            raise ValueError(f"site-year setup audit did not pass: {unit_root}")

        checkpoint_audit_path = (
            unit_root
            / "all_checkpoints_v1"
            / "swap_season_checkpoint_equivalence_v1.csv"
        )
        checkpoint_audit = pd.read_csv(checkpoint_audit_path)
        checkpoint_audit["decision_date"] = pd.to_datetime(
            checkpoint_audit["decision_date"], errors="raise"
        ).dt.strftime("%Y-%m-%d")
        checkpoint_audit["checkpoint_date"] = pd.to_datetime(
            checkpoint_audit["checkpoint_date"], errors="raise"
        ).dt.strftime("%Y-%m-%d")
        if checkpoint_audit["decision_date"].duplicated().any():
            raise ValueError(f"checkpoint audit contains duplicate dates: {unit_root}")

        trunk_crop = unit_root / "workspace" / f"trunk{year}.crp"
        trajectory = read_crop_trajectory(trunk_crop)
        trajectory_by_date = trajectory.set_index("Date", verify_integrity=True)
        audit_by_date = checkpoint_audit.set_index("decision_date", verify_integrity=True)
        for item in unit_plan.itertuples(index=False):
            if item.decision_date not in audit_by_date.index:
                raise ValueError(
                    f"checkpoint audit is missing {site}/{item.decision_date}"
                )
            audited = audit_by_date.loc[item.decision_date]
            passed = str(audited["checkpoint_equivalence_passed"]).strip().lower()
            if passed not in {"true", "1", "yes"}:
                raise ValueError(
                    f"checkpoint equivalence did not pass for {site}/{item.decision_date}"
                )
            if str(audited["checkpoint_date"]) != item.checkpoint_date:
                raise ValueError(
                    f"checkpoint audit date mismatch for {site}/{item.decision_date}"
                )
            if (
                float(audited["maximum_absolute_crop_state_error"]) > 1.0e-6
                or float(audited["maximum_absolute_profile_state_error"]) > 1.0e-6
            ):
                raise ValueError(
                    f"checkpoint equivalence error exceeds tolerance for "
                    f"{site}/{item.decision_date}"
                )
            checkpoint = pd.Timestamp(item.checkpoint_date)
            if checkpoint not in trajectory_by_date.index:
                raise ValueError(
                    f"season trunk is missing {site}/{item.checkpoint_date}"
                )
            state = crop_state_from_row(trajectory_by_date.loc[checkpoint])
            rows.append(
                {
                    "target_year": int(year),
                    "site_id": site,
                    "decision_date": item.decision_date,
                    "checkpoint_date": item.checkpoint_date,
                    "checkpoint_crop_path": str(trunk_crop),
                    "checkpoint_crop_source": "verified_continuous_season_trunk",
                    **state,
                }
            )
    result = pd.DataFrame(rows)
    keys = ["target_year", "site_id", "decision_date"]
    if result[keys].duplicated().any():
        raise ValueError("checkpoint state table contains duplicate cycle keys")
    return result


def aggregate_ensemble_mean_weather(weather: pd.DataFrame) -> pd.DataFrame:
    keys = ["target_year", "site_id", "decision_date", "lead_day"]
    required = {*keys, "gefs_member", *WEATHER_VARIABLES}
    if missing := sorted(required - set(weather.columns)):
        raise ValueError(f"formal weather is missing fields: {missing}")
    data = weather.copy()
    data["target_year"] = pd.to_numeric(data["target_year"], errors="raise").astype(int)
    data["decision_date"] = pd.to_datetime(
        data["decision_date"], errors="raise"
    ).dt.strftime("%Y-%m-%d")
    data["lead_day"] = pd.to_numeric(data["lead_day"], errors="raise").astype(int)
    if data[keys + ["gefs_member"]].duplicated().any():
        raise ValueError("formal weather contains duplicate member-lead keys")
    counts = data.groupby(keys).size()
    if not counts.eq(5).all():
        raise ValueError("every weather day must contain exactly five GEFS members")
    member_sets = data.groupby(keys)["gefs_member"].agg(
        lambda values: tuple(sorted(str(value) for value in values))
    )
    if any(members != EXPECTED_GEFS_MEMBERS for members in member_sets):
        raise ValueError("every weather day must contain the frozen five-member set")
    numeric = data[WEATHER_VARIABLES].apply(pd.to_numeric, errors="coerce")
    if numeric.isna().any().any() or not np.isfinite(numeric.to_numpy()).all():
        raise ValueError("formal corrected weather contains missing or nonfinite values")
    data[WEATHER_VARIABLES] = numeric
    daily = data.groupby(keys, as_index=False)[WEATHER_VARIABLES].mean()
    cycle_keys = ["target_year", "site_id", "decision_date"]
    cycles = daily.groupby(cycle_keys).size()
    if not cycles.eq(7).all():
        raise ValueError("every weather cycle must contain seven ensemble-mean days")
    lead_sets = daily.groupby(cycle_keys)["lead_day"].agg(
        lambda values: tuple(sorted(int(value) for value in values))
    )
    if any(leads != tuple(range(1, 8)) for leads in lead_sets):
        raise ValueError("every weather cycle must contain lead days 1 through 7")
    wide = daily.set_index(keys)[WEATHER_VARIABLES].unstack("lead_day")
    wide.columns = [
        f"weather_{variable}_day{int(day):02d}"
        for variable, day in wide.columns
    ]
    return wide.reset_index()


def boolean_mask(values: pd.Series) -> pd.Series:
    return values.map(
        lambda value: (
            bool(value)
            if isinstance(value, (bool, np.bool_))
            else str(value).strip().lower() in {"1", "true", "yes"}
        )
    )


def build_learning_table(
    candidates: pd.DataFrame,
    weather_wide: pd.DataFrame,
    checkpoint_states: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    candidate_required = {
        "target_year",
        "site",
        "decision_date",
        "checkpoint_date",
        "ir",
        "requested_ir_mm",
        "simulated_ir_mm",
        "numerical_irrigation_fallback",
        "numerical_irrigation_fallback_delta_mm",
        "numerical_irrigation_fallback_attempt_count",
        "predecision_root_depth_cm",
        "predecision_soil_vwc_0_100cm",
        "predecision_soil_storage_0_100cm_mm",
        "net_gain_7d",
        "aet_7d_mm",
        "rain_7d_mm",
        "residual_flux_7d_mm",
        "water_balance_residual_0_100cm_7d_mm",
        *[f"soil_vwc_0_100cm_day{day:02d}" for day in range(1, 8)],
    }
    if missing := sorted(candidate_required - set(candidates.columns)):
        raise ValueError(f"formal candidates are missing fields: {missing}")
    data = candidates.copy().rename(columns={"site": "site_id"})
    data["target_year"] = pd.to_numeric(data["target_year"], errors="raise").astype(int)
    data["decision_date"] = pd.to_datetime(
        data["decision_date"], errors="raise"
    ).dt.strftime("%Y-%m-%d")
    data["checkpoint_date"] = pd.to_datetime(
        data["checkpoint_date"], errors="raise"
    ).dt.strftime("%Y-%m-%d")
    keys = ["target_year", "site_id", "decision_date"]
    if data[keys + ["ir"]].duplicated().any():
        raise ValueError("formal candidates contain duplicate sample keys")
    counts = data.groupby(keys).size()
    if not counts.eq(8).all():
        raise ValueError("every formal cycle must contain eight candidates")
    requested_error = (
        pd.to_numeric(data["ir"], errors="raise")
        - pd.to_numeric(data["requested_ir_mm"], errors="raise")
    ).abs()
    if float(requested_error.max()) > 1.0e-9:
        raise ValueError("logical irrigation and requested irrigation differ")
    for _, group in data.groupby(keys):
        if sorted(group["ir"].astype(float).tolist()) != IRRIGATION_GRID_MM:
            raise ValueError("formal cycle irrigation grid mismatch")

    joined = data.merge(
        checkpoint_states,
        on=keys,
        how="left",
        validate="many_to_one",
        suffixes=("", "_checkpoint"),
        indicator="checkpoint_join",
    )
    if set(joined.pop("checkpoint_join")) != {"both"}:
        raise ValueError("not every candidate matched a checkpoint state")
    if not (
        joined["checkpoint_date"] == joined["checkpoint_date_checkpoint"]
    ).all():
        raise ValueError("candidate and checkpoint dates differ")
    root_error = (
        joined["predecision_root_depth_cm"].astype(float)
        - joined["predecision_crop_root_depth_cm"].astype(float)
    ).abs()
    if float(root_error.max()) > 1.0e-6:
        raise ValueError("candidate and checkpoint predecision root depth differ")
    joined = joined.merge(
        weather_wide,
        on=keys,
        how="left",
        validate="many_to_one",
        indicator="weather_join",
    )
    if set(joined.pop("weather_join")) != {"both"}:
        raise ValueError("not every candidate matched ensemble-mean weather")

    decision = pd.to_datetime(joined["decision_date"])
    phase = 2.0 * math.pi * decision.dt.dayofyear.astype(float) / 365.25
    fallback = boolean_mask(joined["numerical_irrigation_fallback"])
    weather_columns = [
        f"weather_{variable}_day{day:02d}"
        for variable in WEATHER_VARIABLES
        for day in range(1, 8)
    ]
    feature_columns = [
        "decision_doy_sin",
        "decision_doy_cos",
        "irrigation_mm",
        "predecision_dvs",
        "predecision_lai",
        "predecision_crop_root_depth_cm",
        "predecision_cwdm_kg_ha",
        "predecision_cwso_kg_ha",
        "predecision_soil_vwc_0_100cm",
        *weather_columns,
    ]
    output = pd.DataFrame(
        {
            "sample_id": [
                f"Y{year}_{site}_{date.replace('-', '')}_ir{float(irrigation):04.1f}"
                for year, site, date, irrigation in zip(
                    joined["target_year"],
                    joined["site_id"],
                    joined["decision_date"],
                    joined["ir"],
                    strict=True,
                )
            ],
            "target_year": joined["target_year"].astype(int),
            "split": np.where(joined["target_year"].le(2018), "train", "validation"),
            "site_id": joined["site_id"].astype(str),
            "decision_date": joined["decision_date"],
            "checkpoint_date": joined["checkpoint_date"],
            "decision_doy_sin": np.sin(phase),
            "decision_doy_cos": np.cos(phase),
            "irrigation_mm": joined["ir"].astype(float),
            "predecision_dvs": joined["predecision_dvs"].astype(float),
            "predecision_lai": joined["predecision_lai"].astype(float),
            "predecision_crop_root_depth_cm": joined[
                "predecision_crop_root_depth_cm"
            ].astype(float),
            "predecision_cwdm_kg_ha": joined["predecision_cwdm_kg_ha"].astype(float),
            "predecision_cwso_kg_ha": joined["predecision_cwso_kg_ha"].astype(float),
            "predecision_soil_vwc_0_100cm": joined[
                "predecision_soil_vwc_0_100cm"
            ].astype(float),
            "physics_initial_soil_storage_0_100cm_mm": joined[
                "predecision_soil_storage_0_100cm_mm"
            ].astype(float),
            **{column: joined[column].astype(float) for column in weather_columns},
            "target_net_gain_7d": joined["net_gain_7d"].astype(float),
            "target_aet_7d_mm": joined["aet_7d_mm"].astype(float),
            **{
                f"target_soil_vwc_0_100cm_day{day:02d}": joined[
                    f"soil_vwc_0_100cm_day{day:02d}"
                ].astype(float)
                for day in range(1, 8)
            },
            "physics_target_residual_flux_7d_mm": joined[
                "residual_flux_7d_mm"
            ].astype(float),
            "physics_target_water_balance_residual_0_100cm_7d_mm": joined[
                "water_balance_residual_0_100cm_7d_mm"
            ].astype(float),
            "audit_requested_ir_mm": joined["requested_ir_mm"].astype(float),
            "audit_simulated_ir_mm": joined["simulated_ir_mm"].astype(float),
            "audit_numerical_irrigation_fallback": fallback,
            "audit_numerical_irrigation_fallback_delta_mm": joined[
                "numerical_irrigation_fallback_delta_mm"
            ].astype(float),
            "audit_numerical_irrigation_fallback_attempt_count": joined[
                "numerical_irrigation_fallback_attempt_count"
            ].astype(int),
        }
    )
    if output["sample_id"].duplicated().any():
        raise ValueError("learning table contains duplicate sample ids")
    numeric_required = [
        *feature_columns,
        "physics_initial_soil_storage_0_100cm_mm",
        *TARGET_COLUMNS,
        "physics_target_residual_flux_7d_mm",
        "physics_target_water_balance_residual_0_100cm_7d_mm",
    ]
    numeric = output[numeric_required]
    if numeric.isna().any().any() or not np.isfinite(numeric.to_numpy()).all():
        raise ValueError("learning table contains missing or nonfinite numeric values")
    rain_columns = [
        f"weather_precipitation_mm_day{day:02d}" for day in range(1, 8)
    ]
    rain_error = (
        output[rain_columns].sum(axis=1) - joined["rain_7d_mm"].astype(float)
    ).abs()
    if float(rain_error.max()) > 0.01:
        raise ValueError("ensemble-mean weather and formal label rain differ by >0.01 mm")
    audit = {
        "maximum_absolute_checkpoint_root_depth_error_cm": float(root_error.max()),
        "maximum_absolute_weather_rain_7d_error_mm": float(rain_error.max()),
        "fallback_row_count": int(fallback.sum()),
        "second_level_fallback_row_count": int(
            joined.loc[fallback, "numerical_irrigation_fallback_delta_mm"]
            .abs()
            .gt(0.100001)
            .sum()
        ),
        "feature_columns": feature_columns,
        "target_columns": TARGET_COLUMNS,
    }
    return output, audit


def run(args: argparse.Namespace) -> dict[str, Path]:
    frozen_dir = args.frozen_label_dir.resolve()
    formal_dir = args.formal_generation_dir.resolve()
    output_dir = args.output_dir.resolve()
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite output directory: {output_dir}")
    candidate_path = frozen_dir / "gefs_exact_schedule_formal_candidates_v1.csv"
    weather_path = frozen_dir / "gefs_exact_schedule_2015_2019_formal_weather_v1.csv"
    plan_path = frozen_dir / "gefs_exact_schedule_formal_label_plan_v1.csv"
    candidates = pd.read_csv(candidate_path)
    weather = pd.read_csv(weather_path)
    plan = pd.read_csv(plan_path)
    checkpoints = collect_checkpoint_states(plan, formal_dir)
    weather_wide = aggregate_ensemble_mean_weather(weather)
    learning, audit_details = build_learning_table(
        candidates, weather_wide, checkpoints
    )
    if len(learning) != 2704 or learning["sample_id"].nunique() != 2704:
        raise ValueError("formal learning table must contain 2704 unique samples")
    if learning.loc[learning["split"].eq("train")].shape[0] != 2152:
        raise ValueError("training split must contain 2152 candidate rows")
    if learning.loc[learning["split"].eq("validation")].shape[0] != 552:
        raise ValueError("validation split must contain 552 candidate rows")

    output_dir.mkdir(parents=True)
    outputs = {
        "dataset": output_dir / "gefs_exact_schedule_surrogate_dataset_v1.csv",
        "checkpoints": output_dir / "gefs_exact_schedule_predecision_checkpoint_states_v1.csv",
        "weather": output_dir / "gefs_exact_schedule_ensemble_mean_weather_wide_v1.csv",
        "contract": output_dir / "gefs_exact_schedule_surrogate_dataset_contract_v1.json",
        "audit": output_dir / "gefs_exact_schedule_surrogate_dataset_audit_v1.json",
        "manifest": output_dir / "gefs_exact_schedule_surrogate_dataset_manifest_v1.json",
    }
    learning.to_csv(outputs["dataset"], index=False)
    checkpoints.to_csv(outputs["checkpoints"], index=False)
    weather_wide.to_csv(outputs["weather"], index=False)
    contract = {
        "contract_id": "gefs_exact_schedule_surrogate_dataset_v1",
        "sample_key": ["target_year", "site_id", "decision_date", "irrigation_mm"],
        "split": {"train": [2015, 2016, 2017, 2018], "validation": [2019]},
        "categorical_feature_columns": ["site_id"],
        "continuous_feature_columns": audit_details["feature_columns"],
        "physics_known_input_columns": [
            "physics_initial_soil_storage_0_100cm_mm"
        ],
        "formal_target_columns": audit_details["target_columns"],
        "physics_audit_target_columns": [
            "physics_target_residual_flux_7d_mm",
            "physics_target_water_balance_residual_0_100cm_7d_mm",
        ],
        "weather_driver": "frozen_corrected_GEFS_5member_ensemble_mean",
        "weather_sequence_days": 7,
        "irrigation_input": "logical_requested_irrigation_mm",
        "predecision_crop_state_source": (
            "verified_continuous_season_trunk_at_checkpoint_date"
        ),
        "forbidden_future_feature_columns": [
            "dvs",
            "lai",
            "rootd",
            "cwdm_value",
            "cwso_value",
            "final_root_depth_cm",
            "final_soil_storage_0_100cm_mm",
        ],
        "preprocessing_statistics_fitted": False,
        "preprocessing_statistics_must_use_training_split_only": True,
        "training_loss_policy_locked": False,
        "surrogate_training_performed": False,
        "tta_performed": False,
    }
    outputs["contract"].write_text(
        json.dumps(contract, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    audit = {
        "status": "gefs_exact_schedule_surrogate_dataset_leakage_gate_passed",
        "mandatory_gate_passed": True,
        "candidate_rows": int(len(learning)),
        "site_cycle_count": int(
            learning[["target_year", "site_id", "decision_date"]]
            .drop_duplicates()
            .shape[0]
        ),
        "training_rows": int(learning["split"].eq("train").sum()),
        "validation_rows": int(learning["split"].eq("validation").sum()),
        "checkpoint_state_rows": int(len(checkpoints)),
        "checkpoint_crop_state_source": (
            "verified_continuous_season_trunk_at_checkpoint_date"
        ),
        "ensemble_mean_weather_cycle_rows": int(len(weather_wide)),
        "future_restart_crop_fields_used_as_features": False,
        "simulated_fallback_irrigation_used_as_model_input": False,
        "weather_member_rows_used_as_independent_label_samples": False,
        **{key: value for key, value in audit_details.items() if not key.endswith("columns")},
        "dataset_construction_gate_passed": True,
        "training_eligible": False,
        "surrogate_training_performed": False,
        "tta_performed": False,
        "next_gate": "review_dataset_contract_before_no_tta_baseline_training",
    }
    outputs["audit"].write_text(
        json.dumps(audit, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    manifest = {
        "status": audit["status"],
        "inputs": {
            "frozen_candidates": {
                "path": str(candidate_path),
                "sha256": sha256_file(candidate_path),
            },
            "frozen_weather": {
                "path": str(weather_path),
                "sha256": sha256_file(weather_path),
            },
            "frozen_plan": {
                "path": str(plan_path),
                "sha256": sha256_file(plan_path),
            },
            "formal_generation_dir": str(formal_dir),
        },
        "outputs": {
            name: {"path": path.name, "sha256": sha256_file(path)}
            for name, path in outputs.items()
            if name != "manifest"
        },
        "network_download_performed": False,
        "swap_simulation_performed": False,
        "surrogate_training_performed": False,
        "tta_performed": False,
    }
    outputs["manifest"].write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return outputs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frozen-label-dir", type=Path, required=True)
    parser.add_argument("--formal-generation-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


if __name__ == "__main__":
    generated = run(parse_args())
    for name, path in generated.items():
        print(f"{name}: {path}")
