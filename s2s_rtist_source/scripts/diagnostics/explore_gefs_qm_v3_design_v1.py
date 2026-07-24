"""Offline exploration of seasonal grouping, occurrence ablation, and hierarchical QM shrinkage."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from run_gefs_qm_training_cv_v1 import (
    CONTRACT_PATH,
    _load_contract,
    _metric_bundle,
    _tail_audit,
    prepare_training_period_paired,
)
from s2s_rtist.weather.gefs_quantile_mapping import (
    CONTRACT_ID_TRAINING_CV,
    CONTRACT_VERSION_TRAINING_CV,
    GEFS_REFORECAST_MEMBERS,
    UPPER_TAIL_CONSTANT_ADDITIVE,
    UTC_DAY_BOUNDARY,
    _group_artifact_key,
    apply_empirical_precipitation_qm,
    fit_empirical_precipitation_qm,
    verify_quantile_mapping_artifact,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT_DIR = (
    PROJECT_ROOT / "site_general_surrogate_eval" / "gefs_qm_v3_design_exploration_v1"
)


CANDIDATES: list[dict[str, Any]] = [
    {
        "candidate_id": "global_month_occurrence",
        "group_keys": ["init_month"],
        "parent_group_keys": None,
        "occurrence_correction": True,
        "shrink_lambda_independent": None,
    },
    {
        "candidate_id": "site_month_occurrence",
        "group_keys": ["site_id", "init_month"],
        "parent_group_keys": None,
        "occurrence_correction": True,
        "shrink_lambda_independent": None,
    },
    {
        "candidate_id": "site_month_shrink_lambda18_occurrence",
        "group_keys": ["site_id", "init_month"],
        "parent_group_keys": ["init_month"],
        "occurrence_correction": True,
        "shrink_lambda_independent": 18.0,
    },
    {
        "candidate_id": "site_month_shrink_lambda36_occurrence",
        "group_keys": ["site_id", "init_month"],
        "parent_group_keys": ["init_month"],
        "occurrence_correction": True,
        "shrink_lambda_independent": 36.0,
    },
    {
        "candidate_id": "site_only_no_occurrence",
        "group_keys": ["site_id"],
        "parent_group_keys": None,
        "occurrence_correction": False,
        "shrink_lambda_independent": None,
    },
    {
        "candidate_id": "site_only_shrink_lambda18_occurrence",
        "group_keys": ["site_id"],
        "parent_group_keys": [],
        "occurrence_correction": True,
        "shrink_lambda_independent": 18.0,
    },
    {
        "candidate_id": "site_only_shrink_lambda18_no_occurrence",
        "group_keys": ["site_id"],
        "parent_group_keys": [],
        "occurrence_correction": False,
        "shrink_lambda_independent": 18.0,
    },
    {
        "candidate_id": "global_month_no_occurrence",
        "group_keys": ["init_month"],
        "parent_group_keys": None,
        "occurrence_correction": False,
        "shrink_lambda_independent": None,
    },
    {
        "candidate_id": "site_lead_shrink_lambda18_occurrence",
        "group_keys": ["site_id", "lead_day"],
        "parent_group_keys": ["site_id"],
        "occurrence_correction": True,
        "shrink_lambda_independent": 18.0,
    },
]


def _fit_cached(
    paired: pd.DataFrame,
    fit_years: tuple[int, ...],
    validation_year: int,
    candidate_id: str,
    group_keys: tuple[str, ...],
    occurrence_correction: bool,
    cache: dict[tuple[tuple[str, ...], bool], dict[str, Any]],
) -> dict[str, Any]:
    cache_key = (group_keys, occurrence_correction)
    if cache_key not in cache:
        fit = paired.loc[paired["validation_year"].isin(fit_years)].copy()
        artifact = fit_empirical_precipitation_qm(
            fit,
            fit_years=fit_years,
            expected_members=GEFS_REFORECAST_MEMBERS,
            contract_id=CONTRACT_ID_TRAINING_CV,
            contract_version=CONTRACT_VERSION_TRAINING_CV,
            aggregation_day_boundary=UTC_DAY_BOUNDARY,
            canonical_valid_date_column="valid_date_utc",
            upper_tail_policy=UPPER_TAIL_CONSTANT_ADDITIVE,
            group_keys=group_keys,
            occurrence_correction=occurrence_correction,
            artifact_context={
                "candidate_id": candidate_id,
                "fold_id": f"F{validation_year}",
                "validation_year": validation_year,
            },
        )
        verify_quantile_mapping_artifact(artifact)
        cache[cache_key] = artifact
    return cache[cache_key]


def _blend_with_parent(
    child: pd.DataFrame,
    parent: pd.DataFrame,
    child_artifact: dict[str, Any],
    lambda_independent: float,
) -> pd.DataFrame:
    output = child.copy()
    child_keys = tuple(child_artifact["group_keys"])
    weights = []
    for _, row in child.iterrows():
        values = [row[column] for column in child_keys]
        key = _group_artifact_key(child_keys, values)
        sample_count = float(child_artifact["groups"][key]["sample_count"])
        independent_count = sample_count / float(len(GEFS_REFORECAST_MEMBERS))
        weights.append(independent_count / (independent_count + lambda_independent))
    weight = np.asarray(weights, dtype=float)
    output["precipitation_mm_qm"] = (
        weight * child["precipitation_mm_qm"].to_numpy(dtype=float)
        + (1.0 - weight) * parent["precipitation_mm_qm"].to_numpy(dtype=float)
    )
    output["qm_extrapolated_upper"] = (
        child["qm_extrapolated_upper"].to_numpy(dtype=bool)
        | parent["qm_extrapolated_upper"].to_numpy(dtype=bool)
    )
    output["qm_forecast_wet_threshold_mm"] = child[
        "qm_forecast_wet_threshold_mm"
    ].to_numpy(dtype=float)
    return output


def _raw_frame(frame: pd.DataFrame) -> pd.DataFrame:
    output = frame.copy()
    output["precipitation_mm_qm"] = output["precipitation_mm_raw"]
    output["qm_extrapolated_upper"] = False
    output["qm_forecast_wet_threshold_mm"] = 0.1
    return output


def _summary_row(candidate_id: str, year: int | str, bundle: dict[str, Any]) -> dict[str, Any]:
    gate = bundle["gate"]
    return {
        "candidate_id": candidate_id,
        "validation_year": year,
        "raw_seven_day_mae_mm": gate["raw_seven_day_mae_mm"],
        "candidate_seven_day_mae_mm": gate["qm_seven_day_mae_mm"],
        "seven_day_mae_difference_mm": gate["qm_seven_day_mae_mm"]
        - gate["raw_seven_day_mae_mm"],
        "raw_mean_crps_mm": gate["raw_mean_crps_mm"],
        "candidate_mean_crps_mm": gate["qm_mean_crps_mm"],
        "crps_difference_mm": gate["qm_mean_crps_mm"] - gate["raw_mean_crps_mm"],
        "raw_mean_brier_score": gate["raw_mean_brier_score"],
        "candidate_mean_brier_score": gate["qm_mean_brier_score"],
        "mean_brier_difference": gate["qm_mean_brier_score"]
        - gate["raw_mean_brier_score"],
        "heavy_coverage_not_both_worse": gate["automatic_requirements"][
            "heavy_coverage_not_both_worse"
        ],
    }


def run_exploration(output_dir: Path = DEFAULT_OUTPUT_DIR) -> dict[str, Path]:
    contract = _load_contract(CONTRACT_PATH)
    paired, _ = prepare_training_period_paired(contract)
    paired["init_month"] = pd.to_datetime(paired["decision_date"]).dt.month.astype(int)
    output_dir.mkdir(parents=True, exist_ok=True)

    config_path = output_dir / "v3_exploration_candidate_configurations.json"
    config_path.write_text(
        json.dumps(
            {
                "scope": "2015-2018 leave-one-year-out exploratory OOF only",
                "network_download_performed": False,
                "2019_used": False,
                "2024_used": False,
                "seasonal_definition": "init_month derived from forecast decision_date",
                "independent_sample_count": "member_rows / 5 exchangeable members",
                "candidates": CANDIDATES,
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    oof_parts: list[pd.DataFrame] = []
    pooled_rows: list[dict[str, Any]] = []
    year_rows: list[dict[str, Any]] = []
    tail_summary: dict[str, Any] = {}
    for candidate in CANDIDATES:
        candidate_id = candidate["candidate_id"]
        candidate_oof: list[pd.DataFrame] = []
        for fold in contract["folds"]:
            fit_years = tuple(int(value) for value in fold["fit_years"])
            validation_year = int(fold["validation_year"])
            validation = paired.loc[
                paired["validation_year"].eq(validation_year)
            ].copy()
            cache: dict[tuple[tuple[str, ...], bool], dict[str, Any]] = {}
            child = _fit_cached(
                paired,
                fit_years,
                validation_year,
                candidate_id,
                tuple(candidate["group_keys"]),
                bool(candidate["occurrence_correction"]),
                cache,
            )
            corrected = apply_empirical_precipitation_qm(
                validation, child, split=f"v3_{fold['fold_id']}"
            )
            parent_keys = candidate["parent_group_keys"]
            if parent_keys is not None and candidate["shrink_lambda_independent"] is not None:
                parent = _fit_cached(
                    paired,
                    fit_years,
                    validation_year,
                    candidate_id + "_parent",
                    tuple(parent_keys),
                    bool(candidate["occurrence_correction"]),
                    cache,
                )
                parent_corrected = apply_empirical_precipitation_qm(
                    validation, parent, split=f"v3_{fold['fold_id']}_parent"
                )
                corrected = _blend_with_parent(
                    corrected,
                    parent_corrected,
                    child,
                    float(candidate["shrink_lambda_independent"]),
                )
            corrected["candidate_id"] = candidate_id
            corrected["fold_id"] = fold["fold_id"]
            candidate_oof.append(corrected)
        candidate_frame = pd.concat(candidate_oof, ignore_index=True)
        oof_parts.append(candidate_frame)
        bundle = _metric_bundle(candidate_frame)
        pooled_rows.append(_summary_row(candidate_id, "pooled", bundle))
        tail, audit = _tail_audit(candidate_frame)
        audit["candidate_id"] = candidate_id
        audit["upper_tail_worsened_count"] = int(
            tail["absolute_error_change_mm"].gt(0.0).sum()
        )
        tail_summary[candidate_id] = audit
        for year in contract["source_data"]["allowed_years"]:
            year_frame = candidate_frame.loc[
                candidate_frame["validation_year"].eq(int(year))
            ]
            year_rows.append(
                _summary_row(candidate_id, int(year), _metric_bundle(year_frame))
            )

    oof = pd.concat(oof_parts, ignore_index=True)
    oof_path = output_dir / "v3_exploration_oof_member_predictions.csv"
    oof.to_csv(oof_path, index=False, encoding="utf-8-sig")
    pooled_path = output_dir / "v3_exploration_pooled_metrics.csv"
    pd.DataFrame(pooled_rows).to_csv(pooled_path, index=False, encoding="utf-8-sig")
    year_path = output_dir / "v3_exploration_year_metrics.csv"
    pd.DataFrame(year_rows).to_csv(year_path, index=False, encoding="utf-8-sig")
    tail_path = output_dir / "v3_exploration_upper_tail_audit.json"
    tail_path.write_text(
        json.dumps(tail_summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    report_path = output_dir / "v3_exploration_scope_and_conclusion.md"
    report_lines = [
        "# GEFS QM v3 design exploration",
        "",
        "This is an offline exploratory OOF experiment on 2015-2018 only; it does not select a production candidate or authorize 2019/2024 use.",
        "Seasonal grouping uses forecast initialization month (`init_month`). Shrinkage weights use independent reference count, not repeated member rows.",
        "",
    ]
    for row in sorted(pooled_rows, key=lambda item: item["candidate_id"]):
        report_lines.append(
            f"- `{row['candidate_id']}`: 7-day MAE difference `{row['seven_day_mae_difference_mm']:.6f} mm`, "
            f"CRPS difference `{row['crps_difference_mm']:.6f} mm`, "
            f"Brier difference `{row['mean_brier_difference']:.6f}`, "
            f"heavy coverage gate `{row['heavy_coverage_not_both_worse']}`."
        )
    report_lines.extend(
        [
            "",
            "These results are for design comparison only. A promising variant must receive a new prelocked v3 contract and a fresh training-period CV before any holdout application.",
            "",
        ]
    )
    report_path.write_text("\n".join(report_lines), encoding="utf-8")
    return {
        "config": config_path,
        "oof": oof_path,
        "pooled": pooled_path,
        "year": year_path,
        "tail_audit": tail_path,
        "report": report_path,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    print(json.dumps({key: str(value) for key, value in run_exploration(args.output_dir).items()}, indent=2))


if __name__ == "__main__":
    main()
