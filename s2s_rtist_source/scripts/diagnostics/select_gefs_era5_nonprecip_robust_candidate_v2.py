#!/usr/bin/env python3
"""Select robust nonprecipitation candidates and retain a solar raw sensitivity branch."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from scripts.diagnostics.select_gefs_era5_nonprecip_causal_candidate_v1 import (
    SELECTION_YEARS,
    VALIDATION_YEAR,
    aggregate_metrics,
    compare_to_raw,
    select_candidates,
)


SOLAR_VARIABLE = "solar_kj_m2_day"


def annual_comparisons(metrics: pd.DataFrame, years: tuple[int, ...]) -> pd.DataFrame:
    parts = []
    for year in years:
        part = compare_to_raw(aggregate_metrics(metrics, (year,))).copy()
        part["evaluation_year"] = year
        parts.append(part)
    return pd.concat(parts, ignore_index=True)


def select_robust_candidates(
    metrics: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, object]]:
    selected, strict_audit = select_candidates(metrics)
    pooled = compare_to_raw(aggregate_metrics(metrics, SELECTION_YEARS))
    annual = annual_comparisons(metrics, (*SELECTION_YEARS, VALIDATION_YEAR))

    solar_pooled = pooled.loc[
        pooled["variable"].eq(SOLAR_VARIABLE)
        & pooled["candidate_id"].ne("raw_gefs")
        & pooled["all_metric_gates_passed"]
    ].copy()
    solar_selection_years = annual.loc[
        annual["variable"].eq(SOLAR_VARIABLE)
        & annual["evaluation_year"].isin(SELECTION_YEARS)
        & annual["candidate_id"].ne("raw_gefs")
    ].copy()
    stable_ids = (
        solar_selection_years.groupby(
            ["candidate_id", "shrinkage_alpha"], as_index=False
        )[["mae_not_worse_than_raw", "rmse_not_worse_than_raw"]]
        .all()
    )
    stable_ids = stable_ids.loc[
        stable_ids["mae_not_worse_than_raw"]
        & stable_ids["rmse_not_worse_than_raw"]
    ]
    solar_candidates = solar_pooled.merge(
        stable_ids[["candidate_id", "shrinkage_alpha"]],
        on=["candidate_id", "shrinkage_alpha"],
        how="inner",
        validate="one_to_one",
    ).sort_values(["shrinkage_alpha", "rmse", "mae"])
    if solar_candidates.empty:
        raise ValueError("no solar candidate improves annual MAE and RMSE in every selection year")

    solar_choice = solar_candidates.iloc[0]
    solar_validation = annual.loc[
        annual["variable"].eq(SOLAR_VARIABLE)
        & annual["evaluation_year"].eq(VALIDATION_YEAR)
        & annual["candidate_id"].eq(solar_choice["candidate_id"])
        & annual["shrinkage_alpha"].eq(solar_choice["shrinkage_alpha"])
    ]
    if len(solar_validation) != 1:
        raise ValueError("missing unique solar validation row")
    solar_validation = solar_validation.iloc[0]
    primary_validation_passed = bool(
        solar_validation["mae_not_worse_than_raw"]
        and solar_validation["rmse_not_worse_than_raw"]
    )

    replacement = {
        "variable": SOLAR_VARIABLE,
        "selection_status": (
            "selected_for_five_member_raw_sensitivity_bias_tradeoff"
            if primary_validation_passed
            else "blocked_solar_primary_validation_failed"
        ),
        "candidate_id": solar_choice["candidate_id"],
        "shrinkage_alpha": float(solar_choice["shrinkage_alpha"]),
        "selection_rmse": float(solar_choice["rmse"]),
        "selection_raw_rmse": float(solar_choice["raw_rmse"]),
        "selection_mae": float(solar_choice["mae"]),
        "selection_raw_mae": float(solar_choice["raw_mae"]),
        "validation_rmse": float(solar_validation["rmse"]),
        "validation_raw_rmse": float(solar_validation["raw_rmse"]),
        "validation_mae": float(solar_validation["mae"]),
        "validation_raw_mae": float(solar_validation["raw_mae"]),
        "validation_absolute_bias": float(
            abs(solar_validation["bias_corrected_minus_era5"])
        ),
        "validation_raw_absolute_bias": float(abs(solar_validation["raw_bias"])),
        "validation_absolute_bias_delta": float(
            abs(solar_validation["bias_corrected_minus_era5"])
            - abs(solar_validation["raw_bias"])
        ),
        "raw_solar_sensitivity_required": True,
    }
    selected = selected.loc[~selected["variable"].eq(SOLAR_VARIABLE)].copy()
    selected = pd.concat([selected, pd.DataFrame([replacement])], ignore_index=True)
    selected = selected.sort_values("variable").reset_index(drop=True)

    non_solar_passed = bool(
        selected.loc[~selected["variable"].eq(SOLAR_VARIABLE), "selection_status"]
        .eq("selected_and_2019_confirmed")
        .all()
    )
    sensitivity_allowed = non_solar_passed and primary_validation_passed
    audit = {
        "status": (
            "nonprecip_ready_for_five_member_validation_with_solar_raw_sensitivity"
            if sensitivity_allowed
            else "nonprecip_robust_candidate_selection_blocked"
        ),
        "selection_years": list(SELECTION_YEARS),
        "validation_year": VALIDATION_YEAR,
        "selection_uses_2019": False,
        "strict_all_variables_passed": bool(strict_audit["all_variables_passed"]),
        "five_member_sensitivity_allowed": sensitivity_allowed,
        "solar_candidate_selected_by": (
            "smallest_nonzero_alpha_passing_pooled_gates_and_improving_annual_mae_rmse_"
            "in_every_2015_2018_year"
        ),
        "solar_selected_alpha": float(solar_choice["shrinkage_alpha"]),
        "solar_2019_primary_metrics_passed": primary_validation_passed,
        "solar_2019_strict_bias_gate_passed": bool(
            solar_validation["absolute_bias_not_worse_than_raw"]
        ),
        "raw_solar_sensitivity_required": True,
        "five_member_validation_performed": False,
        "swap_simulation_performed": False,
        "surrogate_training_performed": False,
        "tta_performed": False,
        "next_gate": (
            "five_member_validation_compare_raw_and_affine_solar"
            if sensitivity_allowed
            else "revise_blocked_variable_correction_method"
        ),
    }
    return selected, annual, audit


def run(args: argparse.Namespace) -> dict[str, Path]:
    selected, annual, audit = select_robust_candidates(pd.read_csv(args.metrics))
    args.output_dir.mkdir(parents=True, exist_ok=False)
    outputs = {
        "selection": args.output_dir / "gefs_era5_nonprecip_robust_selection_v2.csv",
        "annual": args.output_dir / "gefs_era5_nonprecip_annual_candidate_gates_v2.csv",
        "audit": args.output_dir / "gefs_era5_nonprecip_robust_selection_audit_v2.json",
    }
    selected.to_csv(outputs["selection"], index=False)
    annual.to_csv(outputs["annual"], index=False)
    outputs["audit"].write_text(
        json.dumps(audit, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return outputs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metrics", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


if __name__ == "__main__":
    generated = run(parse_args())
    print(json.dumps({key: str(value) for key, value in generated.items()}, indent=2))
