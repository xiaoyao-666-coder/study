#!/usr/bin/env python3
"""Select separate shrinkage for GEFS temperature center and diurnal range."""

from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd

from scripts.diagnostics.audit_gefs_era5_nonprecip_five_member_application_smoke_v1 import (
    KEYS,
    add_temperature_structure,
    count_order_inversions,
    prepare_member_pairs,
)


SELECTION_YEARS = (2015, 2016, 2017, 2018)
VALIDATION_YEAR = 2019
TEMPERATURE_VARIABLES = ("temperature_min_c", "temperature_max_c")
DEFAULT_ALPHAS = (0.0, 0.25, 0.5, 0.75, 1.0)


def candidate_id(center_alpha: float, range_alpha: float) -> str:
    return f"temperature_center_a{center_alpha:g}_range_a{range_alpha:g}"


def apply_temperature_candidate(
    member_pairs: pd.DataFrame,
    factors: pd.DataFrame,
    *,
    center_alpha: float,
    range_alpha: float,
) -> pd.DataFrame:
    data = member_pairs.merge(
        factors,
        on=["target_year", "site_id", "lead_day"],
        how="left",
        validate="many_to_one",
    )
    if data["fit_sample_count"].isna().any():
        raise ValueError("temperature factor coverage is incomplete")
    center = data["temperature_center_c_gefs"] + center_alpha * data[
        "temperature_center_additive_delta_c"
    ]
    range_factor = 1.0 + range_alpha * (data["temperature_range_ratio"] - 1.0)
    temperature_range = data["temperature_range_c_gefs"] * range_factor
    output = data[KEYS].copy()
    output["temperature_min_c"] = center - temperature_range / 2.0
    output["temperature_max_c"] = center + temperature_range / 2.0
    output["candidate_id"] = candidate_id(center_alpha, range_alpha)
    output["center_alpha"] = float(center_alpha)
    output["range_alpha"] = float(range_alpha)
    return output


def aggregate_metrics(metrics: pd.DataFrame, years: Sequence[int]) -> pd.DataFrame:
    selected = metrics.loc[metrics["target_year"].isin(years)]
    rows = []
    keys = ["candidate_id", "center_alpha", "range_alpha", "variable"]
    for key, group in selected.groupby(keys, sort=False):
        candidate, center_alpha, range_alpha, variable = key
        weights = group["sample_count"].to_numpy(dtype=float)
        total = float(weights.sum())
        rows.append(
            {
                "candidate_id": candidate,
                "center_alpha": float(center_alpha),
                "range_alpha": float(range_alpha),
                "variable": variable,
                "sample_count": int(total),
                "bias": float(np.sum(weights * group["bias"]) / total),
                "mae": float(np.sum(weights * group["mae"]) / total),
                "rmse": float(
                    np.sqrt(np.sum(weights * np.square(group["rmse"])) / total)
                ),
            }
        )
    return pd.DataFrame(rows)


def compare_to_raw(frame: pd.DataFrame) -> pd.DataFrame:
    raw = frame.loc[
        np.isclose(frame["center_alpha"], 0.0)
        & np.isclose(frame["range_alpha"], 0.0),
        ["variable", "bias", "mae", "rmse"],
    ].rename(columns={"bias": "raw_bias", "mae": "raw_mae", "rmse": "raw_rmse"})
    compared = frame.merge(raw, on="variable", how="left", validate="many_to_one")
    compared["bias_gate"] = compared["bias"].abs() <= compared["raw_bias"].abs() + 1e-12
    compared["mae_gate"] = compared["mae"] <= compared["raw_mae"] + 1e-12
    compared["rmse_gate"] = compared["rmse"] <= compared["raw_rmse"] + 1e-12
    compared["all_metric_gates"] = compared[["bias_gate", "mae_gate", "rmse_gate"]].all(axis=1)
    return compared


def select_candidate(metrics: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    pooled = compare_to_raw(aggregate_metrics(metrics, SELECTION_YEARS))
    validation = compare_to_raw(aggregate_metrics(metrics, (VALIDATION_YEAR,)))
    annual_parts = []
    for year in SELECTION_YEARS:
        part = compare_to_raw(aggregate_metrics(metrics, (year,)))
        part["target_year"] = year
        annual_parts.append(part)
    annual = pd.concat(annual_parts, ignore_index=True)

    pooled_pass = (
        pooled.loc[
            ~(
                np.isclose(pooled["center_alpha"], 0.0)
                & np.isclose(pooled["range_alpha"], 0.0)
            )
        ]
        .groupby(["candidate_id", "center_alpha", "range_alpha"], as_index=False)[
            "all_metric_gates"
        ]
        .all()
    )
    pooled_pass = pooled_pass.loc[pooled_pass["all_metric_gates"]]
    stable_rows = []
    for key, group in annual.groupby(
        ["candidate_id", "center_alpha", "range_alpha"], sort=False
    ):
        candidate, center_alpha, range_alpha = key
        variable_counts = group.groupby("variable").agg(
            mae_improved_years=("mae_gate", "sum"),
            rmse_improved_years=("rmse_gate", "sum"),
        )
        stable_rows.append(
            {
                "candidate_id": candidate,
                "center_alpha": center_alpha,
                "range_alpha": range_alpha,
                "annual_stability_passed": bool(
                    (variable_counts["mae_improved_years"] >= 3).all()
                    and (variable_counts["rmse_improved_years"] >= 3).all()
                ),
            }
        )
    stable = pd.DataFrame(stable_rows)
    eligible = pooled_pass.merge(
        stable.loc[stable["annual_stability_passed"]],
        on=["candidate_id", "center_alpha", "range_alpha"],
        how="inner",
        validate="one_to_one",
    )
    if eligible.empty:
        raise ValueError("no separate temperature shrinkage candidate passed selection gates")

    scores = []
    for _, row in eligible.iterrows():
        candidate_metrics = pooled.loc[pooled["candidate_id"].eq(row["candidate_id"])]
        scores.append(
            {
                **row.to_dict(),
                "normalized_rmse_score": float(
                    (candidate_metrics["rmse"] / candidate_metrics["raw_rmse"]).sum()
                ),
                "shrinkage_sum": float(row["center_alpha"] + row["range_alpha"]),
            }
        )
    ranking = pd.DataFrame(scores).sort_values(
        ["normalized_rmse_score", "shrinkage_sum", "center_alpha", "range_alpha"]
    )
    chosen = ranking.iloc[0]
    confirmation = validation.loc[validation["candidate_id"].eq(chosen["candidate_id"])]
    confirmed = len(confirmation) == 2 and bool(confirmation["all_metric_gates"].all())
    selected = pooled.loc[pooled["candidate_id"].eq(chosen["candidate_id"])].copy()
    selected = selected.merge(
        confirmation[
            ["variable", "bias", "mae", "rmse", "raw_bias", "raw_mae", "raw_rmse", "all_metric_gates"]
        ].rename(
            columns={
                "bias": "validation_bias",
                "mae": "validation_mae",
                "rmse": "validation_rmse",
                "raw_bias": "validation_raw_bias",
                "raw_mae": "validation_raw_mae",
                "raw_rmse": "validation_raw_rmse",
                "all_metric_gates": "validation_all_metric_gates",
            }
        ),
        on="variable",
        how="left",
        validate="one_to_one",
    )
    audit = {
        "status": (
            "temperature_center_range_shrinkage_selected_and_2019_confirmed"
            if confirmed
            else "temperature_center_range_selected_candidate_failed_2019"
        ),
        "selection_years": list(SELECTION_YEARS),
        "validation_year": VALIDATION_YEAR,
        "selection_uses_2019": False,
        "selected_candidate_id": chosen["candidate_id"],
        "selected_center_alpha": float(chosen["center_alpha"]),
        "selected_range_alpha": float(chosen["range_alpha"]),
        "selected_candidate_2019_confirmed": confirmed,
        "candidate_grid_size": int(metrics[["candidate_id", "center_alpha", "range_alpha"]].drop_duplicates().shape[0]),
        "formal_five_member_temperature_selection_performed": True,
        "swap_simulation_performed": False,
        "surrogate_training_performed": False,
        "tta_performed": False,
        "next_gate": (
            "rerun_formal_multicycle_validation_with_separate_temperature_shrinkage"
            if confirmed
            else "revise_temperature_correction_method"
        ),
    }
    return selected, audit


def build_candidates(
    *,
    five_member_weather: pd.DataFrame,
    cycle_era5: pd.DataFrame,
    factors: pd.DataFrame,
    alphas: Sequence[float],
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    weather = five_member_weather.copy()
    reference = cycle_era5.copy()
    for frame in (weather, reference):
        frame["decision_date"] = pd.to_datetime(frame["decision_date"]).dt.strftime("%Y-%m-%d")
        frame["local_date"] = pd.to_datetime(frame["local_date"]).dt.strftime("%Y-%m-%d")
    output_parts = []
    metric_rows = []
    structure_inversions = 0
    temperature_order_errors = 0
    missing_values = 0
    nonfinite_values = 0
    for cycle in sorted(weather["decision_date"].unique()):
        cycle_weather = weather.loc[weather["decision_date"].eq(cycle)]
        cycle_reference = reference.loc[reference["decision_date"].eq(cycle)]
        member_pairs = prepare_member_pairs(cycle_weather, cycle_reference)
        cycle_factors = factors.loc[factors["validation_cycle"].eq(cycle)].copy()
        raw = member_pairs[KEYS].copy()
        raw["temperature_min_c"] = member_pairs["temperature_min_c_gefs"].to_numpy()
        raw["temperature_max_c"] = member_pairs["temperature_max_c_gefs"].to_numpy()
        raw_structure = add_temperature_structure(raw)
        for center_alpha, range_alpha in itertools.product(alphas, repeat=2):
            candidate = apply_temperature_candidate(
                member_pairs,
                cycle_factors,
                center_alpha=center_alpha,
                range_alpha=range_alpha,
            )
            candidate.insert(0, "target_year", pd.Timestamp(cycle).year)
            output_parts.append(candidate)
            candidate_structure = add_temperature_structure(candidate)
            for variable in ("temperature_center_c", "temperature_range_c"):
                structure_inversions += count_order_inversions(
                    raw_structure, candidate_structure, variable
                )
            temperature_order_errors += int(
                (candidate["temperature_min_c"] > candidate["temperature_max_c"]).sum()
            )
            missing_values += int(candidate[list(TEMPERATURE_VARIABLES)].isna().sum().sum())
            nonfinite_values += int(
                (~np.isfinite(candidate[list(TEMPERATURE_VARIABLES)].to_numpy(float))).sum()
            )
            ensemble = candidate.groupby(
                ["decision_date", "site_id", "local_date", "lead_day"], as_index=False
            )[list(TEMPERATURE_VARIABLES)].mean()
            paired = ensemble.merge(
                cycle_reference[
                    ["decision_date", "site_id", "local_date", "lead_day", *TEMPERATURE_VARIABLES]
                ],
                on=["decision_date", "site_id", "local_date", "lead_day"],
                suffixes=("_gefs", "_era5"),
                how="inner",
                validate="one_to_one",
            )
            for variable in TEMPERATURE_VARIABLES:
                error = paired[f"{variable}_gefs"] - paired[f"{variable}_era5"]
                metric_rows.append(
                    {
                        "target_year": pd.Timestamp(cycle).year,
                        "decision_date": cycle,
                        "candidate_id": candidate_id(center_alpha, range_alpha),
                        "center_alpha": float(center_alpha),
                        "range_alpha": float(range_alpha),
                        "variable": variable,
                        "sample_count": int(len(error)),
                        "bias": float(error.mean()),
                        "mae": float(error.abs().mean()),
                        "rmse": float(np.sqrt(np.square(error).mean())),
                    }
                )
    candidates = pd.concat(output_parts, ignore_index=True)
    metrics = pd.DataFrame(metric_rows)
    structural_passed = all(
        value == 0
        for value in (
            structure_inversions,
            temperature_order_errors,
            missing_values,
            nonfinite_values,
        )
    )
    structural_audit = {
        "mandatory_structural_gate_passed": structural_passed,
        "temperature_structure_member_order_inversion_count": structure_inversions,
        "temperature_order_error_count": temperature_order_errors,
        "missing_value_count": missing_values,
        "nonfinite_value_count": nonfinite_values,
        "candidate_row_count": int(len(candidates)),
        "metric_row_count": int(len(metrics)),
    }
    return candidates, metrics, structural_audit


def run(args: argparse.Namespace) -> dict[str, Path]:
    candidates, metrics, structural = build_candidates(
        five_member_weather=pd.read_csv(args.five_member_weather),
        cycle_era5=pd.read_csv(args.cycle_era5),
        factors=pd.read_csv(args.factors),
        alphas=args.alphas,
    )
    selected, audit = select_candidate(metrics)
    audit = {**audit, **structural}
    args.output_dir.mkdir(parents=True, exist_ok=False)
    outputs = {
        "candidates": args.output_dir / "gefs_temperature_center_range_candidates_v1.csv",
        "metrics": args.output_dir / "gefs_temperature_center_range_metrics_v1.csv",
        "selection": args.output_dir / "gefs_temperature_center_range_selection_v1.csv",
        "audit": args.output_dir / "gefs_temperature_center_range_selection_audit_v1.json",
    }
    candidates.to_csv(outputs["candidates"], index=False)
    metrics.to_csv(outputs["metrics"], index=False)
    selected.to_csv(outputs["selection"], index=False)
    outputs["audit"].write_text(
        json.dumps(audit, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    if not audit["mandatory_structural_gate_passed"]:
        raise RuntimeError(f"temperature candidate structural gate failed; see {outputs['audit']}")
    return outputs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--five-member-weather", type=Path, required=True)
    parser.add_argument("--cycle-era5", type=Path, required=True)
    parser.add_argument("--factors", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--alphas", nargs="+", type=float, default=list(DEFAULT_ALPHAS))
    return parser.parse_args()


if __name__ == "__main__":
    generated = run(parse_args())
    print(json.dumps({key: str(value) for key, value in generated.items()}, indent=2))
