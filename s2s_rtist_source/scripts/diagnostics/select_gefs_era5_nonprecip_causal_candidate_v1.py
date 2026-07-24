#!/usr/bin/env python3
"""Select per-variable nonprecip correction shrinkage on 2015-2018 and confirm 2019."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


SELECTION_YEARS = (2015, 2016, 2017, 2018)
VALIDATION_YEAR = 2019


def aggregate_metrics(metrics: pd.DataFrame, years: tuple[int, ...]) -> pd.DataFrame:
    selected = metrics.loc[metrics["target_year"].isin(years)].copy()
    rows = []
    keys = ["variable", "candidate_id", "shrinkage_alpha"]
    for key, group in selected.groupby(keys, sort=False):
        variable, candidate_id, alpha = key
        weights = group["sample_count"].to_numpy(dtype=float)
        total = float(weights.sum())
        rows.append(
            {
                "variable": variable,
                "candidate_id": candidate_id,
                "shrinkage_alpha": float(alpha),
                "sample_count": int(total),
                "bias_corrected_minus_era5": float(
                    np.sum(weights * group["bias_corrected_minus_era5"]) / total
                ),
                "mae": float(np.sum(weights * group["mae"]) / total),
                "rmse": float(
                    np.sqrt(np.sum(weights * np.square(group["rmse"])) / total)
                ),
            }
        )
    return pd.DataFrame(rows)


def compare_to_raw(frame: pd.DataFrame) -> pd.DataFrame:
    raw = frame.loc[frame["candidate_id"] == "raw_gefs", [
        "variable", "bias_corrected_minus_era5", "mae", "rmse"
    ]].rename(
        columns={
            "bias_corrected_minus_era5": "raw_bias",
            "mae": "raw_mae",
            "rmse": "raw_rmse",
        }
    )
    compared = frame.merge(raw, on="variable", how="left", validate="many_to_one")
    compared["mae_not_worse_than_raw"] = compared["mae"] <= compared["raw_mae"] + 1e-12
    compared["rmse_not_worse_than_raw"] = compared["rmse"] <= compared["raw_rmse"] + 1e-12
    compared["absolute_bias_not_worse_than_raw"] = (
        compared["bias_corrected_minus_era5"].abs() <= compared["raw_bias"].abs() + 1e-12
    )
    compared["all_metric_gates_passed"] = compared[
        [
            "mae_not_worse_than_raw",
            "rmse_not_worse_than_raw",
            "absolute_bias_not_worse_than_raw",
        ]
    ].all(axis=1)
    return compared


def select_candidates(metrics: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, object]]:
    selection = compare_to_raw(aggregate_metrics(metrics, SELECTION_YEARS))
    validation = compare_to_raw(aggregate_metrics(metrics, (VALIDATION_YEAR,)))
    rows = []
    for variable in sorted(selection["variable"].unique()):
        candidates = selection.loc[
            (selection["variable"] == variable)
            & (selection["candidate_id"] != "raw_gefs")
            & selection["all_metric_gates_passed"]
        ].sort_values(["rmse", "mae", "shrinkage_alpha"])
        if candidates.empty:
            rows.append(
                {
                    "variable": variable,
                    "selection_status": "blocked_no_nonzero_candidate_passed_2015_2018",
                }
            )
            continue
        chosen = candidates.iloc[0]
        confirmation = validation.loc[
            (validation["variable"] == variable)
            & np.isclose(validation["shrinkage_alpha"], chosen["shrinkage_alpha"])
        ]
        if len(confirmation) != 1:
            raise ValueError(f"missing 2019 confirmation row for {variable}")
        confirmed = confirmation.iloc[0]
        rows.append(
            {
                "variable": variable,
                "selection_status": (
                    "selected_and_2019_confirmed"
                    if bool(confirmed["all_metric_gates_passed"])
                    else "blocked_selected_candidate_failed_2019"
                ),
                "candidate_id": chosen["candidate_id"],
                "shrinkage_alpha": float(chosen["shrinkage_alpha"]),
                "selection_rmse": float(chosen["rmse"]),
                "selection_raw_rmse": float(chosen["raw_rmse"]),
                "selection_mae": float(chosen["mae"]),
                "selection_raw_mae": float(chosen["raw_mae"]),
                "validation_rmse": float(confirmed["rmse"]),
                "validation_raw_rmse": float(confirmed["raw_rmse"]),
                "validation_mae": float(confirmed["mae"]),
                "validation_raw_mae": float(confirmed["raw_mae"]),
                "validation_absolute_bias": float(
                    abs(confirmed["bias_corrected_minus_era5"])
                ),
                "validation_raw_absolute_bias": float(abs(confirmed["raw_bias"])),
            }
        )
    selected = pd.DataFrame(rows)
    blocked = int(
        (~selected["selection_status"].eq("selected_and_2019_confirmed")).sum()
    )
    audit = {
        "status": (
            "nonprecip_variable_candidates_selected_and_2019_confirmed"
            if blocked == 0
            else "nonprecip_variable_candidate_selection_blocked"
        ),
        "selection_years": list(SELECTION_YEARS),
        "validation_year": VALIDATION_YEAR,
        "variable_count": int(len(selected)),
        "blocked_variable_count": blocked,
        "all_variables_passed": blocked == 0,
        "raw_candidate_selectable": False,
        "selection_uses_2019": False,
        "candidate_selection_performed": True,
        "five_member_validation_performed": False,
        "swap_simulation_performed": False,
        "surrogate_training_performed": False,
        "tta_performed": False,
        "next_gate": (
            "five_member_validation"
            if blocked == 0
            else "revise_blocked_variable_correction_method"
        ),
    }
    return selected, audit


def run(args: argparse.Namespace) -> dict[str, Path]:
    selected, audit = select_candidates(pd.read_csv(args.metrics))
    args.output_dir.mkdir(parents=True, exist_ok=False)
    outputs = {
        "selection": args.output_dir / "gefs_era5_nonprecip_variable_selection_v1.csv",
        "audit": args.output_dir / "gefs_era5_nonprecip_variable_selection_audit_v1.json",
    }
    selected.to_csv(outputs["selection"], index=False)
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
