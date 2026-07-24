#!/usr/bin/env python3
"""Summarize formal pilot response coverage for date-density design."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


IRRIGATION_OPTIONS_MM = (0.0, 10.0, 15.0, 20.0, 25.0, 30.0, 40.0, 60.0)
GROUP_KEYS = ["target_year", "site", "date_t"]
NUMERIC_COLUMNS = [
    "target_year",
    "ir",
    "cwdm_value",
    "net_gain_7d",
    "best_ir_for_date",
    "best_target_for_date",
    "aet_7d_mm",
    "soil_vwc_0_100cm_day07",
    "water_balance_residual_0_100cm_7d_mm",
    "gefs_corrected_precipitation_7d_mm",
]
REQUIRED_COLUMNS = {"site", "date_t", *NUMERIC_COLUMNS}


def _validate_candidates(
    frame: pd.DataFrame,
    *,
    expected_rows: int,
    expected_site_cycles: int,
) -> pd.DataFrame:
    missing = sorted(REQUIRED_COLUMNS.difference(frame.columns))
    if missing:
        raise ValueError(f"candidate labels missing fields: {missing}")
    data = frame.copy()
    data[NUMERIC_COLUMNS] = data[NUMERIC_COLUMNS].apply(
        pd.to_numeric, errors="coerce"
    )
    if data[NUMERIC_COLUMNS].isna().any().any() or not np.isfinite(
        data[NUMERIC_COLUMNS].to_numpy(dtype=float)
    ).all():
        raise ValueError("candidate labels contain nonfinite required values")
    data["target_year"] = data["target_year"].astype(int)
    if len(data) != expected_rows:
        raise ValueError(f"expected {expected_rows} candidate rows, got {len(data)}")
    if data.duplicated([*GROUP_KEYS, "ir"]).any():
        raise ValueError("candidate labels contain duplicate site-cycle-irrigation rows")
    if data[GROUP_KEYS].drop_duplicates().shape[0] != expected_site_cycles:
        raise ValueError(
            f"expected {expected_site_cycles} site-cycles, got "
            f"{data[GROUP_KEYS].drop_duplicates().shape[0]}"
        )
    expected_options = list(IRRIGATION_OPTIONS_MM)
    for key, group in data.groupby(GROUP_KEYS, sort=True):
        actual = sorted(group["ir"].astype(float).tolist())
        if actual != expected_options:
            raise ValueError(
                f"{key} irrigation candidates must be {expected_options}, got {actual}"
            )
    return data


def analyze_candidate_labels(
    frame: pd.DataFrame,
    *,
    expected_rows: int = 200,
    expected_site_cycles: int = 25,
) -> dict[str, Any]:
    data = _validate_candidates(
        frame,
        expected_rows=expected_rows,
        expected_site_cycles=expected_site_cycles,
    )
    rows: list[dict[str, Any]] = []
    for key, group in data.groupby(GROUP_KEYS, sort=True):
        ordered = group.sort_values("ir").reset_index(drop=True)
        best = ordered.loc[ordered["net_gain_7d"].idxmax()]
        reported_best_ir = ordered["best_ir_for_date"].unique()
        reported_best_gain = ordered["best_target_for_date"].unique()
        if len(reported_best_ir) != 1 or len(reported_best_gain) != 1:
            raise ValueError(f"{key} has inconsistent repeated best-decision fields")
        if not np.isclose(float(best["ir"]), float(reported_best_ir[0])):
            raise ValueError(f"{key} recomputed best irrigation disagrees with labels")
        if not np.isclose(float(best["net_gain_7d"]), float(reported_best_gain[0])):
            raise ValueError(f"{key} recomputed best gain disagrees with labels")
        precipitation = ordered["gefs_corrected_precipitation_7d_mm"]
        if not np.isclose(float(precipitation.max()), float(precipitation.min())):
            raise ValueError(f"{key} precipitation changes across irrigation candidates")
        cwdm_range = float(ordered["cwdm_value"].max() - ordered["cwdm_value"].min())
        best_ir = float(best["ir"])
        best_gain = float(best["net_gain_7d"])
        rows.append(
            {
                "target_year": int(key[0]),
                "site": str(key[1]),
                "date_t": str(key[2]),
                "gefs_corrected_precipitation_7d_mm": float(precipitation.iloc[0]),
                "cwdm_range_kg_ha": cwdm_range,
                "positive_cwdm_response": bool(cwdm_range > 0.0),
                "net_gain_range": float(
                    ordered["net_gain_7d"].max() - ordered["net_gain_7d"].min()
                ),
                "best_ir_mm": best_ir,
                "best_net_gain_7d": best_gain,
                "profitable_nonzero_irrigation": bool(best_ir > 0.0 and best_gain > 0.0),
                "aet_range_mm": float(
                    ordered["aet_7d_mm"].max() - ordered["aet_7d_mm"].min()
                ),
                "final_vwc_range": float(
                    ordered["soil_vwc_0_100cm_day07"].max()
                    - ordered["soil_vwc_0_100cm_day07"].min()
                ),
                "maximum_absolute_water_balance_residual_mm": float(
                    ordered["water_balance_residual_0_100cm_7d_mm"].abs().max()
                ),
            }
        )
    site_cycle = pd.DataFrame(rows).sort_values(GROUP_KEYS).reset_index(drop=True)
    year_summary = (
        site_cycle.groupby("target_year", as_index=False)
        .agg(
            site_cycle_count=("site", "nunique"),
            responsive_site_cycle_count=("positive_cwdm_response", "sum"),
            profitable_nonzero_site_cycle_count=("profitable_nonzero_irrigation", "sum"),
            minimum_corrected_precipitation_7d_mm=(
                "gefs_corrected_precipitation_7d_mm",
                "min",
            ),
            maximum_corrected_precipitation_7d_mm=(
                "gefs_corrected_precipitation_7d_mm",
                "max",
            ),
            median_best_ir_mm=("best_ir_mm", "median"),
            maximum_best_net_gain_7d=("best_net_gain_7d", "max"),
            maximum_cwdm_range_kg_ha=("cwdm_range_kg_ha", "max"),
        )
        .sort_values("target_year")
        .reset_index(drop=True)
    )
    year_summary["cross_site_precipitation_range_7d_mm"] = (
        year_summary["maximum_corrected_precipitation_7d_mm"]
        - year_summary["minimum_corrected_precipitation_7d_mm"]
    )
    site_summary = (
        site_cycle.groupby("site", as_index=False)
        .agg(
            year_count=("target_year", "nunique"),
            responsive_year_count=("positive_cwdm_response", "sum"),
            profitable_nonzero_year_count=("profitable_nonzero_irrigation", "sum"),
            median_best_ir_mm=("best_ir_mm", "median"),
            maximum_best_net_gain_7d=("best_net_gain_7d", "max"),
            maximum_cwdm_range_kg_ha=("cwdm_range_kg_ha", "max"),
        )
        .sort_values("site")
        .reset_index(drop=True)
    )
    best_ir_distribution = (
        site_cycle.groupby("best_ir_mm", as_index=False)
        .size()
        .rename(columns={"size": "site_cycle_count"})
        .sort_values("best_ir_mm")
        .reset_index(drop=True)
    )
    compact_columns = [
        *GROUP_KEYS,
        "ir",
        "cwdm_value",
        "net_gain_7d",
        "aet_7d_mm",
        "soil_vwc_0_100cm_day07",
        "water_balance_residual_0_100cm_7d_mm",
        "gefs_corrected_precipitation_7d_mm",
    ]
    compact = data[compact_columns].sort_values([*GROUP_KEYS, "ir"]).reset_index(
        drop=True
    )
    audit = {
        "status": "pilot_response_coverage_diagnostic_passed_density_not_selected",
        "candidate_rows": int(len(data)),
        "site_cycle_count": int(len(site_cycle)),
        "year_count": int(site_cycle["target_year"].nunique()),
        "site_count": int(site_cycle["site"].nunique()),
        "responsive_site_cycle_count": int(site_cycle["positive_cwdm_response"].sum()),
        "profitable_nonzero_site_cycle_count": int(
            site_cycle["profitable_nonzero_irrigation"].sum()
        ),
        "years_with_profitable_nonzero_irrigation": int(
            site_cycle.groupby("target_year")["profitable_nonzero_irrigation"]
            .any()
            .sum()
        ),
        "maximum_absolute_water_balance_residual_mm": float(
            site_cycle["maximum_absolute_water_balance_residual_mm"].max()
        ),
        "decision_dates_per_year_observed": 1,
        "within_season_density_identifiable_from_this_pilot": False,
        "date_density_parameter_selected": False,
        "additional_swap_generation_started": False,
        "surrogate_training_performed": False,
        "tta_performed": False,
    }
    return {
        "compact_candidates": compact,
        "site_cycle_summary": site_cycle,
        "year_summary": year_summary,
        "site_summary": site_summary,
        "best_ir_distribution": best_ir_distribution,
        "audit": audit,
    }


def write_outputs(result: dict[str, Any], output_dir: Path) -> dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=False)
    outputs: dict[str, Path] = {}
    for key in (
        "compact_candidates",
        "site_cycle_summary",
        "year_summary",
        "site_summary",
        "best_ir_distribution",
    ):
        path = output_dir / f"{key}_v1.csv"
        result[key].to_csv(path, index=False, encoding="utf-8-sig")
        outputs[key] = path
    audit_path = output_dir / "pilot_response_coverage_audit_v1.json"
    audit_path.write_text(
        json.dumps(result["audit"], indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    outputs["audit"] = audit_path
    return outputs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidates", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    analysis = analyze_candidate_labels(pd.read_csv(args.candidates))
    generated = write_outputs(analysis, args.output_dir)
    print(
        json.dumps(
            {
                "outputs": {key: str(value) for key, value in generated.items()},
                **analysis["audit"],
            },
            indent=2,
        )
    )
