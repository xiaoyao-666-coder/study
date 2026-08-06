from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


PROTOCOL_ID = "teacher-guided-robust-envelope-gam-development-v1"
EXPECTED_SITES = ("P1", "P2", "P3", "P4", "P15")
EXPECTED_SITE_CYCLES = {"P1": 38, "P2": 38, "P3": 41, "P4": 39, "P15": 40}
EXPECTED_SOURCE_PROTOCOL = (
    "teacher-guided-hierarchical-gam-nested-shared-selection-development-v2"
)


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def validate_protocol(protocol: dict[str, Any]) -> None:
    """Reject any protocol that changes the frozen first-stage boundary."""
    _require(protocol.get("protocol_id") == PROTOCOL_ID, "unexpected robust envelope protocol")
    data = protocol.get("data", {})
    _require(data.get("development_years") == [2015, 2016, 2017, 2018], "development years changed")
    _require(data.get("reused_context_year") == 2019, "2019 context year changed")
    _require(data.get("sealed_final_test_year") == 2024, "2024 sealed year changed")
    _require(tuple(data.get("sites", ())) == EXPECTED_SITES, "site order changed")
    _require(data.get("candidate_count_per_cycle") == 8, "candidate count changed")
    _require(len(str(data.get("dataset_sha256", ""))) == 64, "dataset hash missing")
    _require(len(str(data.get("contract_sha256", ""))) == 64, "contract hash missing")

    source = protocol.get("source_development", {})
    _require(source.get("protocol_id") == EXPECTED_SOURCE_PROTOCOL, "source protocol changed")
    _require(source.get("formal_variant") == "GAM-VC", "formal source variant changed")
    _require(source.get("must_have_passed_gate") is True, "source gate is not required")
    _require(source.get("reuse_selected_penalties_without_search") is True, "source penalties are not frozen")

    validation = protocol.get("validation", {})
    expected_year_folds = [
        {"train_years": [2015], "validation_year": 2016},
        {"train_years": [2015, 2016], "validation_year": 2017},
        {"train_years": [2015, 2016, 2017], "validation_year": 2018},
    ]
    _require(validation.get("rolling_year_folds") == expected_year_folds, "rolling folds changed")
    _require(validation.get("strict_outer_fold_count") == 15, "outer fold count changed")
    _require(validation.get("validation_cycle_count") == 196, "validation cycle count changed")
    _require(validation.get("site_cycle_counts") == EXPECTED_SITE_CYCLES, "site cycle counts changed")
    _require(validation.get("held_out_site_uses_shared_terms_only") is True, "held-out site uses deviations")

    ensemble = protocol.get("ensemble", {})
    _require(ensemble.get("jackknife_model_count_per_fold") == 4, "jackknife count changed")
    _require(ensemble.get("source_site_count_per_submodel") == 3, "jackknife source count changed")
    _require(
        ensemble.get("decision_target") == "minimum_incremental_net_gain_across_submodels",
        "decision target changed",
    )
    _require(ensemble.get("other_output_aggregation") == "arithmetic_mean", "output aggregation changed")

    optimization = protocol.get("continuous_optimization", {})
    _require(optimization.get("range_mm") == [0.0, 60.0], "irrigation range changed")
    _require(optimization.get("deployment_resolution_mm") == 0.000001, "deployment resolution changed")
    _require(optimization.get("dense_diagnostic_step_mm") == 0.25, "dense diagnostic step changed")
    _require(optimization.get("maximum_dense_gain_gap") == 0.05, "dense diagnostic tolerance changed")
    _require(
        optimization.get("candidate_sources")
        == ["interval_boundaries", "stationary_points", "pairwise_intersections"],
        "analytic candidate sources changed",
    )
    _require(optimization.get("dense_grid_used_for_recommendation") is False, "dense grid selects recommendations")
    _require(optimization.get("tie_break") == "smaller_irrigation", "tie break changed")

    penalties = protocol.get("penalties", {})
    _require(penalties.get("new_search_performed") is False, "new penalty search is forbidden")
    _require(penalties.get("reuse_outer_fold_selected_values") is True, "outer penalties are not frozen")

    forbidden = protocol.get("forbidden", {})
    for key, label in (
        ("2019_rows_read_for_training_or_selection", "2019 training/selection"),
        ("2019_features_read", "2019 features"),
        ("2019_targets_read", "2019 targets"),
        ("2024_rows_read", "2024 rows"),
        ("swap_rerun", "SWAP rerun"),
        ("moe_training", "MoE training"),
        ("tta", "TTA"),
        ("model_promotion", "model promotion"),
    ):
        _require(forbidden.get(key) is True, f"{label} is not frozen as forbidden")

    _require(
        protocol.get("passing_action")
        == "authorize_2015_2018_robust_envelope_development_swap_protocol_design_only",
        "passing action changed",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--source-development-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    arguments = parse_args()
    protocol = json.loads(arguments.protocol.read_text(encoding="utf-8"))
    validate_protocol(protocol)
    raise RuntimeError("robust envelope development runner is not implemented yet")


if __name__ == "__main__":
    main()
