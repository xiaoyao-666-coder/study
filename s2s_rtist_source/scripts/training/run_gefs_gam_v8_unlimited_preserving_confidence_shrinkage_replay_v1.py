#!/usr/bin/env python3
"""Replay v5 while preserving unlimited endpoints and shrinking limited ones."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


PROTOCOL_ID = (
    "teacher-guided-gam-v8-unlimited-preserving-confidence-shrinkage-replay-v1"
)
CONFIDENCE_REFERENCE_LCB_MM = 5.0
EXPECTED_CYCLES = 196
EXPECTED_FOLDS = 15
EXPECTED_SITES = 5
TOLERANCE = 1.0e-6


def _strict_bool(series: pd.Series, name: str) -> np.ndarray:
    if pd.api.types.is_bool_dtype(series):
        return series.to_numpy(dtype=bool)
    normalized = series.astype(str).str.strip().str.lower()
    mapping = {
        "true": True,
        "false": False,
        "1": True,
        "0": False,
        "1.0": True,
        "0.0": False,
    }
    if not normalized.isin(mapping).all():
        raise ValueError(f"{name} must contain only strict boolean values")
    return normalized.map(mapping).to_numpy(dtype=bool)


def validate_protocol(protocol: dict[str, Any]) -> None:
    if protocol.get("protocol_id") != PROTOCOL_ID:
        raise ValueError("v8 protocol identity changed")

    upstream = protocol.get("upstream", {})
    expected_upstream = {
        "required_v5_status": (
            "gam_anchor_trust_region_mixture_replay_v5_passed_pending_"
            "changed_cycle_swap_design"
        ),
        "required_v5_gate_passed": True,
        "required_v6_status": (
            "gam_v6_confidence_protected_anchor_replay_passed_pending_"
            "2019_independent_confirmation_design"
        ),
        "required_v6_gate_passed": True,
        "required_v6_confidence_reference_lcb_mm": 5.0,
        "required_v7_status": (
            "gam_v7_continuous_confidence_shrinkage_replay_failed_"
            "stop_before_swap_or_2019"
        ),
        "required_v7_gate_passed": False,
    }
    for name, expected in expected_upstream.items():
        if upstream.get(name) != expected:
            raise ValueError(f"v8 upstream boundary changed: {name}")

    disclosure = protocol.get("development_disclosure", {})
    if disclosure.get("v5_swap_and_v7_results_reviewed_before_design") is not True:
        raise ValueError("v8 development disclosure changed")
    if disclosure.get("2019_covariates_reviewed_before_design") is not True:
        raise ValueError("2019 covariate disclosure changed")
    if disclosure.get("development_years_reused") != [2015, 2016, 2017, 2018]:
        raise ValueError("development-year disclosure changed")
    if any(
        disclosure.get(name) is not False
        for name in (
            "2019_targets_reviewed_before_design",
            "2019_swap_results_reviewed_before_design",
            "independent_confirmation",
        )
    ):
        raise ValueError("v8 outcome disclosure changed")

    rule = protocol.get("selection_rule", {})
    if rule.get("family") != (
        "unlimited_preserving_limited_lcb_confidence_shrinkage"
    ):
        raise ValueError("v8 selection family changed")
    if float(rule.get("confidence_reference_lcb_mm", np.nan)) != (
        CONFIDENCE_REFERENCE_LCB_MM
    ):
        raise ValueError("v8 confidence reference changed")
    if rule.get("limited_scale_clip") != [0.0, 1.0]:
        raise ValueError("v8 limited scale clip changed")
    if float(rule.get("unchanged_scale", np.nan)) != 0.0:
        raise ValueError("v8 unchanged scale changed")
    if float(rule.get("unlimited_scale", np.nan)) != 1.0:
        raise ValueError("v8 unlimited scale changed")
    if rule.get("allowed_decision_features") != [
        "anchor_recommendation_mm",
        "v5_recommendation_mm",
        "selected_lcb_mm",
        "trust_region_limited",
    ]:
        raise ValueError("v8 decision feature contract changed")
    if any(
        rule.get(name) is not False
        for name in (
            "threshold_search",
            "site_id_as_selection_feature",
            "year_as_selection_feature",
            "target_or_oracle_features",
            "swap_gain_or_regret_features",
            "site_specific_thresholds",
            "year_specific_thresholds",
        )
    ):
        raise ValueError("forbidden v8 selection input or search enabled")

    performance = protocol.get("performance_gate", {})
    thresholds = {
        "minimum_nonworse_outer_folds": 12,
        "minimum_nonworse_target_sites": 4,
        "minimum_changed_irrigation_cycles": 10,
        "minimum_changed_target_sites": 3,
    }
    for name, expected in thresholds.items():
        if int(performance.get(name, -1)) != expected:
            raise ValueError(f"v8 performance threshold changed: {name}")
    required_gates = (
        "overall_mean_peak_distance_nonworse",
        "positive_oracle_mean_peak_distance_nonworse",
        "zero_oracle_mean_peak_distance_nonworse",
        "global_maximum_peak_distance_nonworse",
        "all_numeric_outputs_finite",
    )
    if any(performance.get(name) is not True for name in required_gates):
        raise ValueError("a mandatory v8 peak gate was disabled")
    if float(performance.get("tolerance", np.nan)) != TOLERANCE:
        raise ValueError("v8 gate tolerance changed")

    boundary = protocol.get("evaluation_boundary", {})
    expected_counts = {
        "cycle_count": EXPECTED_CYCLES,
        "outer_fold_count": EXPECTED_FOLDS,
        "target_site_count": EXPECTED_SITES,
    }
    for name, expected in expected_counts.items():
        if int(boundary.get(name, -1)) != expected:
            raise ValueError(f"v8 evaluation structure changed: {name}")
    if boundary.get("target_validation_years") != [2016, 2017, 2018]:
        raise ValueError("v8 target validation years changed")
    if any(
        boundary.get(name) is not False
        for name in (
            "2019_access",
            "2024_access",
            "automatic_model_promotion",
            "expert_reselection",
            "gam_bspline_refit",
            "gate_model_fit",
            "recommendation_optimization",
            "swap_outcome_interpolation",
            "swap_rerun",
        )
    ):
        raise ValueError("v8 evaluation isolation changed")


def apply_unlimited_preserving_confidence_shrinkage(
    frame: pd.DataFrame,
) -> pd.DataFrame:
    required = {
        "anchor_recommendation_mm",
        "recommendation_mm",
        "selected_lcb_mm",
        "trust_region_limited",
        "true_peak_mm",
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"v8 replay fields are missing: {missing}")

    replay = frame.copy()
    anchor = pd.to_numeric(
        replay["anchor_recommendation_mm"], errors="raise"
    ).to_numpy(float)
    v5 = pd.to_numeric(replay["recommendation_mm"], errors="raise").to_numpy(
        float
    )
    lcb = pd.to_numeric(replay["selected_lcb_mm"], errors="raise").to_numpy(
        float
    )
    truth = pd.to_numeric(replay["true_peak_mm"], errors="raise").to_numpy(
        float
    )
    limited = _strict_bool(
        replay["trust_region_limited"], "trust_region_limited"
    )
    if not np.isfinite(np.column_stack([anchor, v5, lcb, truth])).all():
        raise ValueError("v8 replay inputs are non-finite")
    if (
        (anchor < -TOLERANCE).any()
        or (anchor > 60.0 + TOLERANCE).any()
        or (v5 < -TOLERANCE).any()
        or (v5 > 60.0 + TOLERANCE).any()
        or (truth < -TOLERANCE).any()
        or (truth > 60.0 + TOLERANCE).any()
    ):
        raise ValueError("v8 irrigation or peak is outside [0, 60] mm")

    v5_movement = v5 - anchor
    if "v5_movement_from_anchor_mm" in replay:
        frozen_movement = pd.to_numeric(
            replay["v5_movement_from_anchor_mm"], errors="raise"
        ).to_numpy(float)
        if not np.isfinite(frozen_movement).all() or not np.allclose(
            frozen_movement, v5_movement, rtol=0.0, atol=TOLERANCE
        ):
            raise ValueError("frozen v5 movement differs from its endpoints")

    changed = ~np.isclose(v5_movement, 0.0, rtol=0.0, atol=TOLERANCE)
    limited_scale = np.clip(
        lcb / CONFIDENCE_REFERENCE_LCB_MM, 0.0, 1.0
    )
    scale = np.where(~changed, 0.0, np.where(limited, limited_scale, 1.0))
    v8_movement = v5_movement * scale
    raw_recommendation = anchor + v8_movement
    recommendation = np.clip(raw_recommendation, 0.0, 60.0)
    if not np.allclose(
        recommendation, raw_recommendation, rtol=0.0, atol=TOLERANCE
    ):
        raise ValueError("v8 convex recommendation unexpectedly exceeded bounds")
    if (np.abs(v8_movement) > np.abs(v5_movement) + TOLERANCE).any():
        raise ValueError("v8 movement exceeded the frozen v5 movement")

    replay["v5_recommendation_mm"] = v5
    replay["v5_selected_peak_distance_mm"] = (
        pd.to_numeric(
            replay["selected_peak_distance_mm"], errors="raise"
        ).to_numpy(float)
        if "selected_peak_distance_mm" in replay
        else np.abs(v5 - truth)
    )
    replay["v5_irrigation_changed"] = changed
    replay["v5_movement_from_anchor_mm"] = v5_movement
    replay["v8_confidence_reference_lcb_mm"] = CONFIDENCE_REFERENCE_LCB_MM
    replay["v8_confidence_scale"] = scale
    replay["v8_movement_from_anchor_mm"] = v8_movement
    replay["v8_recommendation_mm"] = recommendation
    replay["recommendation_mm"] = recommendation
    replay["selected_peak_distance_mm"] = np.abs(recommendation - truth)
    replay["irrigation_changed"] = ~np.isclose(
        recommendation, anchor, rtol=0.0, atol=TOLERANCE
    )
    numeric = replay.select_dtypes(include=[np.number]).to_numpy(float)
    if not np.isfinite(numeric).all():
        raise ValueError("v8 replay outputs are non-finite")
    return replay
