#!/usr/bin/env python3
"""Replay v5 while preserving unlimited endpoints and shrinking limited ones."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
for import_root in (PROJECT_ROOT, PROJECT_ROOT / "src"):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

from scripts.training.run_gefs_gam_expert_stability_gate_development_v2 import (
    build_performance_gate,
)


PROTOCOL_ID = (
    "teacher-guided-gam-v8-unlimited-preserving-confidence-shrinkage-replay-v1"
)
CONFIDENCE_REFERENCE_LCB_MM = 5.0
EXPECTED_CYCLES = 196
EXPECTED_FOLDS = 15
EXPECTED_SITES = 5
TOLERANCE = 1.0e-6

V5_VALIDATION_NAME = (
    "gefs_gam_anchor_trust_region_mixture_validation_decisions_v5.csv"
)
V5_GATE_NAME = "gefs_gam_anchor_trust_region_mixture_replay_gate_v5.json"
V5_AUDIT_NAME = "gefs_gam_anchor_trust_region_mixture_replay_audit_v5.json"
V5_MANIFEST_NAME = (
    "gefs_gam_anchor_trust_region_mixture_replay_manifest_v5.json"
)
V6_GATE_NAME = "gefs_gam_v6_confidence_protected_anchor_replay_gate_v1.json"
V6_AUDIT_NAME = "gefs_gam_v6_confidence_protected_anchor_replay_audit_v1.json"
V6_MANIFEST_NAME = (
    "gefs_gam_v6_confidence_protected_anchor_replay_manifest_v1.json"
)
V7_GATE_NAME = (
    "gefs_gam_v7_continuous_confidence_shrinkage_replay_gate_v1.json"
)
V7_AUDIT_NAME = (
    "gefs_gam_v7_continuous_confidence_shrinkage_replay_audit_v1.json"
)
V7_MANIFEST_NAME = (
    "gefs_gam_v7_continuous_confidence_shrinkage_replay_manifest_v1.json"
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def require_manifest_hash(
    manifest: dict[str, Any], output_name: str, path: Path
) -> None:
    entry = manifest.get("outputs", {}).get(output_name)
    if not isinstance(entry, dict) or not isinstance(entry.get("sha256"), str):
        raise ValueError(f"manifest entry is missing: {output_name}")
    if entry["sha256"].lower() != sha256_file(path).lower():
        raise ValueError(f"manifest hash mismatch: {output_name}")


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


def load_and_validate_upstream(
    protocol: dict[str, Any],
    v5_dir: Path,
    v6_dir: Path,
    v7_dir: Path,
) -> tuple[dict[str, Path], dict[str, Any]]:
    paths = {
        "v5_validation": v5_dir / V5_VALIDATION_NAME,
        "v5_gate": v5_dir / V5_GATE_NAME,
        "v5_audit": v5_dir / V5_AUDIT_NAME,
        "v5_manifest": v5_dir / V5_MANIFEST_NAME,
        "v6_gate": v6_dir / V6_GATE_NAME,
        "v6_audit": v6_dir / V6_AUDIT_NAME,
        "v6_manifest": v6_dir / V6_MANIFEST_NAME,
        "v7_gate": v7_dir / V7_GATE_NAME,
        "v7_audit": v7_dir / V7_AUDIT_NAME,
        "v7_manifest": v7_dir / V7_MANIFEST_NAME,
    }
    missing = [str(path) for path in paths.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"required v8 replay inputs are missing: {missing}")

    v5_manifest = read_json(paths["v5_manifest"])
    for name in ("validation", "gate", "audit"):
        require_manifest_hash(v5_manifest, name, paths[f"v5_{name}"])
    v6_manifest = read_json(paths["v6_manifest"])
    for name in ("gate", "audit"):
        require_manifest_hash(v6_manifest, name, paths[f"v6_{name}"])
    v7_manifest = read_json(paths["v7_manifest"])
    for name in ("gate", "audit"):
        require_manifest_hash(v7_manifest, name, paths[f"v7_{name}"])

    upstream = protocol["upstream"]
    v5_gate = read_json(paths["v5_gate"])
    v5_audit = read_json(paths["v5_audit"])
    if v5_audit.get("status") != upstream["required_v5_status"]:
        raise ValueError("frozen v5 development status changed")
    if (
        v5_gate.get("passed") is not upstream["required_v5_gate_passed"]
        or v5_audit.get("mandatory_execution_gate_passed") is not True
        or v5_audit.get("predeclared_performance_gate_passed") is not True
    ):
        raise ValueError("frozen v5 development gate changed")
    if (
        int(v5_audit.get("2019_rows_read", -1)) != 0
        or int(v5_audit.get("2024_rows_read", -1)) != 0
    ):
        raise ValueError("frozen v5 development accessed held-out years")
    inner_complete = v5_gate.get("conditions", {}).get(
        "all_inner_source_site_holds_complete"
    )
    if inner_complete is not True:
        raise ValueError("frozen v5 inner source-site holds changed")

    v6_gate = read_json(paths["v6_gate"])
    v6_audit = read_json(paths["v6_audit"])
    if v6_audit.get("status") != upstream["required_v6_status"]:
        raise ValueError("frozen v6 development status changed")
    if (
        v6_gate.get("passed") is not upstream["required_v6_gate_passed"]
        or v6_audit.get("mandatory_execution_gate_passed") is not True
        or v6_audit.get("predeclared_performance_gate_passed") is not True
    ):
        raise ValueError("frozen v6 development gate changed")
    if float(
        v6_gate.get("limited_candidate_required_lcb_mm", np.nan)
    ) != float(upstream["required_v6_confidence_reference_lcb_mm"]):
        raise ValueError("frozen v6 confidence reference changed")
    if (
        int(v6_audit.get("2019_rows_read", -1)) != 0
        or int(v6_audit.get("2024_rows_read", -1)) != 0
    ):
        raise ValueError("frozen v6 development accessed held-out years")

    v7_gate = read_json(paths["v7_gate"])
    v7_audit = read_json(paths["v7_audit"])
    if v7_audit.get("status") != upstream["required_v7_status"]:
        raise ValueError("frozen v7 development status changed")
    if (
        v7_gate.get("passed") is not upstream["required_v7_gate_passed"]
        or v7_audit.get("mandatory_execution_gate_passed") is not True
        or v7_audit.get("predeclared_performance_gate_passed") is not False
    ):
        raise ValueError("frozen v7 development gate changed")
    if (
        int(v7_audit.get("2019_target_rows_read_by_replay", -1)) != 0
        or int(v7_audit.get("2019_swap_results_read_by_replay", -1)) != 0
        or int(v7_audit.get("2024_rows_read", -1)) != 0
    ):
        raise ValueError("frozen v7 replay accessed held-out outcomes")
    return paths, {"inner_holds_complete": inner_complete}


def run(args: argparse.Namespace) -> dict[str, Path]:
    protocol_path = Path(args.protocol).resolve()
    v5_dir = Path(args.v5_dir).resolve()
    v6_dir = Path(args.v6_dir).resolve()
    v7_dir = Path(args.v7_dir).resolve()
    output_dir = Path(args.output_dir).resolve()
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite output directory: {output_dir}")

    protocol = read_json(protocol_path)
    validate_protocol(protocol)
    input_paths, upstream_state = load_and_validate_upstream(
        protocol, v5_dir, v6_dir, v7_dir
    )
    validation = pd.read_csv(input_paths["v5_validation"])
    boundary = protocol["evaluation_boundary"]
    if (
        len(validation) != EXPECTED_CYCLES
        or validation["fold_id"].nunique() != EXPECTED_FOLDS
        or validation["target_site"].nunique() != EXPECTED_SITES
    ):
        raise ValueError("frozen v5 validation structure changed")
    years = sorted(
        pd.to_numeric(validation["validation_year"], errors="raise")
        .astype(int)
        .unique()
        .tolist()
    )
    if years != boundary["target_validation_years"]:
        raise ValueError("frozen v5 validation years changed")
    keys = ["fold_id", "target_site", "site_id", "decision_date"]
    if validation.duplicated(keys).any():
        raise ValueError("frozen v5 validation cycle keys are not unique")

    replay = apply_unlimited_preserving_confidence_shrinkage(validation)
    performance = protocol["performance_gate"]
    gate, fold_metrics, site_metrics = build_performance_gate(
        replay,
        inner_holds_complete=bool(upstream_state["inner_holds_complete"]),
        minimum_nonworse_folds=int(
            performance["minimum_nonworse_outer_folds"]
        ),
        minimum_nonworse_sites=int(
            performance["minimum_nonworse_target_sites"]
        ),
        minimum_changed_cycles=int(
            performance["minimum_changed_irrigation_cycles"]
        ),
        minimum_changed_sites=int(performance["minimum_changed_target_sites"]),
    )
    gate.update(
        {
            "protocol_id": PROTOCOL_ID,
            "gate_type": (
                "predeclared_post_v7_unlimited_preserving_confidence_"
                "shrinkage_development_replay_gate"
            ),
            "confidence_reference_lcb_mm": CONFIDENCE_REFERENCE_LCB_MM,
            "unlimited_endpoint_preserved": True,
            "recommendation_interpolation": True,
            "swap_outcome_interpolation": False,
            "2019_access_authorized": False,
            "automatic_model_promotion": False,
            "passing_action": (
                "authorize_separate_v8_changed_cycle_exact_paired_swap_"
                "protocol_design_only"
            ),
            "failing_action": (
                "freeze_v8_unlimited_preserving_confidence_shrinkage_"
                "as_negative_stop_before_swap_or_2019"
            ),
        }
    )

    output_dir.mkdir(parents=True)
    outputs = {
        "decisions": output_dir
        / "gefs_gam_v8_unlimited_preserving_confidence_shrinkage_decisions_v1.csv",
        "fold_metrics": output_dir
        / "gefs_gam_v8_unlimited_preserving_confidence_shrinkage_fold_metrics_v1.csv",
        "site_metrics": output_dir
        / "gefs_gam_v8_unlimited_preserving_confidence_shrinkage_site_metrics_v1.csv",
        "gate": output_dir
        / "gefs_gam_v8_unlimited_preserving_confidence_shrinkage_replay_gate_v1.json",
        "audit": output_dir
        / "gefs_gam_v8_unlimited_preserving_confidence_shrinkage_replay_audit_v1.json",
        "manifest": output_dir
        / "gefs_gam_v8_unlimited_preserving_confidence_shrinkage_replay_manifest_v1.json",
    }
    replay.to_csv(outputs["decisions"], index=False)
    fold_metrics.to_csv(outputs["fold_metrics"], index=False)
    site_metrics.to_csv(outputs["site_metrics"], index=False)
    write_json(outputs["gate"], gate)

    changed = replay["irrigation_changed"].astype(bool)
    audit = {
        "status": (
            "gam_v8_unlimited_preserving_confidence_shrinkage_replay_"
            "passed_pending_changed_cycle_exact_swap_design"
            if gate["passed"]
            else "gam_v8_unlimited_preserving_confidence_shrinkage_replay_"
            "failed_stop_before_swap_or_2019"
        ),
        "protocol_id": PROTOCOL_ID,
        "mandatory_execution_gate_passed": True,
        "predeclared_performance_gate_passed": bool(gate["passed"]),
        "post_v7_development_iteration": True,
        "v5_swap_and_v7_results_reviewed_before_design": True,
        "2019_covariates_reviewed_before_design": True,
        "2019_targets_reviewed_before_design": False,
        "2019_swap_results_reviewed_before_design": False,
        "independent_confirmation": False,
        "selection_uses_site_id": False,
        "selection_uses_year": False,
        "selection_uses_target_or_oracle": False,
        "selection_uses_swap_gain_or_regret": False,
        "threshold_search_performed": False,
        "model_training_performed": False,
        "gate_model_fit_performed": False,
        "expert_reselection_performed": False,
        "gam_bspline_refit_performed": False,
        "recommendation_optimization_performed": False,
        "recommendation_interpolation_performed": True,
        "swap_outcome_interpolation_performed": False,
        "swap_rerun_performed": False,
        "automatic_model_promotion": False,
        "cycle_count": int(len(replay)),
        "outer_fold_count": int(replay["fold_id"].nunique()),
        "target_site_count": int(replay["target_site"].nunique()),
        "target_validation_peak_labels_read": int(len(replay)),
        "irrigation_changed_count": int(changed.sum()),
        "irrigation_changed_site_count": int(
            replay.loc[changed, "target_site"].nunique()
        ),
        "unlimited_changed_cycle_count": int(
            (changed & ~_strict_bool(replay["trust_region_limited"], "trust_region_limited")).sum()
        ),
        "limited_changed_cycle_count": int(
            (changed & _strict_bool(replay["trust_region_limited"], "trust_region_limited")).sum()
        ),
        "2019_rows_read_by_replay": 0,
        "2019_target_rows_read_by_replay": 0,
        "2019_swap_results_read_by_replay": 0,
        "2024_rows_read": 0,
        "next_gate": gate["passing_action"] if gate["passed"] else gate["failing_action"],
        "inputs": {
            "protocol_sha256": sha256_file(protocol_path),
            **{name: sha256_file(path) for name, path in input_paths.items()},
        },
    }
    write_json(outputs["audit"], audit)
    write_json(
        outputs["manifest"],
        {
            "status": audit["status"],
            "inputs": {
                "protocol": {
                    "path": str(protocol_path),
                    "sha256": sha256_file(protocol_path),
                },
                **{
                    name: {"path": str(path), "sha256": sha256_file(path)}
                    for name, path in input_paths.items()
                },
            },
            "outputs": {
                name: {"path": path.name, "sha256": sha256_file(path)}
                for name, path in outputs.items()
                if name != "manifest"
            },
        },
    )
    if not gate["passed"]:
        raise RuntimeError(
            f"v8 unlimited-preserving confidence shrinkage replay gate failed; "
            f"see {outputs['gate']}"
        )
    return outputs


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--v5-dir", type=Path, required=True)
    parser.add_argument("--v6-dir", type=Path, required=True)
    parser.add_argument("--v7-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args(argv)


if __name__ == "__main__":
    for output_name, output_path in run(parse_args()).items():
        print(f"{output_name}: {output_path}", flush=True)
