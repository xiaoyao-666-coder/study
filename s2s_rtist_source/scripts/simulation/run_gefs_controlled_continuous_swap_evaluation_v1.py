#!/usr/bin/env python3
"""Rerun SWAP for frozen continuous recommendations and an adaptive oracle."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from scripts.simulation.run_gefs_checkpoint_one_date_eight_ir_smoke_v1 import (
    build_ensemble_mean_weather,
    copy_checkpoint,
    endpoint_fallback_metadata,
    inject_future_weather,
    patch_workspace_dtmax,
    run_checkpoint_branches,
    sha256_file,
    validate_checkpoint,
)


DEFAULT_RECOMMENDATION_DIR = (
    Path("site_general_surrogate_eval")
    / "gefs_teacher_aligned_controlled_continuous_recommendations_v1"
)
DEFAULT_FORMAL_LABEL_DIR = (
    Path("site_general_surrogate_eval")
    / "gefs_exact_schedule_formal_labels_frozen_v1"
)
DEFAULT_OUTPUT_DIR = (
    Path("site_general_surrogate_eval")
    / "gefs_teacher_aligned_controlled_continuous_swap_evaluation_v1"
)
RECOMMENDATION_PLAN_NAME = "gefs_controlled_continuous_swap_plan_v1.csv"
FORMAL_WEATHER_NAME = "gefs_exact_schedule_2015_2019_formal_weather_v1.csv"
FIXED_IRRIGATION_MM = (0.0, 10.0, 15.0, 20.0, 25.0, 30.0, 40.0, 60.0)
TARGET_YEAR = 2019
SITES = ("P1", "P15")


def numeric_grid(start: float, stop: float, step: float) -> list[float]:
    if step <= 0.0 or stop < start:
        raise ValueError("invalid irrigation grid controls")
    values = np.arange(start, stop + step / 2.0, step, dtype=float)
    return sorted(
        set(float(value) for value in np.round(np.clip(values, start, stop), 6))
    )


def fallback_policy(values: list[float]) -> dict[float, float]:
    policy = {}
    for requested in values:
        requested = float(requested)
        if requested <= 0.0:
            continue
        candidate = round(max(0.0, requested - 0.2), 1)
        if candidate < requested:
            policy[requested] = candidate
    return policy


def unit_resources(
    formal_label_dir: Path, site_id: str, decision_date: str
) -> tuple[Path, Path, Path]:
    unit = formal_label_dir / "units" / f"Y{TARGET_YEAR}" / site_id
    checkpoint_root = unit / "all_checkpoints_v1"
    checkpoint = checkpoint_root / "checkpoints" / pd.Timestamp(
        decision_date
    ).strftime("%Y%m%d")
    checkpoint_audit = (
        checkpoint_root / "swap_season_checkpoint_equivalence_v1.csv"
    )
    workspace = unit / "workspace"
    required = [workspace, checkpoint, checkpoint_audit]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(f"formal reusable site-year resources are missing: {missing}")
    return workspace, checkpoint, checkpoint_audit


def stage_paths(root: Path) -> dict[str, Path]:
    return {
        "candidates": root / "swap_candidates.csv",
        "daily_weather": root / "ensemble_mean_weather.csv",
        "injection": root / "weather_injection_audit.csv",
        "audit": root / "stage_audit.json",
    }


def load_passed_stage(
    root: Path, requested_values: list[float]
) -> pd.DataFrame | None:
    paths = stage_paths(root)
    if not paths["audit"].is_file() or not paths["candidates"].is_file():
        return None
    audit = json.loads(paths["audit"].read_text(encoding="utf-8"))
    expected = sorted(float(value) for value in requested_values)
    observed = sorted(float(value) for value in audit.get("requested_irrigation_mm", []))
    if audit.get("mandatory_gate_passed", False) and observed == expected:
        return pd.read_csv(paths["candidates"])
    return None


def passed_stage_attempt(
    base: Path, requested_values: list[float]
) -> tuple[pd.DataFrame, Path] | None:
    candidates = [base]
    candidates.extend(sorted(base.parent.glob(f"{base.name}_attempt_*")))
    for root in candidates:
        loaded = load_passed_stage(root, requested_values)
        if loaded is not None:
            return loaded, root
    return None


def stage_attempt_root(base: Path, *, resume: bool) -> Path:
    if not base.exists():
        return base
    if not resume:
        raise FileExistsError(f"SWAP evaluation stage already exists: {base}")
    attempt = 2
    while base.with_name(f"{base.name}_attempt_{attempt:03d}").exists():
        attempt += 1
    return base.with_name(f"{base.name}_attempt_{attempt:03d}")


def validate_stage_candidates(
    candidates: pd.DataFrame,
    *,
    requested_values: list[float],
    fallbacks: dict[float, float],
) -> dict[str, Any]:
    required = {
        "ir",
        "target_value",
        "water_balance_residual_0_100cm_7d_mm",
    }
    if missing := sorted(required - set(candidates.columns)):
        raise ValueError(f"SWAP candidates are missing fields: {missing}")
    irrigation = pd.to_numeric(candidates["ir"], errors="raise").astype(float)
    target = pd.to_numeric(candidates["target_value"], errors="raise").astype(float)
    residual = pd.to_numeric(
        candidates["water_balance_residual_0_100cm_7d_mm"], errors="raise"
    ).astype(float)
    requested = sorted(float(value) for value in requested_values)
    observed = sorted(irrigation.tolist())
    fallback = endpoint_fallback_metadata(candidates, fallbacks)
    mandatory = bool(
        len(candidates) == len(requested)
        and observed == requested
        and not irrigation.duplicated().any()
        and np.isfinite(target).all()
        and np.isfinite(residual).all()
        and float(np.max(np.abs(residual))) <= 0.5
        and fallback["valid"]
    )
    return {
        "mandatory_gate_passed": mandatory,
        "candidate_rows": len(candidates),
        "requested_irrigation_mm": requested,
        "maximum_absolute_water_balance_residual_mm": float(
            np.max(np.abs(residual))
        ),
        "numerical_irrigation_fallback_metadata": fallback,
        "numerical_irrigation_fallback_strategy": (
            "legacy_workspace_compatible_single_downward_0.2mm"
        ),
    }


def run_swap_stage(
    *,
    base_root: Path,
    requested_values: list[float],
    source_workspace: Path,
    checkpoint_dir: Path,
    checkpoint_audit: Path,
    weather: pd.DataFrame,
    site_id: str,
    decision_date: str,
    resume: bool,
    restart_nprintday: int,
    swap_dtmax_days: float,
    target_year: int = TARGET_YEAR,
    sowing_month_day: str = "04-26",
) -> tuple[pd.DataFrame, Path]:
    requested_values = sorted(set(round(float(value), 6) for value in requested_values))
    existing = passed_stage_attempt(base_root, requested_values)
    if existing is not None:
        return existing
    root = stage_attempt_root(base_root, resume=resume)
    root.mkdir(parents=True)
    paths = stage_paths(root)
    workspace = root / "workspace"
    checkpoint = validate_checkpoint(
        checkpoint_dir, checkpoint_audit, decision_date
    )
    daily = build_ensemble_mean_weather(
        weather, site_id=site_id, decision_date=decision_date
    )
    shutil.copytree(source_workspace, workspace)
    observed_dtmax = patch_workspace_dtmax(workspace, swap_dtmax_days)
    injection, _ = inject_future_weather(workspace, daily, year=target_year)
    copy_checkpoint(checkpoint_dir, workspace)
    fallbacks = fallback_policy(requested_values)
    candidates = run_checkpoint_branches(
        workspace=workspace,
        checkpoint_dir=checkpoint_dir,
        decision_date=decision_date,
        year=target_year,
        sowing_month_day=sowing_month_day,
        restart_nprintday=restart_nprintday,
        irrigation_options_mm=requested_values,
        numerical_endpoint_fallbacks_mm=fallbacks,
    )
    candidates.insert(0, "site", site_id)
    candidates["target_year"] = int(target_year)
    candidates["checkpoint_date"] = checkpoint["checkpoint_date"]
    candidates["prestate_swap_rerun"] = False
    candidates["swap_dtmax_days"] = float(observed_dtmax)
    candidates["weather_driver_source"] = (
        "frozen_corrected_GEFS_5member_ensemble_mean"
    )
    candidates["training_eligible"] = False
    validation = validate_stage_candidates(
        candidates,
        requested_values=requested_values,
        fallbacks=fallbacks,
    )
    audit = {
        "status": (
            "controlled_continuous_swap_stage_passed"
            if validation["mandatory_gate_passed"]
            else "controlled_continuous_swap_stage_failed"
        ),
        **validation,
        "site_id": site_id,
        "target_year": int(target_year),
        "decision_date": decision_date,
        "checkpoint_equivalence": checkpoint,
        "restart_nprintday": restart_nprintday,
        "swap_dtmax_days": float(observed_dtmax),
        "prestate_swap_rerun_count": 0,
        "surrogate_training_performed": False,
        "tta_performed": False,
    }
    candidates.to_csv(paths["candidates"], index=False)
    daily.to_csv(paths["daily_weather"], index=False)
    injection.to_csv(paths["injection"], index=False)
    paths["audit"].write_text(
        json.dumps(audit, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    if not validation["mandatory_gate_passed"]:
        raise RuntimeError(f"SWAP stage gate failed; see {paths['audit']}")
    return candidates, root


def lookup_gain(candidates: pd.DataFrame, irrigation_mm: float) -> float:
    irrigation = pd.to_numeric(candidates["ir"], errors="raise").to_numpy(float)
    matches = np.isclose(irrigation, float(irrigation_mm), rtol=0.0, atol=1.0e-6)
    if int(matches.sum()) != 1:
        raise ValueError(f"SWAP result does not uniquely contain {irrigation_mm:g} mm")
    return float(
        pd.to_numeric(candidates.loc[matches, "target_value"], errors="raise").iloc[0]
    )


def evaluate_cycle(
    candidates: pd.DataFrame,
    recommendations: pd.DataFrame,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    data = candidates.copy()
    data["ir"] = pd.to_numeric(data["ir"], errors="raise").astype(float)
    data["target_value"] = pd.to_numeric(
        data["target_value"], errors="raise"
    ).astype(float)
    data = data.sort_values(["ir", "evaluation_stage"]).drop_duplicates(
        "ir", keep="last"
    )
    fixed = data.loc[
        data["ir"].map(
            lambda value: any(
                np.isclose(value, option, rtol=0.0, atol=1.0e-6)
                for option in FIXED_IRRIGATION_MM
            )
        )
    ]
    if len(fixed) != len(FIXED_IRRIGATION_MM):
        raise ValueError("adaptive SWAP evaluation does not contain all fixed candidates")
    fixed_best = fixed.loc[fixed["target_value"].idxmax()]
    oracle = data.loc[data["target_value"].idxmax()]
    rows = []
    for rec in recommendations.itertuples(index=False):
        irrigation = float(rec.continuous_irrigation_mm)
        gain = lookup_gain(data, irrigation)
        rows.append(
            {
                "model_id": rec.model_id,
                "model_role": rec.model_role,
                "site_id": rec.site_id,
                "decision_date": rec.decision_date,
                "continuous_irrigation_mm": irrigation,
                "continuous_recommendation_swap_gain_7d": gain,
                "fixed_eight_swap_oracle_irrigation_mm": float(fixed_best["ir"]),
                "fixed_eight_swap_oracle_gain_7d": float(
                    fixed_best["target_value"]
                ),
                "regret_vs_fixed_eight_swap_7d": float(
                    fixed_best["target_value"] - gain
                ),
                "adaptive_swap_oracle_irrigation_mm": float(oracle["ir"]),
                "adaptive_swap_oracle_gain_7d": float(oracle["target_value"]),
                "regret_vs_adaptive_swap_oracle_7d": float(
                    oracle["target_value"] - gain
                ),
                "continuous_beats_fixed_eight": bool(
                    gain > float(fixed_best["target_value"]) + 1.0e-9
                ),
            }
        )
    summary = {
        "site_id": str(recommendations["site_id"].iloc[0]),
        "decision_date": str(recommendations["decision_date"].iloc[0]),
        "evaluated_irrigation_count": len(data),
        "fixed_eight_swap_oracle_irrigation_mm": float(fixed_best["ir"]),
        "fixed_eight_swap_oracle_gain_7d": float(fixed_best["target_value"]),
        "adaptive_swap_oracle_irrigation_mm": float(oracle["ir"]),
        "adaptive_swap_oracle_gain_7d": float(oracle["target_value"]),
        "adaptive_oracle_gain_over_fixed_eight": float(
            oracle["target_value"] - fixed_best["target_value"]
        ),
    }
    return rows, summary


def run(
    args: argparse.Namespace,
    *,
    artifact_version: str = "v1",
    recommendation_plan_name: str = RECOMMENDATION_PLAN_NAME,
) -> dict[str, Path]:
    recommendation_dir = args.recommendation_dir.resolve()
    formal_label_dir = args.formal_label_dir.resolve()
    output_dir = args.output_dir.resolve()
    if output_dir.exists() and not args.resume:
        raise FileExistsError(f"refusing to overwrite output directory: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=args.resume)
    plan_path = recommendation_dir / recommendation_plan_name
    plan = pd.read_csv(plan_path)
    required = {
        "model_id",
        "model_role",
        "site_id",
        "decision_date",
        "continuous_irrigation_mm",
        "checkpoint_sha256",
    }
    if missing := sorted(required - set(plan.columns)):
        raise ValueError(f"continuous recommendation plan is missing fields: {missing}")
    if set(plan["site_id"].astype(str)) != set(SITES):
        raise ValueError("continuous SWAP plan must contain only P1 and P15")
    if not pd.to_datetime(plan["decision_date"]).dt.year.eq(TARGET_YEAR).all():
        raise ValueError("continuous SWAP plan must contain only 2019 cycles")
    if not plan["continuous_irrigation_mm"].between(0.0, 60.0).all():
        raise ValueError("continuous recommendations must remain within [0, 60] mm")
    cycle_plan = plan[["site_id", "decision_date"]].drop_duplicates()
    cycle_plan = cycle_plan.sort_values(["site_id", "decision_date"])
    if len(cycle_plan) != 27:
        raise ValueError("continuous SWAP evaluation requires 27 P1/P15 cycles")
    if args.max_cycles is not None:
        if args.max_cycles <= 0:
            raise ValueError("max_cycles must be positive")
        cycle_plan = cycle_plan.head(args.max_cycles)

    weather_path = formal_label_dir / FORMAL_WEATHER_NAME
    weather = pd.read_csv(weather_path)
    coarse_values = numeric_grid(0.0, 60.0, args.coarse_step_mm)
    if not all(
        any(np.isclose(value, option, rtol=0.0, atol=1.0e-9) for value in coarse_values)
        for option in FIXED_IRRIGATION_MM
    ):
        raise ValueError("global coarse grid must contain all eight fixed candidates")
    if args.refine_radius_mm <= 0.0:
        raise ValueError("refine radius must be positive")
    decision_rows = []
    cycle_rows = []
    candidate_frames = []
    branch_roots = []
    for index, cycle in enumerate(cycle_plan.itertuples(index=False), start=1):
        site_id = str(cycle.site_id)
        decision_date = str(cycle.decision_date)
        print(
            f"cycle={index}/{len(cycle_plan)} site={site_id} date={decision_date}",
            flush=True,
        )
        recommendations = plan.loc[
            plan["site_id"].astype(str).eq(site_id)
            & plan["decision_date"].astype(str).eq(decision_date)
        ].copy()
        source_workspace, checkpoint_dir, checkpoint_audit = unit_resources(
            formal_label_dir, site_id, decision_date
        )
        cycle_root = (
            output_dir
            / "branches"
            / site_id
            / pd.Timestamp(decision_date).strftime("%Y%m%d")
        )
        coarse, coarse_root = run_swap_stage(
            base_root=cycle_root / "coarse_5mm",
            requested_values=coarse_values,
            source_workspace=source_workspace,
            checkpoint_dir=checkpoint_dir,
            checkpoint_audit=checkpoint_audit,
            weather=weather,
            site_id=site_id,
            decision_date=decision_date,
            resume=args.resume,
            restart_nprintday=args.restart_nprintday,
            swap_dtmax_days=args.swap_dtmax_days,
        )
        coarse_target = pd.to_numeric(coarse["target_value"], errors="raise")
        coarse_best_irrigation = float(
            pd.to_numeric(coarse.loc[coarse_target.idxmax(), "ir"], errors="raise")
        )
        refine_values = numeric_grid(
            max(0.0, coarse_best_irrigation - args.refine_radius_mm),
            min(60.0, coarse_best_irrigation + args.refine_radius_mm),
            args.refine_step_mm,
        )
        refine_values.extend(
            recommendations["continuous_irrigation_mm"].astype(float).tolist()
        )
        refine_values.append(0.0)
        refine_values = sorted(set(round(float(value), 6) for value in refine_values))
        refine, refine_root = run_swap_stage(
            base_root=cycle_root / "refine_0p5mm",
            requested_values=refine_values,
            source_workspace=source_workspace,
            checkpoint_dir=checkpoint_dir,
            checkpoint_audit=checkpoint_audit,
            weather=weather,
            site_id=site_id,
            decision_date=decision_date,
            resume=args.resume,
            restart_nprintday=args.restart_nprintday,
            swap_dtmax_days=args.swap_dtmax_days,
        )
        coarse = coarse.assign(evaluation_stage="global_coarse")
        refine = refine.assign(evaluation_stage="local_refine_and_recommendations")
        combined = pd.concat([coarse, refine], ignore_index=True)
        decisions, cycle_summary = evaluate_cycle(combined, recommendations)
        decision_rows.extend(decisions)
        cycle_rows.append(cycle_summary)
        combined.insert(0, "evaluation_cycle_id", f"{site_id}_{decision_date}")
        candidate_frames.append(combined)
        branch_roots.extend([coarse_root, refine_root])

    decisions = pd.DataFrame(decision_rows)
    cycles = pd.DataFrame(cycle_rows)
    candidates = pd.concat(candidate_frames, ignore_index=True)
    summary = decisions.groupby(
        ["model_id", "model_role", "site_id"], as_index=False
    ).agg(
        cycle_count=("decision_date", "size"),
        mean_continuous_irrigation_mm=("continuous_irrigation_mm", "mean"),
        mean_regret_vs_fixed_eight_swap_7d=(
            "regret_vs_fixed_eight_swap_7d",
            "mean",
        ),
        maximum_regret_vs_fixed_eight_swap_7d=(
            "regret_vs_fixed_eight_swap_7d",
            "max",
        ),
        negative_regret_vs_fixed_eight_count=(
            "regret_vs_fixed_eight_swap_7d",
            lambda values: int((values < -1.0e-9).sum()),
        ),
        mean_regret_vs_adaptive_swap_oracle_7d=(
            "regret_vs_adaptive_swap_oracle_7d",
            "mean",
        ),
        maximum_regret_vs_adaptive_swap_oracle_7d=(
            "regret_vs_adaptive_swap_oracle_7d",
            "max",
        ),
    )
    full_run = len(cycle_plan) == 27
    expected_decisions = len(plan) if full_run else len(decisions)
    mandatory = bool(
        len(decisions) == expected_decisions
        and np.isfinite(
            decisions[
                [
                    "regret_vs_fixed_eight_swap_7d",
                    "regret_vs_adaptive_swap_oracle_7d",
                ]
            ].to_numpy(float)
        ).all()
        and (
            decisions["regret_vs_adaptive_swap_oracle_7d"].to_numpy(float)
            >= -1.0e-6
        ).all()
    )
    outputs = {
        "decisions": output_dir
        / f"gefs_controlled_continuous_swap_decisions_{artifact_version}.csv",
        "summary": output_dir
        / f"gefs_controlled_continuous_swap_summary_{artifact_version}.csv",
        "cycles": output_dir
        / f"gefs_controlled_continuous_swap_oracles_{artifact_version}.csv",
        "candidates": output_dir
        / f"gefs_controlled_continuous_swap_candidates_{artifact_version}.csv",
        "audit": output_dir
        / f"gefs_controlled_continuous_swap_evaluation_audit_{artifact_version}.json",
        "manifest": output_dir
        / f"gefs_controlled_continuous_swap_evaluation_manifest_{artifact_version}.json",
    }
    decisions.to_csv(outputs["decisions"], index=False)
    summary.to_csv(outputs["summary"], index=False)
    cycles.to_csv(outputs["cycles"], index=False)
    candidates.to_csv(outputs["candidates"], index=False)
    audit = {
        "status": (
            "controlled_continuous_swap_evaluation_complete_passed"
            if mandatory and full_run
            else "controlled_continuous_swap_evaluation_partial_passed"
            if mandatory
            else "controlled_continuous_swap_evaluation_failed"
        ),
        "mandatory_gate_passed": mandatory,
        "full_run": full_run,
        "evaluated_site_cycle_count": len(cycle_plan),
        "decision_rows": len(decisions),
        "adaptive_oracle_definition": "global_SWAP_scan_then_local_SWAP_refinement",
        "global_coarse_step_mm": args.coarse_step_mm,
        "local_refine_radius_mm": args.refine_radius_mm,
        "local_refine_step_mm": args.refine_step_mm,
        "continuous_oracle_is_numerically_approximated": True,
        "continuous_oracle_uses_surrogate_predictions": False,
        "continuous_recommendations_rerun_in_swap": True,
        "fixed_eight_baseline_rerun_in_same_swap_protocol": True,
        "interpolation_used_for_formal_metrics": False,
        "negative_fixed_eight_regret_allowed": True,
        "exact_irrigation_match_metric_reported": False,
        "target_years_read": [2019],
        "test_2024_rows_read": 0,
        "surrogate_training_performed": False,
        "tta_performed": False,
        "restart_nprintday": args.restart_nprintday,
        "swap_dtmax_days": args.swap_dtmax_days,
        "branch_stage_roots": [str(path) for path in branch_roots],
        "next_gate": "review_continuous_swap_regret_and_oracle_gap",
    }
    outputs["audit"].write_text(
        json.dumps(audit, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    manifest = {
        "status": audit["status"],
        "inputs": {
            "recommendation_plan": {
                "path": str(plan_path),
                "sha256": sha256_file(plan_path),
            },
            "formal_weather": {
                "path": str(weather_path),
                "sha256": sha256_file(weather_path),
            },
        },
        "outputs": {
            name: {"path": path.name, "sha256": sha256_file(path)}
            for name, path in outputs.items()
            if name != "manifest"
        },
    }
    outputs["manifest"].write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    if not mandatory:
        raise RuntimeError(f"continuous SWAP evaluation failed; see {outputs['audit']}")
    return outputs


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--recommendation-dir", type=Path, default=DEFAULT_RECOMMENDATION_DIR
    )
    parser.add_argument("--formal-label-dir", type=Path, default=DEFAULT_FORMAL_LABEL_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--coarse-step-mm", type=float, default=5.0)
    parser.add_argument("--refine-radius-mm", type=float, default=5.0)
    parser.add_argument("--refine-step-mm", type=float, default=0.5)
    parser.add_argument("--restart-nprintday", type=int, default=48)
    parser.add_argument("--swap-dtmax-days", type=float, default=0.01)
    parser.add_argument("--max-cycles", type=int)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args(argv)


if __name__ == "__main__":
    generated = run(parse_args())
    for name, path in generated.items():
        print(f"{name}: {path}", flush=True)
