#!/usr/bin/env python3
"""Run the P3 B2 regret-sensitive rolling leave-site-out router screen."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd
import torch

from scripts.evaluation.freeze_gefs_p3_2021_dual_gate_five_member_protocol_v1 import (
    DEFAULT_DATASET_DIR,
    DEFAULT_FINAL_MODEL_DIR,
    DEFAULT_OUTPUT_DIR as DEFAULT_DUAL_PROTOCOL_DIR,
    DEFAULT_SCREEN_DIR,
)
from scripts.evaluation.freeze_gefs_p3_strict_loso_cross_year_gate_protocol_v1 import (
    DEVELOPMENT_FOLDS,
)
from scripts.training.run_gefs_p3_strict_loso_cross_year_gate_screen_v1 import (
    actionable_gate_rows,
    build_gate_feature_frame,
    predict_gate_probabilities,
    summarize_policies,
    validation_policy_rows,
)
from scripts.training.train_gefs_exact_schedule_no_tta_baseline_v1 import (
    CONTRACT_NAME,
    DATASET_NAME,
    EXPECTED_IRRIGATION_GRID,
    resolve_device,
    sha256_file,
    write_json,
)
from scripts.training.train_gefs_p3_source_only_router_v1 import (
    expert_decisions,
    load_development_checkpoints,
    load_final_checkpoints,
    read_json,
    require_file,
    router_required_columns,
    validate_dual_protocol,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
B2_POLICY_NAME = "p3_b2_regret_sensitive_router"
BASELINE_POLICY = "always_P15"
ROUTER_SITES = ("P2", "P4")
ALLOWED_YEARS = tuple(range(2015, 2020))
DEFAULT_OUTPUT_DIR = Path(
    "site_general_surrogate_eval/gefs_p3_b2_regret_sensitive_loso_router_screen_v1"
)
OUTPUT_PREFIX = "gefs_p3_b2_regret_sensitive"


def development_folds() -> list[dict[str, Any]]:
    source_folds = {
        int(item["validation_year"]): str(item["protocol_fold"])
        for item in DEVELOPMENT_FOLDS
    }
    folds: list[dict[str, Any]] = []
    for year in range(2016, 2020):
        for held_out_site in ROUTER_SITES:
            training_site = "P4" if held_out_site == "P2" else "P2"
            folds.append(
                {
                    "protocol_fold": f"holdout_{held_out_site}_to_{year}",
                    "source_protocol_fold": source_folds[year],
                    "training_site": training_site,
                    "held_out_site": held_out_site,
                    "train_years": tuple(range(2015, year)),
                    "validation_year": year,
                }
            )
    return folds


def load_b2_router_rows(
    dataset_path: Path, required_columns: Sequence[str]
) -> tuple[pd.DataFrame, dict[str, int]]:
    required = list(dict.fromkeys(str(column) for column in required_columns))
    retained: list[list[str]] = []
    audit = {
        "rows_scanned": 0,
        "rows_retained": 0,
        "rows_skipped_before_materialization": 0,
        "P3_rows_skipped_before_materialization": 0,
        "year_2020_rows_skipped_before_materialization": 0,
        "year_2021_rows_skipped_before_materialization": 0,
        "year_2024_rows_skipped_before_materialization": 0,
        "disallowed_rows_retained": 0,
    }
    with dataset_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.reader(handle)
        try:
            header = next(reader)
        except StopIteration as error:
            raise ValueError("historical dataset is empty") from error
        if missing := sorted(set(required) - set(header)):
            raise ValueError(f"historical dataset is missing B2 router columns: {missing}")
        indices = [header.index(column) for column in required]
        site_index = header.index("site_id")
        year_index = header.index("target_year")
        for row in reader:
            if len(row) != len(header):
                raise ValueError("historical dataset contains a malformed CSV row")
            audit["rows_scanned"] += 1
            site_id = row[site_index]
            try:
                target_year = int(row[year_index])
            except ValueError as error:
                raise ValueError("historical dataset contains an invalid identity year") from error
            if site_id == "P3":
                audit["P3_rows_skipped_before_materialization"] += 1
            if target_year in (2020, 2021, 2024):
                audit[f"year_{target_year}_rows_skipped_before_materialization"] += 1
            if site_id not in ROUTER_SITES or target_year not in ALLOWED_YEARS:
                audit["rows_skipped_before_materialization"] += 1
                continue
            retained.append([row[index] for index in indices])
    frame = pd.DataFrame(retained, columns=required)
    if frame.empty:
        raise ValueError("B2 router scope is empty")
    text_columns = {"sample_id", "split", "site_id", "decision_date"}
    for column in frame.columns:
        if column not in text_columns:
            frame[column] = pd.to_numeric(frame[column], errors="raise")
    frame["target_year"] = frame["target_year"].astype(int)
    disallowed = ~frame["site_id"].isin(ROUTER_SITES) | ~frame["target_year"].isin(
        ALLOWED_YEARS
    )
    audit["disallowed_rows_retained"] = int(disallowed.sum())
    if audit["disallowed_rows_retained"]:
        raise ValueError("disallowed site/year rows survived B2 physical filtering")
    counts = frame.groupby(
        ["target_year", "site_id", "decision_date"], sort=False
    ).size()
    if counts.empty or not counts.eq(len(EXPECTED_IRRIGATION_GRID)).all():
        raise ValueError("B2 router contains incomplete irrigation cycles")
    for _, cycle in frame.groupby(
        ["target_year", "site_id", "decision_date"], sort=False
    ):
        if tuple(sorted(cycle["irrigation_mm"].astype(float))) != EXPECTED_IRRIGATION_GRID:
            raise ValueError("B2 router irrigation grid changed")
    audit["rows_retained"] = len(frame)
    return frame, audit


def attach_regret_weights(rows: pd.DataFrame) -> pd.DataFrame:
    weighted = rows.copy()
    delta = (
        pd.to_numeric(weighted["selected_true_net_gain_7d_p1"], errors="coerce")
        - pd.to_numeric(weighted["selected_true_net_gain_7d_p15"], errors="coerce")
    )
    usable = weighted["usable_for_gate"].astype(bool)
    weighted["realized_gain_delta"] = np.where(usable, delta, np.nan)
    weighted["gate_label_p1_wins"] = np.where(
        usable, (delta > 0.0).astype(float), np.nan
    )
    weighted["raw_regret_weight"] = np.where(usable, np.abs(delta), np.nan)
    raw = weighted.loc[usable, "raw_regret_weight"].to_numpy(dtype=float)
    mean_raw = float(np.mean(raw)) if len(raw) else float("nan")
    if np.isfinite(mean_raw) and mean_raw > 0.0:
        weighted["normalized_regret_weight"] = np.where(
            usable, weighted["raw_regret_weight"] / mean_raw, np.nan
        )
    else:
        weighted["normalized_regret_weight"] = np.nan
    return weighted


def _fallback_gate(
    feature_names: Sequence[str], reason: str, rows: int
) -> dict[str, Any]:
    return {
        "model": "always_P15_fallback",
        "objective": "regret_weighted_sum_binary_cross_entropy_plus_0.5_weight_l2",
        "fallback": True,
        "fallback_reason": reason,
        "feature_names": list(feature_names),
        "training_rows": int(rows),
        "threshold": 0.5,
        "tie_route": "P15",
        "constant_probability_p1": 0.0,
        "weighting": "abs_realized_gain_delta_normalized_to_fold_mean_one",
        "converged": False,
    }


def _sigmoid(values: np.ndarray) -> np.ndarray:
    clipped = np.clip(values, -40.0, 40.0)
    return 1.0 / (1.0 + np.exp(-clipped))


def fit_regret_weighted_logistic_gate(
    rows: pd.DataFrame,
    feature_names: Sequence[str],
    *,
    minimum_rows: int = 8,
    max_iter: int = 100,
    tolerance: float = 1.0e-10,
) -> dict[str, Any]:
    usable = rows.loc[rows["usable_for_gate"].astype(bool)].reset_index(drop=True)
    if len(usable) < int(minimum_rows):
        return _fallback_gate(feature_names, "insufficient_actionable_cycles", len(usable))
    labels = usable["gate_label_p1_wins"].to_numpy(dtype=float)
    values = usable[list(feature_names)].to_numpy(dtype=float)
    deltas = usable["realized_gain_delta"].to_numpy(dtype=float)
    raw_weights = usable["raw_regret_weight"].to_numpy(dtype=float)
    arrays = (values, labels, deltas, raw_weights)
    if not all(np.isfinite(value).all() for value in arrays):
        return _fallback_gate(feature_names, "nonfinite_gate_training_data", len(usable))
    raw_mean = float(raw_weights.mean())
    if raw_mean <= 1.0e-12:
        return _fallback_gate(feature_names, "negligible_mean_absolute_gain_delta", len(usable))
    sample_weights = usable["normalized_regret_weight"].to_numpy(dtype=float)
    if not np.isfinite(sample_weights).all():
        return _fallback_gate(feature_names, "nonfinite_gate_training_data", len(usable))
    if not np.isclose(sample_weights.mean(), 1.0, atol=1.0e-12, rtol=0.0):
        return _fallback_gate(feature_names, "nonunit_normalized_weight_mean", len(usable))
    if len(np.unique(labels)) < 2:
        return _fallback_gate(feature_names, "single_winner_class", len(usable))
    mean = values.mean(axis=0)
    std = values.std(axis=0)
    std[std <= 1.0e-12] = 1.0
    design = np.column_stack([np.ones(len(values)), (values - mean) / std])
    coefficients = np.zeros(design.shape[1], dtype=float)
    regularization = np.eye(design.shape[1], dtype=float)
    regularization[0, 0] = 0.0
    converged = False
    iterations = 0
    for iteration in range(1, int(max_iter) + 1):
        probabilities = _sigmoid(design @ coefficients)
        gradient = design.T @ (sample_weights * (probabilities - labels))
        gradient += regularization @ coefficients
        curvature = sample_weights * np.clip(
            probabilities * (1.0 - probabilities), 1.0e-12, None
        )
        hessian = design.T @ (curvature[:, None] * design) + regularization
        try:
            step = np.linalg.solve(hessian, gradient)
        except np.linalg.LinAlgError:
            return _fallback_gate(feature_names, "singular_logistic_hessian", len(usable))
        if not np.isfinite(step).all():
            return _fallback_gate(feature_names, "nonfinite_logistic_step", len(usable))
        coefficients -= step
        iterations = iteration
        if float(np.max(np.abs(step))) <= float(tolerance):
            converged = True
            break
    if not converged:
        return _fallback_gate(feature_names, "logistic_nonconvergence", len(usable))
    if not np.isfinite(coefficients).all():
        return _fallback_gate(feature_names, "nonfinite_logistic_coefficients", len(usable))
    return {
        "model": "numpy_newton_irls_regret_weighted_l2_logistic",
        "objective": "regret_weighted_sum_binary_cross_entropy_plus_0.5_weight_l2",
        "fallback": False,
        "fallback_reason": "",
        "feature_names": list(feature_names),
        "feature_mean": mean.tolist(),
        "feature_std": std.tolist(),
        "coefficients": coefficients.tolist(),
        "training_rows": len(usable),
        "positive_rows": int(labels.sum()),
        "negative_rows": int((1.0 - labels).sum()),
        "raw_weight_mean": raw_mean,
        "normalized_weight_mean": float(sample_weights.mean()),
        "threshold": 0.5,
        "tie_route": "P15",
        "weighting": "abs_realized_gain_delta_normalized_to_fold_mean_one",
        "feature_standardization": "unweighted_training_fold_mean_and_std",
        "max_iter": int(max_iter),
        "parameter_tolerance": float(tolerance),
        "iterations": iterations,
        "converged": True,
    }


def model_sha256(model: dict[str, Any]) -> str:
    hashable = {key: value for key, value in model.items() if key != "model_sha256"}
    payload = json.dumps(
        hashable, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _policy_rows(
    p1: pd.DataFrame,
    p15: pd.DataFrame,
    probabilities: np.ndarray,
    *,
    fold: dict[str, Any],
    gate_model: dict[str, Any],
) -> pd.DataFrame:
    rows = validation_policy_rows(
        p1,
        p15,
        probabilities,
        protocol_fold=str(fold["protocol_fold"]),
        gate_model=gate_model,
    )
    rows.loc[rows["policy"].eq("p3_logistic_gate"), "policy"] = B2_POLICY_NAME
    rows["held_out_site"] = str(fold["held_out_site"])
    rows["training_site"] = str(fold["training_site"])
    rows["source_protocol_fold"] = str(fold["source_protocol_fold"])
    return rows


def per_site_metrics(per_fold_metrics: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for (site, policy), part in per_fold_metrics.groupby(
        ["held_out_site", "policy"], sort=False
    ):
        rows.append(
            {
                "held_out_site": site,
                "policy": policy,
                "fold_count": len(part),
                "mean_fold_mean_regret_7d": float(part["mean_regret_7d"].mean()),
                "worst_fold_maximum_regret_7d": float(part["maximum_regret_7d"].max()),
                "total_false_positive_count": int(part["false_positive_count"].sum()),
                "total_predicted_60mm_count": int(part["predicted_60mm_count"].sum()),
                "total_effective_P1_route_count": int(
                    part["effective_P1_route_count"].sum()
                ),
            }
        )
    return pd.DataFrame(rows)


def b2_development_gate(
    per_fold_metrics: pd.DataFrame, validation_decisions: pd.DataFrame
) -> pd.DataFrame:
    baseline = per_fold_metrics.loc[
        per_fold_metrics["policy"].eq(BASELINE_POLICY)
    ].set_index("protocol_fold")
    candidate = per_fold_metrics.loc[
        per_fold_metrics["policy"].eq(B2_POLICY_NAME)
    ].set_index("protocol_fold")
    if len(baseline) != 8 or set(baseline.index) != set(candidate.index):
        raise ValueError("B2 development gate requires eight matched folds")
    baseline_macro = float(baseline["mean_regret_7d"].mean())
    candidate_macro = float(candidate["mean_regret_7d"].mean())
    nonworse = candidate["mean_regret_7d"] <= baseline["mean_regret_7d"] + 1.0e-6
    site_results: dict[str, tuple[float, float]] = {}
    for site in ROUTER_SITES:
        fold_ids = baseline.loc[baseline["held_out_site"].eq(site)].index
        if len(fold_ids) != 4:
            raise ValueError(f"B2 development gate requires four folds for {site}")
        site_results[site] = (
            float(baseline.loc[fold_ids, "mean_regret_7d"].mean()),
            float(candidate.loc[fold_ids, "mean_regret_7d"].mean()),
        )
    baseline_worst = float(baseline["maximum_regret_7d"].max())
    candidate_worst = float(candidate["maximum_regret_7d"].max())
    baseline_fp = int(baseline["false_positive_count"].sum())
    candidate_fp = int(candidate["false_positive_count"].sum())
    baseline_60 = int(baseline["predicted_60mm_count"].sum())
    candidate_60 = int(candidate["predicted_60mm_count"].sum())
    candidate_rows = validation_decisions.loc[
        validation_decisions["policy"].eq(B2_POLICY_NAME)
    ]
    effective_p1 = int(
        (
            candidate_rows["selected_source_site_id"].eq("P1")
            & candidate_rows["recommendations_differ"].astype(bool)
        ).sum()
    )
    nonfallback_sites = {
        site: bool(
            (~candidate_rows.loc[
                candidate_rows["held_out_site"].eq(site), "gate_fallback"
            ].astype(bool)).any()
        )
        for site in ROUTER_SITES
    }
    gates = {
        "macro_mean_regret_gate_passed": candidate_macro <= baseline_macro - 1.0e-6,
        "P2_site_mean_regret_gate_passed": site_results["P2"][1]
        <= site_results["P2"][0] + 1.0e-6,
        "P4_site_mean_regret_gate_passed": site_results["P4"][1]
        <= site_results["P4"][0] + 1.0e-6,
        "six_of_eight_nonworse_gate_passed": int(nonworse.sum()) >= 6,
        "worst_maximum_regret_gate_passed": candidate_worst <= baseline_worst + 1.0e-6,
        "false_positive_gate_passed": candidate_fp <= baseline_fp,
        "predicted_60mm_gate_passed": candidate_60 <= baseline_60,
        "P1_route_gate_passed": effective_p1 >= 1,
        "each_site_nonfallback_gate_passed": all(nonfallback_sites.values()),
    }
    return pd.DataFrame(
        [
            {
                "baseline_policy": BASELINE_POLICY,
                "candidate_policy": B2_POLICY_NAME,
                "baseline_macro_mean_regret_7d": baseline_macro,
                "candidate_macro_mean_regret_7d": candidate_macro,
                "P2_baseline_mean_regret_7d": site_results["P2"][0],
                "P2_candidate_mean_regret_7d": site_results["P2"][1],
                "P4_baseline_mean_regret_7d": site_results["P4"][0],
                "P4_candidate_mean_regret_7d": site_results["P4"][1],
                "folds_with_nonworse_mean_regret": int(nonworse.sum()),
                "baseline_worst_fold_maximum_regret_7d": baseline_worst,
                "candidate_worst_fold_maximum_regret_7d": candidate_worst,
                "baseline_total_false_positive_count": baseline_fp,
                "candidate_total_false_positive_count": candidate_fp,
                "baseline_total_predicted_60mm_count": baseline_60,
                "candidate_total_predicted_60mm_count": candidate_60,
                "effective_P1_route_count": effective_p1,
                "P2_nonfallback_fold_present": nonfallback_sites["P2"],
                "P4_nonfallback_fold_present": nonfallback_sites["P4"],
                **gates,
                "development_gate_passed": all(gates.values()),
            }
        ]
    )


def _annotate_samples(samples: pd.DataFrame, fold: dict[str, Any]) -> pd.DataFrame:
    result = attach_regret_weights(samples)
    result["protocol_fold"] = str(fold["protocol_fold"])
    result["source_protocol_fold"] = str(fold["source_protocol_fold"])
    result["training_site"] = str(fold["training_site"])
    result["held_out_site"] = str(fold["held_out_site"])
    result["training_years"] = ",".join(str(year) for year in fold["train_years"])
    result["validation_year"] = int(fold["validation_year"])
    return result


def run(args: argparse.Namespace) -> dict[str, Path]:
    output_dir = args.output_dir
    if output_dir.exists():
        raise FileExistsError(f"output directory already exists: {output_dir}")
    frozen = validate_dual_protocol(args.protocol_dir)
    feature_contract = frozen["protocol"]["B_source_only_router"][
        "gate_feature_contract"
    ]
    feature_names = list(feature_contract["gate_feature_names"])
    dataset_contract_path = require_file(
        args.dataset_dir / CONTRACT_NAME, "historical dataset contract"
    )
    dataset_contract = read_json(dataset_contract_path)
    continuous = [
        str(column)
        for column in dataset_contract.get("continuous_feature_columns", [])
    ]
    if not continuous:
        raise ValueError("historical dataset contract has no continuous feature columns")
    dataset_path = require_file(args.dataset_dir / DATASET_NAME, "historical dataset")
    frame, scope_audit = load_b2_router_rows(
        dataset_path, router_required_columns(continuous)
    )
    if set(frame["site_id"].astype(str)) != set(ROUTER_SITES) or sorted(
        frame["target_year"].astype(int).unique()
    ) != list(ALLOWED_YEARS):
        raise ValueError("B2 runner requires complete P2/P4 2015-2019 scope")
    checkpoints, checkpoint_summary = load_development_checkpoints(
        args.development_screen_dir
    )
    device = resolve_device(args.device)

    fold_models: dict[str, dict[str, Any]] = {}
    sample_frames: list[pd.DataFrame] = []
    validation_frames: list[pd.DataFrame] = []
    for fold in development_folds():
        history = frame.loc[
            frame["site_id"].eq(fold["training_site"])
            & frame["target_year"].isin(fold["train_years"])
        ].copy()
        validation = frame.loc[
            frame["site_id"].eq(fold["held_out_site"])
            & frame["target_year"].eq(fold["validation_year"])
        ].copy()
        if history.empty or validation.empty:
            raise ValueError(f"B2 fold is empty: {fold['protocol_fold']}")
        if set(history["site_id"].astype(str)) != {fold["training_site"]}:
            raise ValueError("B2 training site scope changed")
        if set(validation["site_id"].astype(str)) != {fold["held_out_site"]}:
            raise ValueError("B2 validation site scope changed")
        if int(history["target_year"].max()) >= int(fold["validation_year"]):
            raise ValueError("B2 fold contains future gate-training rows")
        source_fold = str(fold["source_protocol_fold"])
        history_features = build_gate_feature_frame(history, feature_contract)
        validation_features = build_gate_feature_frame(validation, feature_contract)
        history_decisions, _ = expert_decisions(
            checkpoints[source_fold], history, device
        )
        samples = _annotate_samples(
            actionable_gate_rows(
                history_features, history_decisions["P1"], history_decisions["P15"]
            ),
            fold,
        )
        model = fit_regret_weighted_logistic_gate(samples, feature_names)
        model.update(
            {
                "protocol_fold": fold["protocol_fold"],
                "source_protocol_fold": source_fold,
                "training_site": fold["training_site"],
                "held_out_site": fold["held_out_site"],
                "training_years": list(fold["train_years"]),
                "validation_year": fold["validation_year"],
            }
        )
        model["model_sha256"] = model_sha256(model)
        # Validation truth is not materialized through expert decisions until the gate is frozen.
        validation_decisions, _ = expert_decisions(
            checkpoints[source_fold], validation, device
        )
        probabilities = predict_gate_probabilities(model, validation_features)
        decisions = _policy_rows(
            validation_decisions["P1"],
            validation_decisions["P15"],
            probabilities,
            fold=fold,
            gate_model=model,
        )
        fold_models[str(fold["protocol_fold"])] = model
        sample_frames.append(samples)
        validation_frames.append(decisions)
        print(
            f"b2_fold={fold['protocol_fold']} training_site={fold['training_site']} "
            f"held_out_site={fold['held_out_site']} gate_rows={model['training_rows']} "
            f"fallback={model['fallback']}",
            flush=True,
        )

    samples = pd.concat(sample_frames, ignore_index=True)
    validation = pd.concat(validation_frames, ignore_index=True)
    metrics = summarize_policies(validation)
    fold_scope = pd.DataFrame(development_folds())[
        ["protocol_fold", "held_out_site", "training_site", "validation_year"]
    ]
    metrics = metrics.merge(fold_scope, on="protocol_fold", validate="many_to_one")
    site_metrics = per_site_metrics(metrics)
    development_gate = b2_development_gate(metrics, validation)
    development_passed = bool(development_gate.loc[0, "development_gate_passed"])

    final_gate: dict[str, Any] | None = None
    final_policy: dict[str, Any] | None = None
    final_checkpoint_paths: list[tuple[str, Path]] = []
    if development_passed:
        final_checkpoints, final_policy = load_final_checkpoints(args.final_model_dir)
        final_features = build_gate_feature_frame(frame, feature_contract)
        final_decisions, _ = expert_decisions(final_checkpoints, frame, device)
        final_samples = _annotate_samples(
            actionable_gate_rows(
                final_features, final_decisions["P1"], final_decisions["P15"]
            ),
            {
                "protocol_fold": "final_P2_P4_2015_2019",
                "source_protocol_fold": "final_full_history",
                "training_site": "P2+P4",
                "held_out_site": "none",
                "train_years": ALLOWED_YEARS,
                "validation_year": 0,
            },
        )
        final_gate = fit_regret_weighted_logistic_gate(final_samples, feature_names)
        final_gate.update(
            {
                "policy_state": "pending_new_independent_freeze",
                "training_sites": list(ROUTER_SITES),
                "training_years": list(ALLOWED_YEARS),
                "P3_rows_used": 0,
                "year_2021_rows_used": 0,
                "year_2024_rows_used": 0,
            }
        )
        final_gate["model_sha256"] = model_sha256(final_gate)
        if final_gate["fallback"]:
            final_gate = None
        else:
            for source, record in final_policy["source_experts"].items():
                final_checkpoint_paths.append(
                    (source, args.final_model_dir / str(record["checkpoint_path"]))
                )

    final_eligible = development_passed and final_gate is not None
    status = (
        "b2_regret_sensitive_loso_development_passed_pending_new_independent_freeze"
        if final_eligible
        else "b2_regret_sensitive_loso_development_failed_frozen_negative"
    )
    output_dir.mkdir(parents=True)
    outputs = {
        "training_samples": output_dir / f"{OUTPUT_PREFIX}_training_samples_v1.csv",
        "validation_decisions": output_dir / f"{OUTPUT_PREFIX}_validation_decisions_v1.csv",
        "per_fold_metrics": output_dir / f"{OUTPUT_PREFIX}_per_fold_metrics_v1.csv",
        "per_site_metrics": output_dir / f"{OUTPUT_PREFIX}_per_site_metrics_v1.csv",
        "development_gate": output_dir / f"{OUTPUT_PREFIX}_development_gate_v1.csv",
        "fold_models": output_dir / f"{OUTPUT_PREFIX}_fold_models_v1.json",
        "audit": output_dir / f"{OUTPUT_PREFIX}_router_screen_audit_v1.json",
        "manifest": output_dir / f"{OUTPUT_PREFIX}_router_screen_manifest_v1.csv",
    }
    samples.to_csv(outputs["training_samples"], index=False)
    validation.to_csv(outputs["validation_decisions"], index=False)
    metrics.to_csv(outputs["per_fold_metrics"], index=False)
    site_metrics.to_csv(outputs["per_site_metrics"], index=False)
    development_gate.to_csv(outputs["development_gate"], index=False)
    write_json(outputs["fold_models"], fold_models)
    if final_gate is not None:
        outputs["final_gate"] = output_dir / f"{OUTPUT_PREFIX}_final_gate_v1.json"
        write_json(outputs["final_gate"], final_gate)
    audit = {
        "status": status,
        "mandatory_gate_passed": final_eligible,
        "development_gate_passed": development_passed,
        "final_gate_written": final_gate is not None,
        "formal_promotion": False,
        "next_gate": "freeze_and_run_a_new_independent_site_year_evaluation"
        if final_eligible
        else "B2_frozen_negative_no_independent_evaluation",
        "target_site": "P3",
        "router_training_sites": list(ROUTER_SITES),
        "router_training_years": list(ALLOWED_YEARS),
        "development_fold_count": 8,
        "development_folds": development_folds(),
        "scope_filter": scope_audit,
        "P3_rows_retained": 0,
        "P3_target_values_used": 0,
        "year_2020_rows_retained": 0,
        "year_2021_rows_retained": 0,
        "year_2024_rows_retained": 0,
        "source_checkpoint_retraining_performed": False,
        "source_checkpoint_reselection_performed": False,
        "gate_feature_search_performed": False,
        "gate_weight_search_performed": False,
        "gate_threshold_search_performed": False,
        "gate_threshold": 0.5,
        "gate_tie_route": "P15",
        "regret_weighting": "abs_realized_gain_delta_normalized_to_fold_mean_one",
        "fold_fallback_reasons": {
            fold: model["fallback_reason"] for fold, model in fold_models.items()
        },
        "development_conditions": {
            column: bool(development_gate.loc[0, column])
            for column in development_gate.columns
            if column.endswith("_gate_passed")
            and column != "development_gate_passed"
        },
        "network_access_performed": False,
    }
    write_json(outputs["audit"], audit)

    input_paths: list[tuple[str, Path]] = [
        ("historical_dataset", dataset_path),
        ("historical_dataset_contract", dataset_contract_path),
        ("dual_protocol_manifest", frozen["paths"]["manifest"]),
        (
            "development_checkpoint_summary",
            args.development_screen_dir
            / "gefs_p3_strict_loso_source_checkpoint_summary_v1.csv",
        ),
    ]
    for row in checkpoint_summary.itertuples(index=False):
        input_paths.append(
            (
                "development_source_checkpoint",
                args.development_screen_dir / str(row.checkpoint_path),
            )
        )
    if final_gate is not None:
        input_paths.append(
            (
                "final_full_history_policy",
                args.final_model_dir
                / "gefs_p3_2021_final_full_history_policy_v1r2.json",
            )
        )
        input_paths.extend(
            (f"final_source_checkpoint_{source}", path)
            for source, path in final_checkpoint_paths
        )
    manifest_rows: list[dict[str, Any]] = []
    for role, path in input_paths:
        manifest_rows.append(
            {
                "role": f"input_{role}",
                "path": str(path),
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
    code_path = Path(__file__).resolve()
    manifest_rows.append(
        {
            "role": "code",
            "path": code_path.relative_to(PROJECT_ROOT).as_posix(),
            "bytes": code_path.stat().st_size,
            "sha256": sha256_file(code_path),
        }
    )
    for name, path in outputs.items():
        if name != "manifest":
            manifest_rows.append(
                {
                    "role": f"output_{name}",
                    "path": path.name,
                    "bytes": path.stat().st_size,
                    "sha256": sha256_file(path),
                }
            )
    pd.DataFrame(manifest_rows).to_csv(outputs["manifest"], index=False)
    print(json.dumps(audit, indent=2, sort_keys=True), flush=True)
    return outputs


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", type=Path, default=DEFAULT_DATASET_DIR)
    parser.add_argument("--protocol-dir", type=Path, default=DEFAULT_DUAL_PROTOCOL_DIR)
    parser.add_argument(
        "--development-screen-dir", type=Path, default=DEFAULT_SCREEN_DIR
    )
    parser.add_argument("--final-model-dir", type=Path, default=DEFAULT_FINAL_MODEL_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args(argv)


if __name__ == "__main__":
    generated = run(parse_args())
    for name, path in generated.items():
        print(f"{name}: {path}", flush=True)
