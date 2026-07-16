#!/usr/bin/env python3
"""Evaluate a nested LOSO selector over binary-trigger calibration policies.

The nearest-1 oracle-rate transfer is close to the paper fixed-list baseline,
but different held-out sites appear to prefer different calibration policies.
This script tests whether a deployable selector can choose among calibration
policies without using labels from the evaluated site.

For each outer held-out site:

1. remove the held-out site from calibration;
2. run an inner leave-one-site-out validation over the remaining sites;
3. select the candidate calibration policy with the lowest inner mean regret;
4. apply that selected policy to the outer held-out site.

The script also reports a non-deployable oracle selector over the same candidate
policy set. If the oracle selector cannot beat the fixed list, this candidate
calibration family is probably exhausted.
"""

from __future__ import annotations

import argparse
import errno
from pathlib import Path

import numpy as np
import pandas as pd

from evaluate_binary_trigger_loso_calibration_policies_v1 import (
    DEFAULT_PREDICTIONS,
    DEFAULT_SAMPLES,
    DEFAULT_THRESHOLDS,
    build_site_features,
    choose_best_threshold,
    compute_site_oracle,
    evaluate_fixed_threshold,
    parse_thresholds,
    static_distances,
    summarize_decisions,
    threshold_for_target_rate,
)
from train_confirmed_5site_true_input_surrogate_baseline_v1 import bool_series, markdown_table


DEFAULT_OUT = (
    Path("site_general_surrogate_eval")
    / "continuous_ir_12site_10k_binary_trigger_nested_calibration_selector_v1"
)
DEFAULT_CANDIDATES = (
    "global_threshold,"
    "mean_oracle_rate,"
    "median_oracle_rate,"
    "nearest_1_oracle_rate,"
    "nearest_3_oracle_rate,"
    "nearest_5_oracle_rate"
)
PAPER_FIXED_LIST_REGRET = 0.614875609


def candidate_list(text: str) -> list[str]:
    values = [part.strip() for part in text.split(",") if part.strip()]
    if not values:
        raise ValueError("At least one candidate policy is required")
    return values


def int_list(text: str) -> list[int]:
    values = [int(part.strip()) for part in text.split(",") if part.strip()]
    if not values:
        raise ValueError("At least one integer value is required")
    return values


def candidate_threshold(
    *,
    candidate: str,
    eval_site: str,
    train_sites: set[str],
    pred_df: pd.DataFrame,
    site_features: pd.DataFrame,
    site_oracle: pd.DataFrame,
    thresholds: list[float],
) -> dict:
    eval_rows = pred_df.loc[pred_df["site_id"] == str(eval_site)].copy()
    train_rows = pred_df.loc[pred_df["site_id"].isin(train_sites)].copy()
    train_oracle = site_oracle.loc[site_oracle["site_id"].isin(train_sites)].copy()
    if eval_rows.empty:
        raise ValueError(f"No prediction rows for eval_site={eval_site}")
    if train_rows.empty or train_oracle.empty:
        raise ValueError(f"No calibration rows for eval_site={eval_site}")

    if candidate == "global_threshold":
        threshold, _summary = choose_best_threshold(train_rows, thresholds)
        return {
            "assigned_threshold": threshold,
            "target_irrigation_rate": np.nan,
            "achieved_irrigation_rate": np.nan,
            "nearest_sites": "",
            "nearest_distances": "",
            "nearest_oracle_rates": "",
        }

    if candidate == "mean_oracle_rate":
        target_rate = float(train_oracle["oracle_predicted_irrigation_rate"].mean())
    elif candidate == "median_oracle_rate":
        target_rate = float(train_oracle["oracle_predicted_irrigation_rate"].median())
    elif candidate.startswith("nearest_") and candidate.endswith("_oracle_rate"):
        k = int(candidate.replace("nearest_", "").replace("_oracle_rate", ""))
        neighbors = static_distances(site_features, str(eval_site), train_sites).head(k).copy()
        train_oracle_by_site = train_oracle.set_index("site_id")
        neighbor_rates = [
            float(train_oracle_by_site.loc[str(site_id), "oracle_predicted_irrigation_rate"])
            for site_id in neighbors["site_id"].astype(str).tolist()
        ]
        target_rate = float(np.median(neighbor_rates))
        threshold, achieved_rate = threshold_for_target_rate(eval_rows, target_rate, thresholds)
        return {
            "assigned_threshold": threshold,
            "target_irrigation_rate": target_rate,
            "achieved_irrigation_rate": achieved_rate,
            "nearest_sites": ",".join(neighbors["site_id"].astype(str).tolist()),
            "nearest_distances": ",".join(f"{v:.6g}" for v in neighbors["distance"].tolist()),
            "nearest_oracle_rates": ",".join(f"{v:.6g}" for v in neighbor_rates),
        }
    else:
        raise ValueError(f"Unknown candidate policy: {candidate}")

    threshold, achieved_rate = threshold_for_target_rate(eval_rows, target_rate, thresholds)
    return {
        "assigned_threshold": threshold,
        "target_irrigation_rate": target_rate,
        "achieved_irrigation_rate": achieved_rate,
        "nearest_sites": "",
        "nearest_distances": "",
        "nearest_oracle_rates": "",
    }


def evaluate_candidate_on_site(
    *,
    candidate: str,
    eval_site: str,
    train_sites: set[str],
    pred_df: pd.DataFrame,
    site_features: pd.DataFrame,
    site_oracle: pd.DataFrame,
    thresholds: list[float],
) -> tuple[pd.DataFrame, dict]:
    meta = candidate_threshold(
        candidate=candidate,
        eval_site=eval_site,
        train_sites=train_sites,
        pred_df=pred_df,
        site_features=site_features,
        site_oracle=site_oracle,
        thresholds=thresholds,
    )
    eval_rows = pred_df.loc[pred_df["site_id"] == str(eval_site)].copy()
    decisions = evaluate_fixed_threshold(eval_rows, float(meta["assigned_threshold"]))
    decisions["candidate_policy"] = candidate
    decisions["target_irrigation_rate"] = meta["target_irrigation_rate"]
    decisions["achieved_irrigation_rate"] = meta["achieved_irrigation_rate"]
    meta = {
        "eval_site": str(eval_site),
        "candidate_policy": candidate,
        **meta,
    }
    return decisions, meta


def write_csv(path: Path, df: pd.DataFrame) -> bool:
    try:
        df.to_csv(path, index=False)
        return True
    except OSError as exc:
        if exc.errno == errno.ENOSPC:
            print(f"[warn] No space left on device; skipped writing {path}")
            return False
        raise


def write_text(path: Path, text: str) -> bool:
    try:
        path.write_text(text, encoding="utf-8")
        return True
    except OSError as exc:
        if exc.errno == errno.ENOSPC:
            print(f"[warn] No space left on device; skipped writing {path}")
            return False
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", default=str(DEFAULT_PREDICTIONS))
    parser.add_argument("--samples", default=str(DEFAULT_SAMPLES))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUT))
    parser.add_argument("--thresholds", default=DEFAULT_THRESHOLDS)
    parser.add_argument("--candidate-policies", default=DEFAULT_CANDIDATES)
    parser.add_argument("--selector-neighbor-ks", default="1,3,5")
    args = parser.parse_args()

    pred_path = Path(args.predictions)
    samples_path = Path(args.samples)
    if not pred_path.exists():
        raise FileNotFoundError(f"Missing prediction file: {pred_path}")
    if not samples_path.exists():
        raise FileNotFoundError(f"Missing samples file: {samples_path}")
    out_dir = Path(args.output_dir)
    can_write = True
    try:
        out_dir.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        if exc.errno == errno.ENOSPC:
            print("[warn] No space left on device; will print summary only and skip writing")
            can_write = False
        else:
            raise

    pred_df = pd.read_csv(pred_path)
    for col in ["should_irrigate", "target_collapse", "same_date_duplicate_target_curve"]:
        if col in pred_df.columns:
            pred_df[col] = bool_series(pred_df[col])
    pred_df["site_id"] = pred_df["site_id"].astype(str)
    samples = pd.read_csv(samples_path)
    site_features = build_site_features(samples)
    site_features["site_id"] = site_features["site_id"].astype(str)
    thresholds = parse_thresholds(args.thresholds)
    candidates = candidate_list(args.candidate_policies)
    selector_neighbor_ks = int_list(args.selector_neighbor_ks)
    site_oracle = compute_site_oracle(pred_df, thresholds)
    all_sites = sorted(pred_df["site_id"].unique().tolist())

    selector_decision_parts: list[pd.DataFrame] = []
    fixed_candidate_parts: list[pd.DataFrame] = []
    oracle_selector_parts: list[pd.DataFrame] = []
    assignment_rows: list[dict] = []
    inner_rows: list[dict] = []
    inner_site_rows: list[dict] = []

    for outer_site in all_sites:
        outer_train_sites = set(all_sites) - {outer_site}
        inner_scores = []
        outer_inner_site_rows = []
        for candidate in candidates:
            inner_parts = []
            for val_site in sorted(outer_train_sites):
                inner_train_sites = outer_train_sites - {val_site}
                decisions, _meta = evaluate_candidate_on_site(
                    candidate=candidate,
                    eval_site=val_site,
                    train_sites=inner_train_sites,
                    pred_df=pred_df,
                    site_features=site_features,
                    site_oracle=site_oracle,
                    thresholds=thresholds,
                )
                inner_parts.append(decisions)
                val_score = summarize_decisions(decisions.assign(policy=candidate), candidate)
                inner_site_row = {
                    "outer_site": outer_site,
                    "validation_site": val_site,
                    "candidate_policy": candidate,
                    "validation_mean_regret": float(val_score["mean_decision_regret_oracle_amount"]),
                    "validation_balanced_accuracy": float(val_score["trigger_balanced_accuracy"]),
                }
                inner_site_rows.append(inner_site_row)
                outer_inner_site_rows.append(inner_site_row)
            inner_decisions = pd.concat(inner_parts, ignore_index=True)
            score = summarize_decisions(inner_decisions, candidate)
            inner_rows.append({"outer_site": outer_site, **score})
            inner_scores.append(score)
        inner_score_df = pd.DataFrame(inner_scores)
        selected_candidate = str(
            inner_score_df.sort_values(
                ["mean_decision_regret_oracle_amount", "trigger_balanced_accuracy"],
                ascending=[True, False],
            ).iloc[0]["policy"]
        )
        selected_inner_score = inner_score_df.loc[inner_score_df["policy"] == selected_candidate].iloc[0].to_dict()

        outer_candidate_scores = []
        outer_candidate_decisions: dict[str, pd.DataFrame] = {}
        outer_candidate_meta: dict[str, dict] = {}
        for candidate in candidates:
            decisions, meta = evaluate_candidate_on_site(
                candidate=candidate,
                eval_site=outer_site,
                train_sites=outer_train_sites,
                pred_df=pred_df,
                site_features=site_features,
                site_oracle=site_oracle,
                thresholds=thresholds,
            )
            candidate_decisions = decisions.copy()
            candidate_decisions["policy"] = f"candidate_{candidate}"
            fixed_candidate_parts.append(candidate_decisions)

            score = summarize_decisions(decisions.assign(policy=candidate), candidate)
            outer_candidate_scores.append(score)
            outer_candidate_decisions[candidate] = decisions
            outer_candidate_meta[candidate] = meta

            if candidate == selected_candidate:
                selected_decisions = decisions.copy()
                selected_decisions["policy"] = "nested_inner_cv_policy_selector"
                selected_decisions["selected_candidate_policy"] = selected_candidate
                selector_decision_parts.append(selected_decisions)
                assignment_rows.append(
                    {
                        "site_id": outer_site,
                        "selector_policy": "nested_inner_cv_policy_selector",
                        "selected_candidate_policy": selected_candidate,
                        "inner_cv_mean_regret": float(selected_inner_score["mean_decision_regret_oracle_amount"]),
                        **meta,
                    }
                )

        outer_inner_site_df = pd.DataFrame(outer_inner_site_rows)
        for k in selector_neighbor_ks:
            neighbors = static_distances(site_features, outer_site, outer_train_sites).head(k)
            neighbor_sites = neighbors["site_id"].astype(str).tolist()
            neighbor_scores = outer_inner_site_df.loc[
                outer_inner_site_df["validation_site"].astype(str).isin(neighbor_sites)
            ].copy()
            site_specific_scores = (
                neighbor_scores.groupby("candidate_policy")
                .agg(
                    validation_mean_regret=("validation_mean_regret", "mean"),
                    validation_balanced_accuracy=("validation_balanced_accuracy", "mean"),
                )
                .reset_index()
            )
            site_selected_candidate = str(
                site_specific_scores.sort_values(
                    ["validation_mean_regret", "validation_balanced_accuracy"],
                    ascending=[True, False],
                ).iloc[0]["candidate_policy"]
            )
            site_selected_decisions = outer_candidate_decisions[site_selected_candidate].copy()
            selector_policy = f"nearest_{k}_inner_site_policy_selector"
            site_selected_decisions["policy"] = selector_policy
            site_selected_decisions["selected_candidate_policy"] = site_selected_candidate
            selector_decision_parts.append(site_selected_decisions)
            site_selected_score = site_specific_scores.loc[
                site_specific_scores["candidate_policy"] == site_selected_candidate
            ].iloc[0]
            assignment_rows.append(
                {
                    "site_id": outer_site,
                    "selector_policy": selector_policy,
                    "selected_candidate_policy": site_selected_candidate,
                    "inner_cv_mean_regret": float(site_selected_score["validation_mean_regret"]),
                    "inner_cv_neighbor_sites": ",".join(neighbor_sites),
                    "inner_cv_neighbor_distances": ",".join(f"{v:.6g}" for v in neighbors["distance"].tolist()),
                    **outer_candidate_meta[site_selected_candidate],
                }
            )

        best_idx = int(
            pd.DataFrame(outer_candidate_scores)
            .sort_values(["mean_decision_regret_oracle_amount", "trigger_balanced_accuracy"], ascending=[True, False])
            .index[0]
        )
        best_candidate = candidates[best_idx]
        oracle_decisions = outer_candidate_decisions[best_candidate].copy()
        oracle_decisions["policy"] = "oracle_candidate_policy_selector"
        oracle_decisions["selected_candidate_policy"] = outer_candidate_meta[best_candidate]["candidate_policy"]
        oracle_selector_parts.append(oracle_decisions)
        assignment_rows.append(
            {
                "site_id": outer_site,
                "selector_policy": "oracle_candidate_policy_selector",
                "selected_candidate_policy": outer_candidate_meta[best_candidate]["candidate_policy"],
                "inner_cv_mean_regret": np.nan,
                **outer_candidate_meta[best_candidate],
            }
        )

    all_decisions = pd.concat(
        [*fixed_candidate_parts, *selector_decision_parts, *oracle_selector_parts],
        ignore_index=True,
    )
    assignments = pd.DataFrame(assignment_rows)
    inner_cv = pd.DataFrame(inner_rows)
    inner_site_cv = pd.DataFrame(inner_site_rows)
    summary = pd.DataFrame(
        [summarize_decisions(group, policy) for policy, group in all_decisions.groupby("policy", sort=False)]
    ).sort_values("mean_decision_regret_oracle_amount")
    summary["paper_fixed_list_gap"] = summary["mean_decision_regret_oracle_amount"] - PAPER_FIXED_LIST_REGRET
    by_site = (
        all_decisions.groupby(["policy", "site_id"])
        .agg(
            mean_decision_regret_oracle_amount=("trigger_decision_regret_oracle_amount", "mean"),
            max_decision_regret_oracle_amount=("trigger_decision_regret_oracle_amount", "max"),
            trigger_accuracy=("trigger_correct", "mean"),
            assigned_threshold=("assigned_threshold", "first"),
            predicted_irrigation_rate=("pred_should_irrigate", "mean"),
            n_site_dates=("site_date_id", "count"),
        )
        .reset_index()
        .sort_values(["policy", "mean_decision_regret_oracle_amount"], ascending=[True, False])
    )

    summary_path = out_dir / "binary_trigger_nested_calibration_selector_summary_v1.csv"
    assignments_path = out_dir / "binary_trigger_nested_calibration_selector_assignments_v1.csv"
    inner_cv_path = out_dir / "binary_trigger_nested_calibration_selector_inner_cv_v1.csv"
    inner_site_cv_path = out_dir / "binary_trigger_nested_calibration_selector_inner_site_cv_v1.csv"
    by_site_path = out_dir / "binary_trigger_nested_calibration_selector_by_site_v1.csv"
    decisions_path = out_dir / "binary_trigger_nested_calibration_selector_decisions_v1.csv"
    report_path = out_dir / "binary_trigger_nested_calibration_selector_v1.md"

    lines = [
        "# Binary Trigger Nested Calibration Selector V1",
        "",
        "## Inputs",
        "",
        f"- Predictions: `{pred_path}`",
        f"- Samples: `{samples_path}`",
        f"- Paper fixed-list mean regret: `{PAPER_FIXED_LIST_REGRET}`",
        "",
        "## Policy Summary",
        "",
        markdown_table(summary),
        "",
        "## Selector Assignments",
        "",
        markdown_table(assignments),
        "",
        "## Inner CV Scores",
        "",
        markdown_table(inner_cv),
        "",
        "## Inner Site CV Scores",
        "",
        markdown_table(inner_site_cv),
        "",
        "## By Site",
        "",
        markdown_table(by_site),
        "",
        "## Outputs",
        "",
        f"- `{summary_path}`",
        f"- `{assignments_path}`",
        f"- `{inner_cv_path}`",
        f"- `{inner_site_cv_path}`",
        f"- `{by_site_path}`",
        f"- `{decisions_path}`",
    ]
    report_text = "\n".join(lines) + "\n"

    if can_write:
        write_csv(summary_path, summary)
        write_csv(assignments_path, assignments)
        write_csv(inner_cv_path, inner_cv)
        write_csv(inner_site_cv_path, inner_site_cv)
        write_csv(by_site_path, by_site)
        write_csv(decisions_path, all_decisions)
        write_text(report_path, report_text)

    print("Binary trigger nested calibration selector v1")
    print(f"summary: {summary_path}")
    print(f"assignments: {assignments_path}")
    print(f"inner_cv: {inner_cv_path}")
    print(f"inner_site_cv: {inner_site_cv_path}")
    print(f"report: {report_path}")
    print("")
    print(summary.to_string(index=False))
    print("")
    print(assignments.to_string(index=False))


if __name__ == "__main__":
    main()
