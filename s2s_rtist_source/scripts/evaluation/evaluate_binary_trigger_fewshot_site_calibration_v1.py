#!/usr/bin/env python3
"""Evaluate few-shot target-site calibration for binary-trigger policies.

Zero-shot site-level policy selection failed with only 12 sites, but the oracle
candidate selector showed that the candidate calibration family can beat the
fixed list if the target-site policy is chosen correctly. This script evaluates
a practical compromise: use a small number of labeled site-dates from the target
site to choose among deployable calibration policies, then evaluate the chosen
policy on the remaining site-dates.

This is not zero-shot LOSO. It represents a site onboarding workflow where a
small offline SWAP calibration set is allowed for a new site before real-time
surrogate deployment.
"""

from __future__ import annotations

import argparse
import errno
from pathlib import Path

import numpy as np
import pandas as pd

from train_confirmed_5site_true_input_surrogate_baseline_v1 import bool_series, markdown_table


DEFAULT_ROOT = (
    Path("site_general_surrogate_eval")
    / "continuous_ir_12site_10k_binary_trigger_nested_calibration_selector_v1"
)
DEFAULT_DECISIONS = DEFAULT_ROOT / "binary_trigger_nested_calibration_selector_decisions_v1.csv"
DEFAULT_OUT = (
    Path("site_general_surrogate_eval")
    / "continuous_ir_12site_10k_binary_trigger_fewshot_site_calibration_v1"
)
PAPER_FIXED_LIST_REGRET = 0.614875609


def int_list(text: str) -> list[int]:
    values = [int(part.strip()) for part in text.split(",") if part.strip()]
    if not values:
        raise ValueError("At least one integer value is required")
    return values


def safe_div(num: float, den: float) -> float:
    return float(num / den) if den else float("nan")


def summarize_decisions(decisions: pd.DataFrame, policy: str) -> dict:
    pred = decisions["pred_should_irrigate"].astype(bool)
    actual = decisions["should_irrigate"].astype(bool)
    tp = int((pred & actual).sum())
    fp = int((pred & ~actual).sum())
    tn = int((~pred & ~actual).sum())
    fn = int((~pred & actual).sum())
    recall = safe_div(tp, tp + fn)
    specificity = safe_div(tn, tn + fp)
    return {
        "policy": policy,
        "n_decisions": int(len(decisions)),
        "mean_regret": float(decisions["trigger_decision_regret_oracle_amount"].mean()),
        "median_regret": float(decisions["trigger_decision_regret_oracle_amount"].median()),
        "paper_fixed_list_gap": float(decisions["trigger_decision_regret_oracle_amount"].mean() - PAPER_FIXED_LIST_REGRET),
        "trigger_accuracy": float((pred == actual).mean()),
        "trigger_recall": recall,
        "trigger_specificity": specificity,
        "true_positive": tp,
        "false_positive": fp,
        "true_negative": tn,
        "false_negative": fn,
        "predicted_irrigation_rate": float(pred.mean()),
        "true_irrigation_rate": float(actual.mean()),
    }


def choose_policy(calibration: pd.DataFrame) -> tuple[str, pd.DataFrame]:
    scores = (
        calibration.groupby("candidate_policy")
        .agg(
            calibration_mean_regret=("trigger_decision_regret_oracle_amount", "mean"),
            calibration_accuracy=("trigger_correct", "mean"),
            calibration_predicted_irrigation_rate=("pred_should_irrigate", "mean"),
            n_calibration_decisions=("site_date_id", "count"),
        )
        .reset_index()
        .sort_values(["calibration_mean_regret", "calibration_accuracy"], ascending=[True, False])
    )
    return str(scores.iloc[0]["candidate_policy"]), scores


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
    parser.add_argument("--decisions", default=str(DEFAULT_DECISIONS))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUT))
    parser.add_argument("--calibration-site-dates", default="1,2,3,5,8,13")
    parser.add_argument("--seeds", default="0,1,2,3,4,5,6,7,8,9")
    args = parser.parse_args()

    decisions_path = Path(args.decisions)
    if not decisions_path.exists():
        raise FileNotFoundError(f"Missing decisions file: {decisions_path}")
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

    decisions = pd.read_csv(decisions_path)
    for col in ["should_irrigate", "pred_should_irrigate", "trigger_correct"]:
        if col in decisions.columns:
            decisions[col] = bool_series(decisions[col])
    required = {"policy", "site_id", "site_date_id", "trigger_decision_regret_oracle_amount"}
    missing = required.difference(decisions.columns)
    if missing:
        raise ValueError(f"Decisions file is missing columns: {sorted(missing)}")

    candidates = decisions.loc[decisions["policy"].astype(str).str.startswith("candidate_")].copy()
    if candidates.empty:
        raise ValueError("No candidate_* policies found in decisions")
    candidates["candidate_policy"] = candidates["policy"].astype(str).str.replace("candidate_", "", regex=False)
    candidates["site_id"] = candidates["site_id"].astype(str)

    calibration_ks = int_list(args.calibration_site_dates)
    seeds = int_list(args.seeds)
    rng_rows = []
    selected_parts = []
    calibration_score_rows = []

    for k in calibration_ks:
        for seed in seeds:
            rng = np.random.default_rng(seed)
            for site_id, site_rows in candidates.groupby("site_id", sort=False):
                site_dates = np.array(sorted(site_rows["site_date_id"].astype(str).unique().tolist()))
                if len(site_dates) <= k:
                    continue
                calibration_ids = set(rng.choice(site_dates, size=k, replace=False).tolist())
                calibration = site_rows.loc[site_rows["site_date_id"].astype(str).isin(calibration_ids)].copy()
                test = site_rows.loc[~site_rows["site_date_id"].astype(str).isin(calibration_ids)].copy()
                selected_policy, scores = choose_policy(calibration)
                selected_test = test.loc[test["candidate_policy"] == selected_policy].copy()
                selected_test["policy"] = "fewshot_site_calibrated_policy"
                selected_test["selected_candidate_policy"] = selected_policy
                selected_test["calibration_site_dates"] = int(k)
                selected_test["seed"] = int(seed)
                selected_parts.append(selected_test)
                scores.insert(0, "seed", int(seed))
                scores.insert(0, "calibration_site_dates", int(k))
                scores.insert(0, "site_id", site_id)
                scores["selected_candidate_policy"] = selected_policy
                calibration_score_rows.append(scores)
                rng_rows.append(
                    {
                        "site_id": site_id,
                        "seed": int(seed),
                        "calibration_site_dates": int(k),
                        "selected_candidate_policy": selected_policy,
                        "calibration_ids": ",".join(sorted(calibration_ids)),
                        "n_test_site_dates": int(len(site_dates) - k),
                    }
                )

    if not selected_parts:
        raise ValueError("No few-shot evaluation rows were produced")
    selected = pd.concat(selected_parts, ignore_index=True)
    assignment = pd.DataFrame(rng_rows)
    calibration_scores = pd.concat(calibration_score_rows, ignore_index=True)
    summary_rows = []
    for (k, seed), group in selected.groupby(["calibration_site_dates", "seed"], sort=True):
        summary_rows.append(
            {
                "calibration_site_dates": int(k),
                "seed": int(seed),
                **summarize_decisions(group, "fewshot_site_calibrated_policy"),
            }
        )
    seed_summary = pd.DataFrame(summary_rows)
    summary = (
        seed_summary.groupby("calibration_site_dates")
        .agg(
            mean_regret=("mean_regret", "mean"),
            std_regret=("mean_regret", "std"),
            min_regret=("mean_regret", "min"),
            max_regret=("mean_regret", "max"),
            mean_paper_fixed_list_gap=("paper_fixed_list_gap", "mean"),
            mean_trigger_accuracy=("trigger_accuracy", "mean"),
            mean_trigger_recall=("trigger_recall", "mean"),
            mean_trigger_specificity=("trigger_specificity", "mean"),
            mean_predicted_irrigation_rate=("predicted_irrigation_rate", "mean"),
            mean_true_irrigation_rate=("true_irrigation_rate", "mean"),
            n_seeds=("seed", "nunique"),
        )
        .reset_index()
        .sort_values("calibration_site_dates")
    )
    policy_counts = (
        assignment.groupby(["calibration_site_dates", "selected_candidate_policy"])
        .agg(n_site_seed_selections=("site_id", "count"))
        .reset_index()
        .sort_values(["calibration_site_dates", "n_site_seed_selections"], ascending=[True, False])
    )
    by_site = (
        selected.groupby(["calibration_site_dates", "site_id"])
        .agg(
            mean_regret=("trigger_decision_regret_oracle_amount", "mean"),
            selected_policy_modes=("selected_candidate_policy", lambda s: ",".join(s.value_counts().index.astype(str).tolist())),
            n_rows=("site_date_id", "count"),
        )
        .reset_index()
        .sort_values(["calibration_site_dates", "mean_regret"], ascending=[True, False])
    )

    summary_path = out_dir / "binary_trigger_fewshot_site_calibration_summary_v1.csv"
    seed_summary_path = out_dir / "binary_trigger_fewshot_site_calibration_seed_summary_v1.csv"
    assignment_path = out_dir / "binary_trigger_fewshot_site_calibration_assignments_v1.csv"
    policy_counts_path = out_dir / "binary_trigger_fewshot_site_calibration_policy_counts_v1.csv"
    by_site_path = out_dir / "binary_trigger_fewshot_site_calibration_by_site_v1.csv"
    calibration_scores_path = out_dir / "binary_trigger_fewshot_site_calibration_scores_v1.csv"
    report_path = out_dir / "binary_trigger_fewshot_site_calibration_v1.md"

    lines = [
        "# Binary Trigger Few-Shot Site Calibration V1",
        "",
        "## Inputs",
        "",
        f"- Decisions: `{decisions_path}`",
        f"- Paper fixed-list global mean regret: `{PAPER_FIXED_LIST_REGRET}`",
        "",
        "## Summary By Calibration Budget",
        "",
        markdown_table(summary),
        "",
        "## Selected Policy Counts",
        "",
        markdown_table(policy_counts),
        "",
        "## By Site",
        "",
        markdown_table(by_site),
        "",
        "## Seed Summary",
        "",
        markdown_table(seed_summary),
        "",
        "## Outputs",
        "",
        f"- `{summary_path}`",
        f"- `{seed_summary_path}`",
        f"- `{assignment_path}`",
        f"- `{policy_counts_path}`",
        f"- `{by_site_path}`",
        f"- `{calibration_scores_path}`",
    ]
    report_text = "\n".join(lines) + "\n"

    if can_write:
        write_csv(summary_path, summary)
        write_csv(seed_summary_path, seed_summary)
        write_csv(assignment_path, assignment)
        write_csv(policy_counts_path, policy_counts)
        write_csv(by_site_path, by_site)
        write_csv(calibration_scores_path, calibration_scores)
        write_text(report_path, report_text)

    print("Binary trigger few-shot site calibration v1")
    print(f"summary: {summary_path}")
    print(f"assignments: {assignment_path}")
    print(f"policy_counts: {policy_counts_path}")
    print(f"by_site: {by_site_path}")
    print(f"report: {report_path}")
    print("")
    print(summary.to_string(index=False))
    print("")
    print(policy_counts.to_string(index=False))


if __name__ == "__main__":
    main()
