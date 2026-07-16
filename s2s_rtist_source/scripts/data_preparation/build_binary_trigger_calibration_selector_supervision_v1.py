#!/usr/bin/env python3
"""Build a site-level supervision table for calibration-policy selection.

This starts the follow-up line after the binary-trigger calibration branch:
the candidate policy family has a diagnostic upper bound below the paper fixed
list, but deployable selectors did not learn to choose the right policy. This
script turns the nested-selector outputs into a compact site-level training
table for future calibration-selector work.

It does not train a selector. With only 12 sites, the immediate goal is to
audit whether there is enough supervised signal to justify a learned selector
or whether the next step must be more sites / more SWAP-labeled site-date
curves.
"""

from __future__ import annotations

import argparse
import errno
from pathlib import Path

import pandas as pd

from evaluate_binary_trigger_loso_calibration_policies_v1 import build_site_features
from train_confirmed_5site_true_input_surrogate_baseline_v1 import markdown_table


DEFAULT_ROOT = (
    Path("site_general_surrogate_eval")
    / "continuous_ir_12site_10k_binary_trigger_nested_calibration_selector_v1"
)
DEFAULT_ASSIGNMENTS = DEFAULT_ROOT / "binary_trigger_nested_calibration_selector_assignments_v1.csv"
DEFAULT_BY_SITE = DEFAULT_ROOT / "binary_trigger_nested_calibration_selector_by_site_v1.csv"
DEFAULT_SAMPLES = (
    Path("site_general_surrogate_eval")
    / "continuous_ir_12site_10k_surrogate_sequence_wide_features_v1"
    / "continuous_ir_12site_surrogate_sequence_wide_samples_v1.csv"
)
DEFAULT_OUT = (
    Path("site_general_surrogate_eval")
    / "continuous_ir_12site_10k_binary_trigger_calibration_selector_supervision_v1"
)
PAPER_FIXED_LIST_REGRET = 0.614875609


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


def policy_short_name(policy: str) -> str:
    return str(policy).replace("candidate_", "")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--assignments", default=str(DEFAULT_ASSIGNMENTS))
    parser.add_argument("--by-site", default=str(DEFAULT_BY_SITE))
    parser.add_argument("--samples", default=str(DEFAULT_SAMPLES))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUT))
    args = parser.parse_args()

    assignments_path = Path(args.assignments)
    by_site_path = Path(args.by_site)
    samples_path = Path(args.samples)
    if not assignments_path.exists():
        raise FileNotFoundError(f"Missing assignments file: {assignments_path}")
    if not by_site_path.exists():
        raise FileNotFoundError(f"Missing by-site file: {by_site_path}")
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

    assignments = pd.read_csv(assignments_path)
    by_site = pd.read_csv(by_site_path)
    samples = pd.read_csv(samples_path)
    site_features = build_site_features(samples)
    site_features["site_id"] = site_features["site_id"].astype(str)

    oracle_assignments = assignments.loc[
        assignments["selector_policy"].astype(str) == "oracle_candidate_policy_selector"
    ].copy()
    if oracle_assignments.empty:
        raise ValueError("No oracle_candidate_policy_selector rows found in assignments")
    oracle_assignments["site_id"] = oracle_assignments["site_id"].astype(str)
    oracle_assignments["oracle_best_candidate_policy"] = oracle_assignments[
        "selected_candidate_policy"
    ].astype(str)

    deployable_assignments = assignments.loc[
        assignments["selector_policy"].astype(str).isin(
            [
                "nested_inner_cv_policy_selector",
                "nearest_1_inner_site_policy_selector",
                "nearest_3_inner_site_policy_selector",
                "nearest_5_inner_site_policy_selector",
            ]
        )
    ].copy()
    deployable_assignments["site_id"] = deployable_assignments["site_id"].astype(str)
    deployable_choice = deployable_assignments.pivot_table(
        index="site_id",
        columns="selector_policy",
        values="selected_candidate_policy",
        aggfunc="first",
    ).reset_index()

    candidate_rows = by_site.loc[by_site["policy"].astype(str).str.startswith("candidate_")].copy()
    candidate_rows["site_id"] = candidate_rows["site_id"].astype(str)
    candidate_rows["candidate_policy"] = candidate_rows["policy"].map(policy_short_name)
    regret_matrix = candidate_rows.pivot_table(
        index="site_id",
        columns="candidate_policy",
        values="mean_decision_regret_oracle_amount",
        aggfunc="first",
    ).reset_index()
    regret_matrix.columns = [
        "site_id" if col == "site_id" else f"regret__{col}" for col in regret_matrix.columns
    ]

    oracle_regret = by_site.loc[
        by_site["policy"].astype(str) == "oracle_candidate_policy_selector",
        ["site_id", "mean_decision_regret_oracle_amount", "assigned_threshold", "predicted_irrigation_rate"],
    ].copy()
    oracle_regret["site_id"] = oracle_regret["site_id"].astype(str)
    oracle_regret = oracle_regret.rename(
        columns={
            "mean_decision_regret_oracle_amount": "oracle_candidate_mean_regret",
            "assigned_threshold": "oracle_candidate_assigned_threshold",
            "predicted_irrigation_rate": "oracle_candidate_predicted_irrigation_rate",
        }
    )

    nearest1_regret = by_site.loc[
        by_site["policy"].astype(str) == "candidate_nearest_1_oracle_rate",
        ["site_id", "mean_decision_regret_oracle_amount"],
    ].copy()
    nearest1_regret["site_id"] = nearest1_regret["site_id"].astype(str)
    nearest1_regret = nearest1_regret.rename(
        columns={"mean_decision_regret_oracle_amount": "nearest1_candidate_mean_regret"}
    )

    supervision = (
        oracle_assignments[
            [
                "site_id",
                "oracle_best_candidate_policy",
                "assigned_threshold",
                "target_irrigation_rate",
                "achieved_irrigation_rate",
            ]
        ]
        .rename(
            columns={
                "assigned_threshold": "oracle_best_candidate_assigned_threshold",
                "target_irrigation_rate": "oracle_best_candidate_target_irrigation_rate",
                "achieved_irrigation_rate": "oracle_best_candidate_achieved_irrigation_rate",
            }
        )
        .merge(oracle_regret, on="site_id", how="left")
        .merge(nearest1_regret, on="site_id", how="left")
        .merge(deployable_choice, on="site_id", how="left")
        .merge(regret_matrix, on="site_id", how="left")
        .merge(site_features, on="site_id", how="left")
    )
    supervision["oracle_candidate_beats_paper_global_mean"] = (
        supervision["oracle_candidate_mean_regret"] < PAPER_FIXED_LIST_REGRET
    )
    supervision["nearest1_minus_oracle_candidate_regret"] = (
        supervision["nearest1_candidate_mean_regret"] - supervision["oracle_candidate_mean_regret"]
    )
    supervision = supervision.sort_values("nearest1_minus_oracle_candidate_regret", ascending=False)

    label_counts = (
        supervision.groupby("oracle_best_candidate_policy")
        .agg(
            n_sites=("site_id", "count"),
            mean_oracle_candidate_regret=("oracle_candidate_mean_regret", "mean"),
            mean_nearest1_gap=("nearest1_minus_oracle_candidate_regret", "mean"),
        )
        .reset_index()
        .sort_values("n_sites", ascending=False)
    )
    deployable_selector_accuracy_rows = []
    for selector in [
        "nested_inner_cv_policy_selector",
        "nearest_1_inner_site_policy_selector",
        "nearest_3_inner_site_policy_selector",
        "nearest_5_inner_site_policy_selector",
    ]:
        if selector in supervision.columns:
            deployable_selector_accuracy_rows.append(
                {
                    "selector_policy": selector,
                    "matches_oracle_candidate_policy": float(
                        (
                            supervision[selector].astype(str)
                            == supervision["oracle_best_candidate_policy"].astype(str)
                        ).mean()
                    ),
                    "n_sites": int(supervision[selector].notna().sum()),
                }
            )
    selector_accuracy = pd.DataFrame(deployable_selector_accuracy_rows)

    supervision_path = out_dir / "binary_trigger_calibration_selector_supervision_by_site_v1.csv"
    regret_matrix_path = out_dir / "binary_trigger_calibration_selector_candidate_regret_matrix_v1.csv"
    label_counts_path = out_dir / "binary_trigger_calibration_selector_label_counts_v1.csv"
    selector_accuracy_path = out_dir / "binary_trigger_calibration_selector_accuracy_v1.csv"
    report_path = out_dir / "binary_trigger_calibration_selector_supervision_v1.md"

    lines = [
        "# Binary Trigger Calibration Selector Supervision V1",
        "",
        "## Inputs",
        "",
        f"- Assignments: `{assignments_path}`",
        f"- By site: `{by_site_path}`",
        f"- Samples: `{samples_path}`",
        f"- Paper fixed-list global mean regret: `{PAPER_FIXED_LIST_REGRET}`",
        "",
        "## Oracle Candidate Policy Labels",
        "",
        markdown_table(label_counts),
        "",
        "## Deployable Selector Policy Match Rate",
        "",
        markdown_table(selector_accuracy),
        "",
        "## Supervision By Site",
        "",
        markdown_table(supervision),
        "",
        "## Candidate Regret Matrix",
        "",
        markdown_table(regret_matrix),
        "",
        "## Interpretation",
        "",
        "This table is the starting point for a future calibration-selector line. "
        "With only one supervised label per site, the current 12-site dataset is "
        "too small for a reliable learned site-level selector unless more sites "
        "or more independent site groups are generated.",
        "",
        "## Outputs",
        "",
        f"- `{supervision_path}`",
        f"- `{regret_matrix_path}`",
        f"- `{label_counts_path}`",
        f"- `{selector_accuracy_path}`",
    ]
    report_text = "\n".join(lines) + "\n"

    if can_write:
        write_csv(supervision_path, supervision)
        write_csv(regret_matrix_path, regret_matrix)
        write_csv(label_counts_path, label_counts)
        write_csv(selector_accuracy_path, selector_accuracy)
        write_text(report_path, report_text)

    print("Binary trigger calibration selector supervision v1")
    print(f"supervision: {supervision_path}")
    print(f"regret_matrix: {regret_matrix_path}")
    print(f"label_counts: {label_counts_path}")
    print(f"selector_accuracy: {selector_accuracy_path}")
    print(f"report: {report_path}")
    print("")
    print(label_counts.to_string(index=False))
    print("")
    print(selector_accuracy.to_string(index=False))
    print("")
    print(supervision.to_string(index=False))


if __name__ == "__main__":
    main()
