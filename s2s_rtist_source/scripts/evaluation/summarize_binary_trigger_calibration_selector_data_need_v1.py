#!/usr/bin/env python3
"""Summarize site-level data needs for calibration-selector learning.

The calibration-selector follow-up line needs supervised labels at the site
level: which calibration policy should be selected for each held-out site. This
script reads the supervision table and estimates the minimum additional
site-level labels needed to make selector learning less brittle.
"""

from __future__ import annotations

import argparse
import errno
from pathlib import Path

import pandas as pd

from train_confirmed_5site_true_input_surrogate_baseline_v1 import markdown_table


DEFAULT_ROOT = (
    Path("site_general_surrogate_eval")
    / "continuous_ir_12site_10k_binary_trigger_calibration_selector_supervision_v1"
)
DEFAULT_SUPERVISION = DEFAULT_ROOT / "binary_trigger_calibration_selector_supervision_by_site_v1.csv"
DEFAULT_SELECTOR_ACCURACY = DEFAULT_ROOT / "binary_trigger_calibration_selector_accuracy_v1.csv"
DEFAULT_OUT = (
    Path("site_general_surrogate_eval")
    / "continuous_ir_12site_10k_binary_trigger_calibration_selector_data_need_v1"
)


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
    parser.add_argument("--supervision", default=str(DEFAULT_SUPERVISION))
    parser.add_argument("--selector-accuracy", default=str(DEFAULT_SELECTOR_ACCURACY))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUT))
    parser.add_argument("--min-sites-per-class", type=int, default=8)
    parser.add_argument("--target-total-sites", type=int, default=32)
    args = parser.parse_args()

    supervision_path = Path(args.supervision)
    selector_accuracy_path = Path(args.selector_accuracy)
    if not supervision_path.exists():
        raise FileNotFoundError(f"Missing supervision file: {supervision_path}")
    if not selector_accuracy_path.exists():
        raise FileNotFoundError(f"Missing selector accuracy file: {selector_accuracy_path}")
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

    supervision = pd.read_csv(supervision_path)
    selector_accuracy = pd.read_csv(selector_accuracy_path)
    if "oracle_best_candidate_policy" not in supervision.columns:
        raise ValueError("Supervision table is missing oracle_best_candidate_policy")

    n_sites = int(len(supervision))
    n_classes = int(supervision["oracle_best_candidate_policy"].nunique())
    label_counts = (
        supervision.groupby("oracle_best_candidate_policy")
        .agg(
            current_sites=("site_id", "count"),
            mean_oracle_candidate_regret=("oracle_candidate_mean_regret", "mean"),
            mean_nearest1_gap=("nearest1_minus_oracle_candidate_regret", "mean"),
        )
        .reset_index()
        .sort_values("current_sites", ascending=False)
    )
    label_counts["min_sites_per_class"] = int(args.min_sites_per_class)
    label_counts["additional_sites_for_min_class"] = (
        args.min_sites_per_class - label_counts["current_sites"]
    ).clip(lower=0)
    lower_bound_additional_for_min_class = int(label_counts["additional_sites_for_min_class"].sum())
    lower_bound_total_for_min_class = int(n_sites + lower_bound_additional_for_min_class)
    additional_for_target_total = max(0, int(args.target_total_sites) - n_sites)

    summary = pd.DataFrame(
        [
            {
                "metric": "current_site_labels",
                "value": n_sites,
                "note": "One selector label per site.",
            },
            {
                "metric": "oracle_policy_classes",
                "value": n_classes,
                "note": "Distinct oracle candidate policies in the current supervision table.",
            },
            {
                "metric": "min_sites_per_class_target",
                "value": int(args.min_sites_per_class),
                "note": "Conservative minimum before fitting a learned multi-class selector.",
            },
            {
                "metric": "additional_sites_lower_bound_for_class_balance",
                "value": lower_bound_additional_for_min_class,
                "note": "Lower bound assuming new sites can be chosen to fill rare classes.",
            },
            {
                "metric": "total_sites_lower_bound_for_class_balance",
                "value": lower_bound_total_for_min_class,
                "note": "Current sites plus the class-balance lower bound.",
            },
            {
                "metric": "target_total_sites",
                "value": int(args.target_total_sites),
                "note": "Practical first target for a calibration-selector follow-up dataset.",
            },
            {
                "metric": "additional_sites_for_target_total",
                "value": additional_for_target_total,
                "note": "New site labels needed to reach the practical target.",
            },
        ]
    )
    rare_classes = label_counts.loc[label_counts["current_sites"] < args.min_sites_per_class].copy()
    selector_accuracy = selector_accuracy.sort_values("matches_oracle_candidate_policy", ascending=False)

    summary_path = out_dir / "binary_trigger_calibration_selector_data_need_summary_v1.csv"
    label_need_path = out_dir / "binary_trigger_calibration_selector_label_need_v1.csv"
    rare_path = out_dir / "binary_trigger_calibration_selector_rare_classes_v1.csv"
    report_path = out_dir / "binary_trigger_calibration_selector_data_need_v1.md"

    lines = [
        "# Binary Trigger Calibration Selector Data Need V1",
        "",
        "## Inputs",
        "",
        f"- Supervision: `{supervision_path}`",
        f"- Selector accuracy: `{selector_accuracy_path}`",
        "",
        "## Summary",
        "",
        markdown_table(summary),
        "",
        "## Label Need By Oracle Policy Class",
        "",
        markdown_table(label_counts),
        "",
        "## Rare Classes",
        "",
        markdown_table(rare_classes),
        "",
        "## Existing Selector Accuracy",
        "",
        markdown_table(selector_accuracy),
        "",
        "## Interpretation",
        "",
        "The current 12-site table is enough to diagnose the selector problem, "
        "but not enough to train a reliable multi-class site-level selector. "
        "A practical follow-up should target roughly 32 or more labeled sites, "
        "with special attention to adding examples of non-nearest-1 policies.",
        "",
        "## Outputs",
        "",
        f"- `{summary_path}`",
        f"- `{label_need_path}`",
        f"- `{rare_path}`",
    ]
    report_text = "\n".join(lines) + "\n"

    if can_write:
        write_csv(summary_path, summary)
        write_csv(label_need_path, label_counts)
        write_csv(rare_path, rare_classes)
        write_text(report_path, report_text)

    print("Binary trigger calibration selector data need v1")
    print(f"summary: {summary_path}")
    print(f"label_need: {label_need_path}")
    print(f"rare_classes: {rare_path}")
    print(f"report: {report_path}")
    print("")
    print(summary.to_string(index=False))
    print("")
    print(label_counts.to_string(index=False))
    print("")
    print(selector_accuracy.to_string(index=False))


if __name__ == "__main__":
    main()
