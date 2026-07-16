#!/usr/bin/env python3
"""Summarize the binary-trigger mainline diagnostics.

This reads the output CSVs from the binary-trigger branch and writes a compact
comparison table. The goal is to make the branch conclusion explicit before any
amount-ranking work is attempted.
"""

from __future__ import annotations

import argparse
import errno
from pathlib import Path

import pandas as pd

from train_confirmed_5site_true_input_surrogate_baseline_v1 import markdown_table


DEFAULT_ROOT = Path("site_general_surrogate_eval")
PAPER_FIXED_LIST_REGRET = 0.614875609
POINTWISE_LSTM_REGRET = 4.419856
WEIGHTED_LSTM_REGRET = 5.011677
RANKER_LSTM_REGRET = 5.880401
TWOSTAGE_REGRET = 27.471759


def read_best(
    path: Path,
    metric_col: str,
    exclude_policies: set[str] | None = None,
    exclude_policy_prefixes: tuple[str, ...] = (),
) -> dict | None:
    if not path.exists():
        return None
    df = pd.read_csv(path)
    if exclude_policies and "policy" in df.columns:
        df = df.loc[~df["policy"].astype(str).isin(exclude_policies)].copy()
    if exclude_policy_prefixes and "policy" in df.columns:
        policy = df["policy"].astype(str)
        keep = ~policy.apply(lambda value: value.startswith(exclude_policy_prefixes))
        df = df.loc[keep].copy()
    if df.empty or metric_col not in df.columns:
        return None
    row = df.sort_values(metric_col).iloc[0].to_dict()
    return row


def add_row(rows: list[dict], *, policy: str, regret: float, deployable: bool, note: str, **extra: object) -> None:
    row = {
        "policy": policy,
        "mean_regret": float(regret),
        "deployable_loso": bool(deployable),
        "note": note,
    }
    row.update(extra)
    rows.append(row)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=str(DEFAULT_ROOT))
    parser.add_argument(
        "--output-dir",
        default=str(DEFAULT_ROOT / "continuous_ir_12site_10k_binary_trigger_mainline_summary_v1"),
    )
    args = parser.parse_args()

    root = Path(args.root)
    out_dir = Path(args.output_dir)
    write_outputs = True
    try:
        out_dir.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        if exc.errno != errno.ENOSPC:
            raise
        write_outputs = False
        print(f"[warn] No space left on device; will print summary only and skip writing: {out_dir}")
    rows: list[dict] = []

    add_row(
        rows,
        policy="paper_fixed_list",
        regret=PAPER_FIXED_LIST_REGRET,
        deployable=True,
        note="Strong baseline from fixed paper-style candidate list.",
    )
    add_row(
        rows,
        policy="pointwise_lstm_argmax",
        regret=POINTWISE_LSTM_REGRET,
        deployable=True,
        note="Original learned continuous candidate argmax.",
    )
    add_row(
        rows,
        policy="weighted_lstm_argmax",
        regret=WEIGHTED_LSTM_REGRET,
        deployable=True,
        note="Weighted pointwise gain regressor; worsened decision behavior.",
    )
    add_row(
        rows,
        policy="lstm_ranker_argmax",
        regret=RANKER_LSTM_REGRET,
        deployable=True,
        note="Listwise ranker; worse than pointwise LSTM.",
    )
    add_row(
        rows,
        policy="lstm_twostage_existing_predictions",
        regret=TWOSTAGE_REGRET,
        deployable=True,
        note="Thresholding existing gain regressor as trigger; failed.",
    )

    low = read_best(
        root
        / "continuous_ir_12site_10k_binary_trigger_low_threshold_resweep_v1"
        / "binary_trigger_low_threshold_resweep_summary_v1.csv",
        "mean_decision_regret_oracle_amount",
    )
    if low:
        add_row(
            rows,
            policy="binary_trigger_global_low_threshold",
            regret=low["mean_decision_regret_oracle_amount"],
            deployable=True,
            note="Best deployable raw-probability threshold found so far.",
            threshold=low.get("threshold", low.get("site_rate", "")),
            recall=low.get("trigger_recall", ""),
            specificity=low.get("trigger_specificity", ""),
        )

    loso_calibration = read_best(
        root
        / "continuous_ir_12site_10k_binary_trigger_loso_calibration_policies_v1"
        / "binary_trigger_loso_calibration_policy_summary_v1.csv",
        "mean_decision_regret_oracle_amount",
        exclude_policies={"site_oracle_threshold"},
    )
    if loso_calibration:
        add_row(
            rows,
            policy=f"binary_trigger_{loso_calibration.get('policy', 'loso_calibration')}",
            regret=loso_calibration["mean_decision_regret_oracle_amount"],
            deployable=True,
            note="LOSO calibration using only non-held-out site labels; best learned trigger calibration so far.",
            threshold=loso_calibration.get("mean_assigned_threshold", ""),
            recall=loso_calibration.get("trigger_recall", ""),
            specificity=loso_calibration.get("trigger_specificity", ""),
        )

    nested_selector = read_best(
        root
        / "continuous_ir_12site_10k_binary_trigger_nested_calibration_selector_v1"
        / "binary_trigger_nested_calibration_selector_summary_v1.csv",
        "mean_decision_regret_oracle_amount",
        exclude_policies={"oracle_candidate_policy_selector"},
        exclude_policy_prefixes=("candidate_",),
    )
    if nested_selector:
        add_row(
            rows,
            policy=f"binary_trigger_{nested_selector.get('policy', 'nested_calibration_selector')}",
            regret=nested_selector["mean_decision_regret_oracle_amount"],
            deployable=True,
            note="Nested selector over deployable calibration policies; uses only non-held-out site labels.",
            threshold=nested_selector.get("mean_assigned_threshold", ""),
            recall=nested_selector.get("trigger_recall", ""),
            specificity=nested_selector.get("trigger_specificity", ""),
        )

    oracle_candidate_selector = read_best(
        root
        / "continuous_ir_12site_10k_binary_trigger_nested_calibration_selector_v1"
        / "binary_trigger_nested_calibration_selector_summary_v1.csv",
        "mean_decision_regret_oracle_amount",
    )
    if oracle_candidate_selector and oracle_candidate_selector.get("policy") == "oracle_candidate_policy_selector":
        add_row(
            rows,
            policy="binary_trigger_oracle_candidate_policy_selector",
            regret=oracle_candidate_selector["mean_decision_regret_oracle_amount"],
            deployable=False,
            note="Diagnostic selector over deployable candidate policies; uses held-out labels to choose the policy.",
            threshold=oracle_candidate_selector.get("mean_assigned_threshold", ""),
            recall=oracle_candidate_selector.get("trigger_recall", ""),
            specificity=oracle_candidate_selector.get("trigger_specificity", ""),
        )

    site_oracle = read_best(
        root
        / "continuous_ir_12site_10k_binary_trigger_site_threshold_oracle_v1"
        / "binary_trigger_site_threshold_oracle_summary_v1.csv",
        "mean_regret",
    )
    if site_oracle:
        add_row(
            rows,
            policy="binary_trigger_site_oracle_threshold",
            regret=site_oracle["mean_regret"],
            deployable=False,
            note="Diagnostic upper bound; uses held-out site labels.",
            threshold=site_oracle.get("threshold", ""),
            recall=site_oracle.get("trigger_recall", ""),
            specificity=site_oracle.get("trigger_specificity", ""),
        )

    site_rate = read_best(
        root
        / "continuous_ir_12site_10k_binary_trigger_site_rate_policy_v1"
        / "binary_trigger_site_rate_policy_summary_v1.csv",
        "mean_decision_regret_oracle_amount",
    )
    if site_rate:
        add_row(
            rows,
            policy="binary_trigger_site_rate_policy",
            regret=site_rate["mean_decision_regret_oracle_amount"],
            deployable=True,
            note="Within-site top-rate policy; did not preserve site-oracle advantage.",
            threshold=site_rate.get("site_rate", ""),
            recall=site_rate.get("trigger_recall", ""),
            specificity=site_rate.get("trigger_specificity", ""),
        )

    transfer = read_best(
        root
        / "continuous_ir_12site_10k_binary_trigger_threshold_transfer_v1"
        / "binary_trigger_threshold_transfer_summary_v1.csv",
        "mean_decision_regret_oracle_amount",
        exclude_policies={"site_oracle_threshold"},
    )
    if transfer:
        add_row(
            rows,
            policy=f"binary_trigger_{transfer.get('policy', 'threshold_transfer')}",
            regret=transfer["mean_decision_regret_oracle_amount"],
            deployable=transfer.get("policy") != "site_oracle_threshold",
            note="Static-feature threshold transfer diagnostic.",
            recall=transfer.get("trigger_recall", ""),
            specificity=transfer.get("trigger_specificity", ""),
        )

    weighted_trigger = read_best(
        root
        / "continuous_ir_12site_10k_binary_irrigation_trigger_weighted_loso_v1"
        / "continuous_irrigation_binary_trigger_weighted_lstm_v1_threshold_sweep.csv",
        "mean_decision_regret_oracle_amount",
    )
    if weighted_trigger:
        add_row(
            rows,
            policy="regret_weighted_binary_trigger",
            regret=weighted_trigger["mean_decision_regret_oracle_amount"],
            deployable=True,
            note="Regret-weighted BCE trigger; did not beat global low threshold.",
            threshold=weighted_trigger.get("threshold", ""),
            recall=weighted_trigger.get("trigger_recall", ""),
            specificity=weighted_trigger.get("trigger_specificity", ""),
        )

    summary = pd.DataFrame(rows).sort_values(["deployable_loso", "mean_regret"], ascending=[False, True])
    deployable = summary.loc[summary["deployable_loso"]]
    best_deployable = deployable.iloc[0]
    conclusion = (
        "The paper fixed-list baseline remains the best deployable policy in this branch."
        if best_deployable["policy"] == "paper_fixed_list"
        else f"The best deployable learned-trigger policy is {best_deployable['policy']}."
    )

    lines = [
        "# Binary Trigger Mainline Summary V1",
        "",
        "## Conclusion",
        "",
        conclusion,
        "",
        "Do not train a positive-amount ranker unless a deployable trigger first beats or matches the fixed-list baseline.",
        "",
        "## Comparison",
        "",
        markdown_table(summary),
        "",
        "## Outputs",
        "",
    ]
    report_text = "\n".join(lines) + "\n"

    summary_path = out_dir / "binary_trigger_mainline_summary_v1.csv"
    report_path = out_dir / "binary_trigger_mainline_summary_v1.md"
    if write_outputs:
        summary.to_csv(summary_path, index=False)
        report_path.write_text(report_text, encoding="utf-8")

    print("Binary trigger mainline summary v1")
    if write_outputs:
        print(f"summary: {summary_path}")
        print(f"report: {report_path}")
    else:
        print("summary: <not written; no space left on device>")
        print("report: <not written; no space left on device>")
    print("")
    print(summary.to_string(index=False))
    print("")
    print(conclusion)
    if not write_outputs:
        print("")
        print("Markdown report:")
        print(report_text)


if __name__ == "__main__":
    main()
