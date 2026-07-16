#!/usr/bin/env python3
"""Evaluate conservative fallback policies for per-site curve ranker CV output.

The curve-aware ranker passes capacity but fails held-out-date CV. This script
does not retrain; it asks whether simple, conservative guards could make the
ranker deployable by falling back to the paper fixed-list oracle decision.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from train_confirmed_5site_true_input_surrogate_baseline_v1 import markdown_table


DEFAULT_BASE = (
    Path("site_general_surrogate_eval")
    / "continuous_ir_12site_10k_persite_curve_mlp_ranker_cv_v1"
)


def safe_mean(values: pd.Series | np.ndarray) -> float:
    return float(np.mean(values)) if len(values) else float("nan")


def mask_to_array(use_ranker: pd.Series | np.ndarray, index: pd.Index) -> np.ndarray:
    if isinstance(use_ranker, pd.Series):
        mask = use_ranker.reindex(index).fillna(False).astype(bool).to_numpy()
    else:
        mask = np.asarray(use_ranker, dtype=bool)
    if len(mask) != len(index):
        raise ValueError(f"Policy mask length {len(mask)} does not match rows {len(index)}")
    return mask


def policy_summary(df: pd.DataFrame, *, policy_name: str, use_ranker: pd.Series | np.ndarray) -> dict:
    use_ranker_mask = mask_to_array(use_ranker, df.index)
    regret = np.where(
        use_ranker_mask,
        df["continuous_ranker_regret_vs_dense_oracle"].to_numpy(dtype=float),
        df["paper_regret_vs_dense_oracle"].to_numpy(dtype=float),
    )
    gain_over_paper = df["paper_regret_vs_dense_oracle"].to_numpy(dtype=float) - regret
    return {
        "policy": policy_name,
        "site_dates": int(len(df)),
        "mean_regret_vs_dense": float(np.mean(regret)),
        "median_regret_vs_dense": float(np.median(regret)),
        "mean_gain_over_paper": float(np.mean(gain_over_paper)),
        "better_than_paper_rate": safe_mean(gain_over_paper > 1e-9),
        "worse_than_paper_rate": safe_mean(gain_over_paper < -1e-9),
        "zero_regret_rate": safe_mean(np.abs(regret) <= 1e-9),
        "large_regret_rate_gt_5": safe_mean(regret > 5.0),
        "use_ranker_rate": safe_mean(use_ranker_mask),
    }


def attach_policy_regret(
    df: pd.DataFrame,
    policy_name: str,
    use_ranker: pd.Series | np.ndarray,
) -> pd.DataFrame:
    use_ranker_mask = mask_to_array(use_ranker, df.index)
    out = df[["site_id", "site_date_id", "date_t"]].copy()
    out["policy"] = policy_name
    out["use_ranker"] = use_ranker_mask
    out["policy_regret_vs_dense"] = np.where(
        use_ranker_mask,
        df["continuous_ranker_regret_vs_dense_oracle"].to_numpy(dtype=float),
        df["paper_regret_vs_dense_oracle"].to_numpy(dtype=float),
    )
    out["policy_gain_over_paper"] = df["paper_regret_vs_dense_oracle"].to_numpy(dtype=float) - out[
        "policy_regret_vs_dense"
    ].to_numpy(dtype=float)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-dir", default=str(DEFAULT_BASE))
    parser.add_argument("--output-dir", default="")
    parser.add_argument("--top-n", type=int, default=30)
    args = parser.parse_args()

    base_dir = Path(args.base_dir)
    out_dir = Path(args.output_dir) if args.output_dir else base_dir / "guard_policy_eval_v1"
    out_dir.mkdir(parents=True, exist_ok=True)

    decisions_path = base_dir / "persite_curve_mlp_ranker_decisions_v1.csv"
    sampled_path = base_dir / "persite_curve_mlp_ranker_sampled_rank_eval_v1.csv"
    if not decisions_path.exists():
        raise FileNotFoundError(f"Missing decisions table: {decisions_path}")
    if not sampled_path.exists():
        raise FileNotFoundError(f"Missing sampled-rank table: {sampled_path}")

    decisions = pd.read_csv(decisions_path)
    sampled = pd.read_csv(sampled_path)
    if "eval_mode" in decisions.columns:
        decisions = decisions.loc[decisions["eval_mode"].astype(str) == "heldout_date_cv"].copy()
    if "eval_mode" in sampled.columns:
        sampled = sampled.loc[sampled["eval_mode"].astype(str) == "heldout_date_cv"].copy()
    merged = decisions.merge(
        sampled[["site_date_id", "sampled_top1_correct", "sampled_curve_regret", "true_best_pred_rank"]],
        on="site_date_id",
        how="left",
    )
    if merged.empty:
        raise ValueError("No heldout_date_cv rows found")

    by_site_mean = (
        merged.groupby("site_id")
        .agg(
            paper_mean_regret=("paper_regret_vs_dense_oracle", "mean"),
            ranker_mean_regret=("continuous_ranker_regret_vs_dense_oracle", "mean"),
        )
        .reset_index()
    )
    safe_sites = set(
        by_site_mean.loc[
            by_site_mean["ranker_mean_regret"] < by_site_mean["paper_mean_regret"],
            "site_id",
        ].astype(str)
    )

    policies: dict[str, pd.Series] = {
        "paper_fixed_list": pd.Series(False, index=merged.index),
        "raw_continuous_ranker": pd.Series(True, index=merged.index),
        "fixed_candidate_guard": ~merged["continuous_ranker_nonfixed_ir"].astype(bool),
        "cv_site_oracle_guard": merged["site_id"].astype(str).isin(safe_sites),
        "sampled_top1_oracle_guard": merged["sampled_top1_correct"].astype(bool),
        "regret_oracle_guard": merged["continuous_ranker_gain_over_paper"].to_numpy(dtype=float) > 1e-9,
    }

    summary = pd.DataFrame(
        [policy_summary(merged, policy_name=name, use_ranker=mask) for name, mask in policies.items()]
    ).sort_values("mean_regret_vs_dense")

    detail_parts = [attach_policy_regret(merged, name, mask) for name, mask in policies.items()]
    detail = pd.concat(detail_parts, ignore_index=True)
    by_site = (
        detail.groupby(["policy", "site_id"])
        .agg(
            site_dates=("site_date_id", "count"),
            mean_regret_vs_dense=("policy_regret_vs_dense", "mean"),
            mean_gain_over_paper=("policy_gain_over_paper", "mean"),
            use_ranker_rate=("use_ranker", "mean"),
            large_regret_rate_gt_5=("policy_regret_vs_dense", lambda s: float((s > 5.0).mean())),
        )
        .reset_index()
        .sort_values(["policy", "mean_regret_vs_dense"], ascending=[True, False])
    )
    worst = (
        detail.sort_values("policy_regret_vs_dense", ascending=False)
        .head(args.top_n)
        .reset_index(drop=True)
    )
    safe_site_table = by_site_mean.copy()
    safe_site_table["cv_site_oracle_uses_ranker"] = safe_site_table["site_id"].astype(str).isin(safe_sites)

    summary_path = out_dir / "persite_curve_ranker_guard_policy_summary_v1.csv"
    by_site_path = out_dir / "persite_curve_ranker_guard_policy_by_site_v1.csv"
    worst_path = out_dir / "persite_curve_ranker_guard_policy_worst_v1.csv"
    safe_sites_path = out_dir / "persite_curve_ranker_guard_policy_safe_sites_v1.csv"
    report_path = out_dir / "persite_curve_ranker_guard_policy_eval_v1.md"
    summary.to_csv(summary_path, index=False)
    by_site.to_csv(by_site_path, index=False)
    worst.to_csv(worst_path, index=False)
    safe_site_table.to_csv(safe_sites_path, index=False)

    lines = [
        "# Per-Site Curve Ranker Guard Policy Eval V1",
        "",
        "## Policy Summary",
        "",
        markdown_table(summary),
        "",
        "## Site-Level Safety Table",
        "",
        markdown_table(safe_site_table),
        "",
        "## By Site",
        "",
        markdown_table(by_site),
        "",
        "## Worst Policy Rows",
        "",
        markdown_table(worst),
        "",
        "## Outputs",
        "",
        f"- `{summary_path}`",
        f"- `{by_site_path}`",
        f"- `{worst_path}`",
        f"- `{safe_sites_path}`",
    ]
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print("Per-site curve ranker guard policy eval v1")
    print(f"summary: {summary_path}")
    print(f"by_site: {by_site_path}")
    print(f"worst: {worst_path}")
    print(f"safe_sites: {safe_sites_path}")
    print(f"report: {report_path}")
    print("")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
