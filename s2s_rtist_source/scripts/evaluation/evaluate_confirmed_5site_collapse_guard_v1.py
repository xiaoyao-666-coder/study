#!/usr/bin/env python3
"""Evaluate simple prestate guards for target-collapse irrigation decisions.

This diagnostic sits on top of the conservative prediction policy. It first
selects the smallest irrigation candidate within a prediction threshold of the
predicted best score. If a prestate guard fires for the site-date, it instead
forces the 0 mm candidate. The goal is to test whether simple P2-like collapse
boundaries reduce LOSO regret without false-zeroing true non-collapse dates.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


DEFAULT_BASE_THRESHOLD = 75.0
DEFAULT_DVS_MIN = 1.50
DEFAULT_LAI_MAX = 3.53
DEFAULT_CWSO_MIN = 3455.0


def bool_series(series: pd.Series) -> pd.Series:
    if series.dtype == bool:
        return series
    return series.astype(str).str.lower().isin(["true", "1", "yes"])


def markdown_table(df: pd.DataFrame) -> str:
    if df.empty:
        return "_No rows._"
    cols = list(df.columns)
    rows = ["| " + " | ".join(cols) + " |", "| " + " | ".join(["---"] * len(cols)) + " |"]
    for row in df.itertuples(index=False):
        rows.append("| " + " | ".join(str(v) for v in row) + " |")
    return "\n".join(rows)


def select_conservative(group: pd.DataFrame, threshold: float) -> pd.Series:
    ranked = group.sort_values(["pred_net_gain_7d", "candidate_ir"], ascending=[False, True])
    max_pred = float(ranked["pred_net_gain_7d"].iloc[0])
    close = group[group["pred_net_gain_7d"] >= max_pred - threshold].sort_values("candidate_ir")
    return close.iloc[0]


def load_predictions(pred_path: Path, features_path: Path | None) -> pd.DataFrame:
    df = pd.read_csv(pred_path)
    state_cols = ["state_dvs", "state_lai", "state_cwso"]
    missing_state = [c for c in state_cols if c not in df.columns]
    if missing_state:
        if features_path is None:
            raise ValueError(
                "Predictions are missing state columns. Pass --features pointing "
                "to confirmed_5site_true_input_surrogate_features_samples_v1.csv."
            )
        features = pd.read_csv(features_path)
        merge_cols = ["sample_id"]
        add_cols = merge_cols + [c for c in state_cols if c in features.columns]
        missing_from_features = [c for c in state_cols if c not in features.columns]
        if missing_from_features:
            raise ValueError(f"Features CSV is missing state columns: {missing_from_features}")
        df = df.merge(features[add_cols], on=merge_cols, how="left", validate="one_to_one")
    return df


def guard_mask(row: pd.Series, guard_name: str, dvs_min: float, lai_max: float, cwso_min: float) -> bool:
    dvs = float(row["state_dvs"])
    lai = float(row["state_lai"])
    cwso = float(row["state_cwso"])
    parts = {
        "dvs": dvs >= dvs_min,
        "lai": lai <= lai_max,
        "cwso": cwso >= cwso_min,
    }
    if guard_name == "none":
        return False
    if guard_name == "dvs":
        return parts["dvs"]
    if guard_name == "lai":
        return parts["lai"]
    if guard_name == "cwso":
        return parts["cwso"]
    if guard_name == "dvs_lai":
        return parts["dvs"] and parts["lai"]
    if guard_name == "dvs_cwso":
        return parts["dvs"] and parts["cwso"]
    if guard_name == "lai_cwso":
        return parts["lai"] and parts["cwso"]
    if guard_name == "dvs_lai_cwso":
        return parts["dvs"] and parts["lai"] and parts["cwso"]
    raise ValueError(f"Unknown guard: {guard_name}")


def evaluate_guard(
    df: pd.DataFrame,
    guard_name: str,
    base_threshold: float,
    dvs_min: float,
    lai_max: float,
    cwso_min: float,
    max_ir: float | None,
) -> pd.DataFrame:
    rows = []
    for site_date_id, group in df.groupby("site_date_id", sort=False):
        true_best = group.loc[group["net_gain_7d"].idxmax()]
        base_chosen = select_conservative(group, base_threshold)
        triggered = guard_mask(true_best, guard_name, dvs_min, lai_max, cwso_min)
        if triggered:
            zero = group[group["candidate_ir"] == 0.0]
            chosen = zero.iloc[0] if not zero.empty else base_chosen
        else:
            chosen = base_chosen
        cap_applied = False
        if max_ir is not None and float(chosen["candidate_ir"]) > max_ir:
            capped = group[group["candidate_ir"] <= max_ir].sort_values("candidate_ir", ascending=False)
            if not capped.empty:
                chosen = capped.iloc[0]
                cap_applied = True
        rows.append(
            {
                "guard": guard_name,
                "base_threshold": base_threshold,
                "max_ir_cap": max_ir if max_ir is not None else "",
                "site_date_id": site_date_id,
                "site_id": str(true_best["site_id"]),
                "date_t": str(true_best["date_t"]),
                "decision_doy": int(true_best["decision_doy"]),
                "guard_triggered": bool(triggered),
                "cap_applied": bool(cap_applied),
                "target_collapse": bool(true_best["target_collapse"]),
                "true_best_ir": float(true_best["candidate_ir"]),
                "chosen_ir": float(chosen["candidate_ir"]),
                "base_chosen_ir": float(base_chosen["candidate_ir"]),
                "true_best_net_gain": float(true_best["net_gain_7d"]),
                "chosen_true_net_gain": float(chosen["net_gain_7d"]),
                "decision_correct": float(true_best["candidate_ir"]) == float(chosen["candidate_ir"]),
                "decision_regret": float(true_best["net_gain_7d"] - chosen["net_gain_7d"]),
                "false_zero_noncollapse": (
                    (not bool(true_best["target_collapse"]))
                    and float(true_best["candidate_ir"]) != 0.0
                    and float(chosen["candidate_ir"]) == 0.0
                ),
                "state_dvs": float(true_best["state_dvs"]),
                "state_lai": float(true_best["state_lai"]),
                "state_cwso": float(true_best["state_cwso"]),
            }
        )
    return pd.DataFrame(rows)


def summarize(decisions: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (base_threshold, max_ir_cap, guard_name), group in decisions.groupby(
        ["base_threshold", "max_ir_cap", "guard"], sort=False, dropna=False
    ):
        collapse = group[group["target_collapse"]]
        noncollapse = group[~group["target_collapse"]]
        rows.append(
            {
                "base_threshold": base_threshold,
                "max_ir_cap": max_ir_cap,
                "guard": guard_name,
                "decision_accuracy": float(group["decision_correct"].mean()),
                "mean_decision_regret": float(group["decision_regret"].mean()),
                "median_decision_regret": float(group["decision_regret"].median()),
                "max_decision_regret": float(group["decision_regret"].max()),
                "collapse_accuracy": float(collapse["decision_correct"].mean()) if not collapse.empty else float("nan"),
                "noncollapse_accuracy": float(noncollapse["decision_correct"].mean()) if not noncollapse.empty else float("nan"),
                "guard_trigger_rate": float(group["guard_triggered"].mean()),
                "guard_precision_for_collapse": float(group.loc[group["guard_triggered"], "target_collapse"].mean())
                if group["guard_triggered"].any()
                else float("nan"),
                "cap_applied_rate": float(group["cap_applied"].mean()),
                "false_zero_noncollapse": int(group["false_zero_noncollapse"].sum()),
                "zero_choice_rate": float((group["chosen_ir"] == 0.0).mean()),
            }
        )
    return pd.DataFrame(rows)


def summarize_by_site(decisions: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (base_threshold, max_ir_cap, guard_name, site_id), group in decisions.groupby(
        ["base_threshold", "max_ir_cap", "guard", "site_id"], sort=False, dropna=False
    ):
        rows.append(
            {
                "base_threshold": base_threshold,
                "max_ir_cap": max_ir_cap,
                "guard": guard_name,
                "site_id": site_id,
                "decision_accuracy": float(group["decision_correct"].mean()),
                "mean_decision_regret": float(group["decision_regret"].mean()),
                "max_decision_regret": float(group["decision_regret"].max()),
                "guard_triggered": int(group["guard_triggered"].sum()),
                "cap_applied": int(group["cap_applied"].sum()),
                "target_collapse": int(group["target_collapse"].sum()),
                "false_zero_noncollapse": int(group["false_zero_noncollapse"].sum()),
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", required=True, help="Surrogate predictions CSV.")
    parser.add_argument("--features", help="Feature samples CSV, needed when predictions lack state columns.")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--base-threshold", type=float, default=DEFAULT_BASE_THRESHOLD)
    parser.add_argument("--dvs-min", type=float, default=DEFAULT_DVS_MIN)
    parser.add_argument("--lai-max", type=float, default=DEFAULT_LAI_MAX)
    parser.add_argument("--cwso-min", type=float, default=DEFAULT_CWSO_MIN)
    parser.add_argument("--max-ir", type=float, default=None, help="Optional hard cap on chosen irrigation.")
    parser.add_argument(
        "--guards",
        nargs="+",
        default=["none", "dvs", "lai", "cwso", "dvs_lai", "dvs_cwso", "lai_cwso", "dvs_lai_cwso"],
    )
    args = parser.parse_args()

    pred_path = Path(args.predictions)
    features_path = Path(args.features) if args.features else None
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = load_predictions(pred_path, features_path)
    required = {
        "site_date_id",
        "site_id",
        "date_t",
        "decision_doy",
        "candidate_ir",
        "net_gain_7d",
        "pred_net_gain_7d",
        "target_collapse",
        "state_dvs",
        "state_lai",
        "state_cwso",
    }
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")
    for col in ["candidate_ir", "net_gain_7d", "pred_net_gain_7d", "state_dvs", "state_lai", "state_cwso"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df["target_collapse"] = bool_series(df["target_collapse"])

    decisions = pd.concat(
        [
            evaluate_guard(df, guard, args.base_threshold, args.dvs_min, args.lai_max, args.cwso_min, args.max_ir)
            for guard in args.guards
        ],
        ignore_index=True,
    )
    summary = summarize(decisions)
    by_site = summarize_by_site(decisions)
    best = summary.sort_values(
        ["mean_decision_regret", "false_zero_noncollapse", "decision_accuracy"],
        ascending=[True, True, False],
    ).head(1)

    decision_path = out_dir / "confirmed_5site_collapse_guard_v1_decision_eval.csv"
    summary_path = out_dir / "confirmed_5site_collapse_guard_v1_summary.csv"
    by_site_path = out_dir / "confirmed_5site_collapse_guard_v1_by_site.csv"
    report_path = out_dir / "confirmed_5site_collapse_guard_v1.md"
    decisions.to_csv(decision_path, index=False)
    summary.to_csv(summary_path, index=False)
    by_site.to_csv(by_site_path, index=False)

    lines = [
        "# Confirmed 5-Site Collapse Guard V1",
        "",
        f"Base conservative threshold: `{args.base_threshold}`",
        f"Max irrigation cap: `{args.max_ir if args.max_ir is not None else 'none'}`",
        f"Guard constants: `DVS >= {args.dvs_min}`, `LAI <= {args.lai_max}`, `CWSO >= {args.cwso_min}`",
        "",
        "## Summary",
        "",
        markdown_table(summary),
        "",
        "## Best By Mean Regret",
        "",
        markdown_table(best),
        "",
        "## By Site",
        "",
        markdown_table(by_site),
        "",
        "## Outputs",
        "",
        f"- `{decision_path}`",
        f"- `{summary_path}`",
        f"- `{by_site_path}`",
    ]
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print("Confirmed 5-site collapse guard v1")
    print(f"predictions: {pred_path}")
    if features_path:
        print(f"features: {features_path}")
    print(f"decision_eval: {decision_path}")
    print(f"summary: {summary_path}")
    print(f"by_site: {by_site_path}")
    print(f"report: {report_path}")
    print(summary.to_string(index=False))
    print("")
    print("best_by_mean_regret:")
    print(best.to_string(index=False))


if __name__ == "__main__":
    main()
