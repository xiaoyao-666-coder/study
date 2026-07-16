#!/usr/bin/env python3
"""Diagnose state/features for one expanded decision date.

Use this after the expanded tree and learned-trigger outputs exist. The default
target date is 18-Jul-2024 because it remains the largest regret date after the
plateau amount policy.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import pandas as pd


OUT_DIR = Path("Maize_shortterm_surrogate_expanded_v1")

FEATURE_CANDIDATES = [
    "decision_doy",
    "state_dvs",
    "state_lai",
    "state_rootd",
    "state_cwdm",
    "state_cwso",
    "soil_layer_count",
    "soil_h_mean_0_30_cm",
    "soil_h_mean_30_60_cm",
    "soil_h_mean_60_100_cm",
    "soil_h_mean_0_100_cm",
    "soil_h_min_0_100_cm",
    "soil_h_max_0_100_cm",
    "hist_precip_sum",
    "hist_solar_mean",
    "hist_tmax_mean",
    "hist_tmin_mean",
    "hist_relhum_mean",
    "hist_windspeed_mean",
    "future_precip_sum",
    "future_solar_mean",
    "future_tmax_mean",
    "future_tmin_mean",
    "future_relhum_mean",
    "future_windspeed_mean",
]


def markdown_table(df: pd.DataFrame, max_rows: int | None = None) -> str:
    if max_rows is not None:
        df = df.head(max_rows)
    if df.empty:
        return "_No rows._"
    cols = list(df.columns)
    lines = [
        "| " + " | ".join(cols) + " |",
        "| " + " | ".join(["---"] * len(cols)) + " |",
    ]
    for row in df.itertuples(index=False):
        lines.append("| " + " | ".join(format_value(v) for v in row) + " |")
    return "\n".join(lines)


def format_value(value: object) -> str:
    if isinstance(value, float):
        return f"{value:.3f}"
    return str(value)


def load_inputs(out_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    samples_path = out_dir / "shortterm_surrogate_expanded_samples_v1.csv"
    pred_path = out_dir / "surrogate_tree_nosklearn_expanded_v1_predictions.csv"
    learned_path = out_dir / "learned_trigger_curve_policy_expanded_v1_decision_eval.csv"
    for path in [samples_path, pred_path, learned_path]:
        if not path.exists():
            raise FileNotFoundError(path)

    samples = pd.read_csv(samples_path)
    pred = pd.read_csv(pred_path)
    learned = pd.read_csv(learned_path)
    learned = learned[learned["amount_policy"] == "raw_tree_peak"].copy()

    df = samples.merge(pred[["sample_id", "pred_net_gain_7d"]], on="sample_id", how="left")
    if df["pred_net_gain_7d"].isna().any():
        raise RuntimeError("Some samples did not match tree predictions.")
    return df, pred, learned


def available_features(df: pd.DataFrame) -> list[str]:
    return [col for col in FEATURE_CANDIDATES if col in df.columns]


def build_date_table(df: pd.DataFrame, learned: pd.DataFrame, feature_cols: list[str]) -> pd.DataFrame:
    rows = []
    for date_t, group in df.groupby("date_t", sort=False):
        g = group.sort_values("candidate_ir").reset_index(drop=True)
        first = g.iloc[0]
        true_best = g.loc[g["net_gain_7d"].idxmax()]
        tree_best = g.loc[g["pred_net_gain_7d"].idxmax()]
        learned_row = learned[learned["date_t"] == date_t].iloc[0]
        zero = g[g["candidate_ir"].astype(float) == 0.0].iloc[0]

        row = {
            "date_t": date_t,
            "decision_doy": int(first["decision_doy"]),
            "true_best_ir": float(true_best["candidate_ir"]),
            "tree_raw_ir": float(tree_best["candidate_ir"]),
            "learned_chosen_ir": float(learned_row["chosen_ir"]),
            "true_best_gain": float(true_best["net_gain_7d"]),
            "tree_raw_true_gain": float(tree_best["net_gain_7d"]),
            "zero_true_gain": float(zero["net_gain_7d"]),
            "raw_tree_regret": float(true_best["net_gain_7d"] - tree_best["net_gain_7d"]),
            "learned_regret": float(learned_row["decision_regret"]),
            "trigger_prob": float(learned_row["trigger_prob"]),
            "trigger_threshold": float(learned_row["trigger_threshold"]),
            "triggered": bool(learned_row["triggered"]),
        }
        for col in feature_cols:
            row[col] = float(first[col])
        rows.append(row)
    return pd.DataFrame(rows).sort_values("decision_doy").reset_index(drop=True)


def feature_percentiles(date_table: pd.DataFrame, target_date: str, feature_cols: list[str]) -> pd.DataFrame:
    target = date_table[date_table["date_t"] == target_date].iloc[0]
    rows = []
    for col in feature_cols:
        series = pd.to_numeric(date_table[col], errors="coerce")
        value = float(target[col])
        mean = float(series.mean())
        std = float(series.std())
        z = 0.0 if std <= 1e-12 else (value - mean) / std
        percentile = float((series <= value).mean())
        rows.append(
            {
                "feature": col,
                "target_value": value,
                "mean": mean,
                "std": std,
                "z_score": z,
                "abs_z_score": abs(z),
                "percentile": percentile,
                "min": float(series.min()),
                "max": float(series.max()),
            }
        )
    return pd.DataFrame(rows).sort_values("abs_z_score", ascending=False).reset_index(drop=True)


def standardized_distance(a: pd.Series, b: pd.Series, med: pd.Series, std: pd.Series, feature_cols: list[str]) -> float:
    total = 0.0
    used = 0
    for col in feature_cols:
        scale = float(std[col])
        if scale <= 1e-12 or math.isnan(scale):
            scale = 1.0
        av = float(a[col]) if pd.notna(a[col]) else float(med[col])
        bv = float(b[col]) if pd.notna(b[col]) else float(med[col])
        diff = (av - bv) / scale
        total += diff * diff
        used += 1
    return math.sqrt(total / max(used, 1))


def nearest_dates(date_table: pd.DataFrame, target_date: str, feature_cols: list[str]) -> pd.DataFrame:
    med = date_table[feature_cols].median(numeric_only=True)
    std = date_table[feature_cols].std(numeric_only=True).replace(0, 1.0)
    target = date_table[date_table["date_t"] == target_date].iloc[0]

    rows = []
    for _, row in date_table.iterrows():
        if row["date_t"] == target_date:
            continue
        rows.append(
            {
                "date_t": row["date_t"],
                "distance": standardized_distance(target, row, med, std, feature_cols),
                "decision_doy": int(row["decision_doy"]),
                "true_best_ir": float(row["true_best_ir"]),
                "tree_raw_ir": float(row["tree_raw_ir"]),
                "learned_chosen_ir": float(row["learned_chosen_ir"]),
                "learned_regret": float(row["learned_regret"]),
                "state_dvs": row.get("state_dvs", None),
                "state_lai": row.get("state_lai", None),
                "future_precip_sum": row.get("future_precip_sum", None),
                "soil_h_mean_0_30_cm": row.get("soil_h_mean_0_30_cm", None),
            }
        )
    return pd.DataFrame(rows).sort_values("distance").reset_index(drop=True)


def candidate_curve(df: pd.DataFrame, target_date: str) -> pd.DataFrame:
    g = df[df["date_t"] == target_date].sort_values("candidate_ir").reset_index(drop=True).copy()
    if g.empty:
        raise ValueError(f"Target date not found: {target_date}")
    g["true_rank"] = g["net_gain_7d"].rank(ascending=False, method="first")
    g["pred_rank"] = g["pred_net_gain_7d"].rank(ascending=False, method="first")
    g["pred_error"] = g["pred_net_gain_7d"] - g["net_gain_7d"]
    g["is_true_best"] = g["true_rank"] == 1.0
    g["is_tree_peak"] = g["pred_rank"] == 1.0
    return g[
        [
            "date_t",
            "decision_doy",
            "candidate_ir",
            "net_gain_7d",
            "pred_net_gain_7d",
            "pred_error",
            "true_rank",
            "pred_rank",
            "is_true_best",
            "is_tree_peak",
        ]
    ]


def write_report(
    out_dir: Path,
    target_date: str,
    date_table: pd.DataFrame,
    features: pd.DataFrame,
    neighbors: pd.DataFrame,
    curve: pd.DataFrame,
) -> Path:
    safe_date = target_date.replace("-", "_")
    report_path = out_dir / f"date_feature_state_diagnostics_{safe_date}_v1.md"
    target = date_table[date_table["date_t"] == target_date].copy()
    target_view = target[
        [
            "date_t",
            "decision_doy",
            "true_best_ir",
            "tree_raw_ir",
            "learned_chosen_ir",
            "true_best_gain",
            "tree_raw_true_gain",
            "raw_tree_regret",
            "trigger_prob",
            "trigger_threshold",
            "triggered",
        ]
    ]
    report = [
        f"# Date Feature/State Diagnostics: {target_date}",
        "",
        "## Target Decision",
        "",
        markdown_table(target_view),
        "",
        "## Candidate Curve",
        "",
        markdown_table(curve),
        "",
        "## Most Unusual Features",
        "",
        markdown_table(features[["feature", "target_value", "z_score", "percentile", "min", "max"]], max_rows=18),
        "",
        "## Nearest Dates By State/Weather Features",
        "",
        markdown_table(neighbors, max_rows=10),
        "",
        "## Reading Notes",
        "",
        "- If nearest dates have different true best irrigation, the feature state is ambiguous for the tree model.",
        "- Large positive prediction errors on high-irrigation candidates indicate over-irrigation bias.",
        "- Large negative prediction error on the true best candidate indicates the model is suppressing the correct amount.",
    ]
    report_path.write_text("\n".join(report) + "\n", encoding="utf-8")
    return report_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default=str(OUT_DIR), help="Expanded surrogate result directory.")
    parser.add_argument("--date", default="18-Jul-2024", help="Decision date to diagnose.")
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    df, _, learned = load_inputs(out_dir)
    feature_cols = available_features(df)
    if not feature_cols:
        raise RuntimeError("No known feature columns found in samples.")

    date_table = build_date_table(df, learned, feature_cols)
    if args.date not in set(date_table["date_t"]):
        raise ValueError(f"Date not found: {args.date}")

    safe_date = args.date.replace("-", "_")
    features = feature_percentiles(date_table, args.date, feature_cols)
    neighbors = nearest_dates(date_table, args.date, feature_cols)
    curve = candidate_curve(df, args.date)

    feature_out = out_dir / f"date_feature_state_diagnostics_{safe_date}_features_v1.csv"
    neighbor_out = out_dir / f"date_feature_state_diagnostics_{safe_date}_neighbors_v1.csv"
    curve_out = out_dir / f"date_feature_state_diagnostics_{safe_date}_curve_v1.csv"
    features.to_csv(feature_out, index=False)
    neighbors.to_csv(neighbor_out, index=False)
    curve.to_csv(curve_out, index=False)
    report_out = write_report(out_dir, args.date, date_table, features, neighbors, curve)

    print(f"Date feature/state diagnostics: {args.date}")
    print("")
    print("Target candidate curve:")
    print(curve.to_string(index=False))
    print("")
    print("Most unusual features:")
    print(features.head(18).to_string(index=False))
    print("")
    print("Nearest dates:")
    print(neighbors.head(10).to_string(index=False))
    print("")
    print(f"wrote: {feature_out}")
    print(f"wrote: {neighbor_out}")
    print(f"wrote: {curve_out}")
    print(f"wrote: {report_out}")


if __name__ == "__main__":
    main()
