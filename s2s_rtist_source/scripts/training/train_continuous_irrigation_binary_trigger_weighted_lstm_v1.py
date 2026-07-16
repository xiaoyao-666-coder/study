#!/usr/bin/env python3
"""Train a regret-weighted site-date binary irrigation trigger.

Previous diagnostics showed that:

- A separate binary trigger is useful.
- Low global thresholds improve regret, but still do not match the paper fixed
  list.
- Site-oracle thresholds can beat the fixed list, so the trigger has signal.
- Static threshold transfer is not reliable enough.

This variant keeps the same site-date binary task but weights BCE loss by the
decision regret caused by trigger errors. High-gain positive dates receive more
weight, so the model is pushed to avoid expensive missed irrigation.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from train_confirmed_5site_true_input_surrogate_baseline_v1 import (
    TARGET,
    bool_series,
    build_features,
    markdown_table,
)
from train_continuous_irrigation_binary_trigger_lstm_v1 import (
    BinaryTriggerLstm,
    evaluate_thresholds,
    make_trigger_rows,
    make_trigger_sequence_array,
    parse_thresholds,
    usable_trigger_tabular_columns,
)
from train_continuous_irrigation_surrogate_lstm_v1 import standardize_train_test

try:
    import torch
    from torch import nn
    from torch.utils.data import DataLoader, TensorDataset
except ImportError as exc:  # pragma: no cover
    raise SystemExit(
        "PyTorch is required for train_continuous_irrigation_binary_trigger_weighted_lstm_v1.py."
    ) from exc


DEFAULT_DATA = (
    Path("site_general_surrogate_eval")
    / "continuous_ir_12site_10k_surrogate_sequence_wide_features_v1"
    / "continuous_ir_12site_surrogate_sequence_wide_samples_v1.csv"
)
DEFAULT_OUT_DIR = (
    Path("site_general_surrogate_eval")
    / "continuous_ir_12site_10k_binary_irrigation_trigger_weighted_loso_v1"
)
DEFAULT_THRESHOLDS = (
    "0,1e-10,1e-9,1e-8,1e-7,1e-6,5e-6,1e-5,5e-5,1e-4,"
    "5e-4,0.001,0.0025,0.005,0.01,0.02,0.03,0.04,0.05,"
    "0.075,0.1,0.15,0.2,0.25,0.3,0.4,0.5,0.75"
)


def make_regret_weights(
    trigger_df: pd.DataFrame,
    *,
    positive_regret_weight: float,
    negative_regret_weight: float,
    scale_quantile: float,
    max_weight: float,
    class_balance: bool,
) -> np.ndarray:
    y = trigger_df["should_irrigate"].astype(bool).to_numpy()
    pos_regret = pd.to_numeric(trigger_df["zero_if_positive_regret"], errors="coerce").fillna(0.0).to_numpy(dtype=float)
    neg_regret = pd.to_numeric(trigger_df["positive_if_zero_regret"], errors="coerce").fillna(0.0).to_numpy(dtype=float)
    active_regret = np.where(y, pos_regret, neg_regret)
    positive_regrets = active_regret[active_regret > 0]
    scale = float(np.quantile(positive_regrets, scale_quantile)) if len(positive_regrets) else 1.0
    if scale <= 1e-12:
        scale = 1.0

    weights = np.ones(len(trigger_df), dtype=float)
    weights += np.where(
        y,
        positive_regret_weight * np.clip(np.log1p(pos_regret) / np.log1p(scale), 0.0, 1.0),
        negative_regret_weight * np.clip(np.log1p(neg_regret) / np.log1p(scale), 0.0, 1.0),
    )
    if class_balance:
        positives = float(y.sum())
        negatives = float(len(y) - positives)
        if positives > 0:
            weights[y] *= max(1.0, negatives / positives)
    weights = np.clip(weights, 1.0, float(max_weight))
    weights /= weights.mean()
    return weights


def train_fold_weighted_binary(
    seq_train: np.ndarray,
    tab_train: np.ndarray,
    y_train: np.ndarray,
    w_train: np.ndarray,
    seq_test: np.ndarray,
    tab_test: np.ndarray,
    *,
    hidden_size: int,
    tab_hidden: int,
    dropout: float,
    epochs: int,
    batch_size: int,
    lr: float,
    weight_decay: float,
    random_state: int,
    device: str,
) -> np.ndarray:
    torch.manual_seed(random_state)
    model = BinaryTriggerLstm(
        seq_dim=seq_train.shape[2],
        tab_dim=tab_train.shape[1],
        hidden_size=hidden_size,
        tab_hidden=tab_hidden,
        dropout=dropout,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = nn.BCEWithLogitsLoss(reduction="none")
    dataset = TensorDataset(
        torch.tensor(seq_train, dtype=torch.float32),
        torch.tensor(tab_train, dtype=torch.float32),
        torch.tensor(y_train, dtype=torch.float32),
        torch.tensor(w_train, dtype=torch.float32),
    )
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model.train()
    for _epoch in range(epochs):
        for seq_batch, tab_batch, y_batch, w_batch in loader:
            seq_batch = seq_batch.to(device)
            tab_batch = tab_batch.to(device)
            y_batch = y_batch.to(device)
            w_batch = w_batch.to(device)
            optimizer.zero_grad()
            losses = loss_fn(model(seq_batch, tab_batch), y_batch)
            loss = (losses * w_batch).mean()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()

    model.eval()
    probs = []
    with torch.no_grad():
        for start in range(0, len(seq_test), batch_size):
            seq_batch = torch.tensor(seq_test[start : start + batch_size], dtype=torch.float32).to(device)
            tab_batch = torch.tensor(tab_test[start : start + batch_size], dtype=torch.float32).to(device)
            probs.append(torch.sigmoid(model(seq_batch, tab_batch)).cpu().numpy())
    return np.concatenate(probs)


def safe_mean(series: pd.Series) -> float:
    return float(series.mean()) if len(series) else float("nan")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default=str(DEFAULT_DATA))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUT_DIR))
    parser.add_argument("--cv-group-col", default="site_id")
    parser.add_argument("--history-days", type=int, default=14)
    parser.add_argument("--horizon-days", type=int, default=7)
    parser.add_argument("--hidden-size", type=int, default=48)
    parser.add_argument("--tab-hidden", type=int, default=64)
    parser.add_argument("--dropout", type=float, default=0.15)
    parser.add_argument("--epochs", type=int, default=140)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--positive-regret-weight", type=float, default=8.0)
    parser.add_argument("--negative-regret-weight", type=float, default=1.0)
    parser.add_argument("--scale-quantile", type=float, default=0.9)
    parser.add_argument("--max-sample-weight", type=float, default=30.0)
    parser.add_argument("--no-class-balance", action="store_true")
    parser.add_argument("--thresholds", default=DEFAULT_THRESHOLDS)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    args = parser.parse_args()

    data_path = Path(args.input)
    if not data_path.exists():
        raise FileNotFoundError(f"Missing sequence-wide sample table: {data_path}")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but torch.cuda.is_available() is false")

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(data_path)
    for col in ["is_best_ir", "target_collapse", "same_date_duplicate_target_curve"]:
        if col in df.columns:
            df[col] = bool_series(df[col])
    if TARGET not in df.columns:
        raise ValueError(f"Missing target column: {TARGET}")

    trigger_df = make_trigger_rows(df)
    seq_all, seq_cols = make_trigger_sequence_array(trigger_df, args.history_days, args.horizon_days)
    tab_all_df = build_features(trigger_df)
    tab_cols = usable_trigger_tabular_columns(tab_all_df)
    tab_all = tab_all_df[tab_cols].to_numpy(dtype=float)
    y_all = trigger_df["should_irrigate"].astype(float).to_numpy()
    weights_all = make_regret_weights(
        trigger_df,
        positive_regret_weight=args.positive_regret_weight,
        negative_regret_weight=args.negative_regret_weight,
        scale_quantile=args.scale_quantile,
        max_weight=args.max_sample_weight,
        class_balance=not args.no_class_balance,
    )

    groups = sorted(trigger_df[args.cv_group_col].astype(str).unique())
    pred_parts = []
    print(
        f"[weighted-binary-trigger] site_dates={len(trigger_df)} groups={len(groups)} "
        f"positive_rate={y_all.mean():.6f} mean_weight={weights_all.mean():.6f}",
        flush=True,
    )
    for i, group in enumerate(groups):
        print(f"[weighted-binary-trigger] fold {i + 1}/{len(groups)} holdout {args.cv_group_col}={group}", flush=True)
        test_mask = trigger_df[args.cv_group_col].astype(str).to_numpy() == str(group)
        train_mask = ~test_mask
        seq_train, seq_test, _seq_stats = standardize_train_test(seq_all[train_mask], seq_all[test_mask])
        tab_train, tab_test, _tab_stats = standardize_train_test(tab_all[train_mask], tab_all[test_mask])
        probs = train_fold_weighted_binary(
            seq_train,
            tab_train,
            y_all[train_mask],
            weights_all[train_mask],
            seq_test,
            tab_test,
            hidden_size=args.hidden_size,
            tab_hidden=args.tab_hidden,
            dropout=args.dropout,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            weight_decay=args.weight_decay,
            random_state=args.random_state + i,
            device=args.device,
        )
        keep_cols = [
            "site_date_id",
            "site_id",
            "date_t",
            "decision_doy",
            "target_collapse",
            "same_date_duplicate_target_curve",
            "should_irrigate",
            "true_best_ir",
            "true_best_net_gain",
            "zero_true_net_gain",
            "oracle_positive_ir",
            "oracle_positive_true_net_gain",
            "zero_if_positive_regret",
            "positive_if_zero_regret",
        ]
        part = trigger_df.loc[test_mask, keep_cols].copy()
        part["cv_group_col"] = args.cv_group_col
        part["cv_group_value"] = str(group)
        part["model"] = "lstm_binary_irrigation_trigger_weighted"
        part["sample_weight"] = weights_all[test_mask]
        part["pred_irrigate_prob"] = probs
        pred_parts.append(part)
        print(f"[weighted-binary-trigger] fold {i + 1}/{len(groups)} completed site_dates={int(test_mask.sum())}", flush=True)

    pred_df = pd.concat(pred_parts, ignore_index=True)
    decisions, summary = evaluate_thresholds(pred_df, parse_thresholds(args.thresholds))
    summary = summary.sort_values(
        ["mean_decision_regret_oracle_amount", "trigger_balanced_accuracy"],
        ascending=[True, False],
    )
    best_threshold = float(summary.iloc[0]["threshold"])
    best_decisions = decisions.loc[decisions["threshold"] == best_threshold].copy()
    by_site = (
        best_decisions.groupby("site_id")
        .agg(
            trigger_accuracy=("trigger_correct", "mean"),
            mean_decision_regret_oracle_amount=("trigger_decision_regret_oracle_amount", "mean"),
            max_decision_regret_oracle_amount=("trigger_decision_regret_oracle_amount", "max"),
            predicted_irrigation_rate=("pred_should_irrigate", "mean"),
            true_irrigation_rate=("should_irrigate", "mean"),
            n_site_dates=("site_date_id", "count"),
        )
        .reset_index()
        .sort_values("mean_decision_regret_oracle_amount", ascending=False)
    )
    worst = best_decisions.sort_values("trigger_decision_regret_oracle_amount", ascending=False).head(40)

    pred_path = out_dir / "continuous_irrigation_binary_trigger_weighted_lstm_v1_predictions.csv"
    decisions_path = out_dir / "continuous_irrigation_binary_trigger_weighted_lstm_v1_threshold_decisions.csv"
    summary_path = out_dir / "continuous_irrigation_binary_trigger_weighted_lstm_v1_threshold_sweep.csv"
    by_site_path = out_dir / "continuous_irrigation_binary_trigger_weighted_lstm_v1_by_site.csv"
    feature_path = out_dir / "continuous_irrigation_binary_trigger_weighted_lstm_v1_features.json"
    report_path = out_dir / "continuous_irrigation_binary_trigger_weighted_lstm_v1.md"
    pred_df.to_csv(pred_path, index=False)
    decisions.to_csv(decisions_path, index=False)
    summary.to_csv(summary_path, index=False)
    by_site.to_csv(by_site_path, index=False)
    feature_path.write_text(
        json.dumps({"sequence_columns": seq_cols, "tabular_columns": tab_cols}, indent=2),
        encoding="utf-8",
    )

    lines = [
        "# Continuous Irrigation Binary Trigger Weighted LSTM V1",
        "",
        "## Scope",
        "",
        "- Site-date binary trigger with regret-weighted BCE loss.",
        "- Positive missed-irrigation regret receives higher weight than low-cost false positives.",
        "- Decision regret still uses oracle positive amount after a positive trigger.",
        f"- Input table: `{data_path}`.",
        f"- CV group column: `{args.cv_group_col}`.",
        "",
        "## Weighting",
        "",
        f"- positive_regret_weight = `{args.positive_regret_weight}`",
        f"- negative_regret_weight = `{args.negative_regret_weight}`",
        f"- scale_quantile = `{args.scale_quantile}`",
        f"- max_sample_weight = `{args.max_sample_weight}`",
        f"- class_balance = `{not args.no_class_balance}`",
        "",
        "## Threshold Sweep",
        "",
        markdown_table(summary),
        "",
        f"Best threshold by oracle-amount mean regret: `{best_threshold}`",
        "",
        "## Best Threshold By Site",
        "",
        markdown_table(by_site),
        "",
        "## Worst Decisions At Best Threshold",
        "",
        markdown_table(worst),
        "",
        "## Outputs",
        "",
        f"- `{pred_path}`",
        f"- `{decisions_path}`",
        f"- `{summary_path}`",
        f"- `{by_site_path}`",
        f"- `{feature_path}`",
    ]
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print("Continuous irrigation binary trigger weighted LSTM v1")
    print(f"predictions: {pred_path}")
    print(f"threshold_decisions: {decisions_path}")
    print(f"threshold_sweep: {summary_path}")
    print(f"by_site: {by_site_path}")
    print(f"report: {report_path}")
    print("")
    print(summary.to_string(index=False))
    print("")
    print(by_site.to_string(index=False))


if __name__ == "__main__":
    main()
