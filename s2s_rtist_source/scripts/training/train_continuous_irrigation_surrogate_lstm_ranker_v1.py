#!/usr/bin/env python3
"""Train a decision-aware LSTM ranker for continuous irrigation candidates.

The regression LSTM has good pointwise correlation but poor decision ranking on
hard site-dates. This script keeps the same LOSO protocol and input features,
but trains with a listwise loss inside each site-date candidate curve.

The model is evaluated by selecting the highest-scored sampled candidate for
each held-out site-date and measuring regret against the dense sampled oracle.
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
from train_continuous_irrigation_surrogate_lstm_v1 import (
    LstmSurrogate,
    make_sequence_array,
    standardize_train_test,
    usable_tabular_columns,
)
from train_continuous_irrigation_surrogate_tree_nosklearn_v1 import score_metrics

try:
    import torch
    from torch import nn
except ImportError as exc:  # pragma: no cover
    raise SystemExit(
        "PyTorch is required for train_continuous_irrigation_surrogate_lstm_ranker_v1.py."
    ) from exc


DEFAULT_DATA = (
    Path("site_general_surrogate_eval")
    / "continuous_ir_12site_10k_surrogate_sequence_wide_features_v1"
    / "continuous_ir_12site_surrogate_sequence_wide_samples_v1.csv"
)
DEFAULT_OUT_DIR = (
    Path("site_general_surrogate_eval")
    / "continuous_ir_12site_10k_surrogate_lstm_ranker_loso_v1"
)


def safe_mean(series: pd.Series) -> float:
    return float(series.mean()) if len(series) else float("nan")


def make_group_indices(site_date_ids: np.ndarray) -> list[np.ndarray]:
    groups = []
    seen = pd.Series(site_date_ids).drop_duplicates().tolist()
    for site_date_id in seen:
        groups.append(np.where(site_date_ids == site_date_id)[0])
    return groups


def listwise_target(y: torch.Tensor, temperature: float) -> torch.Tensor:
    centered = y - torch.max(y)
    return torch.softmax(centered / max(float(temperature), 1e-6), dim=0)


def fit_ranker(
    seq_train: np.ndarray,
    tab_train: np.ndarray,
    y_train: np.ndarray,
    site_date_train: np.ndarray,
    *,
    hidden_size: int,
    tab_hidden: int,
    dropout: float,
    epochs: int,
    lr: float,
    weight_decay: float,
    random_state: int,
    target_temperature: float,
    device: str,
) -> LstmSurrogate:
    torch.manual_seed(random_state)
    rng = np.random.default_rng(random_state)
    model = LstmSurrogate(
        seq_dim=seq_train.shape[2],
        tab_dim=tab_train.shape[1],
        hidden_size=hidden_size,
        tab_hidden=tab_hidden,
        dropout=dropout,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    group_indices = make_group_indices(site_date_train)
    seq_tensor = torch.tensor(seq_train, dtype=torch.float32, device=device)
    tab_tensor = torch.tensor(tab_train, dtype=torch.float32, device=device)
    y_tensor = torch.tensor(y_train, dtype=torch.float32, device=device)
    model.train()
    for _epoch in range(epochs):
        order = rng.permutation(len(group_indices))
        for group_pos in order:
            idx_np = group_indices[int(group_pos)]
            if len(idx_np) < 2:
                continue
            idx = torch.tensor(idx_np, dtype=torch.long, device=device)
            scores = model(seq_tensor[idx], tab_tensor[idx])
            target_dist = listwise_target(y_tensor[idx], target_temperature)
            log_probs = torch.log_softmax(scores, dim=0)
            loss = -(target_dist * log_probs).sum()
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
    return model


def predict_scores(
    model: LstmSurrogate,
    seq: np.ndarray,
    tab: np.ndarray,
    *,
    batch_size: int,
    device: str,
) -> np.ndarray:
    model.eval()
    preds = []
    with torch.no_grad():
        for start in range(0, len(seq), batch_size):
            seq_batch = torch.tensor(seq[start : start + batch_size], dtype=torch.float32, device=device)
            tab_batch = torch.tensor(tab[start : start + batch_size], dtype=torch.float32, device=device)
            preds.append(model(seq_batch, tab_batch).cpu().numpy())
    return np.concatenate(preds)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default=str(DEFAULT_DATA))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUT_DIR))
    parser.add_argument("--cv-group-col", default="site_id")
    parser.add_argument("--history-days", type=int, default=14)
    parser.add_argument("--horizon-days", type=int, default=7)
    parser.add_argument("--hidden-size", type=int, default=64)
    parser.add_argument("--tab-hidden", type=int, default=96)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--target-temperature", type=float, default=20.0)
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

    seq_all, seq_cols = make_sequence_array(df, args.history_days, args.horizon_days)
    tab_all_df = build_features(df)
    tab_cols = usable_tabular_columns(tab_all_df)
    tab_all = tab_all_df[tab_cols].to_numpy(dtype=float)
    y_all = pd.to_numeric(df[TARGET], errors="coerce").to_numpy(dtype=float)
    if np.isnan(y_all).any():
        raise ValueError(f"Target column {TARGET} contains NaN")

    site_date_all = df["site_date_id"].astype(str).to_numpy()
    groups = sorted(df[args.cv_group_col].astype(str).unique())
    pred_parts = []
    print(
        f"[ranker] loaded rows={len(df)} groups={len(groups)} seq_shape={seq_all.shape} tab_cols={len(tab_cols)}",
        flush=True,
    )
    for i, group in enumerate(groups):
        print(f"[ranker] fold {i + 1}/{len(groups)} holdout {args.cv_group_col}={group}", flush=True)
        test_mask = df[args.cv_group_col].astype(str).to_numpy() == str(group)
        train_mask = ~test_mask
        seq_train, seq_test, _seq_stats = standardize_train_test(seq_all[train_mask], seq_all[test_mask])
        tab_train, tab_test, _tab_stats = standardize_train_test(tab_all[train_mask], tab_all[test_mask])
        model = fit_ranker(
            seq_train,
            tab_train,
            y_all[train_mask],
            site_date_all[train_mask],
            hidden_size=args.hidden_size,
            tab_hidden=args.tab_hidden,
            dropout=args.dropout,
            epochs=args.epochs,
            lr=args.lr,
            weight_decay=args.weight_decay,
            random_state=args.random_state + i,
            target_temperature=args.target_temperature,
            device=args.device,
        )
        scores = predict_scores(
            model,
            seq_test,
            tab_test,
            batch_size=args.batch_size,
            device=args.device,
        )
        keep_cols = [
            "sample_id",
            "site_date_id",
            "site_id",
            "date_t",
            "decision_doy",
            "candidate_ir",
            TARGET,
            "target_7d",
            "best_ir_for_date",
            "best_target_for_date",
            "is_best_ir",
            "target_collapse",
            "same_date_duplicate_target_curve",
        ]
        part = df.loc[test_mask, keep_cols].copy()
        part["cv_group_col"] = args.cv_group_col
        part["cv_group_value"] = str(group)
        part["model"] = "lstm_ranker"
        part["pred_rank_score"] = scores
        pred_parts.append(part)
        print(f"[ranker] fold {i + 1}/{len(groups)} completed rows={int(test_mask.sum())}", flush=True)

    pred_df = pd.concat(pred_parts, ignore_index=True)
    decisions = []
    for site_date_id, part in pred_df.groupby("site_date_id", sort=False):
        true_best = part.loc[part[TARGET].idxmax()]
        pred_best = part.loc[part["pred_rank_score"].idxmax()]
        true_rank = int(part["pred_rank_score"].rank(method="min", ascending=False).loc[true_best.name])
        decisions.append(
            {
                "site_date_id": site_date_id,
                "site_id": str(true_best["site_id"]),
                "date_t": str(true_best["date_t"]),
                "decision_doy": int(true_best["decision_doy"]),
                "target_collapse": bool(true_best["target_collapse"]),
                "same_date_duplicate_target_curve": bool(true_best["same_date_duplicate_target_curve"]),
                "true_best_ir": float(true_best["candidate_ir"]),
                "pred_best_ir": float(pred_best["candidate_ir"]),
                "true_best_net_gain": float(true_best[TARGET]),
                "pred_best_true_net_gain": float(pred_best[TARGET]),
                "pred_best_rank_score": float(pred_best["pred_rank_score"]),
                "true_best_rank_score": float(true_best["pred_rank_score"]),
                "true_best_pred_rank": true_rank,
                "decision_correct": float(true_best["candidate_ir"]) == float(pred_best["candidate_ir"]),
                "decision_regret": float(true_best[TARGET] - pred_best[TARGET]),
            }
        )
    decision_df = pd.DataFrame(decisions)
    metrics = score_metrics(
        pred_df[TARGET].to_numpy(dtype=float),
        pred_df["pred_rank_score"].to_numpy(dtype=float),
    )
    metrics.update(
        {
            "cv_group_col": args.cv_group_col,
            "cv_folds": int(len(groups)),
            "history_days": int(args.history_days),
            "horizon_days": int(args.horizon_days),
            "hidden_size": int(args.hidden_size),
            "tab_hidden": int(args.tab_hidden),
            "epochs": int(args.epochs),
            "target_temperature": float(args.target_temperature),
            "decision_correct": int(decision_df["decision_correct"].sum()),
            "decision_total": int(len(decision_df)),
            "decision_accuracy": safe_mean(decision_df["decision_correct"]),
            "mean_decision_regret": float(decision_df["decision_regret"].mean()),
            "median_decision_regret": float(decision_df["decision_regret"].median()),
            "mean_true_best_pred_rank": float(decision_df["true_best_pred_rank"].mean()),
            "median_true_best_pred_rank": float(decision_df["true_best_pred_rank"].median()),
            "collapse_decision_accuracy": safe_mean(decision_df.loc[decision_df["target_collapse"], "decision_correct"]),
            "noncollapse_decision_accuracy": safe_mean(decision_df.loc[~decision_df["target_collapse"], "decision_correct"]),
        }
    )
    metrics_df = pd.DataFrame([metrics])
    by_site = (
        decision_df.groupby("site_id")
        .agg(
            decision_accuracy=("decision_correct", "mean"),
            mean_decision_regret=("decision_regret", "mean"),
            max_decision_regret=("decision_regret", "max"),
            mean_true_best_pred_rank=("true_best_pred_rank", "mean"),
            n_site_dates=("site_date_id", "count"),
        )
        .reset_index()
        .sort_values("mean_decision_regret", ascending=False)
    )
    worst = decision_df.sort_values("decision_regret", ascending=False).head(20)

    pred_path = out_dir / "continuous_irrigation_surrogate_lstm_ranker_v1_predictions.csv"
    decision_path = out_dir / "continuous_irrigation_surrogate_lstm_ranker_v1_decision_eval.csv"
    metrics_path = out_dir / "continuous_irrigation_surrogate_lstm_ranker_v1_metrics.csv"
    by_site_path = out_dir / "continuous_irrigation_surrogate_lstm_ranker_v1_by_site.csv"
    feature_path = out_dir / "continuous_irrigation_surrogate_lstm_ranker_v1_features.json"
    report_path = out_dir / "continuous_irrigation_surrogate_lstm_ranker_v1.md"
    pred_df.to_csv(pred_path, index=False)
    decision_df.to_csv(decision_path, index=False)
    metrics_df.to_csv(metrics_path, index=False)
    by_site.to_csv(by_site_path, index=False)
    feature_path.write_text(
        json.dumps({"sequence_columns": seq_cols, "tabular_columns": tab_cols}, indent=2),
        encoding="utf-8",
    )

    lines = [
        "# Continuous Irrigation Surrogate LSTM Ranker V1",
        "",
        "## Scope",
        "",
        "- Site-general listwise LSTM ranker over each site-date candidate curve.",
        f"- Input table: `{data_path}`.",
        f"- CV group column: `{args.cv_group_col}`.",
        "",
        "## Metrics",
        "",
        markdown_table(metrics_df),
        "",
        "## By Site",
        "",
        markdown_table(by_site),
        "",
        "## Worst Decision Rows",
        "",
        markdown_table(worst),
        "",
        "## Outputs",
        "",
        f"- `{pred_path}`",
        f"- `{decision_path}`",
        f"- `{metrics_path}`",
        f"- `{by_site_path}`",
        f"- `{feature_path}`",
    ]
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print("Continuous irrigation surrogate LSTM ranker v1")
    print(f"predictions: {pred_path}")
    print(f"decision_eval: {decision_path}")
    print(f"metrics: {metrics_path}")
    print(f"by_site: {by_site_path}")
    print(f"report: {report_path}")
    print(metrics_df.to_string(index=False))
    print("")
    print(by_site.to_string(index=False))
    print("")
    print(worst.to_string(index=False))


if __name__ == "__main__":
    main()
