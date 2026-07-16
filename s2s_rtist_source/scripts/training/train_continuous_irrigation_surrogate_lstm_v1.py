#!/usr/bin/env python3
"""Train a PyTorch LSTM site-general continuous-irrigation surrogate.

This consumes the sequence-wide table produced by
build_continuous_ir_sequence_wide_features_v1.py. It keeps the same
leave-one-site-out protocol as the tabular baselines, but preserves the daily
history/future sequence structure:

- 14 historical weather days
- 7 future weather days
- 7 future candidate irrigation values
- tabular static/current-state/candidate features
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

from train_confirmed_5site_true_input_surrogate_baseline_v1 import (
    TARGET,
    bool_series,
    build_features,
    markdown_table,
)
from train_continuous_irrigation_surrogate_tree_nosklearn_v1 import score_metrics

try:
    import torch
    from torch import nn
    from torch.utils.data import DataLoader, TensorDataset
except ImportError as exc:  # pragma: no cover - runtime dependency guard
    raise SystemExit(
        "PyTorch is required for train_continuous_irrigation_surrogate_lstm_v1.py. "
        "Install torch or run the tabular baselines instead."
    ) from exc


DEFAULT_DATA = (
    Path("site_general_surrogate_eval")
    / "continuous_ir_12site_10k_surrogate_sequence_wide_features_v1"
    / "continuous_ir_12site_surrogate_sequence_wide_samples_v1.csv"
)
DEFAULT_OUT_DIR = (
    Path("site_general_surrogate_eval")
    / "continuous_ir_12site_10k_surrogate_lstm_loso_v1"
)
WEATHER_FIELDS = ["solar", "tmax", "tmin", "relhum", "precip", "windspeed"]


class LstmSurrogate(nn.Module):
    def __init__(self, seq_dim: int, tab_dim: int, hidden_size: int, tab_hidden: int, dropout: float):
        super().__init__()
        self.lstm = nn.LSTM(seq_dim, hidden_size, batch_first=True)
        self.tab = nn.Sequential(
            nn.Linear(tab_dim, tab_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(tab_hidden, tab_hidden),
            nn.ReLU(),
        )
        self.head = nn.Sequential(
            nn.Linear(hidden_size + tab_hidden, tab_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(tab_hidden, 1),
        )

    def forward(self, seq: torch.Tensor, tab: torch.Tensor) -> torch.Tensor:
        _, (hidden, _) = self.lstm(seq)
        seq_vec = hidden[-1]
        tab_vec = self.tab(tab)
        return self.head(torch.cat([seq_vec, tab_vec], dim=1)).squeeze(1)


def make_sequence_array(df: pd.DataFrame, history_days: int, horizon_days: int) -> tuple[np.ndarray, list[str]]:
    cols = []
    arrays = []
    # Chronological order: oldest history day -> latest history day -> future days.
    for lag in range(history_days, 0, -1):
        day_cols = [f"hist_lag{lag:02d}_{field}" for field in WEATHER_FIELDS]
        for col in day_cols:
            if col not in df.columns:
                raise ValueError(f"Missing sequence column: {col}")
        arrays.append(df[day_cols].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float))
        cols.extend(day_cols)
    for day in range(1, horizon_days + 1):
        day_cols = [f"future_day{day:02d}_{field}" for field in WEATHER_FIELDS]
        ir_col = f"future_ir_day{day:02d}"
        for col in [*day_cols, ir_col]:
            if col not in df.columns:
                raise ValueError(f"Missing sequence column: {col}")
        arrays.append(df[[*day_cols, ir_col]].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float))
        cols.extend([*day_cols, ir_col])

    # Pad historical rows with an irrigation column so all timesteps share dims.
    seq_parts = []
    for i, arr in enumerate(arrays):
        if i < history_days:
            arr = np.column_stack([arr, np.zeros(len(df), dtype=float)])
        seq_parts.append(arr[:, None, :])
    seq = np.concatenate(seq_parts, axis=1)
    return seq, cols


def usable_tabular_columns(x: pd.DataFrame) -> list[str]:
    seq_prefixes = ("hist_lag", "future_day", "future_ir_day")
    invalid_site_id_prefixes = (
        "site_",
        "candidate_ir_x_site_",
        "candidate_ir_sq_x_site_",
    )
    return [
        col
        for col in x.columns
        if not col.startswith(seq_prefixes) and not x[col].isna().all()
        and not col.startswith(invalid_site_id_prefixes)
    ]


def standardize_train_test(train: np.ndarray, test: np.ndarray) -> tuple[np.ndarray, np.ndarray, dict]:
    med = np.nanmedian(train, axis=0)
    med = np.where(np.isnan(med), 0.0, med)
    train = train.copy()
    test = test.copy()
    train_inds = np.where(np.isnan(train))
    test_inds = np.where(np.isnan(test))
    train[train_inds] = np.take(med, train_inds[-1])
    test[test_inds] = np.take(med, test_inds[-1])
    mean = train.mean(axis=0)
    std = train.std(axis=0)
    std = np.where(std <= 1e-12, 1.0, std)
    return (train - mean) / std, (test - mean) / std, {"median": med, "mean": mean, "std": std}


def train_fold(
    seq_train: np.ndarray,
    tab_train: np.ndarray,
    y_train: np.ndarray,
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
    y_mean = float(y_train.mean())
    y_std = float(y_train.std())
    if y_std <= 1e-12:
        y_std = 1.0
    y_scaled = (y_train - y_mean) / y_std

    model = LstmSurrogate(
        seq_dim=seq_train.shape[2],
        tab_dim=tab_train.shape[1],
        hidden_size=hidden_size,
        tab_hidden=tab_hidden,
        dropout=dropout,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = nn.MSELoss()
    dataset = TensorDataset(
        torch.tensor(seq_train, dtype=torch.float32),
        torch.tensor(tab_train, dtype=torch.float32),
        torch.tensor(y_scaled, dtype=torch.float32),
    )
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    model.train()
    for _epoch in range(epochs):
        for seq_batch, tab_batch, y_batch in loader:
            seq_batch = seq_batch.to(device)
            tab_batch = tab_batch.to(device)
            y_batch = y_batch.to(device)
            optimizer.zero_grad()
            loss = loss_fn(model(seq_batch, tab_batch), y_batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()

    model.eval()
    preds = []
    with torch.no_grad():
        for start in range(0, len(seq_test), batch_size):
            seq_batch = torch.tensor(seq_test[start : start + batch_size], dtype=torch.float32).to(device)
            tab_batch = torch.tensor(tab_test[start : start + batch_size], dtype=torch.float32).to(device)
            preds.append(model(seq_batch, tab_batch).cpu().numpy())
    return np.concatenate(preds) * y_std + y_mean


def safe_mean(series: pd.Series) -> float:
    return float(series.mean()) if len(series) else float("nan")


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
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
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

    groups = sorted(df[args.cv_group_col].astype(str).unique())
    pred_parts = []
    print(
        f"[lstm] loaded rows={len(df)} groups={len(groups)} seq_shape={seq_all.shape} tab_cols={len(tab_cols)}",
        flush=True,
    )
    for i, group in enumerate(groups):
        print(f"[lstm] fold {i + 1}/{len(groups)} holdout {args.cv_group_col}={group}", flush=True)
        test_mask = df[args.cv_group_col].astype(str).to_numpy() == str(group)
        train_mask = ~test_mask
        seq_train, seq_test, _seq_stats = standardize_train_test(seq_all[train_mask], seq_all[test_mask])
        tab_train, tab_test, _tab_stats = standardize_train_test(tab_all[train_mask], tab_all[test_mask])
        preds = train_fold(
            seq_train,
            tab_train,
            y_all[train_mask],
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
        part["model"] = "lstm"
        part["pred_net_gain_7d"] = preds
        pred_parts.append(part)
        print(f"[lstm] fold {i + 1}/{len(groups)} completed rows={int(test_mask.sum())}", flush=True)

    pred_df = pd.concat(pred_parts, ignore_index=True)
    decisions = []
    for site_date_id, part in pred_df.groupby("site_date_id", sort=False):
        true_best = part.loc[part[TARGET].idxmax()]
        pred_best = part.loc[part["pred_net_gain_7d"].idxmax()]
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
                "pred_best_pred_net_gain": float(pred_best["pred_net_gain_7d"]),
                "decision_correct": float(true_best["candidate_ir"]) == float(pred_best["candidate_ir"]),
                "decision_regret": float(true_best[TARGET] - pred_best[TARGET]),
            }
        )
    decision_df = pd.DataFrame(decisions)
    metrics = score_metrics(
        pred_df[TARGET].to_numpy(dtype=float),
        pred_df["pred_net_gain_7d"].to_numpy(dtype=float),
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
            "decision_correct": int(decision_df["decision_correct"].sum()),
            "decision_total": int(len(decision_df)),
            "decision_accuracy": safe_mean(decision_df["decision_correct"]),
            "mean_decision_regret": float(decision_df["decision_regret"].mean()),
            "median_decision_regret": float(decision_df["decision_regret"].median()),
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
            n_site_dates=("site_date_id", "count"),
        )
        .reset_index()
    )
    worst = decision_df.sort_values("decision_regret", ascending=False).head(15)

    pred_path = out_dir / "continuous_irrigation_surrogate_lstm_v1_predictions.csv"
    decision_path = out_dir / "continuous_irrigation_surrogate_lstm_v1_decision_eval.csv"
    metrics_path = out_dir / "continuous_irrigation_surrogate_lstm_v1_metrics.csv"
    by_site_path = out_dir / "continuous_irrigation_surrogate_lstm_v1_by_site.csv"
    feature_path = out_dir / "continuous_irrigation_surrogate_lstm_v1_features.json"
    report_path = out_dir / "continuous_irrigation_surrogate_lstm_v1.md"
    pred_df.to_csv(pred_path, index=False)
    decision_df.to_csv(decision_path, index=False)
    metrics_df.to_csv(metrics_path, index=False)
    by_site.to_csv(by_site_path, index=False)
    feature_path.write_text(
        json.dumps({"sequence_columns": seq_cols, "tabular_columns": tab_cols}, indent=2),
        encoding="utf-8",
    )
    lines = [
        "# Continuous Irrigation Surrogate LSTM V1",
        "",
        "## Scope",
        "",
        "- Site-general LSTM over daily history/future weather and future candidate irrigation.",
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

    print("Continuous irrigation surrogate LSTM v1")
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
