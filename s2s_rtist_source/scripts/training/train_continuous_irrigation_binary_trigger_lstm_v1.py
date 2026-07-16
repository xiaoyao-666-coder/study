#!/usr/bin/env python3
"""Train a site-date binary irrigation trigger with LOSO evaluation.

This is the next mainline step after the LSTM two-stage diagnostic. The gain
regression model is not reused as the trigger; this model directly learns
whether the best true candidate for a site-date is positive irrigation.

For decision-regret diagnostics, a positive trigger uses the oracle best
positive-irrigation amount. That isolates trigger quality before training a
separate amount-ranking model.
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
    WEATHER_FIELDS,
    standardize_train_test,
    usable_tabular_columns,
)

try:
    import torch
    from torch import nn
    from torch.utils.data import DataLoader, TensorDataset
except ImportError as exc:  # pragma: no cover
    raise SystemExit(
        "PyTorch is required for train_continuous_irrigation_binary_trigger_lstm_v1.py."
    ) from exc


DEFAULT_DATA = (
    Path("site_general_surrogate_eval")
    / "continuous_ir_12site_10k_surrogate_sequence_wide_features_v1"
    / "continuous_ir_12site_surrogate_sequence_wide_samples_v1.csv"
)
DEFAULT_OUT_DIR = (
    Path("site_general_surrogate_eval")
    / "continuous_ir_12site_10k_binary_irrigation_trigger_loso_v1"
)


class BinaryTriggerLstm(nn.Module):
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


def safe_mean(series: pd.Series) -> float:
    return float(series.mean()) if len(series) else float("nan")


def safe_div(num: float, den: float) -> float:
    return float(num / den) if den else float("nan")


def make_trigger_rows(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for site_date_id, group in df.groupby("site_date_id", sort=False):
        g = group.copy()
        g["candidate_ir"] = pd.to_numeric(g["candidate_ir"], errors="coerce")
        zero = g.loc[g["candidate_ir"].abs() <= 1e-9]
        if zero.empty:
            raise ValueError(f"Missing zero-irrigation candidate for site_date_id={site_date_id}")
        zero_row = zero.iloc[0].copy()
        positive = g.loc[g["candidate_ir"] > 1e-9]
        true_best = g.loc[g[TARGET].idxmax()]
        if positive.empty:
            best_positive = zero_row
        else:
            best_positive = positive.loc[positive[TARGET].idxmax()]

        zero_gain = float(zero_row[TARGET])
        true_best_gain = float(true_best[TARGET])
        best_positive_gain = float(best_positive[TARGET])
        should_irrigate = float(true_best["candidate_ir"]) > 1e-9
        zero_row["should_irrigate"] = bool(should_irrigate)
        zero_row["true_best_ir"] = float(true_best["candidate_ir"])
        zero_row["true_best_net_gain"] = true_best_gain
        zero_row["zero_true_net_gain"] = zero_gain
        zero_row["oracle_positive_ir"] = float(best_positive["candidate_ir"])
        zero_row["oracle_positive_true_net_gain"] = best_positive_gain
        zero_row["zero_if_positive_regret"] = true_best_gain - zero_gain
        zero_row["positive_if_zero_regret"] = true_best_gain - best_positive_gain
        rows.append(zero_row)
    return pd.DataFrame(rows).reset_index(drop=True)


def make_trigger_sequence_array(
    df: pd.DataFrame,
    history_days: int,
    horizon_days: int,
) -> tuple[np.ndarray, list[str]]:
    cols = []
    arrays = []
    for lag in range(history_days, 0, -1):
        day_cols = [f"hist_lag{lag:02d}_{field}" for field in WEATHER_FIELDS]
        for col in day_cols:
            if col not in df.columns:
                raise ValueError(f"Missing sequence column: {col}")
        arrays.append(df[day_cols].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float))
        cols.extend(day_cols)
    for day in range(1, horizon_days + 1):
        day_cols = [f"future_day{day:02d}_{field}" for field in WEATHER_FIELDS]
        for col in day_cols:
            if col not in df.columns:
                raise ValueError(f"Missing sequence column: {col}")
        arrays.append(df[day_cols].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float))
        cols.extend(day_cols)
    return np.stack(arrays, axis=1), cols


def usable_trigger_tabular_columns(x: pd.DataFrame) -> list[str]:
    base_cols = usable_tabular_columns(x)
    banned_prefixes = (
        "candidate_ir",
        "future_ir_day",
    )
    banned_exact = {
        "is_zero_ir",
        "candidate_ir_sq",
        "candidate_ir_fraction",
        "candidate_ir_fraction_sq",
    }
    return [
        col
        for col in base_cols
        if col not in banned_exact
        and not col.startswith(banned_prefixes)
        and not x[col].isna().all()
    ]


def train_fold_binary(
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
    model = BinaryTriggerLstm(
        seq_dim=seq_train.shape[2],
        tab_dim=tab_train.shape[1],
        hidden_size=hidden_size,
        tab_hidden=tab_hidden,
        dropout=dropout,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    positives = float(y_train.sum())
    negatives = float(len(y_train) - positives)
    pos_weight = torch.tensor([max(1.0, safe_div(negatives, positives))], dtype=torch.float32).to(device)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    dataset = TensorDataset(
        torch.tensor(seq_train, dtype=torch.float32),
        torch.tensor(tab_train, dtype=torch.float32),
        torch.tensor(y_train, dtype=torch.float32),
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
    probs = []
    with torch.no_grad():
        for start in range(0, len(seq_test), batch_size):
            seq_batch = torch.tensor(seq_test[start : start + batch_size], dtype=torch.float32).to(device)
            tab_batch = torch.tensor(tab_test[start : start + batch_size], dtype=torch.float32).to(device)
            logits = model(seq_batch, tab_batch)
            probs.append(torch.sigmoid(logits).cpu().numpy())
    return np.concatenate(probs)


def parse_thresholds(text: str) -> list[float]:
    values = [float(part.strip()) for part in text.split(",") if part.strip()]
    if not values:
        raise ValueError("At least one threshold is required")
    return sorted(set(values))


def evaluate_thresholds(pred_df: pd.DataFrame, thresholds: list[float]) -> tuple[pd.DataFrame, pd.DataFrame]:
    decisions = []
    summaries = []
    for threshold in thresholds:
        pred_positive = pred_df["pred_irrigate_prob"] >= threshold
        actual_positive = pred_df["should_irrigate"].astype(bool)
        chosen_gain = np.where(
            pred_positive,
            pred_df["oracle_positive_true_net_gain"].to_numpy(dtype=float),
            pred_df["zero_true_net_gain"].to_numpy(dtype=float),
        )
        regret = pred_df["true_best_net_gain"].to_numpy(dtype=float) - chosen_gain
        correct = pred_positive.to_numpy(dtype=bool) == actual_positive.to_numpy(dtype=bool)
        tp = int((pred_positive & actual_positive).sum())
        fp = int((pred_positive & ~actual_positive).sum())
        tn = int((~pred_positive & ~actual_positive).sum())
        fn = int((~pred_positive & actual_positive).sum())
        recall = safe_div(tp, tp + fn)
        specificity = safe_div(tn, tn + fp)
        precision = safe_div(tp, tp + fp)

        part = pred_df.copy()
        part["threshold"] = float(threshold)
        part["pred_should_irrigate"] = pred_positive.to_numpy(dtype=bool)
        part["trigger_correct"] = correct
        part["trigger_decision_regret_oracle_amount"] = regret
        part["chosen_ir_oracle_amount"] = np.where(
            pred_positive,
            pred_df["oracle_positive_ir"].to_numpy(dtype=float),
            0.0,
        )
        decisions.append(part)
        summaries.append(
            {
                "threshold": float(threshold),
                "trigger_accuracy": float(correct.mean()),
                "trigger_balanced_accuracy": float(np.nanmean([recall, specificity])),
                "trigger_precision": precision,
                "trigger_recall": recall,
                "trigger_specificity": specificity,
                "true_positive": tp,
                "false_positive": fp,
                "true_negative": tn,
                "false_negative": fn,
                "mean_decision_regret_oracle_amount": float(np.mean(regret)),
                "median_decision_regret_oracle_amount": float(np.median(regret)),
                "predicted_irrigation_rate": float(pred_positive.mean()),
                "true_irrigation_rate": float(actual_positive.mean()),
            }
        )
    return pd.concat(decisions, ignore_index=True), pd.DataFrame(summaries)


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
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--thresholds", default="0.05,0.1,0.15,0.2,0.25,0.3,0.35,0.4,0.45,0.5,0.55,0.6,0.65,0.7,0.75,0.8,0.85,0.9,0.95")
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

    groups = sorted(trigger_df[args.cv_group_col].astype(str).unique())
    pred_parts = []
    print(
        f"[binary-trigger] loaded site_dates={len(trigger_df)} groups={len(groups)} "
        f"seq_shape={seq_all.shape} tab_cols={len(tab_cols)} positive_rate={y_all.mean():.6f}",
        flush=True,
    )
    for i, group in enumerate(groups):
        print(f"[binary-trigger] fold {i + 1}/{len(groups)} holdout {args.cv_group_col}={group}", flush=True)
        test_mask = trigger_df[args.cv_group_col].astype(str).to_numpy() == str(group)
        train_mask = ~test_mask
        seq_train, seq_test, _seq_stats = standardize_train_test(seq_all[train_mask], seq_all[test_mask])
        tab_train, tab_test, _tab_stats = standardize_train_test(tab_all[train_mask], tab_all[test_mask])
        probs = train_fold_binary(
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
        part["model"] = "lstm_binary_irrigation_trigger"
        part["pred_irrigate_prob"] = probs
        pred_parts.append(part)
        print(f"[binary-trigger] fold {i + 1}/{len(groups)} completed site_dates={int(test_mask.sum())}", flush=True)

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
    worst = best_decisions.sort_values("trigger_decision_regret_oracle_amount", ascending=False).head(25)

    pred_path = out_dir / "continuous_irrigation_binary_trigger_lstm_v1_predictions.csv"
    decisions_path = out_dir / "continuous_irrigation_binary_trigger_lstm_v1_threshold_decisions.csv"
    summary_path = out_dir / "continuous_irrigation_binary_trigger_lstm_v1_threshold_sweep.csv"
    by_site_path = out_dir / "continuous_irrigation_binary_trigger_lstm_v1_by_site.csv"
    feature_path = out_dir / "continuous_irrigation_binary_trigger_lstm_v1_features.json"
    report_path = out_dir / "continuous_irrigation_binary_trigger_lstm_v1.md"
    pred_df.to_csv(pred_path, index=False)
    decisions.to_csv(decisions_path, index=False)
    summary.to_csv(summary_path, index=False)
    by_site.to_csv(by_site_path, index=False)
    feature_path.write_text(
        json.dumps({"sequence_columns": seq_cols, "tabular_columns": tab_cols}, indent=2),
        encoding="utf-8",
    )

    lines = [
        "# Continuous Irrigation Binary Trigger LSTM V1",
        "",
        "## Scope",
        "",
        "- Site-date binary trigger: predicts whether the true best candidate uses positive irrigation.",
        "- Candidate irrigation amount is excluded from the trigger features.",
        "- Decision regret uses oracle best positive amount after a positive trigger, isolating trigger quality.",
        f"- Input table: `{data_path}`.",
        f"- CV group column: `{args.cv_group_col}`.",
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
        "## Worst Trigger Decisions At Best Threshold",
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

    print("Continuous irrigation binary trigger LSTM v1")
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
