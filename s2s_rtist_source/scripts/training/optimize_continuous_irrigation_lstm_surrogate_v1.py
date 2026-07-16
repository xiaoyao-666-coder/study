#!/usr/bin/env python3
"""Optimize irrigation with the LOSO LSTM surrogate on a dense irrigation grid.

This is the first continuous-optimization layer after the LSTM surrogate
baseline. For each leave-one-site-out fold, it trains the LSTM on the other
sites, creates dense candidate irrigation rows for the held-out site, predicts
surrogate net gain, and selects the irrigation amount with the highest predicted
net gain.

Evaluation uses linear interpolation over the existing SWAP-sampled response
curve, because SWAP labels are only available at sampled irrigation amounts.
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
    from torch.utils.data import DataLoader, TensorDataset
except ImportError as exc:  # pragma: no cover
    raise SystemExit("PyTorch is required for LSTM surrogate optimization.") from exc


DEFAULT_INPUT = (
    Path("site_general_surrogate_eval")
    / "continuous_ir_12site_10k_surrogate_sequence_wide_features_v1"
    / "continuous_ir_12site_surrogate_sequence_wide_samples_v1.csv"
)
DEFAULT_OUT_DIR = (
    Path("site_general_surrogate_eval")
    / "continuous_ir_12site_10k_lstm_continuous_optimization_v1"
)


def safe_mean(series: pd.Series) -> float:
    return float(series.mean()) if len(series) else float("nan")


def sample_id(site: str, date_text: str, ir: float) -> str:
    date_token = str(date_text).replace("-", "").replace("/", "").replace(" ", "").lower()
    ir_token = f"{float(ir):.6g}".replace("-", "m").replace(".", "p")
    return f"{site}_{date_token}_dense_ir{ir_token}"


def candidate_sequence(ir: float, horizon_days: int) -> str:
    return json.dumps([float(ir)] + [0.0] * max(horizon_days - 1, 0), separators=(",", ":"))


def dense_values(min_ir: float, max_ir: float, step: float, existing: np.ndarray) -> np.ndarray:
    grid = np.arange(min_ir, max_ir + step / 2.0, step, dtype=float)
    values = np.unique(np.round(np.concatenate([grid, existing, np.array([max_ir])]), 6))
    return values[(values >= min_ir - 1e-9) & (values <= max_ir + 1e-9)]


def update_candidate_columns(row: pd.Series, ir: float, horizon_days: int) -> pd.Series:
    row = row.copy()
    site = str(row["site_id"])
    row["candidate_ir"] = float(ir)
    row["ir"] = float(ir)
    row["candidate_ir_sq"] = float(ir) ** 2
    row["is_zero_ir"] = 1 if abs(float(ir)) <= 1e-12 else 0
    site_ir_max = float(row["site_ir_max"])
    row["candidate_ir_fraction"] = float(ir) / site_ir_max if site_ir_max > 0 else 0.0
    row["candidate_ir_fraction_sq"] = row["candidate_ir_fraction"] ** 2
    row["candidate_ir_sequence"] = candidate_sequence(float(ir), horizon_days)
    row["sample_id"] = sample_id(site, str(row["date_t"]), float(ir))
    for day in range(1, horizon_days + 1):
        row[f"future_ir_day{day:02d}"] = float(ir) if day == 1 else 0.0
    return row


def build_dense_candidates(df: pd.DataFrame, site_mask: np.ndarray, step: float, horizon_days: int) -> pd.DataFrame:
    rows = []
    heldout = df.loc[site_mask].copy()
    for site_date_id, part in heldout.groupby("site_date_id", sort=False):
        base = part.sort_values("candidate_ir").iloc[0]
        existing_ir = pd.to_numeric(part["candidate_ir"], errors="coerce").to_numpy(dtype=float)
        min_ir = float(np.nanmin(existing_ir))
        max_ir = float(part["site_ir_max"].iloc[0])
        for ir in dense_values(min_ir, max_ir, step, existing_ir):
            rows.append(update_candidate_columns(base, float(ir), horizon_days))
    return pd.DataFrame(rows).reset_index(drop=True)


def fit_lstm_model(
    seq_train: np.ndarray,
    tab_train: np.ndarray,
    y_train: np.ndarray,
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
) -> tuple[LstmSurrogate, float, float]:
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
    return model, y_mean, y_std


def predict_lstm(
    model: LstmSurrogate,
    seq: np.ndarray,
    tab: np.ndarray,
    *,
    y_mean: float,
    y_std: float,
    batch_size: int,
    device: str,
) -> np.ndarray:
    model.eval()
    preds = []
    with torch.no_grad():
        for start in range(0, len(seq), batch_size):
            seq_batch = torch.tensor(seq[start : start + batch_size], dtype=torch.float32).to(device)
            tab_batch = torch.tensor(tab[start : start + batch_size], dtype=torch.float32).to(device)
            preds.append(model(seq_batch, tab_batch).cpu().numpy())
    return np.concatenate(preds) * y_std + y_mean


def interpolate_true_gain(curve: pd.DataFrame, ir: float) -> float:
    tmp = curve[["candidate_ir", TARGET]].copy()
    tmp["candidate_ir"] = pd.to_numeric(tmp["candidate_ir"], errors="coerce")
    tmp[TARGET] = pd.to_numeric(tmp[TARGET], errors="coerce")
    tmp = tmp.dropna().sort_values("candidate_ir")
    xs = tmp["candidate_ir"].to_numpy(dtype=float)
    ys = tmp[TARGET].to_numpy(dtype=float)
    return float(np.interp(float(ir), xs, ys))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default=str(DEFAULT_INPUT))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUT_DIR))
    parser.add_argument("--cv-group-col", default="site_id")
    parser.add_argument("--history-days", type=int, default=14)
    parser.add_argument("--horizon-days", type=int, default=7)
    parser.add_argument("--grid-step", type=float, default=1.0)
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
    tab_cols = [
        col
        for col in tab_all_df.columns
        if not col.startswith(("hist_lag", "future_day", "future_ir_day"))
        and not col.startswith(("site_", "candidate_ir_x_site_", "candidate_ir_sq_x_site_"))
        and not tab_all_df[col].isna().all()
    ]
    tab_all = tab_all_df[tab_cols].to_numpy(dtype=float)
    y_all = pd.to_numeric(df[TARGET], errors="coerce").to_numpy(dtype=float)
    groups = sorted(df[args.cv_group_col].astype(str).unique())

    dense_parts = []
    decisions = []
    print(
        f"[opt] loaded rows={len(df)} groups={len(groups)} grid_step={args.grid_step} tab_cols={len(tab_cols)}",
        flush=True,
    )
    for i, group in enumerate(groups):
        print(f"[opt] fold {i + 1}/{len(groups)} holdout {args.cv_group_col}={group}", flush=True)
        group_values = df[args.cv_group_col].astype(str).to_numpy()
        test_mask = group_values == str(group)
        train_mask = ~test_mask

        dense = build_dense_candidates(df, test_mask, args.grid_step, args.horizon_days)
        seq_dense, _ = make_sequence_array(dense, args.history_days, args.horizon_days)
        tab_dense_df = build_features(dense)
        tab_dense = tab_dense_df[tab_cols].to_numpy(dtype=float)

        seq_train, seq_dense_std, _seq_stats = standardize_train_test(seq_all[train_mask], seq_dense)
        tab_train, tab_dense_std, _tab_stats = standardize_train_test(tab_all[train_mask], tab_dense)
        model, y_mean, y_std = fit_lstm_model(
            seq_train,
            tab_train,
            y_all[train_mask],
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
        dense["pred_net_gain_7d"] = predict_lstm(
            model,
            seq_dense_std,
            tab_dense_std,
            y_mean=y_mean,
            y_std=y_std,
            batch_size=args.batch_size,
            device=args.device,
        )
        dense["cv_group_col"] = args.cv_group_col
        dense["cv_group_value"] = str(group)
        dense_parts.append(dense)

        original = df.loc[test_mask].copy()
        for site_date_id, part in dense.groupby("site_date_id", sort=False):
            pred_best = part.loc[part["pred_net_gain_7d"].idxmax()]
            curve = original.loc[original["site_date_id"] == site_date_id].copy()
            true_best = curve.loc[curve[TARGET].idxmax()]
            pred_ir = float(pred_best["candidate_ir"])
            interp_gain = interpolate_true_gain(curve, pred_ir)
            nearest_idx = (pd.to_numeric(curve["candidate_ir"], errors="coerce") - pred_ir).abs().idxmin()
            nearest = curve.loc[nearest_idx]
            decisions.append(
                {
                    "site_date_id": site_date_id,
                    "site_id": str(true_best["site_id"]),
                    "date_t": str(true_best["date_t"]),
                    "decision_doy": int(true_best["decision_doy"]),
                    "target_collapse": bool(true_best["target_collapse"]),
                    "same_date_duplicate_target_curve": bool(true_best["same_date_duplicate_target_curve"]),
                    "true_best_ir_sampled": float(true_best["candidate_ir"]),
                    "true_best_net_gain_sampled": float(true_best[TARGET]),
                    "surrogate_opt_ir": pred_ir,
                    "surrogate_opt_pred_net_gain": float(pred_best["pred_net_gain_7d"]),
                    "surrogate_opt_interp_true_net_gain": interp_gain,
                    "surrogate_opt_interp_regret": float(true_best[TARGET] - interp_gain),
                    "nearest_sampled_ir": float(nearest["candidate_ir"]),
                    "nearest_sampled_true_net_gain": float(nearest[TARGET]),
                    "nearest_sampled_regret": float(true_best[TARGET] - nearest[TARGET]),
                    "exact_sampled_match": abs(float(nearest["candidate_ir"]) - pred_ir) <= 1e-9,
                }
            )
        print(f"[opt] fold {i + 1}/{len(groups)} completed dense_rows={len(dense)}", flush=True)

    dense_df = pd.concat(dense_parts, ignore_index=True)
    decision_df = pd.DataFrame(decisions)
    metrics = {
        "cv_group_col": args.cv_group_col,
        "cv_folds": int(len(groups)),
        "grid_step": float(args.grid_step),
        "history_days": int(args.history_days),
        "horizon_days": int(args.horizon_days),
        "hidden_size": int(args.hidden_size),
        "tab_hidden": int(args.tab_hidden),
        "epochs": int(args.epochs),
        "decision_total": int(len(decision_df)),
        "mean_interp_regret": float(decision_df["surrogate_opt_interp_regret"].mean()),
        "median_interp_regret": float(decision_df["surrogate_opt_interp_regret"].median()),
        "mean_nearest_sampled_regret": float(decision_df["nearest_sampled_regret"].mean()),
        "median_nearest_sampled_regret": float(decision_df["nearest_sampled_regret"].median()),
        "collapse_zero_rate": safe_mean(
            decision_df.loc[decision_df["target_collapse"], "surrogate_opt_ir"].abs() <= 1e-9
        ),
        "noncollapse_exact_sampled_best_rate": safe_mean(
            decision_df.loc[~decision_df["target_collapse"], "surrogate_opt_ir"]
            == decision_df.loc[~decision_df["target_collapse"], "true_best_ir_sampled"]
        ),
    }
    metrics_df = pd.DataFrame([metrics])
    by_site = (
        decision_df.groupby("site_id")
        .agg(
            mean_interp_regret=("surrogate_opt_interp_regret", "mean"),
            median_interp_regret=("surrogate_opt_interp_regret", "median"),
            max_interp_regret=("surrogate_opt_interp_regret", "max"),
            mean_surrogate_opt_ir=("surrogate_opt_ir", "mean"),
            n_site_dates=("site_date_id", "count"),
        )
        .reset_index()
        .sort_values("mean_interp_regret", ascending=False)
    )
    worst = decision_df.sort_values("surrogate_opt_interp_regret", ascending=False).head(20)

    dense_path = out_dir / "continuous_ir_lstm_surrogate_dense_predictions_v1.csv"
    decision_path = out_dir / "continuous_ir_lstm_surrogate_continuous_decisions_v1.csv"
    metrics_path = out_dir / "continuous_ir_lstm_surrogate_continuous_optimization_metrics_v1.csv"
    by_site_path = out_dir / "continuous_ir_lstm_surrogate_continuous_optimization_by_site_v1.csv"
    report_path = out_dir / "continuous_ir_lstm_surrogate_continuous_optimization_v1.md"
    dense_df.to_csv(dense_path, index=False)
    decision_df.to_csv(decision_path, index=False)
    metrics_df.to_csv(metrics_path, index=False)
    by_site.to_csv(by_site_path, index=False)

    lines = [
        "# Continuous Irrigation LSTM Surrogate Optimization V1",
        "",
        "## Scope",
        "",
        "- Dense-grid continuous optimization using LOSO LSTM surrogate folds.",
        "- Evaluation uses interpolation over available SWAP-sampled curves.",
        "",
        "## Metrics",
        "",
        markdown_table(metrics_df),
        "",
        "## By Site",
        "",
        markdown_table(by_site),
        "",
        "## Worst Decisions",
        "",
        markdown_table(worst),
        "",
        "## Outputs",
        "",
        f"- `{dense_path}`",
        f"- `{decision_path}`",
        f"- `{metrics_path}`",
        f"- `{by_site_path}`",
    ]
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print("Continuous irrigation LSTM surrogate optimization v1")
    print(f"dense_predictions: {dense_path}")
    print(f"decisions: {decision_path}")
    print(f"metrics: {metrics_path}")
    print(f"by_site: {by_site_path}")
    print(f"report: {report_path}")
    print("")
    print(metrics_df.to_string(index=False))
    print("")
    print(by_site.to_string(index=False))
    print("")
    print(worst.to_string(index=False))


if __name__ == "__main__":
    main()
