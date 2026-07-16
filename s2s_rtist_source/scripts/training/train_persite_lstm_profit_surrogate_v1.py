#!/usr/bin/env python3
"""Train per-site LSTM profit surrogates and test continuous irrigation search.

This is the teacher-aligned first-stage experiment before MoE/TTA:

1. For each site, train a site-specific surrogate on SWAP-labeled continuous
   irrigation curves.
2. Check whether the surrogate matches SWAP profit labels on the same
   paper-style fixed-list irrigation inputs.
3. Use the surrogate to search continuous irrigation amounts, then evaluate the
   chosen amount by interpolation over the SWAP-sampled response curve.

The script intentionally does not test cross-site generalization. It establishes
whether a per-site SWAP profit function can be learned well enough before using
these site experts inside a later MoE model.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

from compare_discrete_vs_continuous_ir_optimization_v1 import (
    DEFAULT_PAPER_CANDIDATES,
    TARGET,
    candidate_set_for_site,
    interp_gain,
    parse_candidates,
)
from train_confirmed_5site_true_input_surrogate_baseline_v1 import (
    bool_series,
    build_features,
    markdown_table,
)
from train_continuous_irrigation_surrogate_lstm_v1 import (
    LstmSurrogate,
    make_sequence_array,
    standardize_train_test,
)
from train_continuous_irrigation_surrogate_tree_nosklearn_v1 import score_metrics

try:
    import torch
    from torch import nn
    from torch.utils.data import DataLoader, TensorDataset
except ImportError as exc:  # pragma: no cover
    raise SystemExit("PyTorch is required for per-site LSTM surrogate training.") from exc


DEFAULT_INPUT = (
    Path("site_general_surrogate_eval")
    / "continuous_ir_12site_10k_surrogate_sequence_wide_features_v1"
    / "continuous_ir_12site_surrogate_sequence_wide_samples_v1.csv"
)
DEFAULT_OUT = (
    Path("site_general_surrogate_eval")
    / "continuous_ir_12site_10k_persite_lstm_profit_surrogate_v1"
)


def safe_mean(values: pd.Series | np.ndarray) -> float:
    return float(np.mean(values)) if len(values) else float("nan")


def safe_median(values: pd.Series | np.ndarray) -> float:
    return float(np.median(values)) if len(values) else float("nan")


def sample_id(site: str, date_text: str, ir: float, prefix: str) -> str:
    date_token = str(date_text).replace("-", "").replace("/", "").replace(" ", "").lower()
    ir_token = f"{float(ir):.6g}".replace("-", "m").replace(".", "p")
    return f"{prefix}_{site}_{date_token}_ir{ir_token}"


def candidate_sequence(ir: float, horizon_days: int) -> str:
    return json.dumps([float(ir)] + [0.0] * max(horizon_days - 1, 0), separators=(",", ":"))


def update_candidate_columns(row: pd.Series, ir: float, horizon_days: int, prefix: str) -> pd.Series:
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
    row["sample_id"] = sample_id(site, str(row["date_t"]), float(ir), prefix)
    for day in range(1, horizon_days + 1):
        row[f"future_ir_day{day:02d}"] = float(ir) if day == 1 else 0.0
    return row


def dense_values(max_ir: float, step: float, extra_values: list[float]) -> np.ndarray:
    grid = np.arange(0.0, float(max_ir) + step * 0.5, step, dtype=float)
    values = np.concatenate([grid, np.array(extra_values + [float(max_ir)], dtype=float)])
    values = np.unique(np.round(values, 6))
    return values[(values >= -1e-9) & (values <= float(max_ir) + 1e-9)]


def build_candidate_rows(
    curve: pd.DataFrame,
    irrigation_values: list[float] | np.ndarray,
    *,
    horizon_days: int,
    prefix: str,
) -> pd.DataFrame:
    curve = curve.sort_values("candidate_ir")
    base = curve.iloc[0]
    rows = [update_candidate_columns(base, float(ir), horizon_days, prefix) for ir in irrigation_values]
    return pd.DataFrame(rows).reset_index(drop=True)


def usable_tabular_columns(x: pd.DataFrame) -> list[str]:
    seq_prefixes = ("hist_lag", "future_day", "future_ir_day")
    site_identity_prefixes = ("site_", "candidate_ir_x_site_", "candidate_ir_sq_x_site_")
    return [
        col
        for col in x.columns
        if not col.startswith(seq_prefixes)
        and not col.startswith(site_identity_prefixes)
        and not x[col].isna().all()
    ]


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


def make_group_folds(groups: list[str], folds: int, random_state: int) -> list[list[str]]:
    if folds <= 1:
        raise ValueError("--folds-per-site must be at least 2")
    if len(groups) < 2:
        raise ValueError("Each site needs at least two site-date groups for validation")
    folds = min(int(folds), len(groups))
    rng = np.random.default_rng(random_state)
    shuffled = np.array(groups, dtype=object)
    rng.shuffle(shuffled)
    parts = [list(part.astype(str)) for part in np.array_split(shuffled, folds)]
    return [part for part in parts if part]


def sanitize_name(value: str) -> str:
    keep = []
    for char in str(value):
        keep.append(char if char.isalnum() or char in {"-", "_"} else "_")
    return "".join(keep)


def add_interp_truth(rows: pd.DataFrame, curve: pd.DataFrame) -> pd.DataFrame:
    rows = rows.copy()
    rows["interp_true_net_gain_7d"] = [
        interp_gain(curve, float(ir)) for ir in rows["candidate_ir"].to_numpy(dtype=float)
    ]
    return rows


def prediction_inputs(
    rows: pd.DataFrame,
    tab_cols: list[str],
    seq_train_raw: np.ndarray,
    tab_train_raw: np.ndarray,
    *,
    history_days: int,
    horizon_days: int,
) -> tuple[np.ndarray, np.ndarray]:
    seq_raw, _ = make_sequence_array(rows, history_days, horizon_days)
    tab_raw_df = build_features(rows)
    tab_raw = tab_raw_df[tab_cols].to_numpy(dtype=float)
    _seq_train, seq_std, _seq_stats = standardize_train_test(seq_train_raw, seq_raw)
    _tab_train, tab_std, _tab_stats = standardize_train_test(tab_train_raw, tab_raw)
    return seq_std, tab_std


def summarize_decisions(decisions: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    if decisions.empty:
        return pd.DataFrame(), pd.DataFrame()
    overall = pd.DataFrame(
        [
            {
                "sites": int(decisions["site_id"].nunique()),
                "site_dates": int(decisions["site_date_id"].nunique()),
                "paper_fixed_list_mean_regret_vs_dense": float(decisions["paper_regret_vs_dense_oracle"].mean()),
                "fixed_list_surrogate_mean_regret_vs_fixed_oracle": float(
                    decisions["fixed_list_surrogate_regret_vs_fixed_oracle"].mean()
                ),
                "continuous_surrogate_mean_regret_vs_dense": float(
                    decisions["continuous_surrogate_regret_vs_dense_oracle"].mean()
                ),
                "continuous_surrogate_mean_gain_over_paper": float(
                    decisions["continuous_surrogate_gain_over_paper"].mean()
                ),
                "continuous_surrogate_better_than_paper_rate": safe_mean(
                    decisions["continuous_surrogate_gain_over_paper"] > 1e-9
                ),
                "continuous_surrogate_worse_than_paper_rate": safe_mean(
                    decisions["continuous_surrogate_gain_over_paper"] < -1e-9
                ),
                "continuous_surrogate_nonfixed_ir_rate": safe_mean(
                    decisions["continuous_surrogate_nonfixed_ir"]
                ),
                "continuous_surrogate_mean_distance_to_nearest_fixed": float(
                    decisions["continuous_surrogate_distance_to_nearest_fixed_ir"].mean()
                ),
            }
        ]
    )
    by_site = (
        decisions.groupby("site_id")
        .agg(
            site_dates=("site_date_id", "nunique"),
            paper_fixed_list_mean_regret_vs_dense=("paper_regret_vs_dense_oracle", "mean"),
            fixed_list_surrogate_mean_regret_vs_fixed_oracle=(
                "fixed_list_surrogate_regret_vs_fixed_oracle",
                "mean",
            ),
            continuous_surrogate_mean_regret_vs_dense=(
                "continuous_surrogate_regret_vs_dense_oracle",
                "mean",
            ),
            continuous_surrogate_mean_gain_over_paper=("continuous_surrogate_gain_over_paper", "mean"),
            continuous_surrogate_better_than_paper_rate=(
                "continuous_surrogate_better_than_paper",
                "mean",
            ),
            continuous_surrogate_nonfixed_ir_rate=("continuous_surrogate_nonfixed_ir", "mean"),
            continuous_surrogate_mean_distance_to_nearest_fixed=(
                "continuous_surrogate_distance_to_nearest_fixed_ir",
                "mean",
            ),
        )
        .reset_index()
        .sort_values("continuous_surrogate_mean_regret_vs_dense", ascending=False)
    )
    return overall, by_site


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default=str(DEFAULT_INPUT))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUT))
    parser.add_argument("--paper-candidates", default=DEFAULT_PAPER_CANDIDATES)
    parser.add_argument("--history-days", type=int, default=14)
    parser.add_argument("--horizon-days", type=int, default=7)
    parser.add_argument("--folds-per-site", type=int, default=5)
    parser.add_argument("--grid-step", type=float, default=0.5)
    parser.add_argument("--hidden-size", type=int, default=64)
    parser.add_argument("--tab-hidden", type=int, default=96)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    parser.add_argument(
        "--skip-final-experts",
        action="store_true",
        help="Skip fitting compact final per-site expert checkpoints after validation.",
    )
    parser.add_argument(
        "--site-limit",
        type=int,
        default=0,
        help="Optional smoke limit on the number of sites. Use 0 for all sites.",
    )
    args = parser.parse_args()

    data_path = Path(args.input)
    if not data_path.exists():
        raise FileNotFoundError(f"Missing sequence-wide sample table: {data_path}")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but torch.cuda.is_available() is false")
    if args.grid_step <= 0:
        raise ValueError("--grid-step must be positive")

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    paper_candidates = parse_candidates(args.paper_candidates)

    df = pd.read_csv(data_path)
    required = {
        "site_id",
        "site_date_id",
        "date_t",
        "decision_doy",
        "candidate_ir",
        "site_ir_max",
        TARGET,
    }
    missing = sorted(required.difference(df.columns))
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    for col in ["is_best_ir", "target_collapse", "same_date_duplicate_target_curve"]:
        if col in df.columns:
            df[col] = bool_series(df[col])
    if pd.to_numeric(df[TARGET], errors="coerce").isna().any():
        raise ValueError(f"Target column {TARGET} contains NaN")

    sites = sorted(df["site_id"].astype(str).unique())
    if args.site_limit and args.site_limit > 0:
        sites = sites[: args.site_limit]

    prediction_parts = []
    fixed_parts = []
    dense_parts = []
    decision_rows = []
    fold_rows = []
    expert_rows = []
    expert_dir = out_dir / "final_site_experts_v1"
    if not args.skip_final_experts:
        expert_dir.mkdir(parents=True, exist_ok=True)
    print(
        f"[persite-lstm] rows={len(df)} sites={len(sites)} folds_per_site={args.folds_per_site} "
        f"grid_step={args.grid_step}",
        flush=True,
    )

    for site_idx, site_id in enumerate(sites):
        site_df = df.loc[df["site_id"].astype(str) == site_id].copy().reset_index(drop=True)
        groups = sorted(site_df["site_date_id"].astype(str).unique())
        folds = make_group_folds(groups, args.folds_per_site, args.random_state + site_idx)
        print(
            f"[persite-lstm] site {site_idx + 1}/{len(sites)} {site_id}: "
            f"rows={len(site_df)} site_dates={len(groups)} folds={len(folds)}",
            flush=True,
        )

        seq_site_raw, seq_cols = make_sequence_array(site_df, args.history_days, args.horizon_days)
        tab_site_df = build_features(site_df)
        tab_cols = usable_tabular_columns(tab_site_df)
        tab_site_raw = tab_site_df[tab_cols].to_numpy(dtype=float)
        y_site = pd.to_numeric(site_df[TARGET], errors="coerce").to_numpy(dtype=float)
        site_group_values = site_df["site_date_id"].astype(str).to_numpy()

        for fold_idx, holdout_groups in enumerate(folds):
            print(
                f"[persite-lstm] site={site_id} fold {fold_idx + 1}/{len(folds)} "
                f"holdout_dates={len(holdout_groups)}",
                flush=True,
            )
            test_mask = np.isin(site_group_values, np.array(holdout_groups, dtype=str))
            train_mask = ~test_mask
            if int(train_mask.sum()) == 0 or int(test_mask.sum()) == 0:
                raise RuntimeError(f"Empty train/test split for site={site_id} fold={fold_idx}")

            seq_train_std, seq_test_std, _seq_stats = standardize_train_test(
                seq_site_raw[train_mask], seq_site_raw[test_mask]
            )
            tab_train_std, tab_test_std, _tab_stats = standardize_train_test(
                tab_site_raw[train_mask], tab_site_raw[test_mask]
            )
            model, y_mean, y_std = fit_lstm_model(
                seq_train_std,
                tab_train_std,
                y_site[train_mask],
                hidden_size=args.hidden_size,
                tab_hidden=args.tab_hidden,
                dropout=args.dropout,
                epochs=args.epochs,
                batch_size=args.batch_size,
                lr=args.lr,
                weight_decay=args.weight_decay,
                random_state=args.random_state + site_idx * 100 + fold_idx,
                device=args.device,
            )

            sampled_pred = predict_lstm(
                model,
                seq_test_std,
                tab_test_std,
                y_mean=y_mean,
                y_std=y_std,
                batch_size=args.batch_size,
                device=args.device,
            )
            sampled_keep = site_df.loc[test_mask].copy()
            sampled_keep["model"] = "persite_lstm_profit_surrogate"
            sampled_keep["site_fold"] = fold_idx
            sampled_keep["pred_net_gain_7d"] = sampled_pred
            prediction_parts.append(
                sampled_keep[
                    [
                        "sample_id",
                        "site_date_id",
                        "site_id",
                        "date_t",
                        "decision_doy",
                        "candidate_ir",
                        TARGET,
                        "pred_net_gain_7d",
                        "model",
                        "site_fold",
                    ]
                ].copy()
            )

            seq_train_raw = seq_site_raw[train_mask]
            tab_train_raw = tab_site_raw[train_mask]
            fold_curve_metrics = score_metrics(
                sampled_keep[TARGET].to_numpy(dtype=float),
                sampled_keep["pred_net_gain_7d"].to_numpy(dtype=float),
            )
            fold_curve_metrics.update(
                {
                    "site_id": site_id,
                    "site_fold": int(fold_idx),
                    "train_rows": int(train_mask.sum()),
                    "test_rows": int(test_mask.sum()),
                    "test_site_dates": int(len(holdout_groups)),
                }
            )
            fold_rows.append(fold_curve_metrics)

            holdout_df = site_df.loc[test_mask].copy()
            for site_date_id, curve in holdout_df.groupby("site_date_id", sort=False):
                curve = curve.copy()
                curve["candidate_ir"] = pd.to_numeric(curve["candidate_ir"], errors="coerce")
                curve[TARGET] = pd.to_numeric(curve[TARGET], errors="coerce")
                curve = curve.dropna(subset=["candidate_ir", TARGET]).sort_values("candidate_ir")
                if curve.empty:
                    continue

                dense_oracle = curve.loc[curve[TARGET].idxmax()]
                dense_oracle_gain = float(dense_oracle[TARGET])
                dense_oracle_ir = float(dense_oracle["candidate_ir"])
                site_ir_max = float(curve["site_ir_max"].iloc[0])
                fixed_values = candidate_set_for_site(site_ir_max, paper_candidates)

                fixed_rows = build_candidate_rows(
                    curve,
                    fixed_values,
                    horizon_days=args.horizon_days,
                    prefix="fixedlist",
                )
                fixed_rows = add_interp_truth(fixed_rows, curve)
                fixed_seq, fixed_tab = prediction_inputs(
                    fixed_rows,
                    tab_cols,
                    seq_train_raw,
                    tab_train_raw,
                    history_days=args.history_days,
                    horizon_days=args.horizon_days,
                )
                fixed_rows["pred_net_gain_7d"] = predict_lstm(
                    model,
                    fixed_seq,
                    fixed_tab,
                    y_mean=y_mean,
                    y_std=y_std,
                    batch_size=args.batch_size,
                    device=args.device,
                )
                fixed_rows["site_fold"] = fold_idx
                fixed_parts.append(
                    fixed_rows[
                        [
                            "sample_id",
                            "site_date_id",
                            "site_id",
                            "date_t",
                            "candidate_ir",
                            "interp_true_net_gain_7d",
                            "pred_net_gain_7d",
                            "site_fold",
                        ]
                    ].copy()
                )
                fixed_oracle = fixed_rows.loc[fixed_rows["interp_true_net_gain_7d"].idxmax()]
                fixed_pred_best = fixed_rows.loc[fixed_rows["pred_net_gain_7d"].idxmax()]

                dense_grid = dense_values(site_ir_max, args.grid_step, fixed_values)
                dense_rows = build_candidate_rows(
                    curve,
                    dense_grid,
                    horizon_days=args.horizon_days,
                    prefix="denseopt",
                )
                dense_rows = add_interp_truth(dense_rows, curve)
                dense_seq, dense_tab = prediction_inputs(
                    dense_rows,
                    tab_cols,
                    seq_train_raw,
                    tab_train_raw,
                    history_days=args.history_days,
                    horizon_days=args.horizon_days,
                )
                dense_rows["pred_net_gain_7d"] = predict_lstm(
                    model,
                    dense_seq,
                    dense_tab,
                    y_mean=y_mean,
                    y_std=y_std,
                    batch_size=args.batch_size,
                    device=args.device,
                )
                dense_rows["site_fold"] = fold_idx
                dense_parts.append(
                    dense_rows[
                        [
                            "sample_id",
                            "site_date_id",
                            "site_id",
                            "date_t",
                            "candidate_ir",
                            "interp_true_net_gain_7d",
                            "pred_net_gain_7d",
                            "site_fold",
                        ]
                    ].copy()
                )
                continuous_pred_best = dense_rows.loc[dense_rows["pred_net_gain_7d"].idxmax()]
                no_ir_gain = interp_gain(curve, 0.0)
                nearest_fixed = min(fixed_values, key=lambda value: abs(float(value) - float(continuous_pred_best["candidate_ir"])))
                continuous_gain = float(continuous_pred_best["interp_true_net_gain_7d"])
                fixed_oracle_gain = float(fixed_oracle["interp_true_net_gain_7d"])
                paper_regret = dense_oracle_gain - fixed_oracle_gain
                continuous_regret = dense_oracle_gain - continuous_gain
                continuous_gain_over_paper = continuous_gain - fixed_oracle_gain
                decision_rows.append(
                    {
                        "site_date_id": str(site_date_id),
                        "site_id": site_id,
                        "date_t": str(dense_oracle["date_t"]),
                        "site_fold": int(fold_idx),
                        "site_ir_max": site_ir_max,
                        "dense_oracle_ir": dense_oracle_ir,
                        "dense_oracle_gain": dense_oracle_gain,
                        "paper_fixed_list_oracle_ir": float(fixed_oracle["candidate_ir"]),
                        "paper_fixed_list_oracle_gain": fixed_oracle_gain,
                        "paper_regret_vs_dense_oracle": paper_regret,
                        "fixed_list_surrogate_ir": float(fixed_pred_best["candidate_ir"]),
                        "fixed_list_surrogate_true_gain": float(fixed_pred_best["interp_true_net_gain_7d"]),
                        "fixed_list_surrogate_pred_gain": float(fixed_pred_best["pred_net_gain_7d"]),
                        "fixed_list_surrogate_regret_vs_fixed_oracle": fixed_oracle_gain
                        - float(fixed_pred_best["interp_true_net_gain_7d"]),
                        "continuous_surrogate_ir": float(continuous_pred_best["candidate_ir"]),
                        "continuous_surrogate_true_gain": continuous_gain,
                        "continuous_surrogate_pred_gain": float(continuous_pred_best["pred_net_gain_7d"]),
                        "continuous_surrogate_regret_vs_dense_oracle": continuous_regret,
                        "continuous_surrogate_gain_over_paper": continuous_gain_over_paper,
                        "continuous_surrogate_better_than_paper": continuous_gain_over_paper > 1e-9,
                        "continuous_surrogate_worse_than_paper": continuous_gain_over_paper < -1e-9,
                        "continuous_surrogate_nearest_fixed_ir": float(nearest_fixed),
                        "continuous_surrogate_distance_to_nearest_fixed_ir": abs(
                            float(continuous_pred_best["candidate_ir"]) - float(nearest_fixed)
                        ),
                        "continuous_surrogate_nonfixed_ir": abs(
                            float(continuous_pred_best["candidate_ir"]) - float(nearest_fixed)
                        )
                        > 1e-9,
                        "continuous_surrogate_gain_minus_no_ir": continuous_gain - float(no_ir_gain),
                    }
                )

        if not args.skip_final_experts:
            print(f"[persite-lstm] fitting final expert for site={site_id}", flush=True)
            seq_final_std, _seq_unused, seq_stats = standardize_train_test(seq_site_raw, seq_site_raw)
            tab_final_std, _tab_unused, tab_stats = standardize_train_test(tab_site_raw, tab_site_raw)
            final_model, final_y_mean, final_y_std = fit_lstm_model(
                seq_final_std,
                tab_final_std,
                y_site,
                hidden_size=args.hidden_size,
                tab_hidden=args.tab_hidden,
                dropout=args.dropout,
                epochs=args.epochs,
                batch_size=args.batch_size,
                lr=args.lr,
                weight_decay=args.weight_decay,
                random_state=args.random_state + site_idx * 1000 + 999,
                device=args.device,
            )
            expert_path = expert_dir / f"persite_lstm_expert_{sanitize_name(site_id)}_v1.pt"
            torch.save(
                {
                    "model": "persite_lstm_profit_surrogate_v1",
                    "site_id": site_id,
                    "state_dict": final_model.cpu().state_dict(),
                    "seq_dim": int(seq_site_raw.shape[2]),
                    "tab_dim": int(tab_site_raw.shape[1]),
                    "hidden_size": int(args.hidden_size),
                    "tab_hidden": int(args.tab_hidden),
                    "dropout": float(args.dropout),
                    "history_days": int(args.history_days),
                    "horizon_days": int(args.horizon_days),
                    "sequence_columns": seq_cols,
                    "tabular_columns": tab_cols,
                    "sequence_stats": seq_stats,
                    "tabular_stats": tab_stats,
                    "target_mean": float(final_y_mean),
                    "target_std": float(final_y_std),
                    "target_column": TARGET,
                    "training_rows": int(len(site_df)),
                    "training_site_dates": int(site_df["site_date_id"].nunique()),
                    "paper_candidates": paper_candidates,
                    "grid_step": float(args.grid_step),
                },
                expert_path,
            )
            expert_rows.append(
                {
                    "site_id": site_id,
                    "expert_checkpoint": str(expert_path),
                    "training_rows": int(len(site_df)),
                    "training_site_dates": int(site_df["site_date_id"].nunique()),
                    "hidden_size": int(args.hidden_size),
                    "tab_hidden": int(args.tab_hidden),
                    "epochs": int(args.epochs),
                }
            )

    sampled_predictions = pd.concat(prediction_parts, ignore_index=True)
    fixed_predictions = pd.concat(fixed_parts, ignore_index=True)
    dense_predictions = pd.concat(dense_parts, ignore_index=True)
    decisions = pd.DataFrame(decision_rows)
    fold_metrics = pd.DataFrame(fold_rows)

    sampled_metrics = pd.DataFrame(
        [
            {
                "prediction_set": "heldout_sampled_curve_rows",
                **score_metrics(
                    sampled_predictions[TARGET].to_numpy(dtype=float),
                    sampled_predictions["pred_net_gain_7d"].to_numpy(dtype=float),
                ),
                "rows": int(len(sampled_predictions)),
            }
        ]
    )
    fixed_metrics = pd.DataFrame(
        [
            {
                "prediction_set": "paper_fixed_list_rows",
                **score_metrics(
                    fixed_predictions["interp_true_net_gain_7d"].to_numpy(dtype=float),
                    fixed_predictions["pred_net_gain_7d"].to_numpy(dtype=float),
                ),
                "rows": int(len(fixed_predictions)),
            }
        ]
    )
    summary, by_site = summarize_decisions(decisions)
    worst_continuous = decisions.sort_values(
        "continuous_surrogate_regret_vs_dense_oracle", ascending=False
    ).head(30)
    best_gains = decisions.sort_values("continuous_surrogate_gain_over_paper", ascending=False).head(30)

    sampled_path = out_dir / "persite_lstm_profit_surrogate_sampled_predictions_v1.csv"
    fixed_path = out_dir / "persite_lstm_profit_surrogate_fixed_list_predictions_v1.csv"
    dense_path = out_dir / "persite_lstm_profit_surrogate_dense_predictions_v1.csv"
    decisions_path = out_dir / "persite_lstm_profit_surrogate_decisions_v1.csv"
    fold_metrics_path = out_dir / "persite_lstm_profit_surrogate_fold_metrics_v1.csv"
    prediction_metrics_path = out_dir / "persite_lstm_profit_surrogate_prediction_metrics_v1.csv"
    summary_path = out_dir / "persite_lstm_profit_surrogate_summary_v1.csv"
    by_site_path = out_dir / "persite_lstm_profit_surrogate_by_site_v1.csv"
    feature_path = out_dir / "persite_lstm_profit_surrogate_features_v1.json"
    expert_manifest_path = out_dir / "persite_lstm_profit_surrogate_expert_manifest_v1.csv"
    report_path = out_dir / "persite_lstm_profit_surrogate_v1.md"

    sampled_predictions.to_csv(sampled_path, index=False)
    fixed_predictions.to_csv(fixed_path, index=False)
    dense_predictions.to_csv(dense_path, index=False)
    decisions.to_csv(decisions_path, index=False)
    fold_metrics.to_csv(fold_metrics_path, index=False)
    pd.concat([sampled_metrics, fixed_metrics], ignore_index=True).to_csv(
        prediction_metrics_path, index=False
    )
    summary.to_csv(summary_path, index=False)
    by_site.to_csv(by_site_path, index=False)
    pd.DataFrame(expert_rows).to_csv(expert_manifest_path, index=False)
    feature_path.write_text(
        json.dumps(
            {
                "sequence_columns": seq_cols if "seq_cols" in locals() else [],
                "tabular_columns": tab_cols if "tab_cols" in locals() else [],
                "paper_candidates": paper_candidates,
                "history_days": args.history_days,
                "horizon_days": args.horizon_days,
                "grid_step": args.grid_step,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    lines = [
        "# Per-Site LSTM Profit Surrogate V1",
        "",
        "## Purpose",
        "",
        "Teacher-aligned first-stage validation before MoE/TTA:",
        "",
        "- Train one profit surrogate per site.",
        "- Check whether predictions match SWAP profits on the same paper fixed-list inputs.",
        "- Optimize a continuous irrigation amount with the surrogate and evaluate it by SWAP-curve interpolation.",
        "",
        "## Prediction Metrics",
        "",
        markdown_table(pd.concat([sampled_metrics, fixed_metrics], ignore_index=True)),
        "",
        "## Decision Summary",
        "",
        markdown_table(summary),
        "",
        "## By Site",
        "",
        markdown_table(by_site),
        "",
        "## Largest Continuous-Optimization Regrets",
        "",
        markdown_table(worst_continuous),
        "",
        "## Largest Continuous Gains Over Paper Fixed List",
        "",
        markdown_table(best_gains),
        "",
        "## Outputs",
        "",
        f"- `{sampled_path}`",
        f"- `{fixed_path}`",
        f"- `{dense_path}`",
        f"- `{decisions_path}`",
        f"- `{fold_metrics_path}`",
        f"- `{prediction_metrics_path}`",
        f"- `{summary_path}`",
        f"- `{by_site_path}`",
        f"- `{expert_manifest_path}`",
        f"- `{feature_path}`",
    ]
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print("Per-site LSTM profit surrogate v1")
    print(f"sampled_predictions: {sampled_path}")
    print(f"fixed_list_predictions: {fixed_path}")
    print(f"dense_predictions: {dense_path}")
    print(f"decisions: {decisions_path}")
    print(f"summary: {summary_path}")
    print(f"by_site: {by_site_path}")
    print(f"expert_manifest: {expert_manifest_path}")
    print(f"report: {report_path}")
    print("")
    print("Prediction metrics")
    print(pd.concat([sampled_metrics, fixed_metrics], ignore_index=True).to_string(index=False))
    print("")
    print("Decision summary")
    print(summary.to_string(index=False))
    print("")
    print("By site")
    print(by_site.to_string(index=False))


if __name__ == "__main__":
    main()
