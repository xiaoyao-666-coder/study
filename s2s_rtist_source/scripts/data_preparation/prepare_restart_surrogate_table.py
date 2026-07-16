"""Prepare first surrogate-model tables from restart_decision_dataset.csv.

This is a lightweight table-shaping step for the current single-site smoke
dataset. It does not pretend to be the final cross-site feature set; it simply
separates candidate-level inputs from SWAP-derived labels so baseline modeling
can start cleanly.
"""

from __future__ import annotations

import math
from pathlib import Path

import pandas as pd


INPUT_CSV = "restart_decision_dataset.csv"
SAMPLE_OUT = "surrogate_samples_restart.csv"
LABEL_OUT = "surrogate_decision_labels_restart.csv"
REPORT_OUT = "surrogate_table_report.txt"

START_DOY = 61
YEAR_LENGTH = 366


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["sample_id"] = [
        f"{row.date_t.replace('-', '')}_ir{int(row.ir):02d}" for row in out.itertuples(index=False)
    ]
    out["days_after_start"] = out["decision_doy"] - START_DOY
    out["horizon_days"] = out["horizon_end_doy"] - out["decision_doy"]
    out["ir_sq"] = out["ir"] ** 2
    out["is_zero_ir"] = (out["ir"] == 0).astype(int)
    out["decision_doy_sin"] = out["decision_doy"].map(lambda x: math.sin(2 * math.pi * x / YEAR_LENGTH))
    out["decision_doy_cos"] = out["decision_doy"].map(lambda x: math.cos(2 * math.pi * x / YEAR_LENGTH))
    out["best_ir_gap"] = out["ir"] - out["best_ir_for_date"]
    out["target_regret"] = out["best_target_for_date"] - out["target_value"]
    return out


def main() -> None:
    path = Path(INPUT_CSV)
    if not path.exists():
        raise FileNotFoundError(f"Missing {INPUT_CSV}; run this in the Maize_restart_dataset directory.")

    df = pd.read_csv(path)
    expected_cols = {
        "date_t",
        "decision_doy",
        "horizon_end_doy",
        "ir",
        "end_daynr",
        "dvs",
        "lai",
        "rootd",
        "cwdm_value",
        "cwso_value",
        "target_value",
        "best_ir_for_date",
        "best_target_for_date",
        "is_best_ir",
    }
    missing = expected_cols.difference(df.columns)
    if missing:
        raise ValueError(f"Missing expected columns: {sorted(missing)}")

    df = add_features(df)
    df["is_best_ir"] = df["is_best_ir"].astype(bool)

    feature_cols = [
        "sample_id",
        "date_t",
        "decision_doy",
        "days_after_start",
        "horizon_days",
        "decision_doy_sin",
        "decision_doy_cos",
        "ir",
        "ir_sq",
        "is_zero_ir",
    ]
    label_cols = [
        "end_daynr",
        "dvs",
        "lai",
        "rootd",
        "cwdm_value",
        "cwso_value",
        "target_value",
        "best_ir_for_date",
        "best_target_for_date",
        "is_best_ir",
        "best_ir_gap",
        "target_regret",
    ]
    sample_cols = feature_cols + label_cols
    df[sample_cols].to_csv(SAMPLE_OUT, index=False)

    decision_labels = (
        df[df["is_best_ir"]][
            [
                "date_t",
                "decision_doy",
                "horizon_end_doy",
                "best_ir_for_date",
                "best_target_for_date",
            ]
        ]
        .sort_values("decision_doy")
        .reset_index(drop=True)
    )
    decision_labels.to_csv(LABEL_OUT, index=False)

    by_date = df.groupby("date_t", sort=False).agg(
        n_candidates=("ir", "size"),
        n_best=("is_best_ir", "sum"),
        best_ir=("best_ir_for_date", "first"),
        best_target=("best_target_for_date", "first"),
        max_target=("target_value", "max"),
        min_target=("target_value", "min"),
    )
    report = [
        "Surrogate restart table report",
        "",
        f"input_rows: {len(df)}",
        f"decision_dates: {df['date_t'].nunique()}",
        f"candidate_options: {sorted(df['ir'].unique().tolist())}",
        "",
        "is_best_ir counts:",
        df["is_best_ir"].value_counts().to_string(),
        "",
        "per-date summary:",
        by_date.to_string(),
        "",
        f"wrote: {SAMPLE_OUT}",
        f"wrote: {LABEL_OUT}",
    ]
    Path(REPORT_OUT).write_text("\n".join(report), encoding="utf-8")

    print("\n".join(report))


if __name__ == "__main__":
    main()
