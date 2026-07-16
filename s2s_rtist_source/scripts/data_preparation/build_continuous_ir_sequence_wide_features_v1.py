#!/usr/bin/env python3
"""Expand weather/irrigation JSON windows into sequence-wide feature columns."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from train_confirmed_5site_true_input_surrogate_baseline_v1 import markdown_table


DEFAULT_INPUT = (
    Path("site_general_surrogate_eval")
    / "continuous_ir_12site_10k_surrogate_features_v1"
    / "continuous_ir_12site_surrogate_features_samples_v1.csv"
)
DEFAULT_OUT_DIR = (
    Path("site_general_surrogate_eval")
    / "continuous_ir_12site_10k_surrogate_sequence_wide_features_v1"
)

WEATHER_FIELDS = [
    ("solar", "Solar"),
    ("tmax", "T-max"),
    ("tmin", "T-min"),
    ("relhum", "RelHum"),
    ("precip", "Precip"),
    ("windspeed", "WindSpeed"),
]


def parse_json_list(value: object) -> list:
    if pd.isna(value):
        return []
    if isinstance(value, list):
        return value
    text = str(value).strip()
    if not text:
        return []
    try:
        obj = json.loads(text)
    except json.JSONDecodeError:
        return []
    return obj if isinstance(obj, list) else []


def expand_history(value: object, history_days: int) -> dict[str, float]:
    seq = parse_json_list(value)
    # Align to the end of the window: hist_lag01 is yesterday.
    seq = seq[-history_days:]
    offset = history_days - len(seq)
    out: dict[str, float] = {}
    for pos in range(history_days):
        lag = history_days - pos
        row = seq[pos - offset] if pos >= offset else {}
        for safe, raw in WEATHER_FIELDS:
            out[f"hist_lag{lag:02d}_{safe}"] = float(row.get(raw, np.nan)) if isinstance(row, dict) else np.nan
    return out


def expand_future(value: object, horizon_days: int) -> dict[str, float]:
    seq = parse_json_list(value)
    out: dict[str, float] = {}
    for i in range(horizon_days):
        row = seq[i] if i < len(seq) else {}
        day = i + 1
        for safe, raw in WEATHER_FIELDS:
            out[f"future_day{day:02d}_{safe}"] = float(row.get(raw, np.nan)) if isinstance(row, dict) else np.nan
    return out


def expand_ir_sequence(value: object, horizon_days: int) -> dict[str, float]:
    seq = parse_json_list(value)
    out = {}
    for i in range(horizon_days):
        item = seq[i] if i < len(seq) else np.nan
        try:
            out[f"future_ir_day{i + 1:02d}"] = float(item)
        except (TypeError, ValueError):
            out[f"future_ir_day{i + 1:02d}"] = np.nan
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default=str(DEFAULT_INPUT))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUT_DIR))
    parser.add_argument("--history-days", type=int, default=14)
    parser.add_argument("--horizon-days", type=int, default=7)
    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        raise FileNotFoundError(f"Missing sample table: {input_path}")
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(input_path)
    required = ["hist_weather_json", "future_weather_json", "candidate_ir_sequence"]
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"Input table is missing sequence JSON columns: {missing}")

    seq_rows = []
    for row in df.itertuples(index=False):
        item = {}
        item.update(expand_history(getattr(row, "hist_weather_json"), args.history_days))
        item.update(expand_future(getattr(row, "future_weather_json"), args.horizon_days))
        item.update(expand_ir_sequence(getattr(row, "candidate_ir_sequence"), args.horizon_days))
        seq_rows.append(item)
    seq_df = pd.DataFrame(seq_rows)

    out = pd.concat([df.reset_index(drop=True), seq_df], axis=1)
    out["sequence_wide_feature_status"] = "built_from_weather_and_ir_json"

    samples_path = out_dir / "continuous_ir_12site_surrogate_sequence_wide_samples_v1.csv"
    report_path = out_dir / "continuous_ir_12site_surrogate_sequence_wide_features_v1.md"
    out.to_csv(samples_path, index=False)

    sequence_cols = [col for col in seq_df.columns]
    summary = pd.DataFrame(
        [
            {
                "input": str(input_path),
                "rows": len(out),
                "base_columns": len(df.columns),
                "sequence_columns_added": len(sequence_cols),
                "output_columns": len(out.columns),
                "history_days": args.history_days,
                "horizon_days": args.horizon_days,
            }
        ]
    )
    missing_rates = (
        out[sequence_cols]
        .isna()
        .mean()
        .reset_index()
        .rename(columns={"index": "sequence_column", 0: "missing_rate"})
        .sort_values("missing_rate", ascending=False)
    )
    missing_path = out_dir / "continuous_ir_12site_surrogate_sequence_wide_missing_v1.csv"
    missing_rates.to_csv(missing_path, index=False)

    lines = [
        "# Continuous Irrigation Sequence-Wide Features V1",
        "",
        "## Summary",
        "",
        markdown_table(summary),
        "",
        "## Purpose",
        "",
        "- Preserve daily history/future weather structure for stronger neural models.",
        "- Keep the original candidate-level sample rows and labels unchanged.",
        "- Add numeric `hist_lagXX_*`, `future_dayXX_*`, and `future_ir_dayXX` columns.",
        "",
        "## Outputs",
        "",
        f"- `{samples_path}`",
        f"- `{missing_path}`",
    ]
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print("Continuous irrigation sequence-wide features v1")
    print(f"samples: {samples_path}")
    print(f"missing: {missing_path}")
    print(f"report: {report_path}")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
