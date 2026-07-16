#!/usr/bin/env python3
"""Build the first 7-day short-term surrogate table from restart results.

This script converts the existing restart decision dataset into a version-1
short-term rolling surrogate dataset. Version 1 is intentionally conservative:
it preserves the SWAP-generated 7-day outcome labels and marks weather/current
state blocks that still need to be expanded in the next experiment.
"""

from __future__ import annotations

from pathlib import Path
import argparse
import json

import pandas as pd


DEFAULT_CANDIDATES = [0, 10, 15, 20, 25, 30, 40, 60]


def find_input_path(base: Path, explicit: str | None) -> Path:
    if explicit:
        path = Path(explicit)
        if not path.is_absolute():
            path = base / path
        return path

    candidates = [
        base / "Maize_restart_dataset" / "restart_decision_dataset.csv",
        base / "restart_decision_dataset.csv",
    ]
    for path in candidates:
        if path.exists():
            return path
    return candidates[0]


def sample_id(date_text: str, ir: float, site_id: str) -> str:
    token = date_text.replace("-", "").replace("/", "").replace(" ", "")
    ir_token = f"{int(ir):02d}" if float(ir).is_integer() else str(ir).replace(".", "p")
    return f"{site_id}_{token}_ir{ir_token}"


def candidate_sequence(ir: float, horizon_days: int) -> str:
    seq = [float(ir)] + [0.0] * max(horizon_days - 1, 0)
    return json.dumps(seq, separators=(",", ":"))


def markdown_table(df: pd.DataFrame) -> str:
    cols = list(df.columns)
    lines = [
        "| " + " | ".join(cols) + " |",
        "| " + " | ".join(["---"] * len(cols)) + " |",
    ]
    for row in df.itertuples(index=False):
        lines.append("| " + " | ".join(str(v) for v in row) + " |")
    return "\n".join(lines)


def build_shortterm_table(df: pd.DataFrame, site_id: str) -> pd.DataFrame:
    required = {
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
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"Input dataset is missing required columns: {missing}")

    work = df.copy()
    work["horizon_days"] = work["horizon_end_doy"] - work["decision_doy"]
    if (work["horizon_days"] <= 0).any():
        raise ValueError("Found non-positive horizon_days; check decision/end DOY columns.")

    no_ir = (
        work[work["ir"].astype(float) == 0.0]
        .loc[:, ["date_t", "target_value"]]
        .rename(columns={"target_value": "no_irrigation_target_7d"})
    )
    work = work.merge(no_ir, on="date_t", how="left")
    if work["no_irrigation_target_7d"].isna().any():
        bad_dates = sorted(work.loc[work["no_irrigation_target_7d"].isna(), "date_t"].unique())
        raise ValueError(f"Missing 0 mm baseline for dates: {bad_dates}")

    out = pd.DataFrame()
    out["sample_id"] = [
        sample_id(str(row.date_t), float(row.ir), site_id)
        for row in work.itertuples(index=False)
    ]
    out["site_id"] = site_id
    out["date_t"] = work["date_t"]
    out["decision_doy"] = work["decision_doy"].astype(int)
    out["horizon_days"] = work["horizon_days"].astype(int)
    out["candidate_ir"] = work["ir"].astype(float)
    out["candidate_ir_sequence"] = [
        candidate_sequence(float(row.ir), int(row.horizon_days))
        for row in work.itertuples(index=False)
    ]

    # These blocks are explicit placeholders in v1. They prevent us from
    # pretending that current-state and sequence features are already complete.
    out["static_feature_status"] = "pending_expand_from_uploaded_base_data"
    out["current_state_status"] = "pending_extract_pre_decision_state"
    out["history_weather_status"] = "pending_extract_history_sequence"
    out["history_irrigation_status"] = "pending_extract_history_sequence"
    out["forecast_weather_status"] = "observed_proxy_pending_gefs"
    out["forecast_irrigation_status"] = "decision_day_amount_then_zero"

    out["end_daynr"] = work["end_daynr"].astype(int)
    out["dvs_7d"] = work["dvs"]
    out["lai_7d"] = work["lai"]
    out["rootd_7d"] = work["rootd"]
    out["cwdm_7d"] = work["cwdm_value"]
    out["cwso_7d"] = work["cwso_value"]
    out["target_7d"] = work["target_value"]
    out["no_irrigation_target_7d"] = work["no_irrigation_target_7d"]
    out["net_gain_7d"] = out["target_7d"] - out["no_irrigation_target_7d"]
    out["best_ir_for_date"] = work["best_ir_for_date"]
    out["best_target_for_date"] = work["best_target_for_date"]
    out["is_best_ir"] = work["is_best_ir"].astype(bool)
    out["target_regret"] = out["best_target_for_date"] - out["target_7d"]
    return out


def write_outputs(out: pd.DataFrame, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    samples_path = output_dir / "shortterm_surrogate_samples_v1.csv"
    labels_path = output_dir / "shortterm_surrogate_labels_v1.csv"
    brief_path = output_dir / "shortterm_surrogate_dataset_brief.md"

    out.to_csv(samples_path, index=False)

    labels = (
        out.loc[out["is_best_ir"], ["site_id", "date_t", "decision_doy", "horizon_days", "best_ir_for_date", "best_target_for_date"]]
        .sort_values(["site_id", "decision_doy"])
        .reset_index(drop=True)
    )
    labels.to_csv(labels_path, index=False)

    summary = (
        out.groupby(["site_id", "date_t", "decision_doy"], as_index=False)
        .agg(
            n_candidates=("candidate_ir", "count"),
            best_ir=("best_ir_for_date", "first"),
            best_target=("best_target_for_date", "first"),
            max_target=("target_7d", "max"),
            min_target=("target_7d", "min"),
        )
        .sort_values(["site_id", "decision_doy"])
    )

    brief = [
        "# Short-Term Surrogate Dataset V1",
        "",
        "This table converts the existing SWAP restart decision results into the first 7-day short-term rolling surrogate format.",
        "",
        "## Summary",
        "",
        f"- Samples: {len(out)}",
        f"- Sites: {out['site_id'].nunique()}",
        f"- Decision dates: {out[['site_id', 'date_t']].drop_duplicates().shape[0]}",
        f"- Candidate irrigation amounts: {sorted(out['candidate_ir'].unique().tolist())}",
        "- Weather block: placeholder, to be replaced by observed/reanalysis sequences and then GEFS forecasts.",
        "- Current-state block: placeholder, to be filled by pre-decision SWAP states in the next experiment.",
        "",
        "## Best Candidate By Date",
        "",
        markdown_table(summary),
        "",
        "## Written Files",
        "",
        f"- `{samples_path.name}`",
        f"- `{labels_path.name}`",
        f"- `{brief_path.name}`",
    ]
    brief_path.write_text("\n".join(brief) + "\n", encoding="utf-8")

    print(f"Wrote {samples_path}")
    print(f"Wrote {labels_path}")
    print(f"Wrote {brief_path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", help="Path to restart_decision_dataset.csv")
    parser.add_argument("--output-dir", default="Maize_shortterm_surrogate_v1")
    parser.add_argument("--site-id", default="maize_test_site")
    args = parser.parse_args()

    base = Path.cwd()
    input_path = find_input_path(base, args.input)
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    df = pd.read_csv(input_path)
    out = build_shortterm_table(df, args.site_id)
    write_outputs(out, Path(args.output_dir))


if __name__ == "__main__":
    main()
