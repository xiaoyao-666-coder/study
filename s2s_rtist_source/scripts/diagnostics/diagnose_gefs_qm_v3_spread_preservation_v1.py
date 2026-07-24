"""Diagnose whether preserving raw GEFS ensemble spread repairs QM coverage."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from run_gefs_qm_training_cv_v1 import _metric_bundle, _tail_audit


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT = (
    PROJECT_ROOT
    / "site_general_surrogate_eval"
    / "gefs_qm_v3_design_exploration_v1"
    / "v3_exploration_oof_member_predictions.csv"
)
DEFAULT_OUTPUT_DIR = (
    PROJECT_ROOT
    / "site_general_surrogate_eval"
    / "gefs_qm_v3_spread_diagnostic_v1"
)
BASE_CANDIDATES = (
    "global_month_occurrence",
    "site_only_shrink_lambda18_occurrence",
    "site_month_shrink_lambda36_occurrence",
)


def _preserve_spread(frame: pd.DataFrame, mode: str) -> tuple[pd.DataFrame, int]:
    output = frame.copy()
    negative_before_clip = 0
    corrected_parts = []
    for _, group in output.groupby(
        ["site_id", "decision_date"], sort=False, dropna=False
    ):
        raw = group["precipitation_mm_raw"].to_numpy(dtype=float)
        qm = group["precipitation_mm_qm"].to_numpy(dtype=float)
        raw_mean = float(raw.mean())
        qm_mean = float(qm.mean())
        if mode == "raw_anomaly":
            corrected = qm_mean + (raw - raw_mean)
        elif mode.startswith("blend_raw_anomaly_alpha_"):
            alpha = float(mode.rsplit("_", 1)[-1])
            corrected = qm_mean + (1.0 - alpha) * (qm - qm_mean) + alpha * (
                raw - raw_mean
            )
        elif mode == "rescale_raw_std":
            raw_std = float(raw.std(ddof=0))
            qm_std = float(qm.std(ddof=0))
            scale = raw_std / qm_std if qm_std > 1.0e-12 else 1.0
            corrected = qm_mean + (qm - qm_mean) * scale
        else:
            raise ValueError(f"unsupported spread preservation mode: {mode}")
        negative_before_clip += int((corrected < 0.0).sum())
        group = group.copy()
        group["precipitation_mm_qm"] = np.maximum(corrected, 0.0)
        corrected_parts.append(group)
    return pd.concat(corrected_parts, ignore_index=True), negative_before_clip


def run_diagnostic(
    input_path: Path = DEFAULT_INPUT, output_dir: Path = DEFAULT_OUTPUT_DIR
) -> dict[str, Path]:
    data = pd.read_csv(
        input_path,
        parse_dates=["decision_date", "valid_date_utc"],
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    audits = {}
    for base_candidate in BASE_CANDIDATES:
        base = data.loc[data["candidate_id"].eq(base_candidate)].copy()
        modes = (
            "raw_anomaly",
            "rescale_raw_std",
            "blend_raw_anomaly_alpha_0.25",
            "blend_raw_anomaly_alpha_0.50",
            "blend_raw_anomaly_alpha_0.75",
        )
        for mode in modes:
            corrected, negative_before_clip = _preserve_spread(base, mode)
            candidate_id = f"{base_candidate}__{mode}"
            bundle = _metric_bundle(corrected)
            gate = bundle["gate"]
            heavy = bundle["observations"].loc[
                bundle["observations"]["reference_condition"].eq("heavy")
            ]
            by_method = heavy.groupby("method").agg(
                mean_spread=("ensemble_std", "mean"),
                p10_p90=("covered_by_p10_p90", "mean"),
                min_max=("covered_by_min_max", "mean"),
            )
            raw_values = by_method.loc["GEFS_raw"]
            qm_values = by_method.loc["GEFS_QM"]
            tail, audit = _tail_audit(corrected)
            audit.update(
                {
                    "candidate_id": candidate_id,
                    "base_candidate": base_candidate,
                    "spread_mode": mode,
                    "negative_before_clip_count": negative_before_clip,
                }
            )
            audits[candidate_id] = audit
            rows.append(
                {
                    "candidate_id": candidate_id,
                    "base_candidate": base_candidate,
                    "spread_mode": mode,
                    "raw_seven_day_mae_mm": gate["raw_seven_day_mae_mm"],
                    "candidate_seven_day_mae_mm": gate["qm_seven_day_mae_mm"],
                    "seven_day_mae_difference_mm": gate["qm_seven_day_mae_mm"]
                    - gate["raw_seven_day_mae_mm"],
                    "raw_mean_crps_mm": gate["raw_mean_crps_mm"],
                    "candidate_mean_crps_mm": gate["qm_mean_crps_mm"],
                    "crps_difference_mm": gate["qm_mean_crps_mm"]
                    - gate["raw_mean_crps_mm"],
                    "raw_mean_brier_score": gate["raw_mean_brier_score"],
                    "candidate_mean_brier_score": gate["qm_mean_brier_score"],
                    "mean_brier_difference": gate["qm_mean_brier_score"]
                    - gate["raw_mean_brier_score"],
                    "raw_heavy_mean_spread": float(raw_values["mean_spread"]),
                    "candidate_heavy_mean_spread": float(qm_values["mean_spread"]),
                    "raw_heavy_p10_p90": float(raw_values["p10_p90"]),
                    "candidate_heavy_p10_p90": float(qm_values["p10_p90"]),
                    "raw_heavy_min_max": float(raw_values["min_max"]),
                    "candidate_heavy_min_max": float(qm_values["min_max"]),
                    "heavy_coverage_not_both_worse": gate[
                        "automatic_requirements"
                    ]["heavy_coverage_not_both_worse"],
                    "negative_before_clip_count": negative_before_clip,
                }
            )
    metrics_path = output_dir / "v3_spread_preservation_metrics.csv"
    pd.DataFrame(rows).to_csv(metrics_path, index=False, encoding="utf-8-sig")
    audit_path = output_dir / "v3_spread_preservation_upper_tail_audit.json"
    audit_path.write_text(
        json.dumps(audits, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    report_path = output_dir / "v3_spread_preservation_scope_and_conclusion.md"
    report_path.write_text(
        "\n".join(
            [
                "# GEFS QM v3 spread preservation diagnostic",
                "",
                "This is an offline 2015-2018 OOF diagnostic. It keeps each base QM ensemble mean and restores raw member anomalies or raw standard deviation.",
                "Negative values before physical nonnegative clipping are reported explicitly; this diagnostic is not a production promotion.",
                "",
            ]
        ),
        encoding="utf-8",
    )
    return {"metrics": metrics_path, "audit": audit_path, "report": report_path}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    print(json.dumps({key: str(value) for key, value in run_diagnostic(args.input, args.output_dir).items()}, indent=2))


if __name__ == "__main__":
    main()
