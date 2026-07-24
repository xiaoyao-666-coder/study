#!/usr/bin/env python3
"""Compare upper-tail policies without refitting the GEFS QM bulk mapping."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from run_gefs_quantile_mapping_validation_v1 import (
    _probabilistic_metrics,
    _promotion_gate,
    _seven_day_metrics,
    _write_csv,
)
from s2s_rtist.weather.gefs_quantile_mapping import (
    GEFS_REFORECAST_MEMBERS,
    read_quantile_mapping_artifact,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
PILOT_DIR = (
    PROJECT_ROOT
    / "site_general_surrogate_eval"
    / "gefs_quantile_mapping_v1"
    / "gefs_qm_2015_2019_pilot_v1"
)
OUTPUT_DIR = PILOT_DIR / "upper_tail_policy_diagnostic_v1"


def apply_tail_variant(
    paired: pd.DataFrame,
    artifact: dict[str, object],
    *,
    policy: str,
) -> pd.DataFrame:
    if policy not in {"multiplicative", "constant_additive", "raw_on_tail"}:
        raise ValueError(f"unsupported tail diagnostic policy: {policy}")
    output = paired.copy()
    if policy == "multiplicative":
        return output
    for (site_id, lead_day), indices in output.loc[
        output["qm_extrapolated_upper"]
    ].groupby(["site_id", "lead_day"]).groups.items():
        group = artifact["groups"][f"{site_id}|{int(lead_day)}"]
        raw = output.loc[indices, "precipitation_mm_raw"].to_numpy(dtype=float)
        if policy == "constant_additive":
            correction = float(group["training_reference_maximum_mm"]) - float(
                group["training_forecast_maximum_mm"]
            )
            corrected = np.maximum(0.0, raw + correction)
        else:
            corrected = raw
        output.loc[indices, "precipitation_mm_qm"] = corrected
    output["tail_diagnostic_policy"] = policy
    return output


def evaluate_variant(
    paired: pd.DataFrame,
    artifact: dict[str, object],
    *,
    policy: str,
) -> tuple[dict[str, object], pd.DataFrame]:
    variant = apply_tail_variant(paired, artifact, policy=policy)
    observations, _, probabilities = _probabilistic_metrics(
        variant, members=GEFS_REFORECAST_MEMBERS
    )
    seven_day = _seven_day_metrics(variant)
    gate = _promotion_gate(
        observations=observations,
        probabilities=probabilities,
        seven_day=seven_day,
        paired=variant,
    )
    return gate, variant.loc[variant["qm_extrapolated_upper"]].copy()


def run_diagnostic(
    *, pilot_dir: Path = PILOT_DIR, output_dir: Path = OUTPUT_DIR
) -> dict[str, Path]:
    cases = {
        "site_local_day": (
            pilot_dir / "paired_raw_and_qm_members_2019.csv",
            pilot_dir / "gefs_precipitation_qm_artifact.json",
        ),
        "utc_day": (
            pilot_dir
            / "alignment_diagnostic_utc_day_v1"
            / "paired_raw_and_qm_members_2019_utc_day.csv",
            pilot_dir
            / "alignment_diagnostic_utc_day_v1"
            / "gefs_precipitation_qm_artifact_utc_day.json",
        ),
    }
    policies = ("multiplicative", "constant_additive", "raw_on_tail")
    summary_rows = []
    tail_parts = []
    detailed: dict[str, object] = {}
    for alignment, (paired_path, artifact_path) in cases.items():
        paired = pd.read_csv(paired_path)
        artifact = read_quantile_mapping_artifact(artifact_path)
        detailed[alignment] = {}
        for policy in policies:
            gate, tail = evaluate_variant(paired, artifact, policy=policy)
            detailed[alignment][policy] = gate
            summary_rows.append(
                {
                    "alignment": alignment,
                    "upper_tail_policy": policy,
                    "raw_seven_day_mae_mm": gate["raw_seven_day_mae_mm"],
                    "qm_seven_day_mae_mm": gate["qm_seven_day_mae_mm"],
                    "raw_mean_crps_mm": gate["raw_mean_crps_mm"],
                    "qm_mean_crps_mm": gate["qm_mean_crps_mm"],
                    "raw_mean_brier_score": gate["raw_mean_brier_score"],
                    "qm_mean_brier_score": gate["qm_mean_brier_score"],
                    "automatic_requirements_passed": gate[
                        "automatic_requirements_passed"
                    ],
                    "promotion_status": gate["promotion_status"],
                }
            )
            tail["alignment"] = alignment
            tail["upper_tail_policy"] = policy
            tail_parts.append(tail)

    summary = pd.DataFrame(summary_rows)
    tail_evidence = pd.concat(tail_parts, ignore_index=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / "upper_tail_policy_metrics.csv"
    tail_path = output_dir / "upper_tail_policy_events.csv"
    json_path = output_dir / "upper_tail_policy_gate_details.json"
    report_path = output_dir / "upper_tail_policy_diagnostic.md"
    _write_csv(summary, summary_path)
    _write_csv(tail_evidence, tail_path)
    json_path.write_text(
        json.dumps(detailed, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    report_lines = [
        "# GEFS QM upper-tail policy diagnostic",
        "",
        "The bulk mapping and wet-day correction are frozen. Only 2019 values above each training-group forecast maximum are changed.",
        "",
        "| Day alignment | Tail policy | QM 7-day MAE (mm) | QM CRPS (mm) | QM mean Brier | Automatic gate |",
        "|---|---|---:|---:|---:|---|",
    ]
    for row in summary.itertuples(index=False):
        report_lines.append(
            f"| {row.alignment} | {row.upper_tail_policy} "
            f"| {row.qm_seven_day_mae_mm:.6f} | {row.qm_mean_crps_mm:.6f} "
            f"| {row.qm_mean_brier_score:.6f} "
            f"| {'pass' if row.automatic_requirements_passed else 'fail'} |"
        )
    report_lines.extend(
        [
            "",
            "`raw_on_tail` is an attribution control, not a proposed operational extrapolation rule.",
            "",
        ]
    )
    report_path.write_text("\n".join(report_lines), encoding="utf-8")
    return {
        "summary": summary_path,
        "tail_events": tail_path,
        "gate_details": json_path,
        "report": report_path,
    }


if __name__ == "__main__":
    print(
        json.dumps(
            {key: str(value) for key, value in run_diagnostic().items()}, indent=2
        )
    )
