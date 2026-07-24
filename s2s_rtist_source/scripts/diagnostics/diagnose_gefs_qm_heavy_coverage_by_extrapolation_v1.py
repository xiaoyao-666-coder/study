"""Split heavy-event ensemble coverage by whether any member used QM upper-tail extrapolation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from s2s_rtist.weather.gefs_ensemble_validation import _ensemble_crps


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT = (
    PROJECT_ROOT
    / "site_general_surrogate_eval"
    / "gefs_quantile_mapping_training_cv_v1"
    / "training_cv_oof_member_predictions.csv"
)
DEFAULT_OUTPUT_DIR = (
    PROJECT_ROOT
    / "site_general_surrogate_eval"
    / "gefs_quantile_mapping_training_cv_v1"
)


def _observation_rows(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    keys = ["candidate_id", "site_id", "decision_date", "valid_date_utc", "lead_day"]
    for values, group in frame.groupby(keys, sort=True, dropna=False):
        candidate_id, site_id, decision_date, valid_date_utc, lead_day = values
        reference = float(group["precipitation_mm_reference"].iloc[0])
        raw = group["precipitation_mm_raw"].to_numpy(dtype=float)
        qm = group["precipitation_mm_qm"].to_numpy(dtype=float)
        if len(raw) != 5:
            raise ValueError(f"expected five members for {values}, got {len(raw)}")
        if group["precipitation_mm_reference"].nunique(dropna=False) != 1:
            raise ValueError(f"reference differs across members for {values}")
        any_upper = bool(group["qm_extrapolated_upper"].astype(bool).any())
        rows.append(
            {
                "candidate_id": candidate_id,
                "site_id": site_id,
                "decision_date": decision_date,
                "valid_date_utc": valid_date_utc,
                "lead_day": int(lead_day),
                "precipitation_mm_reference": reference,
                "heavy_event": reference >= 20.0,
                "extrapolation_group": (
                    "any_member_extrapolated" if any_upper else "no_member_extrapolated"
                ),
                "extrapolated_member_count": int(
                    group["qm_extrapolated_upper"].astype(bool).sum()
                ),
                "raw_mean": float(raw.mean()),
                "qm_mean": float(qm.mean()),
                "raw_spread": float(raw.std(ddof=0)),
                "qm_spread": float(qm.std(ddof=0)),
                "raw_crps": float(_ensemble_crps(raw, reference)),
                "qm_crps": float(_ensemble_crps(qm, reference)),
                "raw_p10": float(np.quantile(raw, 0.1)),
                "raw_p90": float(np.quantile(raw, 0.9)),
                "qm_p10": float(np.quantile(qm, 0.1)),
                "qm_p90": float(np.quantile(qm, 0.9)),
                "raw_min": float(raw.min()),
                "raw_max": float(raw.max()),
                "qm_min": float(qm.min()),
                "qm_max": float(qm.max()),
                "raw_p10_p90_covered": bool(
                    np.quantile(raw, 0.1) <= reference <= np.quantile(raw, 0.9)
                ),
                "qm_p10_p90_covered": bool(
                    np.quantile(qm, 0.1) <= reference <= np.quantile(qm, 0.9)
                ),
                "raw_min_max_covered": bool(raw.min() <= reference <= raw.max()),
                "qm_min_max_covered": bool(qm.min() <= reference <= qm.max()),
            }
        )
    return pd.DataFrame(rows)


def _aggregate(heavy: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (candidate_id, group_name), group in heavy.groupby(
        ["candidate_id", "extrapolation_group"], sort=True, dropna=False
    ):
        rows.append(
            {
                "candidate_id": candidate_id,
                "extrapolation_group": group_name,
                "heavy_observation_count": int(len(group)),
                "extrapolated_member_count": int(group["extrapolated_member_count"].sum()),
                "raw_p10_p90_coverage": float(group["raw_p10_p90_covered"].mean()),
                "qm_p10_p90_coverage": float(group["qm_p10_p90_covered"].mean()),
                "p10_p90_coverage_change_qm_minus_raw": float(
                    group["qm_p10_p90_covered"].mean()
                    - group["raw_p10_p90_covered"].mean()
                ),
                "raw_min_max_coverage": float(group["raw_min_max_covered"].mean()),
                "qm_min_max_coverage": float(group["qm_min_max_covered"].mean()),
                "min_max_coverage_change_qm_minus_raw": float(
                    group["qm_min_max_covered"].mean()
                    - group["raw_min_max_covered"].mean()
                ),
                "raw_mean_spread_mm": float(group["raw_spread"].mean()),
                "qm_mean_spread_mm": float(group["qm_spread"].mean()),
                "spread_change_qm_minus_raw_mm": float(
                    group["qm_spread"].mean() - group["raw_spread"].mean()
                ),
                "raw_mean_crps_mm": float(group["raw_crps"].mean()),
                "qm_mean_crps_mm": float(group["qm_crps"].mean()),
                "crps_change_qm_minus_raw_mm": float(
                    group["qm_crps"].mean() - group["raw_crps"].mean()
                ),
                "raw_mean_absolute_error_mm": float(
                    (group["raw_mean"] - group["precipitation_mm_reference"]).abs().mean()
                ),
                "qm_mean_absolute_error_mm": float(
                    (group["qm_mean"] - group["precipitation_mm_reference"]).abs().mean()
                ),
            }
        )
    return pd.DataFrame(rows)


def run_diagnostic(
    input_path: Path = DEFAULT_INPUT, output_dir: Path = DEFAULT_OUTPUT_DIR
) -> dict[str, Path]:
    frame = pd.read_csv(
        input_path,
        parse_dates=["decision_date", "valid_date_utc"],
    )
    observations = _observation_rows(frame)
    heavy = observations.loc[observations["heavy_event"]].copy()
    summary = _aggregate(heavy)
    output_dir.mkdir(parents=True, exist_ok=True)
    observation_path = output_dir / "training_cv_heavy_events_by_extrapolation.csv"
    summary_path = output_dir / "training_cv_heavy_coverage_by_extrapolation.csv"
    report_path = output_dir / "training_cv_heavy_coverage_by_extrapolation.md"
    heavy.to_csv(observation_path, index=False, encoding="utf-8-sig")
    summary.to_csv(summary_path, index=False, encoding="utf-8-sig")

    lines = [
        "# Heavy-event coverage split by QM upper-tail extrapolation",
        "",
        "A heavy event is an observation with reference precipitation >= 20 mm/day. An observation is classified as extrapolated when any of its five members uses upper-tail extrapolation.",
        "",
    ]
    for row in summary.itertuples(index=False):
        lines.append(
            f"- `{row.candidate_id}` / `{row.extrapolation_group}`: "
            f"n={row.heavy_observation_count}, p10-p90 {row.raw_p10_p90_coverage:.3f}->{row.qm_p10_p90_coverage:.3f}, "
            f"min-max {row.raw_min_max_coverage:.3f}->{row.qm_min_max_coverage:.3f}, "
            f"spread {row.raw_mean_spread_mm:.3f}->{row.qm_mean_spread_mm:.3f} mm, "
            f"CRPS change {row.crps_change_qm_minus_raw_mm:+.3f} mm."
        )
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return {
        "heavy_observations": observation_path,
        "summary": summary_path,
        "report": report_path,
    }


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
