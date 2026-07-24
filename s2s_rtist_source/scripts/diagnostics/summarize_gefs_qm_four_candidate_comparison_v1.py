"""Build one evidence table comparing the four prelocked training-CV QM candidates."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONTRACT = (
    PROJECT_ROOT
    / "site_general_surrogate_eval"
    / "gefs_qm_training_cv_contract_v1.json"
)
DEFAULT_RESULT_DIR = (
    PROJECT_ROOT
    / "site_general_surrogate_eval"
    / "gefs_quantile_mapping_training_cv_v1"
)


def build_comparison(contract_path: Path, result_dir: Path) -> pd.DataFrame:
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    gate = json.loads(
        (result_dir / "training_cv_candidate_gate.json").read_text(encoding="utf-8")
    )
    pooled = pd.read_csv(result_dir / "training_cv_pooled_metrics.csv")
    heavy = pd.read_csv(result_dir / "training_cv_heavy_events_by_extrapolation.csv")
    configs = {
        item["candidate_id"]: item
        for item in contract["candidates"]
        if item["method"] == "empirical_quantile_mapping"
    }
    rows = []
    for candidate_id, config in configs.items():
        result = gate["candidate_results"][candidate_id]
        metrics = pooled.loc[pooled["candidate_id"].eq(candidate_id)].iloc[0]
        heavy_rows = heavy.loc[heavy["candidate_id"].eq(candidate_id)]
        non_tail = heavy_rows.loc[
            heavy_rows["extrapolation_group"].eq("no_member_extrapolated")
        ]
        fit_samples = int(config["fit_samples_per_group"])
        pass_counts = result["year_primary_metric_not_worse_counts"]
        failures = [
            name
            for name, passed in result["pooled_requirements"].items()
            if not passed
        ]
        if not result["year_stability_requirement_passed"]:
            failures.append("year_stability")
        rows.append(
            {
                "candidate_id": candidate_id,
                "group_keys": "+".join(config["group_keys"]) or "global_none",
                "mapping_group_count": int(config["mapping_groups"]),
                "fit_member_rows_per_group": fit_samples,
                "independent_reference_observations_per_group": fit_samples // 5,
                "seven_day_mae_difference_candidate_minus_raw_mm": float(
                    metrics["seven_day_mae_difference_candidate_minus_raw_mm"]
                ),
                "crps_difference_candidate_minus_raw_mm": float(
                    metrics["crps_difference_candidate_minus_raw_mm"]
                ),
                "mean_brier_difference_candidate_minus_raw": float(
                    metrics["mean_brier_difference_candidate_minus_raw"]
                ),
                "seven_day_mae_years_not_worse_out_of_4": int(
                    pass_counts["seven_day_mae_not_worse"]
                ),
                "crps_years_not_worse_out_of_4": int(
                    pass_counts["crps_not_worse"]
                ),
                "mean_brier_years_not_worse_out_of_4": int(
                    pass_counts["mean_brier_not_worse"]
                ),
                "heavy_observation_count": int(len(heavy_rows)),
                "raw_heavy_p10_p90_coverage": float(
                    heavy_rows["raw_p10_p90_covered"].mean()
                ),
                "qm_heavy_p10_p90_coverage": float(
                    heavy_rows["qm_p10_p90_covered"].mean()
                ),
                "raw_heavy_min_max_coverage": float(
                    heavy_rows["raw_min_max_covered"].mean()
                ),
                "qm_heavy_min_max_coverage": float(
                    heavy_rows["qm_min_max_covered"].mean()
                ),
                "non_extrapolated_heavy_observation_count": int(len(non_tail)),
                "non_extrapolated_heavy_spread_change_qm_minus_raw_mm": float(
                    non_tail["qm_spread"].mean() - non_tail["raw_spread"].mean()
                ),
                "upper_tail_event_count": int(
                    result["upper_tail_audit"]["event_count"]
                ),
                "upper_tail_improved_count": int(
                    result["upper_tail_audit"]["improved_count"]
                ),
                "upper_tail_worsened_count": int(
                    result["upper_tail_audit"]["worsened_count"]
                ),
                "year_stability_passed": bool(
                    result["year_stability_requirement_passed"]
                ),
                "eligible": bool(result["eligible"]),
                "failure_reasons": ";".join(failures),
            }
        )
    return pd.DataFrame(rows)


def run_summary(
    contract_path: Path = DEFAULT_CONTRACT, result_dir: Path = DEFAULT_RESULT_DIR
) -> dict[str, Path]:
    comparison = build_comparison(contract_path, result_dir)
    csv_path = result_dir / "training_cv_four_candidate_comparison.csv"
    report_path = result_dir / "training_cv_four_candidate_comparison.md"
    comparison.to_csv(csv_path, index=False, encoding="utf-8-sig")
    lines = [
        "# Four prelocked QM candidate comparison",
        "",
        "Negative metric differences favor QM. Independent reference counts divide repeated member rows by five exchangeable members.",
        "",
        "| Candidate | Grouping | Groups | Independent refs/group | 7d MAE diff | CRPS diff | Brier diff | MAE/CRPS/Brier years not worse | Heavy p10-p90 raw->QM | Heavy min-max raw->QM | Tail improved/worsened | Eligible |",
        "|---|---|---:|---:|---:|---:|---:|---|---|---|---|---|",
    ]
    for row in comparison.itertuples(index=False):
        lines.append(
            f"| `{row.candidate_id}` | `{row.group_keys}` | {row.mapping_group_count} | "
            f"{row.independent_reference_observations_per_group} | "
            f"{row.seven_day_mae_difference_candidate_minus_raw_mm:+.4f} | "
            f"{row.crps_difference_candidate_minus_raw_mm:+.4f} | "
            f"{row.mean_brier_difference_candidate_minus_raw:+.5f} | "
            f"{row.seven_day_mae_years_not_worse_out_of_4}/"
            f"{row.crps_years_not_worse_out_of_4}/"
            f"{row.mean_brier_years_not_worse_out_of_4} | "
            f"{row.raw_heavy_p10_p90_coverage:.3f}->{row.qm_heavy_p10_p90_coverage:.3f} | "
            f"{row.raw_heavy_min_max_coverage:.3f}->{row.qm_heavy_min_max_coverage:.3f} | "
            f"{row.upper_tail_improved_count}/{row.upper_tail_worsened_count} | "
            f"{row.eligible} |"
        )
    lines.extend(
        [
            "",
            "`global_seasonal` has no grouping key and is therefore a single global mapping in this experiment, despite its name.",
            "",
        ]
    )
    report_path.write_text("\n".join(lines), encoding="utf-8")
    return {"csv": csv_path, "report": report_path}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", type=Path, default=DEFAULT_CONTRACT)
    parser.add_argument("--result-dir", type=Path, default=DEFAULT_RESULT_DIR)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    print(json.dumps({key: str(value) for key, value in run_summary(args.contract, args.result_dir).items()}, indent=2))


if __name__ == "__main__":
    main()
