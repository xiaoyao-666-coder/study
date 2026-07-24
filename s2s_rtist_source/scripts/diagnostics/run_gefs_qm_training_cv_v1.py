"""Run the prelocked 2015-2018 GEFS precipitation QM cross-validation offline."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd

from run_gefs_qm_2019_confirmation_v1 import (
    _cycle_metric_differences,
    _stratified_metrics,
)
from run_gefs_quantile_mapping_validation_v1 import (
    _probabilistic_metrics,
    _promotion_gate,
    _seven_day_metrics,
    _write_csv,
)
from s2s_rtist.weather.gefs_quantile_mapping import (
    CONTRACT_ID_TRAINING_CV,
    CONTRACT_VERSION_TRAINING_CV,
    GEFS_REFORECAST_MEMBERS,
    UPPER_TAIL_CONSTANT_ADDITIVE,
    UTC_DAY_BOUNDARY,
    aggregate_reforecast_member_daily_utc,
    apply_empirical_precipitation_qm,
    fit_empirical_precipitation_qm,
    pair_member_and_reference,
    validate_member_daily_precipitation,
    validate_reference_daily_precipitation,
    verify_quantile_mapping_artifact,
    write_quantile_mapping_artifact,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONTRACT_PATH = (
    PROJECT_ROOT
    / "site_general_surrogate_eval"
    / "gefs_qm_training_cv_contract_v1.json"
)
DEFAULT_OUTPUT_DIR = (
    PROJECT_ROOT
    / "site_general_surrogate_eval"
    / "gefs_quantile_mapping_training_cv_v1"
)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_contract(path: Path) -> dict[str, Any]:
    contract = json.loads(path.read_text(encoding="utf-8"))
    if contract.get("contract_id") != CONTRACT_ID_TRAINING_CV:
        raise ValueError("training CV contract id mismatch")
    if contract.get("contract_version") != CONTRACT_VERSION_TRAINING_CV:
        raise ValueError("training CV contract version mismatch")
    if contract["scope"]["new_network_download_allowed"]:
        raise ValueError("training CV contract must prohibit network downloads")
    return contract


def _cycle_dates(contract: dict[str, Any]) -> list[str]:
    years = [int(year) for year in contract["source_data"]["allowed_years"]]
    month_days = contract["source_data"]["month_days_per_year"]
    return [f"{year}-{month_day}" for year in years for month_day in month_days]


def load_training_period_inputs(
    contract: dict[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    source = contract["source_data"]
    allowed_years = {int(year) for year in source["allowed_years"]}
    point_dir = PROJECT_ROOT / source["point_cache_relative_path"]
    selected_paths = sorted(
        path
        for path in point_dir.glob("*.csv")
        if int(path.name[:4]) in allowed_years
    )
    expected_files = int(source["cycles"]) * len(source["members"])
    if len(selected_paths) != expected_files:
        raise ValueError(
            f"training point cache files={len(selected_paths)}, expected={expected_files}"
        )
    points = pd.concat(
        [pd.read_csv(path, parse_dates=["cycle_init_utc"]) for path in selected_paths],
        ignore_index=True,
    )

    manifest = pd.read_csv(PROJECT_ROOT / source["source_manifest_relative_path"])
    manifest_year = pd.to_datetime(manifest["cycle_date"]).dt.year
    manifest = manifest.loc[manifest_year.isin(allowed_years)].copy()
    if len(manifest) != expected_files:
        raise ValueError(
            f"training source manifest rows={len(manifest)}, expected={expected_files}"
        )
    if manifest["source_etag"].astype(str).str.strip().eq("").any():
        raise ValueError("training source manifest contains an empty ETag")

    reference = pd.read_csv(
        PROJECT_ROOT / source["era5_reference_relative_path"],
        parse_dates=["valid_date_utc"],
    )
    reference_year = pd.to_datetime(reference["valid_date_utc"]).dt.year
    reference = reference.loc[reference_year.isin(allowed_years)].copy()
    return points, manifest.reset_index(drop=True), reference.reset_index(drop=True)


def prepare_training_period_paired(
    contract: dict[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    points, manifest, reference = load_training_period_inputs(contract)
    member_daily = aggregate_reforecast_member_daily_utc(points, manifest=manifest)
    source = contract["source_data"]
    cycles = _cycle_dates(contract)
    valid_dates = sorted(pd.to_datetime(reference["valid_date_utc"]).unique())
    validate_member_daily_precipitation(
        member_daily,
        expected_sites=source["sites"],
        expected_members=source["members"],
        expected_cycles=cycles,
        date_column="valid_date_utc",
    )
    validate_reference_daily_precipitation(
        reference,
        expected_sites=source["sites"],
        expected_dates=valid_dates,
        date_column="valid_date_utc",
    )
    if len(member_daily) != int(source["member_rows"]):
        raise ValueError("training member row count does not match the contract")
    if len(reference) != int(source["unique_reference_observations"]):
        raise ValueError("training reference row count does not match the contract")
    paired = pair_member_and_reference(
        member_daily, reference, date_column="valid_date_utc"
    )
    paired["decision_date"] = pd.to_datetime(paired["decision_date"])
    paired["validation_year"] = paired["decision_date"].dt.year.astype(int)
    forbidden = set(source["forbidden_years"]).intersection(
        set(paired["validation_year"])
    )
    if forbidden:
        raise ValueError(f"forbidden years entered training CV: {sorted(forbidden)}")
    return paired, manifest


def _fold_assignment(contract: dict[str, Any]) -> pd.DataFrame:
    rows = []
    month_days = contract["source_data"]["month_days_per_year"]
    for fold in contract["folds"]:
        year = int(fold["validation_year"])
        for month_day in month_days:
            rows.append(
                {
                    "fold_id": fold["fold_id"],
                    "validation_year": year,
                    "forecast_init_utc": f"{year}-{month_day}T00:00:00Z",
                    "fit_years": ",".join(str(value) for value in fold["fit_years"]),
                    "validation_rows_used_for_fit": 0,
                }
            )
    return pd.DataFrame(rows)


def _tail_audit(corrected: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    tail = corrected.loc[corrected["qm_extrapolated_upper"]].copy()
    tail["raw_absolute_error_mm"] = (
        tail["precipitation_mm_raw"] - tail["precipitation_mm_reference"]
    ).abs()
    tail["qm_absolute_error_mm"] = (
        tail["precipitation_mm_qm"] - tail["precipitation_mm_reference"]
    ).abs()
    tail["absolute_error_change_mm"] = (
        tail["qm_absolute_error_mm"] - tail["raw_absolute_error_mm"]
    )
    all_values = corrected["precipitation_mm_qm"].to_numpy(dtype=float)
    tail_values = tail["precipitation_mm_qm"].to_numpy(dtype=float)
    passed = bool(np.isfinite(all_values).all() and np.all(all_values >= 0.0))
    summary = {
        "event_count": int(len(tail)),
        "nonfinite_corrected_count": int((~np.isfinite(all_values)).sum()),
        "negative_corrected_count": int((all_values < 0.0).sum()),
        "maximum_corrected_mm_day": (
            float(tail_values.max()) if len(tail_values) else None
        ),
        "improved_count": int(tail["absolute_error_change_mm"].lt(0.0).sum()),
        "worsened_count": int(tail["absolute_error_change_mm"].gt(0.0).sum()),
        "mean_absolute_error_change_mm_day": (
            float(tail["absolute_error_change_mm"].mean()) if len(tail) else None
        ),
        "numeric_audit_passed": passed,
        "audit_status": (
            "passed_no_events"
            if passed and tail.empty
            else (
                "passed_with_residual_extreme_error_caveat"
                if passed
                else "failed_numeric_audit"
            )
        ),
    }
    return tail, summary


def _metric_bundle(corrected: pd.DataFrame) -> dict[str, Any]:
    metric_input = corrected.rename(columns={"valid_date_utc": "local_date"})
    observations, probabilistic, probabilities = _probabilistic_metrics(
        metric_input, members=GEFS_REFORECAST_MEMBERS
    )
    seven_day = _seven_day_metrics(metric_input)
    gate = _promotion_gate(
        observations=observations,
        probabilities=probabilities,
        seven_day=seven_day,
        paired=metric_input,
    )
    return {
        "metric_input": metric_input,
        "observations": observations,
        "probabilistic": probabilistic,
        "probabilities": probabilities,
        "seven_day": seven_day,
        "gate": gate,
    }


def _metric_row(candidate_id: str, year: int | None, bundle: dict[str, Any]) -> dict[str, Any]:
    gate = bundle["gate"]
    row: dict[str, Any] = {
        "candidate_id": candidate_id,
        "validation_year": year if year is not None else "pooled",
        "raw_seven_day_mae_mm": gate["raw_seven_day_mae_mm"],
        "candidate_seven_day_mae_mm": gate["qm_seven_day_mae_mm"],
        "seven_day_mae_difference_candidate_minus_raw_mm": (
            gate["qm_seven_day_mae_mm"] - gate["raw_seven_day_mae_mm"]
        ),
        "raw_mean_crps_mm": gate["raw_mean_crps_mm"],
        "candidate_mean_crps_mm": gate["qm_mean_crps_mm"],
        "crps_difference_candidate_minus_raw_mm": (
            gate["qm_mean_crps_mm"] - gate["raw_mean_crps_mm"]
        ),
        "raw_mean_brier_score": gate["raw_mean_brier_score"],
        "candidate_mean_brier_score": gate["qm_mean_brier_score"],
        "mean_brier_difference_candidate_minus_raw": (
            gate["qm_mean_brier_score"] - gate["raw_mean_brier_score"]
        ),
        "seven_day_mae_not_worse": gate["automatic_requirements"][
            "seven_day_mae_not_worse"
        ],
        "crps_not_worse": gate["automatic_requirements"]["crps_not_worse"],
        "mean_brier_not_worse": gate["automatic_requirements"][
            "mean_brier_not_worse"
        ],
        "heavy_coverage_not_both_worse": gate["automatic_requirements"][
            "heavy_coverage_not_both_worse"
        ],
    }
    overall = bundle["probabilistic"].loc[
        bundle["probabilistic"]["scope"].eq("overall")
    ]
    for method, prefix in (("GEFS_raw", "raw"), ("GEFS_QM", "candidate")):
        values = overall.loc[overall["method"].eq(method)].iloc[0]
        for metric in (
            "ensemble_mean_bias",
            "ensemble_mean_mae",
            "ensemble_mean_rmse",
            "mean_ensemble_spread",
            "p10_p90_coverage",
            "min_max_coverage",
        ):
            row[f"{prefix}_{metric}"] = float(values[metric])
    return row


def _bootstrap(
    candidate_id: str,
    metric_input: pd.DataFrame,
    *,
    expected_cycles: int,
    replicates: int,
    seed: int,
) -> pd.DataFrame:
    differences = _cycle_metric_differences(metric_input)
    rng = np.random.default_rng(seed)
    rows = []
    for metric_name, frame in differences.items():
        values = frame["difference_qm_minus_raw"].to_numpy(dtype=float)
        if len(values) != expected_cycles:
            raise ValueError(
                f"bootstrap cycles={len(values)}, expected={expected_cycles} for {metric_name}"
            )
        indices = rng.integers(0, len(values), size=(replicates, len(values)))
        means = values[indices].mean(axis=1)
        rows.append(
            {
                "candidate_id": candidate_id,
                "metric": metric_name,
                "cycle_count": int(len(values)),
                "point_difference_candidate_minus_raw": float(values.mean()),
                "bootstrap_ci_lower_95": float(np.quantile(means, 0.025)),
                "bootstrap_ci_upper_95": float(np.quantile(means, 0.975)),
                "bootstrap_replicates": replicates,
                "random_seed": seed,
                "fraction_bootstrap_improved": float((means <= 0.0).mean()),
            }
        )
    return pd.DataFrame(rows)


def _pareto_relationship(
    eligible: Sequence[str], pooled_rows: pd.DataFrame
) -> dict[str, list[str]]:
    metrics = [
        "candidate_seven_day_mae_mm",
        "candidate_mean_crps_mm",
        "candidate_mean_brier_score",
    ]
    indexed = pooled_rows.set_index("candidate_id")
    result: dict[str, list[str]] = {}
    for candidate in eligible:
        candidate_values = indexed.loc[candidate, metrics].to_numpy(dtype=float)
        dominated_by = []
        for other in eligible:
            if other == candidate:
                continue
            other_values = indexed.loc[other, metrics].to_numpy(dtype=float)
            if np.all(other_values <= candidate_values) and np.any(
                other_values < candidate_values
            ):
                dominated_by.append(other)
        result[candidate] = sorted(dominated_by)
    return result


def run_training_cv(
    *, contract_path: Path = CONTRACT_PATH, output_dir: Path = DEFAULT_OUTPUT_DIR
) -> dict[str, Path]:
    contract = _load_contract(contract_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    paired, source_manifest = prepare_training_period_paired(contract)
    folds = contract["folds"]
    qm_candidates = [
        candidate
        for candidate in contract["candidates"]
        if candidate["method"] == "empirical_quantile_mapping"
    ]
    members = tuple(contract["source_data"]["members"])
    expected = contract["per_fold_expected_counts"]
    artifacts_dir = output_dir / "artifacts"

    assignment = _fold_assignment(contract)
    assignment_path = output_dir / "training_cv_fold_assignment.csv"
    _write_csv(assignment, assignment_path)
    configuration_path = output_dir / "training_cv_candidate_configurations.json"
    configuration_path.write_text(
        json.dumps(
            {
                "contract_id": contract["contract_id"],
                "contract_version": contract["contract_version"],
                "candidates": contract["candidates"],
                "folds": folds,
                "network_download_performed": False,
                "source_manifest_sha256": _sha256_file(
                    PROJECT_ROOT
                    / contract["source_data"]["source_manifest_relative_path"]
                ),
                "selected_source_manifest_rows": int(len(source_manifest)),
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    artifact_rows: list[dict[str, Any]] = []
    oof_parts: list[pd.DataFrame] = []
    for candidate in qm_candidates:
        candidate_id = candidate["candidate_id"]
        group_keys = tuple(candidate["group_keys"])
        for fold in folds:
            fit_years = tuple(int(year) for year in fold["fit_years"])
            validation_year = int(fold["validation_year"])
            fit = paired.loc[paired["validation_year"].isin(fit_years)].copy()
            validation = paired.loc[
                paired["validation_year"].eq(validation_year)
            ].copy()
            if len(fit) != int(expected["fit_member_rows"]):
                raise ValueError(f"{candidate_id} {fold['fold_id']} fit row mismatch")
            if len(validation) != int(expected["validation_member_rows"]):
                raise ValueError(
                    f"{candidate_id} {fold['fold_id']} validation row mismatch"
                )
            artifact = fit_empirical_precipitation_qm(
                fit,
                fit_years=fit_years,
                expected_members=members,
                contract_id=CONTRACT_ID_TRAINING_CV,
                contract_version=CONTRACT_VERSION_TRAINING_CV,
                aggregation_day_boundary=UTC_DAY_BOUNDARY,
                canonical_valid_date_column="valid_date_utc",
                upper_tail_policy=UPPER_TAIL_CONSTANT_ADDITIVE,
                group_keys=group_keys,
                artifact_context={
                    "candidate_id": candidate_id,
                    "fold_id": fold["fold_id"],
                    "validation_year": validation_year,
                },
            )
            verify_quantile_mapping_artifact(artifact)
            artifact_path = artifacts_dir / candidate_id / f"{fold['fold_id']}.json"
            write_quantile_mapping_artifact(artifact_path, artifact)
            counts = set(int(value) for value in artifact["group_sample_counts"].values())
            expected_group_count = int(candidate["mapping_groups"])
            expected_samples = int(candidate["fit_samples_per_group"])
            group_checks_passed = bool(
                len(artifact["groups"]) == expected_group_count
                and counts == {expected_samples}
            )
            leakage_rows = int(
                fit["validation_year"].eq(validation_year).sum()
            )
            artifact_rows.append(
                {
                    "candidate_id": candidate_id,
                    "fold_id": fold["fold_id"],
                    "fit_years": ",".join(str(year) for year in fit_years),
                    "validation_year": validation_year,
                    "fit_member_rows": int(len(fit)),
                    "validation_member_rows": int(len(validation)),
                    "mapping_group_count": int(len(artifact["groups"])),
                    "expected_mapping_group_count": expected_group_count,
                    "group_sample_counts": ",".join(
                        str(value) for value in sorted(counts)
                    ),
                    "expected_samples_per_group": expected_samples,
                    "validation_rows_used_for_fit": leakage_rows,
                    "group_checks_passed": group_checks_passed,
                    "hash_check_passed": True,
                    "artifact_sha256": artifact["artifact_sha256"],
                    "training_input_sha256": artifact["training_input_sha256"],
                    "artifact_relative_path": str(
                        artifact_path.relative_to(output_dir)
                    ).replace("\\", "/"),
                }
            )
            corrected = apply_empirical_precipitation_qm(
                validation,
                artifact,
                split=f"training_cv_oof_{fold['fold_id']}",
            )
            corrected["candidate_id"] = candidate_id
            corrected["fold_id"] = fold["fold_id"]
            oof_parts.append(corrected)

    artifact_manifest = pd.DataFrame(artifact_rows)
    artifact_manifest_path = output_dir / "training_cv_fold_artifact_manifest.csv"
    _write_csv(artifact_manifest, artifact_manifest_path)
    oof = pd.concat(oof_parts, ignore_index=True)
    oof_key = [
        "candidate_id",
        "site_id",
        "decision_date",
        "valid_date_utc",
        "gefs_member",
    ]
    if oof.duplicated(oof_key).any():
        raise ValueError("duplicate training CV OOF prediction keys")
    expected_oof = int(
        contract["oof_expected_counts_per_qm_candidate"]["member_rows"]
    )
    candidate_counts = oof.groupby("candidate_id").size().to_dict()
    if any(int(candidate_counts.get(item["candidate_id"], 0)) != expected_oof for item in qm_candidates):
        raise ValueError(f"OOF candidate row counts do not match contract: {candidate_counts}")
    oof_path = output_dir / "training_cv_oof_member_predictions.csv"
    _write_csv(oof, oof_path)

    pooled_rows = []
    year_rows = []
    stratified_parts = []
    bootstrap_parts = []
    tail_parts = []
    tail_summaries: dict[str, Any] = {}
    candidate_bundles: dict[str, dict[str, Any]] = {}
    bootstrap_config = contract["paired_cycle_block_bootstrap"]
    for candidate in qm_candidates:
        candidate_id = candidate["candidate_id"]
        corrected = oof.loc[oof["candidate_id"].eq(candidate_id)].copy()
        bundle = _metric_bundle(corrected)
        candidate_bundles[candidate_id] = bundle
        pooled_rows.append(_metric_row(candidate_id, None, bundle))
        for year in contract["source_data"]["allowed_years"]:
            year_corrected = corrected.loc[
                corrected["validation_year"].eq(int(year))
            ].copy()
            year_bundle = _metric_bundle(year_corrected)
            year_rows.append(
                _metric_row(candidate_id, int(year), year_bundle)
            )
            stratified = _stratified_metrics(
                year_bundle["observations"], year_bundle["probabilistic"]
            )
            stratified["candidate_id"] = candidate_id
            stratified["validation_year"] = int(year)
            stratified["method"] = stratified["method"].replace(
                {"GEFS_raw": "raw_gefs", "GEFS_QM": candidate_id}
            )
            stratified_parts.append(stratified)
        bootstrap_parts.append(
            _bootstrap(
                candidate_id,
                bundle["metric_input"],
                expected_cycles=int(bootstrap_config["oof_cycle_count"]),
                replicates=int(bootstrap_config["replicates"]),
                seed=int(bootstrap_config["random_seed"]),
            )
        )
        tail, tail_summary = _tail_audit(corrected)
        tail["candidate_id"] = candidate_id
        tail_parts.append(tail)
        tail_summaries[candidate_id] = tail_summary

    pooled = pd.DataFrame(pooled_rows)
    years = pd.DataFrame(year_rows)
    pooled_path = output_dir / "training_cv_pooled_metrics.csv"
    year_path = output_dir / "training_cv_year_metrics.csv"
    stratified_path = output_dir / "training_cv_stratified_metrics.csv"
    bootstrap_path = output_dir / "training_cv_cycle_block_bootstrap.csv"
    tail_path = output_dir / "training_cv_upper_tail_events.csv"
    tail_audit_path = output_dir / "training_cv_upper_tail_audit.json"
    _write_csv(pooled, pooled_path)
    _write_csv(years, year_path)
    _write_csv(pd.concat(stratified_parts, ignore_index=True), stratified_path)
    _write_csv(pd.concat(bootstrap_parts, ignore_index=True), bootstrap_path)
    _write_csv(pd.concat(tail_parts, ignore_index=True), tail_path)
    tail_audit_path.write_text(
        json.dumps(tail_summaries, ensure_ascii=False, indent=2, sort_keys=True)
        + "\n",
        encoding="utf-8",
    )

    artifact_ok = {
        candidate_id: bool(
            frame["group_checks_passed"].all()
            and frame["hash_check_passed"].all()
            and frame["validation_rows_used_for_fit"].eq(0).all()
        )
        for candidate_id, frame in artifact_manifest.groupby(
            "candidate_id", sort=True
        )
    }
    candidate_results: dict[str, Any] = {}
    eligible: list[str] = []
    for candidate in qm_candidates:
        candidate_id = candidate["candidate_id"]
        pooled_row = pooled.loc[pooled["candidate_id"].eq(candidate_id)].iloc[0]
        year_subset = years.loc[years["candidate_id"].eq(candidate_id)]
        year_pass_counts = {
            "seven_day_mae_not_worse": int(
                year_subset["seven_day_mae_not_worse"].sum()
            ),
            "crps_not_worse": int(year_subset["crps_not_worse"].sum()),
            "mean_brier_not_worse": int(
                year_subset["mean_brier_not_worse"].sum()
            ),
        }
        pooled_requirements = {
            "seven_day_mae_not_worse": bool(pooled_row["seven_day_mae_not_worse"]),
            "crps_not_worse": bool(pooled_row["crps_not_worse"]),
            "mean_brier_not_worse": bool(pooled_row["mean_brier_not_worse"]),
            "heavy_coverage_not_both_worse": bool(
                pooled_row["heavy_coverage_not_both_worse"]
            ),
            "upper_tail_numeric_audit_passed": bool(
                tail_summaries[candidate_id]["numeric_audit_passed"]
            ),
            "artifact_hash_and_leakage_checks_passed": bool(
                artifact_ok[candidate_id]
            ),
        }
        year_stability = bool(all(count >= 3 for count in year_pass_counts.values()))
        is_eligible = bool(all(pooled_requirements.values()) and year_stability)
        if is_eligible:
            eligible.append(candidate_id)
        candidate_results[candidate_id] = {
            "eligible": is_eligible,
            "pooled_requirements": pooled_requirements,
            "year_primary_metric_not_worse_counts": year_pass_counts,
            "year_stability_requirement_passed": year_stability,
            "upper_tail_audit": tail_summaries[candidate_id],
        }

    statuses = contract["decision_statuses"]
    if not eligible:
        decision_status = statuses["none_eligible"]
    elif len(eligible) == 1:
        decision_status = statuses["one_eligible"]
    else:
        decision_status = statuses["multiple_eligible"]
    gate = {
        "contract_id": contract["contract_id"],
        "contract_version": contract["contract_version"],
        "decision_status": decision_status,
        "eligible_candidate_set": sorted(eligible),
        "pareto_dominated_by": _pareto_relationship(eligible, pooled),
        "candidate_results": candidate_results,
        "post_hoc_weighted_score_used": False,
        "2019_used": False,
        "2024_used": False,
        "network_download_performed": False,
        "model_training_performed": False,
        "oof_member_rows_per_candidate": candidate_counts,
    }
    gate_path = output_dir / "training_cv_candidate_gate.json"
    gate_path.write_text(
        json.dumps(gate, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    report_path = output_dir / "training_cv_scope_and_conclusion.md"
    report_lines = [
        "# GEFS precipitation QM training-period cross-validation",
        "",
        "This offline analysis uses only the prelocked 2015-2018 data and four leave-one-year-out folds.",
        "It does not use 2019 or 2024, perform network downloads, train the surrogate model, or alter TTA.",
        "",
        f"- Decision status: `{decision_status}`.",
        f"- Eligible candidate set: `{', '.join(sorted(eligible)) or 'none'}`.",
        f"- OOF member rows per QM candidate: `{expected_oof}`.",
        f"- Fold artifacts: `{len(artifact_manifest)}`; all validation rows used for fit: `0`.",
        "",
    ]
    for row in pooled.sort_values("candidate_id").itertuples(index=False):
        report_lines.append(
            f"- `{row.candidate_id}`: 7-day MAE difference "
            f"`{row.seven_day_mae_difference_candidate_minus_raw_mm:.6f} mm`, "
            f"CRPS difference `{row.crps_difference_candidate_minus_raw_mm:.6f} mm`, "
            f"mean Brier difference `{row.mean_brier_difference_candidate_minus_raw:.6f}`."
        )
    report_lines.extend(
        [
            "",
            "Negative differences favor QM. Bootstrap intervals are descriptive and are not a hard gate.",
            "Multiple eligible candidates are retained as a set; no post-hoc weighted score is used.",
            "",
        ]
    )
    report_path.write_text("\n".join(report_lines), encoding="utf-8")

    return {
        "fold_assignment": assignment_path,
        "candidate_configurations": configuration_path,
        "fold_artifact_manifest": artifact_manifest_path,
        "oof_predictions": oof_path,
        "pooled_metrics": pooled_path,
        "year_metrics": year_path,
        "stratified_metrics": stratified_path,
        "bootstrap": bootstrap_path,
        "tail_events": tail_path,
        "tail_audit": tail_audit_path,
        "candidate_gate": gate_path,
        "report": report_path,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", type=Path, default=CONTRACT_PATH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    outputs = run_training_cv(contract_path=args.contract, output_dir=args.output_dir)
    print(json.dumps({key: str(value) for key, value in outputs.items()}, indent=2))


if __name__ == "__main__":
    main()
