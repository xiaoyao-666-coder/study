#!/usr/bin/env python3
"""Validate the frozen v2 QM on the independent 2019 confirmation dates."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from run_gefs_quantile_mapping_validation_v1 import (
    SITE_IDS,
    _deterministic_metrics,
    _probabilistic_metrics,
    _promotion_gate,
    _seven_day_metrics,
    _validation_long,
    _write_csv,
)
from run_gefs_quantile_mapping_validation_v2 import _upper_tail_audit
from s2s_rtist.weather.gefs_ensemble_validation import (
    aggregate_probabilistic_metrics,
    compute_precipitation_probability_metrics,
)
from s2s_rtist.weather.gefs_quantile_mapping import (
    CONTRACT_ID_V2,
    GEFS_REFORECAST_MEMBERS,
    apply_empirical_precipitation_qm,
    pair_member_and_reference,
    read_quantile_mapping_artifact,
    validate_member_daily_precipitation,
    validate_reference_daily_precipitation,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONTRACT_PATH = (
    PROJECT_ROOT
    / "site_general_surrogate_eval"
    / "gefs_quantile_mapping_2019_confirmation_contract_v1.json"
)
DEFAULT_INPUT_DIR = (
    PROJECT_ROOT
    / "site_general_surrogate_eval"
    / "gefs_quantile_mapping_v2"
    / "gefs_qm_2019_confirmation_v1"
)
DEFAULT_OUTPUT_DIR = DEFAULT_INPUT_DIR
DEFAULT_PILOT_DIR = (
    PROJECT_ROOT
    / "site_general_surrogate_eval"
    / "gefs_quantile_mapping_v2"
    / "gefs_qm_2015_2019_pilot_v2"
)
BOOTSTRAP_REPLICATES = 10_000
BOOTSTRAP_SEED = 20260717


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_contract() -> dict[str, object]:
    contract = json.loads(CONTRACT_PATH.read_text(encoding="utf-8"))
    if contract["parent_contract"]["contract_id"] != CONTRACT_ID_V2:
        raise ValueError("confirmation parent contract mismatch")
    return contract


def _load_dates(contract: dict[str, object]) -> tuple[str, ...]:
    selection = set(contract["strategy_selection_dates_2019"])
    dates = tuple(contract["independent_confirmation_dates_2019"])
    if selection.intersection(dates):
        raise ValueError("confirmation dates overlap strategy selection dates")
    return dates


def _rename_for_metrics(frame: pd.DataFrame) -> pd.DataFrame:
    return frame.rename(columns={"valid_date_utc": "local_date"})


def _month_deterministic_metrics(observations: pd.DataFrame) -> pd.DataFrame:
    rows = []
    work = observations.copy()
    work["month"] = pd.to_datetime(work["decision_date"]).dt.month
    for (method, month), group in work.groupby(["method", "month"], sort=True):
        error = group["ensemble_mean"].to_numpy(dtype=float) - group[
            "reference_value"
        ].to_numpy(dtype=float)
        rows.append(
            {
                "metric_family": "deterministic",
                "method": method,
                "scope": "month",
                "scope_value": str(month),
                "n_observations": int(len(group)),
                "bias": float(error.mean()),
                "mae": float(np.abs(error).mean()),
                "rmse": float(np.sqrt(np.mean(error * error))),
                "wet_day_frequency_error": float(
                    group["ensemble_mean"].ge(0.1).mean()
                    - group["reference_value"].ge(0.1).mean()
                ),
            }
        )
    return pd.DataFrame(rows)


def _month_probabilistic_metrics(observations: pd.DataFrame) -> pd.DataFrame:
    work = observations.copy()
    work["month"] = pd.to_datetime(work["decision_date"]).dt.month
    result = aggregate_probabilistic_metrics(
        work, group_columns=("method", "month")
    )
    result["metric_family"] = "probabilistic"
    result["scope"] = "month"
    result["scope_value"] = result["month"].astype(str)
    return result


def _stratified_metrics(
    observations: pd.DataFrame, probabilistic: pd.DataFrame
) -> pd.DataFrame:
    deterministic = _deterministic_metrics(observations).copy()
    deterministic["metric_family"] = "deterministic"
    deterministic = pd.concat(
        [deterministic, _month_deterministic_metrics(observations)],
        ignore_index=True,
        sort=False,
    )
    probability = probabilistic.copy()
    probability["metric_family"] = "probabilistic"
    probability = pd.concat(
        [probability, _month_probabilistic_metrics(observations)],
        ignore_index=True,
        sort=False,
    )
    return pd.concat([deterministic, probability], ignore_index=True, sort=False)


def _cycle_metric_differences(
    metric_input: pd.DataFrame,
) -> dict[str, pd.DataFrame]:
    observations, _, _ = _probabilistic_metrics(
        metric_input, members=GEFS_REFORECAST_MEMBERS
    )
    crps = (
        observations.groupby(["method", "decision_date"], as_index=False)["crps"]
        .mean()
        .pivot(index="decision_date", columns="method", values="crps")
    )
    crps["difference_qm_minus_raw"] = crps["GEFS_QM"] - crps["GEFS_raw"]

    brier_parts = []
    for method, forecast_column in (
        ("GEFS_raw", "precipitation_mm_raw"),
        ("GEFS_QM", "precipitation_mm_qm"),
    ):
        long = _validation_long(
            metric_input, method=method, forecast_column=forecast_column
        )
        brier = compute_precipitation_probability_metrics(
            long,
            expected_members=GEFS_REFORECAST_MEMBERS,
            group_columns=("decision_date",),
        )
        brier = brier.groupby("decision_date", as_index=True)["brier_score"].mean()
        brier_parts.append(brier.rename(method))
    brier_frame = pd.concat(brier_parts, axis=1)
    brier_frame["difference_qm_minus_raw"] = (
        brier_frame["GEFS_QM"] - brier_frame["GEFS_raw"]
    )

    seven_day = _seven_day_metrics(metric_input)
    seven = (
        seven_day.groupby(["method", "decision_date"], as_index=False)[
            "absolute_error_7d_mm"
        ]
        .mean()
        .pivot(index="decision_date", columns="method", values="absolute_error_7d_mm")
    )
    seven["difference_qm_minus_raw"] = seven["GEFS_QM"] - seven["GEFS_raw"]
    return {"crps": crps, "mean_brier": brier_frame, "seven_day_mae": seven}


def _bootstrap_summary(metric_differences: dict[str, pd.DataFrame]) -> pd.DataFrame:
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    rows = []
    for metric_name, frame in metric_differences.items():
        differences = frame["difference_qm_minus_raw"].to_numpy(dtype=float)
        if len(differences) != 12:
            raise ValueError(f"bootstrap requires 12 confirmation cycles for {metric_name}")
        sample_indices = rng.integers(
            0, len(differences), size=(BOOTSTRAP_REPLICATES, len(differences))
        )
        bootstrap_means = differences[sample_indices].mean(axis=1)
        rows.append(
            {
                "metric": metric_name,
                "cycle_count": int(len(differences)),
                "point_difference_qm_minus_raw": float(differences.mean()),
                "bootstrap_ci_lower_95": float(np.quantile(bootstrap_means, 0.025)),
                "bootstrap_ci_upper_95": float(np.quantile(bootstrap_means, 0.975)),
                "bootstrap_replicates": BOOTSTRAP_REPLICATES,
                "random_seed": BOOTSTRAP_SEED,
                "fraction_bootstrap_improved": float((bootstrap_means <= 0.0).mean()),
            }
        )
    return pd.DataFrame(rows)


def _combined_descriptive_metrics(
    confirmation: pd.DataFrame,
    pilot_dir: Path,
) -> pd.DataFrame:
    pilot_path = pilot_dir / "paired_raw_and_qm_members_2019_v2.csv"
    if not pilot_path.is_file():
        raise FileNotFoundError(f"missing pilot v2 paired table: {pilot_path}")
    pilot = pd.read_csv(pilot_path, parse_dates=["forecast_init_utc", "valid_date_utc"])
    combined = pd.concat([pilot, confirmation], ignore_index=True, sort=False)
    metric_input = _rename_for_metrics(combined)
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
    rows = []
    for method in ("GEFS_raw", "GEFS_QM"):
        rows.extend(
            [
                {
                    "population": "combined_18_cycle_descriptive",
                    "method": method,
                    "metric": "seven_day_mae_mm",
                    "value": gate[f"{method.lower().replace('gefs_', '')}_seven_day_mae_mm"],
                },
                {
                    "population": "combined_18_cycle_descriptive",
                    "method": method,
                    "metric": "mean_crps_mm",
                    "value": gate[f"{method.lower().replace('gefs_', '')}_mean_crps_mm"],
                },
                {
                    "population": "combined_18_cycle_descriptive",
                    "method": method,
                    "metric": "mean_brier_score",
                    "value": gate[f"{method.lower().replace('gefs_', '')}_mean_brier_score"],
                },
            ]
        )
    return pd.DataFrame(rows)


def run_confirmation(
    *, input_dir: Path, output_dir: Path, pilot_dir: Path
) -> dict[str, Path]:
    contract = _load_contract()
    dates = _load_dates(contract)
    expected = contract["expected_counts"]
    member_path = input_dir / "gefs_reforecast_member_daily_precipitation_utc_2019_confirmation.csv"
    reference_path = input_dir / "era5_reference_daily_precipitation_utc_2019_confirmation.csv"
    manifest_path = input_dir / "gefs_reforecast_download_manifest_2019_confirmation.csv"
    frozen_manifest_path = input_dir / "frozen_qm_artifact_load_manifest.json"
    if not all(path.is_file() for path in (member_path, reference_path, manifest_path, frozen_manifest_path)):
        raise FileNotFoundError("confirmation extraction outputs are incomplete")

    member = pd.read_csv(member_path, parse_dates=["forecast_init_utc", "valid_date_utc"])
    reference = pd.read_csv(reference_path, parse_dates=["valid_date_utc"])
    manifest = pd.read_csv(manifest_path)
    frozen_manifest = json.loads(frozen_manifest_path.read_text(encoding="utf-8"))
    if frozen_manifest["artifact_sha256"] != contract["frozen_mapping"]["artifact_sha256"]:
        raise ValueError("confirmation frozen artifact manifest hash mismatch")
    if frozen_manifest["refit_performed"] or frozen_manifest["strategy_change_performed"]:
        raise ValueError("confirmation run reports refit or strategy change")
    if len(manifest) != expected["member_archive_tasks"]:
        raise ValueError("confirmation manifest task count does not match contract")
    network_fallback = manifest["network_fallback_used"].astype(str).str.lower().eq("true")
    member_fallback = manifest["member_fallback_used"].astype(str).str.lower().eq("true")
    if network_fallback.any() or member_fallback.any():
        raise ValueError("confirmation manifest reports a fallback")
    if set(manifest["maximum_end_hour"].astype(int)) != {168}:
        raise ValueError("confirmation manifest has a non-168 end hour")
    if set(manifest["expected_selected_message_count"].astype(int)) != {56}:
        raise ValueError("confirmation manifest has a non-56 message task")
    if set(manifest["selected_message_count"].astype(int)) != {56}:
        raise ValueError("confirmation manifest actual message count is not 56")
    if set(manifest["selected_end_step"].astype(int)) != {168}:
        raise ValueError("confirmation manifest actual end step is not 168")
    if len(member) != expected["confirmation_member_rows"]:
        raise ValueError("confirmation member rows do not match contract")
    if len(reference) != expected["confirmation_unique_reference_observations"]:
        raise ValueError("confirmation reference rows do not match contract")
    validate_member_daily_precipitation(
        member,
        expected_sites=SITE_IDS,
        expected_members=GEFS_REFORECAST_MEMBERS,
        expected_cycles=dates,
        date_column="valid_date_utc",
    )
    validate_reference_daily_precipitation(
        reference,
        expected_sites=SITE_IDS,
        expected_dates=sorted(reference["valid_date_utc"].unique()),
        date_column="valid_date_utc",
    )

    frozen_path = PROJECT_ROOT / contract["frozen_mapping"]["artifact_relative_path"]
    artifact = read_quantile_mapping_artifact(
        frozen_path, expected_contract_id=CONTRACT_ID_V2
    )
    if artifact["artifact_sha256"] != contract["frozen_mapping"]["artifact_sha256"]:
        raise ValueError("loaded frozen artifact hash differs from contract")
    paired = pair_member_and_reference(
        member, reference, date_column="valid_date_utc"
    )
    paired["confirmation_role"] = "independent_confirmation_2019"
    corrected = apply_empirical_precipitation_qm(
        paired, artifact, split="independent_confirmation_2019"
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    paired_path = output_dir / "paired_raw_and_qm_members_2019_confirmation.csv"
    _write_csv(corrected, paired_path)

    metric_input = _rename_for_metrics(corrected)
    observations, probabilistic, probabilities = _probabilistic_metrics(
        metric_input, members=GEFS_REFORECAST_MEMBERS
    )
    deterministic = _deterministic_metrics(observations)
    seven_day = _seven_day_metrics(metric_input)
    gate = _promotion_gate(
        observations=observations,
        probabilities=probabilities,
        seven_day=seven_day,
        paired=metric_input,
    )
    tail_events, tail_audit = _upper_tail_audit(corrected)
    metric_differences = _cycle_metric_differences(metric_input)
    bootstrap = _bootstrap_summary(metric_differences)
    stratified = _stratified_metrics(observations, probabilistic)
    combined = _combined_descriptive_metrics(corrected, pilot_dir)

    retained_grib_count = len(list((input_dir / "cache" / "minigrib").glob("*.grib2")))
    frozen_ok = artifact["artifact_sha256"] == contract["frozen_mapping"]["artifact_sha256"]
    dates_ok = not set(dates).intersection(contract["strategy_selection_dates_2019"])
    gate.update(
        {
            "contract_id": contract["contract_id"],
            "contract_version": contract["contract_version"],
            "confirmation_dates": list(dates),
            "strategy_selection_dates_excluded": contract[
                "strategy_selection_dates_2019"
            ],
            "frozen_artifact_hash_match": frozen_ok,
            "confirmation_dates_disjoint": dates_ok,
            "retained_grib_file_count": retained_grib_count,
            "upper_tail_numeric_audit": tail_audit,
            "upper_tail_review_status": tail_audit["audit_status"],
        }
    )
    gate["automatic_requirements_passed"] = bool(
        gate["automatic_requirements_passed"]
        and frozen_ok
        and dates_ok
        and retained_grib_count == 0
        and tail_audit["numeric_audit_passed"]
    )
    gate["promotion_status"] = (
        "passed_independent_2019_confirmation"
        if gate["automatic_requirements_passed"]
        else "failed_independent_2019_confirmation"
    )

    outputs = {
        "deterministic": output_dir / "deterministic_metrics_2019_confirmation.csv",
        "probabilistic": output_dir / "probabilistic_metrics_2019_confirmation.csv",
        "probability": output_dir / "precipitation_probability_2019_confirmation.csv",
        "seven_day": output_dir / "seven_day_precipitation_2019_confirmation.csv",
        "stratified": output_dir / "stratified_metrics_2019_confirmation.csv",
        "bootstrap": output_dir / "paired_cycle_block_bootstrap_2019_confirmation.csv",
        "tail_events": output_dir / "upper_tail_events_2019_confirmation.csv",
        "tail_summary": output_dir / "upper_tail_audit_summary_2019_confirmation.json",
        "gate": output_dir / "promotion_gate_2019_confirmation.json",
        "combined": output_dir / "combined_18_cycle_descriptive_metrics.csv",
        "report": output_dir / "validation_scope_and_conclusion_2019_confirmation.md",
    }
    _write_csv(deterministic, outputs["deterministic"])
    _write_csv(probabilistic, outputs["probabilistic"])
    _write_csv(probabilities, outputs["probability"])
    _write_csv(seven_day, outputs["seven_day"])
    _write_csv(stratified, outputs["stratified"])
    _write_csv(bootstrap, outputs["bootstrap"])
    _write_csv(tail_events, outputs["tail_events"])
    _write_csv(combined, outputs["combined"])
    outputs["tail_summary"].write_text(
        json.dumps(tail_audit, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    outputs["gate"].write_text(
        json.dumps(gate, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    outputs["report"].write_text(
        "\n".join(
            [
                "# GEFS QM v2 independent 2019 confirmation",
                "",
                "Primary metrics use only the 12 dates locked in the confirmation contract.",
                "The six strategy-selection dates are excluded from the primary population.",
                "",
                f"- Confirmation member rows: `{len(member)}`.",
                f"- Confirmation reference observations: `{len(reference)}`.",
                f"- Frozen artifact SHA-256: `{artifact['artifact_sha256']}`.",
                f"- Raw 7-day MAE: `{gate['raw_seven_day_mae_mm']:.6f} mm`.",
                f"- QM 7-day MAE: `{gate['qm_seven_day_mae_mm']:.6f} mm`.",
                f"- Raw mean CRPS: `{gate['raw_mean_crps_mm']:.6f} mm`.",
                f"- QM mean CRPS: `{gate['qm_mean_crps_mm']:.6f} mm`.",
                f"- Raw mean Brier: `{gate['raw_mean_brier_score']:.6f}`.",
                f"- QM mean Brier: `{gate['qm_mean_brier_score']:.6f}`.",
                f"- Promotion status: `{gate['promotion_status']}`.",
                "",
                "Combined 18-cycle metrics are descriptive only. A failure on the confirmation dates must not be repaired by tuning on the same dates.",
                "",
            ]
        ),
        encoding="utf-8",
    )
    return outputs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--pilot-dir", type=Path, default=DEFAULT_PILOT_DIR)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    outputs = run_confirmation(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        pilot_dir=args.pilot_dir,
    )
    print(json.dumps({key: str(value) for key, value in outputs.items()}, indent=2))


if __name__ == "__main__":
    main()
