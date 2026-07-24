"""Offline multiplicative quantile delta mapping for GEFS precipitation."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence
from typing import Any

import numpy as np
import pandas as pd


CONTRACT_ID = "gefs-precipitation-quantile-delta-mapping-offline-v1"
CONTRACT_VERSION = 1
DEFAULT_TRACE_THRESHOLD_MM = 0.05
DEFAULT_RANDOM_SEED = 20260718
SUPPORTED_GROUP_KEYS = {(), ("site_id",)}


def _artifact_hash(artifact: dict[str, Any]) -> str:
    payload = dict(artifact)
    payload.pop("artifact_sha256", None)
    encoded = json.dumps(
        payload,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _group_key(group_keys: Sequence[str], values: Sequence[Any]) -> str:
    if not group_keys:
        return "global"
    return "|".join(
        f"{column}={value}" for column, value in zip(group_keys, values, strict=True)
    )


def _validate_group_keys(group_keys: Sequence[str]) -> tuple[str, ...]:
    result = tuple(str(column) for column in group_keys)
    if result not in SUPPORTED_GROUP_KEYS:
        raise ValueError("QDM v1 supports only global or site_id grouping")
    return result


def _stable_uniform_for_zeros(
    frame: pd.DataFrame,
    values: np.ndarray,
    *,
    threshold_mm: float,
    random_seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    output = np.asarray(values, dtype=float).copy()
    zero = output == 0.0
    if not zero.any():
        return output, zero
    sort_columns = [
        column
        for column in (
            "site_id",
            "forecast_init_utc",
            "decision_date",
            "valid_date_utc",
            "lead_day",
            "gefs_member",
        )
        if column in frame.columns
    ]
    ordered_indices = (
        frame.reset_index(drop=True)
        .sort_values(sort_columns, kind="mergesort")
        .index.to_numpy()
        if sort_columns
        else np.arange(len(frame))
    )
    rng = np.random.default_rng(int(random_seed))
    ordered_zero = zero[ordered_indices]
    draws = rng.uniform(
        np.finfo(float).eps,
        float(threshold_mm),
        size=int(ordered_zero.sum()),
    )
    ordered_values = output[ordered_indices]
    ordered_values[ordered_zero] = draws
    output[ordered_indices] = ordered_values
    return output, zero


def _inverse_empirical(sorted_sample: np.ndarray, probabilities: np.ndarray) -> np.ndarray:
    return np.quantile(
        np.asarray(sorted_sample, dtype=float),
        np.asarray(probabilities, dtype=float),
        method="linear",
    )


def verify_qdm_artifact(artifact: dict[str, Any]) -> None:
    if artifact.get("contract_id") != CONTRACT_ID:
        raise ValueError("QDM artifact contract id mismatch")
    if artifact.get("contract_version") != CONTRACT_VERSION:
        raise ValueError("QDM artifact contract version mismatch")
    if artifact.get("target_cdf_mode") != "offline_complete_withheld_gefs_batch":
        raise ValueError("QDM artifact target CDF mode mismatch")
    _validate_group_keys(artifact.get("group_keys", []))
    expected = str(artifact.get("artifact_sha256", ""))
    if not expected or expected != _artifact_hash(artifact):
        raise ValueError("QDM artifact hash mismatch")
    if {2019, 2024}.intersection(set(artifact.get("fit_years", []))):
        raise ValueError("validation or test years found in QDM fit years")
    for key, group in artifact.get("groups", {}).items():
        model = np.asarray(group.get("historical_model_sorted_mm", []), dtype=float)
        reference = np.asarray(
            group.get("historical_reference_sorted_mm", []), dtype=float
        )
        if len(model) == 0 or len(reference) == 0:
            raise ValueError(f"empty historical QDM distribution for {key}")
        if np.any(~np.isfinite(model)) or np.any(~np.isfinite(reference)):
            raise ValueError(f"nonfinite historical QDM distribution for {key}")
        if np.any(model <= 0.0) or np.any(reference <= 0.0):
            raise ValueError(f"trace replacement failed for QDM group {key}")


def fit_offline_precipitation_qdm(
    frame: pd.DataFrame,
    *,
    fit_years: Sequence[int],
    group_keys: Sequence[str] = (),
    trace_threshold_mm: float = DEFAULT_TRACE_THRESHOLD_MM,
    random_seed: int = DEFAULT_RANDOM_SEED,
) -> dict[str, Any]:
    group_keys = _validate_group_keys(group_keys)
    required = {
        "site_id",
        "decision_date",
        "valid_date_utc",
        "lead_day",
        "gefs_member",
        "precipitation_mm_raw",
        "precipitation_mm_reference",
    }.union(group_keys)
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"missing QDM fit columns: {sorted(missing)}")
    if trace_threshold_mm <= 0.0:
        raise ValueError("trace threshold must be positive")
    data = frame.copy().reset_index(drop=True)
    data["decision_date"] = pd.to_datetime(data["decision_date"])
    data["valid_date_utc"] = pd.to_datetime(data["valid_date_utc"])
    actual_years = set(data["decision_date"].dt.year.astype(int))
    allowed_years = {int(year) for year in fit_years}
    if actual_years.difference(allowed_years):
        raise ValueError("QDM fit frame contains non-fit years")
    if {2019, 2024}.intersection(actual_years):
        raise ValueError("2019 or 2024 cannot be used to fit QDM")
    numeric = data[["precipitation_mm_raw", "precipitation_mm_reference"]].to_numpy(
        dtype=float
    )
    if np.any(~np.isfinite(numeric)) or np.any(numeric < 0.0):
        raise ValueError("QDM precipitation samples must be finite and nonnegative")
    duplicate_keys = [
        "site_id",
        "decision_date",
        "valid_date_utc",
        "gefs_member",
    ]
    if data.duplicated(duplicate_keys).any():
        raise ValueError("duplicate member rows in QDM fit frame")
    reference_keys = ["site_id", "decision_date", "valid_date_utc"]
    for key, group in data.groupby(reference_keys, sort=False):
        if group["precipitation_mm_reference"].nunique(dropna=False) != 1:
            raise ValueError(f"reference precipitation differs across members for {key}")

    groups: dict[str, dict[str, Any]] = {}
    grouped = [((), data)] if not group_keys else data.groupby(list(group_keys), sort=True)
    for raw_key, model_group in grouped:
        values = raw_key if isinstance(raw_key, tuple) else (raw_key,)
        key = _group_key(group_keys, values)
        reference_group = model_group.drop_duplicates(reference_keys).reset_index(drop=True)
        model_jittered, model_zero = _stable_uniform_for_zeros(
            model_group.reset_index(drop=True),
            model_group["precipitation_mm_raw"].to_numpy(dtype=float),
            threshold_mm=trace_threshold_mm,
            random_seed=random_seed + 101,
        )
        reference_jittered, reference_zero = _stable_uniform_for_zeros(
            reference_group,
            reference_group["precipitation_mm_reference"].to_numpy(dtype=float),
            threshold_mm=trace_threshold_mm,
            random_seed=random_seed + 211,
        )
        groups[key] = {
            "group_values": {
                column: str(value)
                for column, value in zip(group_keys, values, strict=True)
            },
            "historical_model_sample_count": int(len(model_jittered)),
            "historical_reference_sample_count": int(len(reference_jittered)),
            "historical_model_zero_count": int(model_zero.sum()),
            "historical_reference_zero_count": int(reference_zero.sum()),
            "historical_model_sorted_mm": np.sort(model_jittered).tolist(),
            "historical_reference_sorted_mm": np.sort(reference_jittered).tolist(),
        }
    artifact: dict[str, Any] = {
        "contract_id": CONTRACT_ID,
        "contract_version": CONTRACT_VERSION,
        "fit_years": sorted(allowed_years),
        "site_ids": sorted(data["site_id"].astype(str).unique()),
        "group_keys": list(group_keys),
        "trace_threshold_mm": float(trace_threshold_mm),
        "random_seed": int(random_seed),
        "zero_handling": "seeded_uniform_jitter_below_trace_then_recensor",
        "change_type": "multiplicative_relative",
        "ratio_cap": None,
        "target_cdf_mode": "offline_complete_withheld_gefs_batch",
        "realtime_deployment_allowed": False,
        "groups": groups,
    }
    artifact["artifact_sha256"] = _artifact_hash(artifact)
    verify_qdm_artifact(artifact)
    return artifact


def apply_offline_precipitation_qdm(
    frame: pd.DataFrame,
    artifact: dict[str, Any],
    *,
    split: str,
) -> pd.DataFrame:
    verify_qdm_artifact(artifact)
    group_keys = tuple(artifact["group_keys"])
    required = {"precipitation_mm_raw"}.union(group_keys)
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"missing QDM apply columns: {sorted(missing)}")
    data = frame.copy().reset_index(drop=True)
    raw = data["precipitation_mm_raw"].to_numpy(dtype=float)
    if np.any(~np.isfinite(raw)) or np.any(raw < 0.0):
        raise ValueError("QDM target precipitation must be finite and nonnegative")
    data["precipitation_mm_qdm"] = np.nan
    data["qdm_nonexceedance_probability"] = np.nan
    data["qdm_relative_quantile_change"] = np.nan
    data["qdm_trace_censored_input"] = False
    data["qdm_trace_censored_output"] = False
    data["qdm_group"] = ""
    grouped = [((), data.index)] if not group_keys else data.groupby(list(group_keys), sort=True).groups.items()
    for raw_key, indices in grouped:
        values = raw_key if isinstance(raw_key, tuple) else (raw_key,)
        key = _group_key(group_keys, values)
        if key not in artifact["groups"]:
            raise ValueError(f"no historical QDM group for {key}")
        positions = np.asarray(list(indices), dtype=int)
        target_group = data.loc[positions].reset_index(drop=True)
        target, target_zero = _stable_uniform_for_zeros(
            target_group,
            target_group["precipitation_mm_raw"].to_numpy(dtype=float),
            threshold_mm=float(artifact["trace_threshold_mm"]),
            random_seed=int(artifact["random_seed"]) + 307,
        )
        ranks = pd.Series(target).rank(method="average").to_numpy(dtype=float)
        probabilities = (ranks - 0.5) / float(len(target))
        historical = artifact["groups"][key]
        historical_model = _inverse_empirical(
            np.asarray(historical["historical_model_sorted_mm"], dtype=float),
            probabilities,
        )
        historical_reference = _inverse_empirical(
            np.asarray(historical["historical_reference_sorted_mm"], dtype=float),
            probabilities,
        )
        relative_delta = target / historical_model
        corrected = historical_reference * relative_delta
        censored_output = corrected < float(artifact["trace_threshold_mm"])
        corrected[censored_output] = 0.0
        if np.any(~np.isfinite(corrected)) or np.any(corrected < 0.0):
            raise ValueError(f"QDM produced invalid precipitation for {key}")
        data.loc[positions, "precipitation_mm_qdm"] = corrected
        data.loc[positions, "qdm_nonexceedance_probability"] = probabilities
        data.loc[positions, "qdm_relative_quantile_change"] = relative_delta
        data.loc[positions, "qdm_trace_censored_input"] = target_zero
        data.loc[positions, "qdm_trace_censored_output"] = censored_output
        data.loc[positions, "qdm_group"] = key
    if data["precipitation_mm_qdm"].isna().any():
        raise ValueError("QDM left missing corrected values")
    data["split"] = str(split)
    data["qdm_artifact_sha256"] = artifact["artifact_sha256"]
    return data


def apply_current_cycle_precipitation_qdm(
    frame: pd.DataFrame,
    artifact: dict[str, Any],
    *,
    split: str,
    cycle_column: str = "decision_date",
    expected_rows_per_cycle: int | None = None,
) -> pd.DataFrame:
    """Apply QDM with a target CDF built independently from each visible cycle."""
    if cycle_column not in frame.columns:
        raise ValueError(f"missing QDM cycle column: {cycle_column}")
    data = frame.copy().reset_index(drop=True)
    data[cycle_column] = pd.to_datetime(data[cycle_column])
    corrected_parts = []
    for cycle, cycle_frame in data.groupby(cycle_column, sort=True):
        if expected_rows_per_cycle is not None and len(cycle_frame) != int(
            expected_rows_per_cycle
        ):
            raise ValueError(
                f"cycle {cycle} rows={len(cycle_frame)}, "
                f"expected={expected_rows_per_cycle}"
            )
        corrected = apply_offline_precipitation_qdm(
            cycle_frame,
            artifact,
            split=split,
        )
        corrected["qdm_target_cdf_mode"] = "causal_current_cycle_global_batch"
        corrected["qdm_target_cdf_cycle"] = pd.Timestamp(cycle)
        corrected["qdm_target_cdf_sample_count"] = int(len(cycle_frame))
        corrected_parts.append(corrected)
    if not corrected_parts:
        raise ValueError("QDM current-cycle input is empty")
    return pd.concat(corrected_parts, ignore_index=True)
