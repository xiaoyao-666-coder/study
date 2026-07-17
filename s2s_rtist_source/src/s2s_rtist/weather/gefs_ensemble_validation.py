"""Probabilistic validation metrics for member-level GEFS forecasts."""

from __future__ import annotations

from collections.abc import Sequence
from math import sqrt

import numpy as np
import pandas as pd

from .gefs_gridmet_bias import gefs_members


OBSERVATION_KEYS = (
    "site",
    "decision_date",
    "local_date",
    "lead_day",
    "variable",
)


def _validated_member_rows(
    frame: pd.DataFrame, expected_members: Sequence[str]
) -> pd.DataFrame:
    required = {
        *OBSERVATION_KEYS,
        "gefs_member",
        "forecast_value",
        "reference_value",
    }
    missing_columns = required.difference(frame.columns)
    if missing_columns:
        raise ValueError(f"missing required columns: {sorted(missing_columns)}")

    expected = tuple(str(member) for member in expected_members)
    if not expected or len(expected) != len(set(expected)):
        raise ValueError("expected_members must contain unique member names")

    data = frame.copy()
    data["gefs_member"] = data["gefs_member"].astype(str)
    for column in ("forecast_value", "reference_value"):
        data[column] = pd.to_numeric(data[column], errors="coerce")
        if data[column].isna().any():
            raise ValueError(f"column {column!r} contains missing values")

    duplicate = data.duplicated([*OBSERVATION_KEYS, "gefs_member"], keep=False)
    if duplicate.any():
        example = data.loc[
            duplicate, [*OBSERVATION_KEYS, "gefs_member"]
        ].iloc[0].to_dict()
        raise ValueError(f"duplicate GEFS member forecast: {example}")

    expected_set = set(expected)
    for observation_key, group in data.groupby(
        list(OBSERVATION_KEYS), sort=False, dropna=False
    ):
        actual_set = set(group["gefs_member"])
        if actual_set != expected_set:
            missing = sorted(expected_set.difference(actual_set))
            extra = sorted(actual_set.difference(expected_set))
            raise ValueError(
                "incomplete GEFS member set for "
                f"{observation_key}: missing={missing}, extra={extra}"
            )
        if group["reference_value"].nunique(dropna=False) != 1:
            raise ValueError(
                f"reference value differs across members for {observation_key}"
            )
    return data


def _ensemble_crps(values: np.ndarray, reference: float) -> float:
    absolute_error = np.abs(values - float(reference)).mean()
    pairwise_distance = np.abs(values[:, None] - values[None, :]).mean()
    return float(absolute_error - 0.5 * pairwise_distance)


def summarize_ensemble_observations(
    frame: pd.DataFrame,
    *,
    expected_members: Sequence[str] | None = None,
) -> pd.DataFrame:
    """Create one probabilistic summary row per forecast observation."""

    members = tuple(expected_members) if expected_members is not None else gefs_members()
    data = _validated_member_rows(frame, members)
    rows: list[dict[str, object]] = []
    for observation_key, group in data.groupby(
        list(OBSERVATION_KEYS), sort=False, dropna=False
    ):
        values = group["forecast_value"].to_numpy(dtype=float)
        reference = float(group["reference_value"].iloc[0])
        p10, p25, median, p75, p90 = np.quantile(
            values, [0.10, 0.25, 0.50, 0.75, 0.90]
        )
        row = dict(zip(OBSERVATION_KEYS, observation_key))
        row.update(
            {
                "reference_value": reference,
                "member_count": int(len(values)),
                "ensemble_mean": float(values.mean()),
                "ensemble_std": float(values.std(ddof=0)),
                "ensemble_min": float(values.min()),
                "ensemble_p10": float(p10),
                "ensemble_p25": float(p25),
                "ensemble_median": float(median),
                "ensemble_p75": float(p75),
                "ensemble_p90": float(p90),
                "ensemble_max": float(values.max()),
                "crps": _ensemble_crps(values, reference),
                "covered_by_p10_p90": bool(p10 <= reference <= p90),
                "covered_by_min_max": bool(values.min() <= reference <= values.max()),
            }
        )
        rows.append(row)
    return pd.DataFrame(rows)


def aggregate_probabilistic_metrics(
    observations: pd.DataFrame, *, group_columns: Sequence[str]
) -> pd.DataFrame:
    """Aggregate ensemble-mean errors, spread, CRPS, and interval coverage."""

    required = {
        *group_columns,
        "reference_value",
        "ensemble_mean",
        "ensemble_std",
        "crps",
        "covered_by_p10_p90",
        "covered_by_min_max",
    }
    missing = required.difference(observations.columns)
    if missing:
        raise ValueError(f"missing probabilistic metric columns: {sorted(missing)}")
    if not group_columns:
        raise ValueError("group_columns must not be empty")

    work = observations.copy()
    work["ensemble_mean_error"] = work["ensemble_mean"] - work["reference_value"]
    rows: list[dict[str, object]] = []
    for keys, group in work.groupby(list(group_columns), sort=True, dropna=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        error = group["ensemble_mean_error"].to_numpy(dtype=float)
        row = dict(zip(group_columns, keys))
        row.update(
            {
                "n_observations": int(len(group)),
                "ensemble_mean_bias": float(error.mean()),
                "ensemble_mean_mae": float(np.abs(error).mean()),
                "ensemble_mean_rmse": sqrt(float(np.mean(error * error))),
                "mean_ensemble_spread": float(group["ensemble_std"].mean()),
                "mean_crps": float(group["crps"].mean()),
                "p10_p90_coverage": float(group["covered_by_p10_p90"].mean()),
                "min_max_coverage": float(group["covered_by_min_max"].mean()),
            }
        )
        rows.append(row)
    return pd.DataFrame(rows)


def compute_precipitation_probability_metrics(
    frame: pd.DataFrame,
    *,
    thresholds_mm: Sequence[float] = (1.0, 5.0, 10.0, 20.0),
    expected_members: Sequence[str] | None = None,
    group_columns: Sequence[str] = (),
) -> pd.DataFrame:
    """Compute Brier scores for member-derived precipitation probabilities."""

    members = tuple(expected_members) if expected_members is not None else gefs_members()
    data = _validated_member_rows(frame, members)
    data = data.loc[data["variable"].eq("precipitation_mm")].copy()
    if data.empty:
        raise ValueError("no precipitation member forecasts were provided")

    event_rows: list[dict[str, object]] = []
    for observation_key, group in data.groupby(
        list(OBSERVATION_KEYS), sort=False, dropna=False
    ):
        reference = float(group["reference_value"].iloc[0])
        forecasts = group["forecast_value"].to_numpy(dtype=float)
        key_values = dict(zip(OBSERVATION_KEYS, observation_key))
        for threshold in thresholds_mm:
            probability = float(np.mean(forecasts >= float(threshold)))
            observed = float(reference >= float(threshold))
            event_rows.append(
                {
                    **key_values,
                    "threshold_mm": float(threshold),
                    "forecast_probability": probability,
                    "observed_event": observed,
                    "brier_error": (probability - observed) ** 2,
                }
            )

    events = pd.DataFrame(event_rows)
    grouping = ["threshold_mm", *group_columns]
    rows: list[dict[str, object]] = []
    for keys, group in events.groupby(grouping, sort=True, dropna=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        row = dict(zip(grouping, keys))
        row.update(
            {
                "n_observations": int(len(group)),
                "observed_event_rate": float(group["observed_event"].mean()),
                "mean_forecast_probability": float(
                    group["forecast_probability"].mean()
                ),
                "brier_score": float(group["brier_error"].mean()),
            }
        )
        rows.append(row)
    return pd.DataFrame(rows)
