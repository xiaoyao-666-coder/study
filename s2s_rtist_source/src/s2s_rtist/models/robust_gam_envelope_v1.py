from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Sequence

import numpy as np
import pandas as pd


ArrayFunction = Callable[[np.ndarray], np.ndarray]


@dataclass(frozen=True)
class EnvelopeOptimizationResult:
    raw_irrigation_mm: float
    irrigation_mm: float
    robust_incremental_gain: float
    mean_incremental_gain: float
    minimum_model_index: int
    candidate_count: int
    stationary_point_count: int
    pairwise_intersection_count: int
    dense_diagnostic_irrigation_mm: float
    dense_diagnostic_robust_gain: float
    dense_diagnostic_gain_gap: float


def _evaluate(function: ArrayFunction, values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    result = np.asarray(function(values), dtype=np.float64)
    if result.shape != values.shape or not np.isfinite(result).all():
        raise ValueError("objective returned nonfinite or misaligned values")
    return result


def _unique_sorted(values: Sequence[float], tolerance: float = 1.0e-8) -> np.ndarray:
    ordered = np.asarray(sorted(float(value) for value in values), dtype=np.float64)
    if len(ordered) == 0:
        return ordered
    keep = np.ones(len(ordered), dtype=bool)
    keep[1:] = np.diff(ordered) > tolerance
    return ordered[keep]


def _polynomial_on_interval(
    function: ArrayFunction,
    lower: float,
    upper: float,
    reconstruction_tolerance: float,
) -> np.ndarray:
    fractions = np.asarray([0.0, 1.0 / 3.0, 2.0 / 3.0, 1.0], dtype=np.float64)
    samples = lower + (upper - lower) * fractions
    values = _evaluate(function, samples)
    coefficients = np.polynomial.polynomial.polyfit(fractions, values, deg=3)
    reconstructed = np.polynomial.polynomial.polyval(fractions, coefficients)
    residual = float(np.max(np.abs(reconstructed - values)))
    if residual > reconstruction_tolerance:
        raise ValueError(
            "objective is not cubic on the frozen spline interval: "
            f"residual={residual:.6g}"
        )
    return coefficients


def _roots_in_unit_interval(
    coefficients: np.ndarray,
    tolerance: float = 1.0e-8,
) -> list[float]:
    roots = np.polynomial.polynomial.polyroots(np.asarray(coefficients, dtype=np.float64))
    values: list[float] = []
    for root in roots:
        if abs(float(np.imag(root))) > tolerance:
            continue
        value = float(np.real(root))
        if -tolerance <= value <= 1.0 + tolerance:
            values.append(float(np.clip(value, 0.0, 1.0)))
    return values


def _select_lower_envelope(
    candidates: np.ndarray,
    objectives: Sequence[ArrayFunction],
    deployment_resolution_mm: float,
) -> tuple[float, float, float, float, int]:
    values = np.vstack([_evaluate(function, candidates) for function in objectives])
    robust = np.min(values, axis=0)
    maximum = float(np.max(robust))
    tied = np.isclose(robust, maximum, rtol=1.0e-12, atol=1.0e-12)
    raw_index = int(np.flatnonzero(tied)[0])
    raw_irrigation = float(candidates[raw_index])
    raw_gain = float(robust[raw_index])
    deployed = float(
        np.round(raw_irrigation / deployment_resolution_mm) * deployment_resolution_mm
    )
    deployed = float(np.clip(deployed, candidates[0], candidates[-1]))
    deployed_values = np.asarray(
        [_evaluate(function, np.asarray([deployed]))[0] for function in objectives],
        dtype=np.float64,
    )
    minimum_index = int(np.argmin(deployed_values))
    return (
        raw_irrigation,
        deployed,
        raw_gain,
        float(np.min(deployed_values)),
        minimum_index,
    )


def optimize_piecewise_cubic_lower_envelope(
    *,
    objectives: Sequence[ArrayFunction],
    interval_boundaries: np.ndarray,
    deployment_resolution_mm: float,
    dense_step_mm: float,
    reconstruction_tolerance: float = 1.0e-8,
    dense_gain_gap_tolerance: float = 0.05,
) -> EnvelopeOptimizationResult:
    """Maximize the lower envelope of piecewise cubic objective curves.

    The dense grid is evaluated only after the analytic solution and is never
    included in the recommendation candidate set.
    """
    if not objectives:
        raise ValueError("at least one objective is required")
    boundaries = np.asarray(interval_boundaries, dtype=np.float64).reshape(-1)
    if len(boundaries) < 2 or not np.isfinite(boundaries).all() or np.any(np.diff(boundaries) <= 0.0):
        raise ValueError("interval boundaries must be finite and strictly increasing")
    if deployment_resolution_mm <= 0.0 or dense_step_mm <= 0.0:
        raise ValueError("optimization resolutions must be positive")

    candidates: list[float] = [float(value) for value in boundaries]
    stationary_count = 0
    pairwise_count = 0
    for lower, upper in zip(boundaries[:-1], boundaries[1:]):
        width = float(upper - lower)
        interval_coefficients = [
            _polynomial_on_interval(function, float(lower), float(upper), reconstruction_tolerance)
            for function in objectives
        ]
        for coefficients in interval_coefficients:
            derivative = np.polynomial.polynomial.polyder(coefficients)
            roots = _roots_in_unit_interval(derivative)
            stationary_count += len(roots)
            candidates.extend(float(lower + width * root) for root in roots)
        for left_index in range(len(interval_coefficients)):
            for right_index in range(left_index + 1, len(interval_coefficients)):
                difference = interval_coefficients[left_index] - interval_coefficients[right_index]
                roots = _roots_in_unit_interval(difference)
                pairwise_count += len(roots)
                candidates.extend(float(lower + width * root) for root in roots)

    analytic_candidates = _unique_sorted(candidates)
    raw_irrigation, deployed_irrigation, raw_robust_gain, robust_gain, minimum_index = _select_lower_envelope(
        analytic_candidates,
        objectives,
        deployment_resolution_mm,
    )
    deployed_values = np.asarray(
        [_evaluate(function, np.asarray([deployed_irrigation]))[0] for function in objectives],
        dtype=np.float64,
    )
    mean_gain = float(np.mean(deployed_values))

    dense = np.arange(
        boundaries[0], boundaries[-1] + 0.5 * dense_step_mm, dense_step_mm, dtype=np.float64
    )
    if len(dense) == 0 or dense[-1] != boundaries[-1]:
        dense = np.append(dense[dense <= boundaries[-1]], boundaries[-1])
    dense_values = np.vstack([_evaluate(function, dense) for function in objectives])
    dense_robust = np.min(dense_values, axis=0)
    dense_maximum = float(np.max(dense_robust))
    dense_tied = np.isclose(dense_robust, dense_maximum, rtol=1.0e-12, atol=1.0e-12)
    dense_index = int(np.flatnonzero(dense_tied)[0])
    dense_irrigation = float(dense[dense_index])
    dense_gap = float(raw_robust_gain - dense_maximum)
    if dense_gap < -dense_gain_gap_tolerance:
        raise ValueError(
            "analytic lower-envelope gain is below dense diagnostic by more than tolerance: "
            f"gap={dense_gap:.6g}"
        )

    return EnvelopeOptimizationResult(
        raw_irrigation_mm=raw_irrigation,
        irrigation_mm=deployed_irrigation,
        robust_incremental_gain=robust_gain,
        mean_incremental_gain=mean_gain,
        minimum_model_index=minimum_index,
        candidate_count=int(len(analytic_candidates)),
        stationary_point_count=int(stationary_count),
        pairwise_intersection_count=int(pairwise_count),
        dense_diagnostic_irrigation_mm=dense_irrigation,
        dense_diagnostic_robust_gain=dense_maximum,
        dense_diagnostic_gain_gap=dense_gap,
    )


def gam_incremental_objective(model: object, state: pd.DataFrame) -> ArrayFunction:
    """Build a shared-term GAM net-gain increment objective anchored at zero irrigation."""
    if len(state) != 1:
        raise ValueError("cycle state must contain exactly one row")

    def probe(values: np.ndarray) -> pd.DataFrame:
        irrigation = np.asarray(values, dtype=np.float64).reshape(-1)
        frame = pd.concat([state] * len(irrigation), ignore_index=True)
        frame["irrigation_mm"] = irrigation
        return frame

    zero = probe(np.asarray([0.0]))
    zero_gain = float(model.predict_shared(zero)["pred_target_net_gain_7d"].iloc[0])

    def objective(values: np.ndarray) -> np.ndarray:
        predictions = model.predict_shared(probe(values))
        gain = predictions["pred_target_net_gain_7d"].to_numpy(dtype=np.float64)
        return gain - zero_gain

    return objective
