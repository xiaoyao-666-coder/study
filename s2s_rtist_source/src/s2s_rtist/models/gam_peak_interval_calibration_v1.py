"""Low-capacity peak-position calibration for frozen GAM recommendations."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np


FEATURE_COLUMNS = (
    "raw_peak_mm",
    "raw_peak_is_zero_boundary",
    "raw_peak_is_sixty_boundary",
    "raw_peak_gain_margin",
    "raw_peak_curvature_abs",
    "delete_one_peak_mean_mm",
    "delete_one_peak_median_mm",
    "delete_one_peak_std_mm",
    "delete_one_peak_range_mm",
    "delete_one_peak_min_mm",
    "delete_one_peak_max_mm",
    "raw_minus_delete_one_median_mm",
    "delete_one_gain_at_raw_mean",
    "delete_one_gain_at_raw_std",
    "delete_one_positive_gain_ratio",
)


def _matrix(values: np.ndarray, *, name: str) -> np.ndarray:
    result = np.asarray(values, dtype=np.float64)
    if result.ndim != 2 or len(result) == 0 or not np.isfinite(result).all():
        raise ValueError(f"{name} must be a nonempty finite matrix")
    return result


def _vector(values: np.ndarray, *, name: str, length: int | None = None) -> np.ndarray:
    result = np.asarray(values, dtype=np.float64).reshape(-1)
    if len(result) == 0 or not np.isfinite(result).all():
        raise ValueError(f"{name} must be a nonempty finite vector")
    if length is not None and len(result) != length:
        raise ValueError(f"{name} is not aligned")
    return result


@dataclass(frozen=True)
class WeightedRidgeModel:
    feature_names: tuple[str, ...]
    feature_means: np.ndarray
    feature_scales: np.ndarray
    intercept: float
    coefficients: np.ndarray
    gamma: float
    ridge_lambda: float
    high_risk_threshold: float
    high_risk_count: int

    def predict(self, features: np.ndarray) -> np.ndarray:
        values = _matrix(features, name="features")
        if values.shape[1] != len(self.feature_names):
            raise ValueError("prediction feature dimension changed")
        standardized = (values - self.feature_means) / self.feature_scales
        result = self.intercept + standardized @ self.coefficients
        if not np.isfinite(result).all():
            raise ValueError("weighted ridge prediction is nonfinite")
        return result


def fit_weighted_ridge(
    features: np.ndarray,
    target_delta: np.ndarray,
    baseline_absolute_error: np.ndarray,
    *,
    gamma: float,
    ridge_lambda: float,
    feature_names: Sequence[str] = FEATURE_COLUMNS,
) -> WeightedRidgeModel:
    values = _matrix(features, name="features")
    target = _vector(target_delta, name="target_delta", length=len(values))
    baseline_error = _vector(
        baseline_absolute_error,
        name="baseline_absolute_error",
        length=len(values),
    )
    names = tuple(str(name) for name in feature_names)
    if len(names) != values.shape[1] or len(set(names)) != len(names):
        raise ValueError("feature names must be unique and aligned")
    if gamma < 0.0 or ridge_lambda < 0.0:
        raise ValueError("gamma and ridge_lambda must be nonnegative")
    means = values.mean(axis=0)
    scales = values.std(axis=0)
    scales = np.where(scales > 1.0e-12, scales, 1.0)
    standardized = (values - means) / scales
    threshold = float(np.quantile(baseline_error, 0.8))
    high_risk = baseline_error >= threshold
    weights = np.ones(len(values), dtype=np.float64)
    weights[high_risk] += float(gamma)
    design = np.column_stack([np.ones(len(values)), standardized])
    weighted = design * np.sqrt(weights)[:, None]
    weighted_target = target * np.sqrt(weights)
    penalty = np.eye(design.shape[1], dtype=np.float64) * float(ridge_lambda)
    penalty[0, 0] = 0.0
    system = weighted.T @ weighted + penalty + 1.0e-12 * np.eye(design.shape[1])
    system[0, 0] -= 1.0e-12
    right = weighted.T @ weighted_target
    try:
        solved = np.linalg.solve(system, right)
    except np.linalg.LinAlgError:
        solved = np.linalg.lstsq(system, right, rcond=None)[0]
    if not np.isfinite(solved).all():
        raise ValueError("weighted ridge coefficients are nonfinite")
    return WeightedRidgeModel(
        feature_names=names,
        feature_means=means,
        feature_scales=scales,
        intercept=float(solved[0]),
        coefficients=solved[1:],
        gamma=float(gamma),
        ridge_lambda=float(ridge_lambda),
        high_risk_threshold=threshold,
        high_risk_count=int(high_risk.sum()),
    )


def grouped_cross_predictions(
    features: np.ndarray,
    target_delta: np.ndarray,
    baseline_absolute_error: np.ndarray,
    groups: Sequence[object],
    *,
    gamma: float,
    ridge_lambda: float,
    feature_names: Sequence[str] = FEATURE_COLUMNS,
) -> np.ndarray:
    values = _matrix(features, name="features")
    target = _vector(target_delta, name="target_delta", length=len(values))
    baseline = _vector(
        baseline_absolute_error,
        name="baseline_absolute_error",
        length=len(values),
    )
    labels = np.asarray([str(group) for group in groups], dtype=object)
    if len(labels) != len(values):
        raise ValueError("groups are not aligned")
    unique = tuple(dict.fromkeys(labels.tolist()))
    if len(unique) < 2:
        raise ValueError("at least two calibration groups are required")
    predictions = np.full(len(values), np.nan, dtype=np.float64)
    for group in unique:
        validation = labels == group
        training = ~validation
        if not validation.any() or not training.any():
            raise ValueError("empty grouped calibration split")
        model = fit_weighted_ridge(
            values[training],
            target[training],
            baseline[training],
            gamma=gamma,
            ridge_lambda=ridge_lambda,
            feature_names=feature_names,
        )
        predictions[validation] = model.predict(values[validation])
    if not np.isfinite(predictions).all():
        raise ValueError("grouped calibration predictions are incomplete")
    return predictions


@dataclass(frozen=True)
class HyperparameterSelection:
    gamma: float
    ridge_lambda: float
    maximum_absolute_error_mm: float
    positive_mean_absolute_error_mm: float
    overall_mean_absolute_error_mm: float
    cross_fitted_delta_mm: np.ndarray
    cross_fitted_peak_mm: np.ndarray
    search_rows: tuple[dict[str, float], ...]


def select_hyperparameters(
    features: np.ndarray,
    target_delta: np.ndarray,
    baseline_absolute_error: np.ndarray,
    raw_peak_mm: np.ndarray,
    true_peak_mm: np.ndarray,
    groups: Sequence[object],
    *,
    gamma_grid: Sequence[float],
    lambda_grid: Sequence[float],
    feature_names: Sequence[str] = FEATURE_COLUMNS,
) -> HyperparameterSelection:
    values = _matrix(features, name="features")
    delta = _vector(target_delta, name="target_delta", length=len(values))
    baseline = _vector(
        baseline_absolute_error,
        name="baseline_absolute_error",
        length=len(values),
    )
    raw = _vector(raw_peak_mm, name="raw_peak_mm", length=len(values))
    truth = _vector(true_peak_mm, name="true_peak_mm", length=len(values))
    candidates: list[tuple[tuple[float, ...], float, float, np.ndarray, np.ndarray]] = []
    rows: list[dict[str, float]] = []
    for gamma in gamma_grid:
        for ridge_lambda in lambda_grid:
            prediction = grouped_cross_predictions(
                values,
                delta,
                baseline,
                groups,
                gamma=float(gamma),
                ridge_lambda=float(ridge_lambda),
                feature_names=feature_names,
            )
            corrected = np.clip(raw + prediction, 0.0, 60.0)
            error = np.abs(corrected - truth)
            positive = truth > 1.0e-6
            maximum = float(error.max())
            positive_mean = float(error[positive].mean()) if positive.any() else 0.0
            overall = float(error.mean())
            key = (maximum, positive_mean, overall, -float(ridge_lambda), float(gamma))
            candidates.append((key, float(gamma), float(ridge_lambda), prediction, corrected))
            rows.append(
                {
                    "gamma": float(gamma),
                    "ridge_lambda": float(ridge_lambda),
                    "maximum_absolute_error_mm": maximum,
                    "positive_mean_absolute_error_mm": positive_mean,
                    "overall_mean_absolute_error_mm": overall,
                }
            )
    if not candidates:
        raise ValueError("calibration hyperparameter grid is empty")
    key, gamma, ridge_lambda, prediction, corrected = min(candidates, key=lambda item: item[0])
    return HyperparameterSelection(
        gamma=gamma,
        ridge_lambda=ridge_lambda,
        maximum_absolute_error_mm=key[0],
        positive_mean_absolute_error_mm=key[1],
        overall_mean_absolute_error_mm=key[2],
        cross_fitted_delta_mm=prediction,
        cross_fitted_peak_mm=corrected,
        search_rows=tuple(rows),
    )


def empirical_signed_quantiles(
    residuals: np.ndarray,
    quantiles: tuple[float, float] = (0.05, 0.95),
) -> tuple[float, float]:
    values = np.sort(_vector(residuals, name="residuals"))
    lower_q, upper_q = quantiles
    if not (0.0 < lower_q < upper_q < 1.0):
        raise ValueError("quantiles must be ordered inside (0, 1)")

    def inverse_cdf(quantile: float) -> float:
        index = max(int(np.ceil(quantile * len(values))) - 1, 0)
        return float(values[index])

    return inverse_cdf(lower_q), inverse_cdf(upper_q)


@dataclass(frozen=True)
class CalibratedInterval:
    point_peak_mm: np.ndarray
    lower_mm: np.ndarray
    upper_mm: np.ndarray
    calibrated_peak_mm: np.ndarray


def calibrated_interval(
    *,
    raw_peak_mm: np.ndarray,
    predicted_delta_mm: np.ndarray,
    q05: float,
    q95: float,
) -> CalibratedInterval:
    raw = _vector(raw_peak_mm, name="raw_peak_mm")
    delta = _vector(predicted_delta_mm, name="predicted_delta_mm", length=len(raw))
    if not np.isfinite([q05, q95]).all() or q05 > q95:
        raise ValueError("signed residual quantiles are invalid")
    unbounded_point = raw + delta
    point = np.clip(unbounded_point, 0.0, 60.0)
    lower = np.clip(unbounded_point + q05, 0.0, 60.0)
    upper = np.clip(unbounded_point + q95, 0.0, 60.0)
    if np.any(lower > upper):
        raise ValueError("calibration interval inverted after clipping")
    midpoint = (lower + upper) / 2.0
    if not np.isfinite(np.column_stack([point, lower, upper, midpoint])).all():
        raise ValueError("calibration interval is nonfinite")
    return CalibratedInterval(point, lower, upper, midpoint)


def aggregate_peak_diagnostics(
    *,
    raw_peak_mm: float,
    raw_gain_margin: float,
    curvature_abs: float,
    delete_one_peaks_mm: np.ndarray,
    delete_one_gains_at_raw: np.ndarray,
) -> dict[str, float]:
    peaks = _vector(delete_one_peaks_mm, name="delete_one_peaks_mm")
    gains = _vector(
        delete_one_gains_at_raw,
        name="delete_one_gains_at_raw",
        length=len(peaks),
    )
    if len(peaks) < 2:
        raise ValueError("at least two delete-one diagnostic models are required")
    raw_peak = float(raw_peak_mm)
    if not np.isfinite([raw_peak, raw_gain_margin, curvature_abs]).all():
        raise ValueError("raw peak diagnostics are nonfinite")
    median = float(np.median(peaks))
    values = {
        "raw_peak_mm": raw_peak,
        "raw_peak_is_zero_boundary": float(np.isclose(raw_peak, 0.0, atol=1.0e-8, rtol=0.0)),
        "raw_peak_is_sixty_boundary": float(np.isclose(raw_peak, 60.0, atol=1.0e-8, rtol=0.0)),
        "raw_peak_gain_margin": float(raw_gain_margin),
        "raw_peak_curvature_abs": float(curvature_abs),
        "delete_one_peak_mean_mm": float(np.mean(peaks)),
        "delete_one_peak_median_mm": median,
        "delete_one_peak_std_mm": float(np.std(peaks)),
        "delete_one_peak_range_mm": float(np.ptp(peaks)),
        "delete_one_peak_min_mm": float(np.min(peaks)),
        "delete_one_peak_max_mm": float(np.max(peaks)),
        "raw_minus_delete_one_median_mm": raw_peak - median,
        "delete_one_gain_at_raw_mean": float(np.mean(gains)),
        "delete_one_gain_at_raw_std": float(np.std(gains)),
        "delete_one_positive_gain_ratio": float(np.mean(gains > 0.0)),
    }
    if tuple(values) != FEATURE_COLUMNS:
        raise RuntimeError("peak diagnostic feature order changed")
    return values
