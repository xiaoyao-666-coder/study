"""Small dependency-free B-spline utilities for penalized GAM models."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


class CubicBSplineBasis:
    def __init__(self, *, lower: float, upper: float, basis_count: int, degree: int = 3) -> None:
        if not np.isfinite(lower) or not np.isfinite(upper) or upper <= lower:
            raise ValueError("B-spline bounds must be finite and increasing")
        if degree != 3:
            raise ValueError("this protocol freezes cubic B-splines")
        if basis_count < degree + 1:
            raise ValueError("basis_count is too small for a cubic spline")
        self.lower = float(lower)
        self.upper = float(upper)
        self.basis_count = int(basis_count)
        self.degree = int(degree)
        interior_count = basis_count - degree - 1
        interior = (
            np.linspace(lower, upper, interior_count + 2, dtype=np.float64)[1:-1]
            if interior_count > 0
            else np.empty(0, dtype=np.float64)
        )
        self.knots = np.concatenate(
            [
                np.repeat(self.lower, degree + 1),
                interior,
                np.repeat(self.upper, degree + 1),
            ]
        )

    def _evaluate_degree(self, values: np.ndarray, degree: int) -> np.ndarray:
        x = np.clip(np.asarray(values, dtype=np.float64).reshape(-1), self.lower, self.upper)
        knot_count = len(self.knots)
        base_count = knot_count - 1
        basis = np.zeros((len(x), base_count), dtype=np.float64)
        for index in range(base_count):
            basis[:, index] = ((x >= self.knots[index]) & (x < self.knots[index + 1])).astype(np.float64)
        basis[x == self.upper, self.basis_count - 1] = 1.0
        for current_degree in range(1, degree + 1):
            next_basis = np.zeros((len(x), base_count - current_degree), dtype=np.float64)
            for index in range(base_count - current_degree):
                left_denominator = self.knots[index + current_degree] - self.knots[index]
                right_denominator = self.knots[index + current_degree + 1] - self.knots[index + 1]
                if left_denominator > 0:
                    next_basis[:, index] += (
                        (x - self.knots[index]) / left_denominator * basis[:, index]
                    )
                if right_denominator > 0:
                    next_basis[:, index] += (
                        (self.knots[index + current_degree + 1] - x)
                        / right_denominator
                        * basis[:, index + 1]
                    )
            basis = next_basis
        if degree == self.degree:
            basis[x == self.upper, :] = 0.0
            basis[x == self.upper, -1] = 1.0
        return basis

    def evaluate(self, values: np.ndarray) -> np.ndarray:
        return self._evaluate_degree(values, self.degree)

    def derivative(self, values: np.ndarray) -> np.ndarray:
        # At the closed upper boundary use the interior one-sided derivative.
        # The degree-3 endpoint basis index is not the degree-2 endpoint index.
        x = np.asarray(values, dtype=np.float64)
        x = np.where(x == self.upper, np.nextafter(self.upper, self.lower), x)
        lower_basis = self._evaluate_degree(x, self.degree - 1)
        derivative = np.zeros((len(lower_basis), self.basis_count), dtype=np.float64)
        for index in range(self.basis_count):
            left_denominator = self.knots[index + self.degree] - self.knots[index]
            right_denominator = self.knots[index + self.degree + 1] - self.knots[index + 1]
            if left_denominator > 0:
                derivative[:, index] += self.degree / left_denominator * lower_basis[:, index]
            if right_denominator > 0:
                derivative[:, index] -= self.degree / right_denominator * lower_basis[:, index + 1]
        return derivative


def second_difference_penalty(basis_count: int) -> np.ndarray:
    if basis_count < 3:
        raise ValueError("second differences require at least three coefficients")
    difference = np.diff(np.eye(basis_count, dtype=np.float64), n=2, axis=0)
    return difference.T @ difference


def centered_tensor_product(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    left_values = np.asarray(left, dtype=np.float64)
    right_values = np.asarray(right, dtype=np.float64)
    if left_values.ndim != 2 or right_values.ndim != 2 or len(left_values) != len(right_values):
        raise ValueError("tensor product inputs must be aligned two-dimensional arrays")
    return np.einsum("ni,nj->nij", left_values, right_values).reshape(len(left_values), -1)


@dataclass(frozen=True)
class InteractionProjector:
    """Remove the pure irrigation component while preserving a zero-I anchor."""

    coefficients: np.ndarray

    @classmethod
    def fit(cls, irrigation: np.ndarray, state: np.ndarray) -> "InteractionProjector":
        irrigation_values = np.asarray(irrigation, dtype=np.float64)
        raw = centered_tensor_product(irrigation_values, state)
        coefficients = np.linalg.lstsq(irrigation_values, raw, rcond=None)[0]
        return cls(coefficients=coefficients)

    def transform(self, irrigation: np.ndarray, state: np.ndarray) -> np.ndarray:
        irrigation_values = np.asarray(irrigation, dtype=np.float64)
        raw = centered_tensor_product(irrigation_values, state)
        return raw - irrigation_values @ self.coefficients
