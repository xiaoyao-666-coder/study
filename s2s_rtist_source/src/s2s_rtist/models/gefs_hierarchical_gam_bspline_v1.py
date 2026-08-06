"""Hierarchical multi-output GAM with explicit irrigation-state B-splines."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd

from s2s_rtist.models.penalized_bspline_v1 import (
    CubicBSplineBasis,
    InteractionProjector,
    second_difference_penalty,
)


NET_GAIN_TARGET = "target_net_gain_7d"
AET_TARGET = "target_aet_7d_mm"
VWC_TARGETS = tuple(f"target_soil_vwc_0_100cm_day{day:02d}" for day in range(1, 8))
FORMAL_TARGETS = (NET_GAIN_TARGET, AET_TARGET, *VWC_TARGETS)
MODEL_OUTPUTS = (*FORMAL_TARGETS, "balance_closure_flux_7d_mm")
INITIAL_STORAGE = "physics_initial_soil_storage_0_100cm_mm"
RAIN_COLUMNS = tuple(f"weather_precipitation_mm_day{day:02d}" for day in range(1, 8))


@dataclass(frozen=True)
class GamVariant:
    name: str
    use_site_deviation: bool
    interaction_mode: str

    @property
    def use_irrigation_state_interaction(self) -> bool:
        return self.interaction_mode != "none"


GAM_0 = GamVariant("GAM-0", False, "none")
GAM_1 = GamVariant("GAM-1", True, "none")
GAM_2 = GamVariant("GAM-2", True, "tensor")
GAM_VC = GamVariant("GAM-VC", True, "varying_coefficient")
VARIANTS = {variant.name: variant for variant in (GAM_0, GAM_1, GAM_2, GAM_VC)}


def _finite(frame: pd.DataFrame, columns: Sequence[str]) -> np.ndarray:
    missing = sorted(set(columns) - set(frame.columns))
    if missing:
        raise ValueError(f"model input columns are missing: {missing}")
    values = frame.loc[:, list(columns)].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=np.float64)
    if not np.isfinite(values).all():
        raise ValueError(f"model input contains nonfinite values: {list(columns)}")
    return values


def _safe_bounds(values: np.ndarray) -> tuple[float, float]:
    lower = float(np.min(values))
    upper = float(np.max(values))
    if upper <= lower:
        width = max(abs(lower) * 1.0e-6, 1.0e-6)
        return lower - width, upper + width
    return lower, upper


def _site_contrast(site_ids: tuple[str, ...], sites: pd.Series) -> tuple[np.ndarray, np.ndarray]:
    counts = sites.astype(str).value_counts()
    if set(counts.index) != set(site_ids):
        raise ValueError("each training site must contain rows")
    weights = np.asarray([counts[site_id] for site_id in site_ids], dtype=np.float64)
    weights /= weights.sum()
    if len(site_ids) == 1:
        return weights, np.zeros((1, 0), dtype=np.float64)
    contrast = np.zeros((len(site_ids), len(site_ids) - 1), dtype=np.float64)
    contrast[:-1, :] = np.eye(len(site_ids) - 1)
    contrast[-1, :] = -weights[:-1] / weights[-1]
    return weights, contrast


class HierarchicalGamBSpline:
    def __init__(
        self,
        *,
        variant: GamVariant,
        state_columns: Sequence[str],
        irrigation_basis_count: int = 6,
        state_basis_count: int = 4,
        irrigation_bounds: tuple[float, float] = (0.0, 60.0),
    ) -> None:
        if variant.name not in VARIANTS or VARIANTS[variant.name] != variant:
            raise ValueError("unknown GAM variant")
        normalized_states = tuple(str(value) for value in state_columns)
        if not normalized_states or len(set(normalized_states)) != len(normalized_states):
            raise ValueError("state_columns must be nonempty and unique")
        if irrigation_bounds != (0.0, 60.0):
            raise ValueError("the development protocol freezes irrigation to [0, 60] mm")
        self.variant = variant
        self.state_columns = normalized_states
        self.irrigation_basis_count = int(irrigation_basis_count)
        self.state_basis_count = int(state_basis_count)
        self.irrigation_bounds = irrigation_bounds
        self.fitted_ = False

    def _fit_feature_state(self, frame: pd.DataFrame) -> None:
        self.irrigation_basis_ = CubicBSplineBasis(
            lower=self.irrigation_bounds[0],
            upper=self.irrigation_bounds[1],
            basis_count=self.irrigation_basis_count,
        )
        irrigation = _finite(frame, ("irrigation_mm",))[:, 0]
        raw_irrigation = self.irrigation_basis_.evaluate(irrigation)
        zero = self.irrigation_basis_.evaluate(np.asarray([0.0]))
        shifted_irrigation = raw_irrigation - zero
        self.irrigation_zero_basis_ = zero
        self.state_bases_: list[CubicBSplineBasis] = []
        self.state_means_: list[np.ndarray] = []
        self.state_scalar_means_: list[float] = []
        self.state_scalar_scales_: list[float] = []
        self.interaction_projectors_: list[InteractionProjector] = []
        for column in self.state_columns:
            values = _finite(frame, (column,))[:, 0]
            lower, upper = _safe_bounds(values)
            basis = CubicBSplineBasis(
                lower=lower,
                upper=upper,
                basis_count=self.state_basis_count,
            )
            raw_state = basis.evaluate(values)
            state_mean = raw_state.mean(axis=0, keepdims=True)
            centered_state = raw_state - state_mean
            self.state_bases_.append(basis)
            self.state_means_.append(state_mean)
            self.state_scalar_means_.append(float(values.mean()))
            self.state_scalar_scales_.append(max(float(values.std()), 1.0e-6))
            self.interaction_projectors_.append(
                InteractionProjector.fit(shifted_irrigation, centered_state)
            )
        self.site_ids_ = tuple(sorted(frame["site_id"].astype(str).unique()))
        self.site_weights_, self.site_contrast_ = _site_contrast(
            self.site_ids_, frame["site_id"]
        )

    def _site_values(self, frame: pd.DataFrame, use_site_deviation: bool) -> np.ndarray:
        output = np.zeros((len(frame), self.site_contrast_.shape[1]), dtype=np.float64)
        if not use_site_deviation or output.shape[1] == 0:
            return output
        lookup = {site_id: index for index, site_id in enumerate(self.site_ids_)}
        for row_index, site_id in enumerate(frame["site_id"].astype(str)):
            if site_id in lookup:
                output[row_index] = self.site_contrast_[lookup[site_id]]
        return output

    def _design(self, frame: pd.DataFrame, *, use_site_deviation: bool) -> tuple[np.ndarray, dict[str, slice]]:
        irrigation = _finite(frame, ("irrigation_mm",))[:, 0]
        if np.any(irrigation < self.irrigation_bounds[0]) or np.any(irrigation > self.irrigation_bounds[1]):
            raise ValueError("irrigation is outside the frozen [0, 60] mm range")
        shifted_irrigation = self.irrigation_basis_.evaluate(irrigation) - self.irrigation_zero_basis_
        blocks: list[np.ndarray] = [np.ones((len(frame), 1)), shifted_irrigation]
        names = ["intercept", "global_irrigation"]
        centered_states: list[np.ndarray] = []
        for column, basis, mean in zip(self.state_columns, self.state_bases_, self.state_means_):
            centered = basis.evaluate(_finite(frame, (column,))[:, 0]) - mean
            centered_states.append(centered)
            blocks.append(centered)
            names.append(f"state:{column}")
        if self.variant.interaction_mode == "tensor":
            for column, centered, projector in zip(
                self.state_columns, centered_states, self.interaction_projectors_
            ):
                blocks.append(projector.transform(shifted_irrigation, centered))
                names.append(f"interaction:{column}")
        elif self.variant.interaction_mode == "varying_coefficient":
            for column, mean, scale in zip(
                self.state_columns, self.state_scalar_means_, self.state_scalar_scales_
            ):
                standardized = (_finite(frame, (column,))[:, 0] - mean) / scale
                blocks.append(standardized[:, None] * shifted_irrigation)
                names.append(f"interaction:{column}")
        if self.variant.use_site_deviation:
            site_values = self._site_values(frame, use_site_deviation)
            blocks.append(site_values)
            names.append("site_intercept")
            blocks.append(
                np.einsum("ns,ni->nsi", site_values, shifted_irrigation).reshape(len(frame), -1)
            )
            names.append("site_irrigation")
        slices: dict[str, slice] = {}
        start = 0
        for name, block in zip(names, blocks):
            slices[name] = slice(start, start + block.shape[1])
            start += block.shape[1]
        return np.column_stack(blocks), slices

    def _irrigation_derivative_design(
        self,
        frame: pd.DataFrame,
        *,
        use_site_deviation: bool,
    ) -> tuple[np.ndarray, dict[str, slice]]:
        irrigation = _finite(frame, ("irrigation_mm",))[:, 0]
        if np.any(irrigation < self.irrigation_bounds[0]) or np.any(irrigation > self.irrigation_bounds[1]):
            raise ValueError("irrigation is outside the frozen [0, 60] mm range")
        irrigation_derivative = self.irrigation_basis_.derivative(irrigation)
        blocks: list[np.ndarray] = [
            np.zeros((len(frame), 1), dtype=np.float64),
            irrigation_derivative,
        ]
        names = ["intercept", "global_irrigation"]
        centered_states: list[np.ndarray] = []
        for column, basis, mean in zip(self.state_columns, self.state_bases_, self.state_means_):
            centered = basis.evaluate(_finite(frame, (column,))[:, 0]) - mean
            centered_states.append(centered)
            blocks.append(np.zeros_like(centered))
            names.append(f"state:{column}")
        if self.variant.interaction_mode == "tensor":
            for column, centered, projector in zip(
                self.state_columns, centered_states, self.interaction_projectors_
            ):
                blocks.append(projector.transform(irrigation_derivative, centered))
                names.append(f"interaction:{column}")
        elif self.variant.interaction_mode == "varying_coefficient":
            for column, mean, scale in zip(
                self.state_columns, self.state_scalar_means_, self.state_scalar_scales_
            ):
                standardized = (_finite(frame, (column,))[:, 0] - mean) / scale
                blocks.append(standardized[:, None] * irrigation_derivative)
                names.append(f"interaction:{column}")
        if self.variant.use_site_deviation:
            site_values = self._site_values(frame, use_site_deviation)
            blocks.append(np.zeros_like(site_values))
            names.append("site_intercept")
            blocks.append(
                np.einsum("ns,ni->nsi", site_values, irrigation_derivative).reshape(len(frame), -1)
            )
            names.append("site_irrigation")
        slices: dict[str, slice] = {}
        start = 0
        for name, block in zip(names, blocks):
            slices[name] = slice(start, start + block.shape[1])
            start += block.shape[1]
        return np.column_stack(blocks), slices

    def _penalty(
        self,
        slices: dict[str, slice],
        *,
        lambda_main: float,
        lambda_site: float,
        lambda_interaction: float,
    ) -> np.ndarray:
        dimension = max(value.stop for value in slices.values())
        penalty = np.zeros((dimension, dimension), dtype=np.float64)

        def add(name: str, value: np.ndarray, weight: float) -> None:
            block = slices[name]
            if value.shape != (block.stop - block.start, block.stop - block.start):
                raise ValueError(f"penalty dimension mismatch for {name}")
            penalty[block, block] += weight * value

        irrigation_penalty = second_difference_penalty(self.irrigation_basis_count)
        state_penalty = second_difference_penalty(self.state_basis_count)
        add("global_irrigation", irrigation_penalty, lambda_main)
        for column in self.state_columns:
            add(f"state:{column}", state_penalty, lambda_main)
        if self.variant.interaction_mode == "tensor":
            interaction_penalty = np.kron(irrigation_penalty, np.eye(self.state_basis_count)) + np.kron(
                np.eye(self.irrigation_basis_count), state_penalty
            )
            for column in self.state_columns:
                add(f"interaction:{column}", interaction_penalty, lambda_interaction)
        elif self.variant.interaction_mode == "varying_coefficient":
            interaction_penalty = irrigation_penalty + 0.1 * np.eye(self.irrigation_basis_count)
            for column in self.state_columns:
                add(f"interaction:{column}", interaction_penalty, lambda_interaction)
        if self.variant.use_site_deviation:
            site_count = self.site_contrast_.shape[1]
            add("site_intercept", np.eye(site_count), lambda_site)
            site_irrigation_penalty = np.kron(np.eye(site_count), irrigation_penalty + 0.1 * np.eye(self.irrigation_basis_count))
            add("site_irrigation", site_irrigation_penalty, lambda_site)
        return penalty

    @staticmethod
    def _balance_closure_target(frame: pd.DataFrame) -> np.ndarray:
        precipitation = _finite(frame, RAIN_COLUMNS).sum(axis=1)
        irrigation = _finite(frame, ("irrigation_mm",))[:, 0]
        aet = _finite(frame, (AET_TARGET,))[:, 0]
        final_storage = 1000.0 * _finite(frame, (VWC_TARGETS[-1],))[:, 0]
        initial_storage = _finite(frame, (INITIAL_STORAGE,))[:, 0]
        return np.maximum(precipitation + irrigation - aet - (final_storage - initial_storage), 0.0)

    def fit(
        self,
        frame: pd.DataFrame,
        *,
        lambda_main: float,
        lambda_site: float,
        lambda_interaction: float,
        lambda_balance: float = 1.0,
    ) -> "HierarchicalGamBSpline":
        if min(lambda_main, lambda_site, lambda_interaction, lambda_balance) < 0:
            raise ValueError("penalty weights must be nonnegative")
        required = {"site_id", "irrigation_mm", *self.state_columns, *FORMAL_TARGETS, INITIAL_STORAGE, *RAIN_COLUMNS}
        missing = sorted(required - set(frame.columns))
        if missing:
            raise ValueError(f"training frame is missing columns: {missing}")
        self._fit_feature_state(frame)
        design, slices = self._design(frame, use_site_deviation=True)
        self.block_slices_ = slices
        penalty = self._penalty(
            slices,
            lambda_main=lambda_main,
            lambda_site=lambda_site,
            lambda_interaction=lambda_interaction,
        )
        formal = _finite(frame, FORMAL_TARGETS)
        targets = np.column_stack([formal, self._balance_closure_target(frame)])
        offsets = targets.mean(axis=0)
        offsets[0] = 0.0
        scales = targets.std(axis=0)
        scales = np.maximum(scales, 1.0e-6)
        standardized = (targets - offsets) / scales
        coefficients = np.zeros((design.shape[1], len(MODEL_OUTPUTS)), dtype=np.float64)
        profit_names = ["global_irrigation"]
        if self.variant.use_irrigation_state_interaction:
            profit_names.extend(f"interaction:{column}" for column in self.state_columns)
        if self.variant.use_site_deviation:
            profit_names.append("site_irrigation")
        profit_indices = np.concatenate(
            [np.arange(slices[name].start, slices[name].stop) for name in profit_names]
        )
        output_masks = [profit_indices] + [np.arange(design.shape[1]) for _ in MODEL_OUTPUTS[1:]]
        for output_index, indices in enumerate(output_masks):
            x = design[:, indices]
            regularization = penalty[np.ix_(indices, indices)].copy()
            if output_index == len(MODEL_OUTPUTS) - 1:
                regularization /= max(lambda_balance, 1.0e-12)
            system = x.T @ x / len(x) + regularization + 1.0e-8 * np.eye(len(indices))
            right = x.T @ standardized[:, output_index] / len(x)
            try:
                solved = np.linalg.solve(system, right)
            except np.linalg.LinAlgError:
                solved = np.linalg.lstsq(system, right, rcond=None)[0]
            coefficients[indices, output_index] = solved
        self.coefficients_ = coefficients
        self.target_offsets_ = offsets
        self.target_scales_ = scales
        self.penalty_weights_ = {
            "lambda_main": float(lambda_main),
            "lambda_site": float(lambda_site),
            "lambda_interaction": float(lambda_interaction),
            "lambda_balance": float(lambda_balance),
        }
        self.fitted_ = True
        return self

    def _predict(self, frame: pd.DataFrame, *, use_site_deviation: bool) -> pd.DataFrame:
        if not self.fitted_:
            raise RuntimeError("model must be fitted before prediction")
        design, slices = self._design(frame, use_site_deviation=use_site_deviation)
        if slices != self.block_slices_:
            raise RuntimeError("prediction design does not match fitted design")
        raw = design @ self.coefficients_
        values = raw * self.target_scales_ + self.target_offsets_
        values[:, 1] = np.maximum(values[:, 1], 0.0)
        values[:, 2:9] = np.clip(values[:, 2:9], 0.0, 1.0)
        values[:, 9] = np.maximum(values[:, 9], 0.0)
        columns = [f"pred_{name}" for name in FORMAL_TARGETS] + ["pred_balance_closure_flux_7d_mm"]
        return pd.DataFrame(values, columns=columns, index=frame.index)

    def predict(self, frame: pd.DataFrame) -> pd.DataFrame:
        return self._predict(frame, use_site_deviation=True)

    def predict_shared(self, frame: pd.DataFrame) -> pd.DataFrame:
        return self._predict(frame, use_site_deviation=False)

    def predict_net_gain_derivative(
        self,
        frame: pd.DataFrame,
        *,
        use_site_deviation: bool,
    ) -> pd.Series:
        if not self.fitted_:
            raise RuntimeError("model must be fitted before derivative prediction")
        design, slices = self._irrigation_derivative_design(
            frame,
            use_site_deviation=use_site_deviation,
        )
        if slices != self.block_slices_:
            raise RuntimeError("derivative design does not match fitted design")
        derivative = design @ self.coefficients_[:, 0]
        derivative *= self.target_scales_[0]
        return pd.Series(
            derivative,
            index=frame.index,
            name="pred_target_net_gain_7d_derivative_per_mm",
        )

    def state_range_diagnostics(self, frame: pd.DataFrame) -> dict[str, float]:
        if not self.fitted_:
            raise RuntimeError("model must be fitted before range diagnostics")
        outside_any = np.zeros(len(frame), dtype=bool)
        diagnostics: dict[str, float] = {}
        for column, basis in zip(self.state_columns, self.state_bases_):
            values = _finite(frame, (column,))[:, 0]
            outside = (values < basis.lower) | (values > basis.upper)
            outside_any |= outside
            diagnostics[f"{column}_out_of_training_range_fraction"] = float(outside.mean())
        diagnostics["any_state_out_of_training_range_fraction"] = float(outside_any.mean())
        return diagnostics

    def interaction_coefficients(self) -> pd.DataFrame:
        rows: list[dict[str, float | str | int]] = []
        if not self.variant.use_irrigation_state_interaction:
            return pd.DataFrame(columns=["state_column", "output", "coefficient_index", "coefficient"])
        for state_column in self.state_columns:
            block = self.block_slices_[f"interaction:{state_column}"]
            for output_index, output in enumerate(MODEL_OUTPUTS):
                for index, value in enumerate(self.coefficients_[block, output_index]):
                    rows.append(
                        {
                            "state_column": state_column,
                            "output": output,
                            "coefficient_index": index,
                            "coefficient": float(value),
                        }
                    )
        return pd.DataFrame(rows)

    def site_deviation_norms(self) -> pd.DataFrame:
        if not self.variant.use_site_deviation:
            return pd.DataFrame(columns=["site_id", "output", "site_intercept", "irrigation_deviation_l2"])
        intercept = self.coefficients_[self.block_slices_["site_intercept"]]
        irrigation = self.coefficients_[self.block_slices_["site_irrigation"]].reshape(
            self.site_contrast_.shape[1], self.irrigation_basis_count, len(MODEL_OUTPUTS)
        )
        rows = []
        for site_index, site_id in enumerate(self.site_ids_):
            contrast = self.site_contrast_[site_index]
            site_intercept = contrast @ intercept
            site_irrigation = np.tensordot(contrast, irrigation, axes=(0, 0))
            for output_index, output in enumerate(MODEL_OUTPUTS):
                rows.append(
                    {
                        "site_id": site_id,
                        "output": output,
                        "site_intercept": float(site_intercept[output_index]),
                        "irrigation_deviation_l2": float(np.linalg.norm(site_irrigation[:, output_index])),
                    }
                )
        return pd.DataFrame(rows)

    def save_coefficients(self, path: Path) -> None:
        if not self.fitted_:
            raise RuntimeError("cannot save an unfitted model")
        path.parent.mkdir(parents=True, exist_ok=True)
        metadata = {
            "variant": self.variant.name,
            "interaction_mode": self.variant.interaction_mode,
            "state_columns": list(self.state_columns),
            "irrigation_basis_count": self.irrigation_basis_count,
            "state_basis_count": self.state_basis_count,
            "site_ids": list(self.site_ids_),
            "state_bounds": [[basis.lower, basis.upper] for basis in self.state_bases_],
            "block_slices": {name: [value.start, value.stop] for name, value in self.block_slices_.items()},
            "penalty_weights": self.penalty_weights_,
            "outputs": list(MODEL_OUTPUTS),
        }
        np.savez_compressed(
            path,
            metadata=np.asarray(json.dumps(metadata, sort_keys=True)),
            coefficients=self.coefficients_,
            target_offsets=self.target_offsets_,
            target_scales=self.target_scales_,
            site_weights=self.site_weights_,
            site_contrast=self.site_contrast_,
            state_means=np.stack(self.state_means_),
            state_scalar_means=np.asarray(self.state_scalar_means_),
            state_scalar_scales=np.asarray(self.state_scalar_scales_),
            interaction_projectors=np.stack(
                [projector.coefficients for projector in self.interaction_projectors_]
            ),
        )

    @classmethod
    def load_coefficients(cls, path: Path) -> "HierarchicalGamBSpline":
        if not path.is_file():
            raise FileNotFoundError(f"saved GAM coefficients are missing: {path}")
        with np.load(path, allow_pickle=False) as saved:
            metadata = json.loads(str(saved["metadata"].item()))
            variant_name = str(metadata["variant"])
            if variant_name not in VARIANTS:
                raise ValueError("saved GAM variant is unknown")
            model = cls(
                variant=VARIANTS[variant_name],
                state_columns=tuple(metadata["state_columns"]),
                irrigation_basis_count=int(metadata["irrigation_basis_count"]),
                state_basis_count=int(metadata["state_basis_count"]),
            )
            model.irrigation_basis_ = CubicBSplineBasis(
                lower=model.irrigation_bounds[0],
                upper=model.irrigation_bounds[1],
                basis_count=model.irrigation_basis_count,
            )
            model.irrigation_zero_basis_ = model.irrigation_basis_.evaluate(
                np.asarray([0.0])
            )
            model.state_bases_ = [
                CubicBSplineBasis(
                    lower=float(bounds[0]),
                    upper=float(bounds[1]),
                    basis_count=model.state_basis_count,
                )
                for bounds in metadata["state_bounds"]
            ]
            state_means = np.asarray(saved["state_means"], dtype=np.float64)
            if state_means.ndim != 3 or state_means.shape[1] != 1:
                raise ValueError("saved GAM state mean dimensions changed")
            model.state_means_ = [value for value in state_means]
            model.state_scalar_means_ = np.asarray(
                saved["state_scalar_means"], dtype=np.float64
            ).tolist()
            model.state_scalar_scales_ = np.asarray(
                saved["state_scalar_scales"], dtype=np.float64
            ).tolist()
            model.interaction_projectors_ = [
                InteractionProjector(coefficients=value)
                for value in np.asarray(saved["interaction_projectors"], dtype=np.float64)
            ]
            model.site_ids_ = tuple(str(value) for value in metadata["site_ids"])
            model.site_weights_ = np.asarray(saved["site_weights"], dtype=np.float64)
            model.site_contrast_ = np.asarray(saved["site_contrast"], dtype=np.float64)
            model.block_slices_ = {
                name: slice(int(bounds[0]), int(bounds[1]))
                for name, bounds in metadata["block_slices"].items()
            }
            model.penalty_weights_ = {
                name: float(value)
                for name, value in metadata["penalty_weights"].items()
            }
            model.coefficients_ = np.asarray(saved["coefficients"], dtype=np.float64)
            model.target_offsets_ = np.asarray(saved["target_offsets"], dtype=np.float64)
            model.target_scales_ = np.asarray(saved["target_scales"], dtype=np.float64)
        expected_shapes = {
            "coefficients": (max(value.stop for value in model.block_slices_.values()), len(MODEL_OUTPUTS)),
            "target_offsets": (len(MODEL_OUTPUTS),),
            "target_scales": (len(MODEL_OUTPUTS),),
        }
        if model.coefficients_.shape != expected_shapes["coefficients"]:
            raise ValueError("saved GAM coefficient dimensions changed")
        if model.target_offsets_.shape != expected_shapes["target_offsets"]:
            raise ValueError("saved GAM target offset dimensions changed")
        if model.target_scales_.shape != expected_shapes["target_scales"]:
            raise ValueError("saved GAM target scale dimensions changed")
        numeric = [
            model.coefficients_,
            model.target_offsets_,
            model.target_scales_,
            model.site_weights_,
            model.site_contrast_,
            *model.state_means_,
        ]
        if not all(np.isfinite(value).all() for value in numeric):
            raise ValueError("saved GAM coefficients contain nonfinite values")
        model.fitted_ = True
        return model
