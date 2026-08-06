from __future__ import annotations

import unittest

import numpy as np

from s2s_rtist.models.gam_peak_interval_calibration_v1 import (
    FEATURE_COLUMNS,
    aggregate_peak_diagnostics,
    calibrated_interval,
    empirical_signed_quantiles,
    fit_weighted_ridge,
    select_hyperparameters,
)


class WeightedRidgeTests(unittest.TestCase):
    def test_intercept_is_not_penalized_and_constant_feature_is_safe(self) -> None:
        features = np.column_stack([np.arange(6.0), np.ones(6)])
        target = np.full(6, 7.5)
        model = fit_weighted_ridge(
            features,
            target,
            np.zeros(6),
            gamma=5.0,
            ridge_lambda=1.0e6,
            feature_names=("trend", "constant"),
        )
        np.testing.assert_allclose(model.predict(features), target, atol=1.0e-8)
        self.assertEqual(float(model.feature_scales[1]), 1.0)

    def test_weights_are_derived_only_from_training_baseline_errors(self) -> None:
        features = np.arange(10.0)[:, None]
        baseline_error = np.arange(10.0)
        model = fit_weighted_ridge(
            features,
            np.zeros(10),
            baseline_error,
            gamma=2.0,
            ridge_lambda=1.0,
            feature_names=("x",),
        )
        self.assertAlmostEqual(model.high_risk_threshold, 7.2)
        self.assertEqual(model.high_risk_count, 2)

    def test_grouped_selection_uses_frozen_lexicographic_tie_break(self) -> None:
        features = np.zeros((8, 1))
        raw_peak = np.full(8, 10.0)
        true_peak = np.full(8, 10.0)
        groups = np.asarray(["a", "a", "b", "b", "c", "c", "d", "d"])
        selection = select_hyperparameters(
            features,
            true_peak - raw_peak,
            np.abs(true_peak - raw_peak),
            raw_peak,
            true_peak,
            groups,
            gamma_grid=(0.0, 2.0, 5.0),
            lambda_grid=(0.1, 1.0, 10.0),
            feature_names=("x",),
        )
        self.assertEqual(selection.gamma, 0.0)
        self.assertEqual(selection.ridge_lambda, 10.0)
        np.testing.assert_allclose(selection.cross_fitted_peak_mm, true_peak)

    def test_high_risk_weighting_reduces_synthetic_worst_error(self) -> None:
        features = np.linspace(-1.0, 1.0, 12)[:, None]
        target = np.asarray(
            [
                -1.827207903967607,
                -1.225554564613057,
                -1.1075087346355792,
                -1.5606695248930897,
                -0.09277661211798655,
                0.04136910436382399,
                -0.08665843586196093,
                0.836013597552722,
                1.091377107183947,
                1.4197935210550359,
                1.650574757021535,
                8.251746792300777,
            ]
        )
        baseline_error = np.abs(target)
        unweighted = fit_weighted_ridge(
            features,
            target,
            baseline_error,
            gamma=0.0,
            ridge_lambda=1.0,
            feature_names=("x",),
        )
        weighted = fit_weighted_ridge(
            features,
            target,
            baseline_error,
            gamma=5.0,
            ridge_lambda=1.0,
            feature_names=("x",),
        )
        unweighted_max = np.max(np.abs(unweighted.predict(features) - target))
        weighted_max = np.max(np.abs(weighted.predict(features) - target))
        self.assertLess(weighted_max, unweighted_max)


class IntervalTests(unittest.TestCase):
    def test_signed_quantiles_use_frozen_inverse_cdf_order_statistics(self) -> None:
        values = np.arange(1.0, 21.0)
        self.assertEqual(empirical_signed_quantiles(values), (1.0, 19.0))

    def test_interval_clips_each_component_and_uses_midpoint(self) -> None:
        result = calibrated_interval(
            raw_peak_mm=np.asarray([2.0, 58.0]),
            predicted_delta_mm=np.asarray([-5.0, 5.0]),
            q05=-4.0,
            q95=6.0,
        )
        np.testing.assert_allclose(result.point_peak_mm, [0.0, 60.0])
        np.testing.assert_allclose(result.lower_mm, [0.0, 59.0])
        np.testing.assert_allclose(result.upper_mm, [3.0, 60.0])
        np.testing.assert_allclose(result.calibrated_peak_mm, [1.5, 59.5])


class FeatureAggregationTests(unittest.TestCase):
    def test_three_and_four_diagnostic_models_share_one_feature_schema(self) -> None:
        three = aggregate_peak_diagnostics(
            raw_peak_mm=20.0,
            raw_gain_margin=4.0,
            curvature_abs=0.5,
            delete_one_peaks_mm=np.asarray([10.0, 20.0, 30.0]),
            delete_one_gains_at_raw=np.asarray([-1.0, 2.0, 3.0]),
        )
        four = aggregate_peak_diagnostics(
            raw_peak_mm=20.0,
            raw_gain_margin=4.0,
            curvature_abs=0.5,
            delete_one_peaks_mm=np.asarray([10.0, 20.0, 30.0, 40.0]),
            delete_one_gains_at_raw=np.asarray([-1.0, 2.0, 3.0, 4.0]),
        )
        self.assertEqual(tuple(three), FEATURE_COLUMNS)
        self.assertEqual(tuple(four), FEATURE_COLUMNS)
        self.assertAlmostEqual(three["delete_one_positive_gain_ratio"], 2.0 / 3.0)
        self.assertAlmostEqual(four["delete_one_positive_gain_ratio"], 0.75)


if __name__ == "__main__":
    unittest.main()
