from __future__ import annotations

import unittest

import numpy as np

from s2s_rtist.models.robust_gam_envelope_v1 import (
    optimize_piecewise_cubic_lower_envelope,
)


class RobustEnvelopeOptimizerTests(unittest.TestCase):
    def test_finds_internal_stationary_point(self) -> None:
        result = optimize_piecewise_cubic_lower_envelope(
            objectives=(lambda x: 12.0 - (np.asarray(x) - 18.0) ** 2 / 18.0,),
            interval_boundaries=np.asarray([0.0, 60.0]),
            deployment_resolution_mm=1.0e-6,
            dense_step_mm=0.25,
        )
        self.assertAlmostEqual(result.irrigation_mm, 18.0, places=6)
        self.assertAlmostEqual(result.robust_incremental_gain, 12.0, places=6)
        self.assertEqual(result.pairwise_intersection_count, 0)

    def test_pairwise_intersection_can_be_lower_envelope_maximum(self) -> None:
        objectives = (
            lambda x: 12.0 - (np.asarray(x) - 18.0) ** 2 / 18.0,
            lambda x: 12.0 - (np.asarray(x) - 42.0) ** 2 / 18.0,
        )
        result = optimize_piecewise_cubic_lower_envelope(
            objectives=objectives,
            interval_boundaries=np.asarray([0.0, 60.0]),
            deployment_resolution_mm=1.0e-6,
            dense_step_mm=0.25,
        )
        self.assertAlmostEqual(result.irrigation_mm, 30.0, places=6)
        self.assertAlmostEqual(result.robust_incremental_gain, 4.0, places=6)
        self.assertGreaterEqual(result.pairwise_intersection_count, 1)

    def test_zero_irrigation_is_conservative_boundary(self) -> None:
        result = optimize_piecewise_cubic_lower_envelope(
            objectives=(lambda x: -np.asarray(x) / 10.0, lambda x: -np.asarray(x) / 20.0),
            interval_boundaries=np.asarray([0.0, 60.0]),
            deployment_resolution_mm=1.0e-6,
            dense_step_mm=0.25,
        )
        self.assertEqual(result.irrigation_mm, 0.0)
        self.assertAlmostEqual(result.robust_incremental_gain, 0.0, places=8)

    def test_equal_objective_uses_smaller_irrigation(self) -> None:
        result = optimize_piecewise_cubic_lower_envelope(
            objectives=(lambda x: np.ones_like(np.asarray(x), dtype=float),),
            interval_boundaries=np.asarray([0.0, 60.0]),
            deployment_resolution_mm=1.0e-6,
            dense_step_mm=0.25,
        )
        self.assertEqual(result.irrigation_mm, 0.0)

    def test_dense_grid_is_diagnostic_only(self) -> None:
        result = optimize_piecewise_cubic_lower_envelope(
            objectives=(lambda x: 12.0 - (np.asarray(x) - 18.0) ** 2 / 18.0,),
            interval_boundaries=np.asarray([0.0, 60.0]),
            deployment_resolution_mm=7.0,
            dense_step_mm=0.25,
        )
        self.assertEqual(result.irrigation_mm, 21.0)
        self.assertNotEqual(result.irrigation_mm, result.dense_diagnostic_irrigation_mm)
        self.assertLessEqual(result.dense_diagnostic_gain_gap, 0.05)

    def test_rejects_nonfinite_or_misaligned_objective(self) -> None:
        with self.assertRaisesRegex(ValueError, "nonfinite"):
            optimize_piecewise_cubic_lower_envelope(
                objectives=(lambda x: np.full_like(np.asarray(x), np.nan, dtype=float),),
                interval_boundaries=np.asarray([0.0, 60.0]),
                deployment_resolution_mm=1.0e-6,
                dense_step_mm=0.25,
            )


if __name__ == "__main__":
    unittest.main()
