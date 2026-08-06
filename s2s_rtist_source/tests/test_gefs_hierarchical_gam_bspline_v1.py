from __future__ import annotations

import unittest
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

from s2s_rtist.models.gefs_hierarchical_gam_bspline_v1 import (
    GAM_2,
    GAM_VC,
    HierarchicalGamBSpline,
)


def training_frame() -> pd.DataFrame:
    rows = []
    for site_index, site_id in enumerate(("P1", "P2")):
        for state in (0.1, 0.5, 0.9):
            for irrigation in (0.0, 20.0, 40.0, 60.0):
                row = {
                    "site_id": site_id,
                    "irrigation_mm": irrigation,
                    "state_a": state,
                    "target_net_gain_7d": irrigation * (1.0 + state) - 0.03 * irrigation**2,
                    "target_aet_7d_mm": 20.0 + 0.2 * irrigation + site_index,
                    "physics_initial_soil_storage_0_100cm_mm": 200.0,
                }
                for day in range(1, 8):
                    row[f"target_soil_vwc_0_100cm_day{day:02d}"] = 0.2 + 0.0005 * irrigation
                    row[f"weather_precipitation_mm_day{day:02d}"] = 1.0
                rows.append(row)
    return pd.DataFrame(rows)


class HierarchicalGamTests(unittest.TestCase):
    def test_saved_coefficients_round_trip_predictions(self) -> None:
        frame = training_frame()
        model = HierarchicalGamBSpline(
            variant=GAM_VC,
            state_columns=("state_a",),
        ).fit(frame, lambda_main=1.0, lambda_site=2.0, lambda_interaction=3.0)
        probe = frame.iloc[[0, 3]].copy()
        probe["site_id"] = "P3"
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "model.npz"
            model.save_coefficients(path)
            restored = HierarchicalGamBSpline.load_coefficients(path)
        np.testing.assert_allclose(
            restored.predict_shared(probe).to_numpy(),
            model.predict_shared(probe).to_numpy(),
        )
        np.testing.assert_allclose(
            restored.predict_net_gain_derivative(
                probe, use_site_deviation=False
            ).to_numpy(),
            model.predict_net_gain_derivative(
                probe, use_site_deviation=False
            ).to_numpy(),
        )
        self.assertEqual(restored.penalty_weights_, model.penalty_weights_)

    def test_unseen_site_uses_shared_terms_and_output_constraints(self) -> None:
        frame = training_frame()
        model = HierarchicalGamBSpline(
            variant=GAM_2,
            state_columns=("state_a",),
            irrigation_basis_count=6,
            state_basis_count=4,
        ).fit(frame, lambda_main=1.0, lambda_site=1.0, lambda_interaction=1.0)
        probe = frame.iloc[[0]].copy()
        probe["site_id"] = "P3"
        probe["irrigation_mm"] = 0.0
        unseen = model.predict(probe)
        shared = model.predict_shared(probe)
        np.testing.assert_allclose(unseen["pred_target_net_gain_7d"], 0.0, atol=1.0e-12)
        np.testing.assert_allclose(unseen.to_numpy(), shared.to_numpy())
        self.assertGreaterEqual(float(unseen["pred_target_aet_7d_mm"].iloc[0]), 0.0)
        vwc = unseen.filter(like="pred_target_soil_vwc").to_numpy()
        self.assertTrue(((vwc >= 0.0) & (vwc <= 1.0)).all())
        self.assertGreaterEqual(float(unseen["pred_balance_closure_flux_7d_mm"].iloc[0]), 0.0)

    def test_site_contrast_has_weighted_zero_mean(self) -> None:
        frame = training_frame()
        model = HierarchicalGamBSpline(
            variant=GAM_2,
            state_columns=("state_a",),
        ).fit(frame, lambda_main=1.0, lambda_site=1.0, lambda_interaction=1.0)
        np.testing.assert_allclose(
            model.site_weights_ @ model.site_contrast_, 0.0, atol=1.0e-12
        )

    def test_state_range_diagnostics_marks_training_extrapolation(self) -> None:
        frame = training_frame()
        model = HierarchicalGamBSpline(
            variant=GAM_2,
            state_columns=("state_a",),
        ).fit(frame, lambda_main=1.0, lambda_site=1.0, lambda_interaction=1.0)
        probe = frame.iloc[[0]].copy()
        probe["state_a"] = 2.0
        diagnostics = model.state_range_diagnostics(probe)
        self.assertEqual(diagnostics["any_state_out_of_training_range_fraction"], 1.0)
        self.assertEqual(diagnostics["state_a_out_of_training_range_fraction"], 1.0)

    def test_varying_coefficient_uses_six_zero_anchored_columns_per_state(self) -> None:
        frame = training_frame()
        frame["state_b"] = 1.0 - frame["state_a"]
        model = HierarchicalGamBSpline(
            variant=GAM_VC,
            state_columns=("state_a", "state_b"),
        ).fit(frame, lambda_main=1.0, lambda_site=1.0, lambda_interaction=1.0)
        self.assertEqual(
            model.block_slices_["interaction:state_a"].stop
            - model.block_slices_["interaction:state_a"].start,
            6,
        )
        self.assertEqual(
            model.block_slices_["interaction:state_b"].stop
            - model.block_slices_["interaction:state_b"].start,
            6,
        )
        zero = frame.iloc[[0]].copy()
        zero["irrigation_mm"] = 0.0
        design, slices = model._design(zero, use_site_deviation=True)
        np.testing.assert_allclose(design[:, slices["interaction:state_a"]], 0.0)
        np.testing.assert_allclose(design[:, slices["interaction:state_b"]], 0.0)

    def test_shared_net_gain_derivative_matches_centered_finite_difference(self) -> None:
        frame = training_frame()
        model = HierarchicalGamBSpline(
            variant=GAM_VC,
            state_columns=("state_a",),
        ).fit(frame, lambda_main=1.0, lambda_site=1.0, lambda_interaction=1.0)
        probe = frame.iloc[[0]].copy()
        probe["site_id"] = "P3"
        probe["irrigation_mm"] = 30.0
        step = 1.0e-4
        lower = probe.copy()
        upper = probe.copy()
        lower["irrigation_mm"] -= step
        upper["irrigation_mm"] += step
        finite_difference = (
            float(model.predict_shared(upper)["pred_target_net_gain_7d"].iloc[0])
            - float(model.predict_shared(lower)["pred_target_net_gain_7d"].iloc[0])
        ) / (2.0 * step)
        derivative = float(
            model.predict_net_gain_derivative(
                probe,
                use_site_deviation=False,
            ).iloc[0]
        )
        self.assertAlmostEqual(derivative, finite_difference, places=6)


if __name__ == "__main__":
    unittest.main()
