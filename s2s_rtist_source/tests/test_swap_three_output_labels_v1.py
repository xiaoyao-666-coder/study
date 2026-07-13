from __future__ import annotations

import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from swap_three_output_labels_v1 import (
    extract_candidate_labels,
    flatten_candidate_labels,
    inclusive_horizon_end_doy,
)


AUDIT_ROOT = (
    ROOT
    / "site_general_surrogate_eval"
    / "three_output_balance_audit_p1_20240716_server_v1"
)


class HorizonTests(unittest.TestCase):
    def test_seven_day_inclusive_window_ends_on_day_six(self) -> None:
        self.assertEqual(inclusive_horizon_end_doy(198, 7), 204)


class CandidateLabelTests(unittest.TestCase):
    def extract(self, case: str):
        folder = AUDIT_ROOT / case
        return extract_candidate_labels(
            pre_crop_path=folder / "result_forec.crp",
            pre_profile_path=folder / "result_forec.vap",
            restart_crop_path=folder / "result_restart.crp",
            restart_profile_path=folder / "result_restart.vap",
            restart_increment_path=folder / "result_restart.inc",
            decision_date="2024-07-16",
            horizon_days=7,
        )

    def test_ir0_uses_exactly_seven_dates(self) -> None:
        result = self.extract("P1_20240716_ir0")

        self.assertEqual(result.summary["horizon_days_actual"], 7)
        self.assertEqual(result.summary["horizon_start_date"], "2024-07-16")
        self.assertEqual(result.summary["horizon_end_date"], "2024-07-22")
        self.assertEqual(len(result.daily), 7)

    def test_ir0_actual_et_includes_interception(self) -> None:
        result = self.extract("P1_20240716_ir0")

        self.assertAlmostEqual(result.summary["tact_7d_mm"], 15.7695, places=6)
        self.assertAlmostEqual(result.summary["eact_7d_mm"], 1.7410, places=6)
        self.assertAlmostEqual(result.summary["interc_7d_mm"], 1.5040, places=6)
        self.assertAlmostEqual(result.summary["aet_7d_mm"], 19.0145, places=6)

    def test_ir0_dynamic_rootzone_balance_matches_audit(self) -> None:
        result = self.extract("P1_20240716_ir0")

        self.assertAlmostEqual(
            result.summary["predecision_rootzone_storage_mm"], 80.39, places=6
        )
        self.assertAlmostEqual(
            result.summary["delta_rootzone_storage_7d_mm"], -11.86, places=6
        )
        self.assertAlmostEqual(
            result.summary["root_boundary_flux_7d_mm"], 0.082607, places=6
        )
        self.assertAlmostEqual(
            result.summary["water_balance_residual_7d_mm"], -0.037107, places=6
        )

    def test_ir30_labels_match_audit(self) -> None:
        result = self.extract("P1_20240716_ir30")

        self.assertAlmostEqual(result.summary["irrigation_7d_mm"], 30.0, places=6)
        self.assertAlmostEqual(result.summary["aet_7d_mm"], 33.1791, places=6)
        self.assertAlmostEqual(
            result.summary["delta_rootzone_storage_7d_mm"], 3.95, places=6
        )
        self.assertAlmostEqual(
            result.summary["root_boundary_flux_7d_mm"], 0.051779, places=6
        )
        self.assertAlmostEqual(
            result.summary["water_balance_residual_7d_mm"], 0.019121, places=6
        )

    def test_daily_output_contains_required_sequences(self) -> None:
        result = self.extract("P1_20240716_ir30")

        required = {
            "date",
            "root_depth_cm",
            "rootzone_vwc",
            "rootzone_storage_mm",
            "tact_mm",
            "eact_mm",
            "interc_mm",
            "aet_mm",
            "runoff_mm",
            "root_drainage_mm",
            "root_boundary_flux_mm",
        }
        self.assertTrue(required.issubset(result.daily.columns))

    def test_flattened_labels_have_seven_numbered_daily_values(self) -> None:
        result = self.extract("P1_20240716_ir30")

        flat = flatten_candidate_labels(result)

        self.assertAlmostEqual(flat["aet_7d_mm"], 33.1791, places=6)
        self.assertIn("rootzone_vwc_day01", flat)
        self.assertIn("rootzone_vwc_day07", flat)
        self.assertIn("root_boundary_flux_day01_mm", flat)
        self.assertIn("root_boundary_flux_day07_mm", flat)
        self.assertNotIn("rootzone_vwc_day08", flat)


if __name__ == "__main__":
    unittest.main()
