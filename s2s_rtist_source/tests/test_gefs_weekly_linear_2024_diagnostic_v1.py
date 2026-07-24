from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from scripts.diagnostics.run_gefs_weekly_linear_2024_diagnostic_v1 import (
    apply_artifact,
    cycle_ensemble_mean_totals,
)


class WeeklyLinear2024DiagnosticTests(unittest.TestCase):
    def setUp(self) -> None:
        rows = []
        for member in ("gec00", "gep01", "gep02"):
            for lead in range(1, 8):
                rows.append(
                    {
                        "site_id": "P1",
                        "decision_date": pd.Timestamp("2024-07-16"),
                        "local_date": pd.Timestamp("2024-07-15") + pd.Timedelta(days=lead),
                        "valid_date_utc": pd.Timestamp("2024-07-15") + pd.Timedelta(days=lead),
                        "lead_day": lead,
                        "gefs_member": member,
                        "precipitation_mm_raw": 10.0,
                    }
                )
        self.frame = pd.DataFrame(rows)
        self.artifact = {
            "candidate_id": "weekly_two_stage_linear_site_factor_shrink_a075",
            "factor_shrinkage_alpha": 0.75,
            "artifact_sha256": "x" * 64,
            "groups": [
                {
                    "site_id": "P1",
                    "raw_ensemble_mean_7d_q90_mm": 50.0,
                    "effective_overall_factor": 0.9,
                    "effective_extreme_factor": 1.1,
                }
            ],
        }

    def test_cycle_total_supports_non_reforecast_member_count(self) -> None:
        cycles = cycle_ensemble_mean_totals(self.frame, expected_member_count=3)
        self.assertEqual(len(cycles), 1)
        self.assertAlmostEqual(cycles["ensemble_mean_raw_7d_mm"].iloc[0], 70.0)

    def test_extreme_cycle_applies_one_frozen_factor_to_all_rows(self) -> None:
        corrected = apply_artifact(self.frame, self.artifact, expected_member_count=3)
        self.assertTrue(corrected["weekly_extreme_regime"].all())
        self.assertTrue(corrected["weekly_linear_scaling_factor"].eq(1.1).all())
        np.testing.assert_allclose(corrected["precipitation_mm_qm"], 11.0)

    def test_incomplete_member_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            cycle_ensemble_mean_totals(self.frame.iloc[:-1], expected_member_count=3)


if __name__ == "__main__":
    unittest.main()
