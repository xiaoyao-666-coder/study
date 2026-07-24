from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from scripts.diagnostics.run_gefs_weekly_two_stage_linear_scaling_cv_v1 import (
    apply_two_stage_factors,
    fit_two_stage_factors,
)


class WeeklyTwoStageLinearScalingTests(unittest.TestCase):
    def test_factor_fit_matches_two_stage_definition(self) -> None:
        cycles = pd.DataFrame(
            {
                "site_id": ["P1"] * 20,
                "ensemble_mean_raw_7d_mm": list(range(1, 21)),
                "reference_7d_mm": [2.0 * value for value in range(1, 21)],
            }
        )
        factors = fit_two_stage_factors(
            cycles,
            group_keys=("site_id",),
            extreme_quantile=0.9,
        )
        row = factors.iloc[0]
        self.assertAlmostEqual(float(row["extreme_factor"]), 2.0)
        expected_overall = 420.0 / 249.0
        self.assertAlmostEqual(float(row["overall_factor"]), expected_overall)
        self.assertAlmostEqual(
            float(row["final_extreme_factor"]), 2.0 * expected_overall
        )

    def test_application_uses_one_factor_for_all_member_days(self) -> None:
        rows = []
        for member, multiplier in (("c00", 1.0), ("p01", 2.0), ("p02", 3.0), ("p03", 4.0), ("p04", 5.0)):
            for lead in range(1, 8):
                rows.append(
                    {
                        "site_id": "P1",
                        "decision_date": "2015-06-01",
                        "valid_date_utc": pd.Timestamp("2015-06-01") + pd.Timedelta(days=lead - 1),
                        "gefs_member": member,
                        "precipitation_mm_raw": multiplier * lead,
                    }
                )
        frame = pd.DataFrame(rows)
        factors = pd.DataFrame(
            {
                "site_id": ["P1"],
                "raw_ensemble_mean_7d_q90_mm": [50.0],
                "extreme_factor": [2.0],
                "overall_factor": [0.5],
                "final_extreme_factor": [1.0],
            }
        )
        corrected = apply_two_stage_factors(
            frame,
            factors,
            candidate="weekly_two_stage_linear_site_only",
            group_keys=("site_id",),
        )
        ratios = corrected["precipitation_mm_qm"] / corrected["precipitation_mm_raw"]
        self.assertEqual(ratios.nunique(), 1)
        self.assertTrue(np.isfinite(corrected["precipitation_mm_qm"]).all())


if __name__ == "__main__":
    unittest.main()
