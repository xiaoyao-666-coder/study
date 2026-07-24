from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from scripts.diagnostics.run_gefs_qdm_raw_allocation_shrinkage_cv_v1 import (
    candidate_id,
    member_total_audit,
    shrink_prediction,
)


class RawAllocationShrinkageTests(unittest.TestCase):
    def setUp(self) -> None:
        self.frame = pd.DataFrame(
            {
                "site_id": ["P1"] * 7,
                "decision_date": ["2018-06-01"] * 7,
                "valid_date_utc": pd.date_range("2018-06-01", periods=7),
                "gefs_member": ["c00"] * 7,
                "validation_year": [2018] * 7,
                "precipitation_mm_raw": [0, 1, 2, 0, 3, 0, 4],
                "precipitation_mm_qm": [0, 2, 1, 0, 1, 0, 6],
            }
        )

    def test_convex_shrinkage_preserves_member_total(self) -> None:
        corrected = shrink_prediction(self.frame, 0.5)
        expected = 0.5 * (
            self.frame["precipitation_mm_raw"] + self.frame["precipitation_mm_qm"]
        )
        np.testing.assert_allclose(corrected["precipitation_mm_qm"], expected)
        audit = member_total_audit(corrected)
        self.assertAlmostEqual(float(audit.iloc[0]["member_total_error_mm"]), 0.0)
        self.assertEqual(corrected["candidate_id"].iloc[0], candidate_id(0.5))

    def test_alpha_one_reproduces_qdm_volume_preserving_prediction(self) -> None:
        corrected = shrink_prediction(self.frame, 1.0)
        np.testing.assert_allclose(
            corrected["precipitation_mm_qm"], self.frame["precipitation_mm_qm"]
        )


if __name__ == "__main__":
    unittest.main()
