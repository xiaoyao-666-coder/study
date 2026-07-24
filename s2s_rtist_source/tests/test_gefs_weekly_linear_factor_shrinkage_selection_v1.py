from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from scripts.diagnostics.run_gefs_weekly_linear_factor_shrinkage_selection_v1 import (
    BASE_CANDIDATE_ID,
    candidate_id,
    shrink_factors,
)


class WeeklyLinearFactorShrinkageTests(unittest.TestCase):
    def setUp(self) -> None:
        self.frame = pd.DataFrame(
            {
                "candidate_id": [BASE_CANDIDATE_ID] * 3,
                "precipitation_mm_raw": [0.0, 10.0, 20.0],
                "precipitation_mm_qm": [0.0, 5.0, 10.0],
                "weekly_linear_scaling_factor": [0.5, 0.5, 0.5],
            }
        )

    def test_alpha_three_quarters_shrinks_factor_toward_one(self) -> None:
        corrected = shrink_factors(self.frame, 0.75, split="test")
        self.assertTrue(corrected["weekly_linear_scaling_factor"].eq(0.625).all())
        np.testing.assert_allclose(
            corrected["precipitation_mm_qm"], [0.0, 6.25, 12.5]
        )
        self.assertEqual(corrected["candidate_id"].iloc[0], candidate_id(0.75))

    def test_alpha_one_reproduces_base_factor(self) -> None:
        corrected = shrink_factors(self.frame, 1.0, split="test")
        np.testing.assert_allclose(
            corrected["weekly_linear_scaling_factor"],
            self.frame["weekly_linear_scaling_factor"],
        )
        np.testing.assert_allclose(
            corrected["precipitation_mm_qm"], self.frame["precipitation_mm_qm"]
        )


if __name__ == "__main__":
    unittest.main()
