from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import pandas as pd

from scripts.diagnostics.review_gefs_exact_schedule_2015_2019_corrected_weather_packaging_v1 import (
    normalize_dates,
    pairwise_inversions,
    paths_for_year,
)


class UnifiedFrozenWeatherReviewTests(unittest.TestCase):
    def test_paths_use_special_2015_and_strict_prior_year_names(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            paths_2015 = paths_for_year(root / "raw", root / "frozen", 2015)
            paths_2019 = paths_for_year(root / "raw", root / "frozen", 2019)

        self.assertEqual(
            paths_2015["causal_audit"].name,
            "gefs_exact_schedule_2015_causal_fit_c00_audit_v1.json",
        )
        self.assertEqual(
            paths_2019["causal_audit"].name,
            "gefs_exact_schedule_2019_prior_year_fit_c00_audit_v1.json",
        )
        self.assertEqual(
            paths_2019["raw"].name,
            "gefs_exact_schedule_2019_raw_full_weather_v1.csv",
        )

    def test_normalization_and_inversion_check(self) -> None:
        frame = normalize_dates(
            pd.DataFrame(
                {
                    "decision_date": ["2019-05-01", "2019-05-01"],
                    "site_id": ["P1", "P1"],
                    "gefs_member": ["c00", "p01"],
                    "local_date": ["2019-05-01", "2019-05-01"],
                    "lead_day": [1, 1],
                    "raw": [1.0, 2.0],
                    "corrected": [10.0, 20.0],
                }
            )
        )
        self.assertEqual(frame["decision_date"].tolist(), ["2019-05-01"] * 2)
        self.assertEqual(pairwise_inversions(frame, "raw", "corrected"), 0)
        frame.loc[1, "corrected"] = 5.0
        self.assertEqual(pairwise_inversions(frame, "raw", "corrected"), 1)


if __name__ == "__main__":
    unittest.main()
