from __future__ import annotations

import unittest

import pandas as pd

from scripts.diagnostics.select_gefs_era5_temperature_center_range_shrinkage_v1 import (
    apply_temperature_candidate,
    select_candidate,
)


class TemperatureCenterRangeShrinkageTests(unittest.TestCase):
    def test_center_and_range_alphas_are_applied_separately(self) -> None:
        pairs = pd.DataFrame(
            [
                {
                    "target_year": 2019,
                    "decision_date": "2019-07-01",
                    "site_id": "P1",
                    "gefs_member": "c00",
                    "local_date": "2019-07-01",
                    "lead_day": 1,
                    "temperature_center_c_gefs": 15.0,
                    "temperature_range_c_gefs": 10.0,
                }
            ]
        )
        factors = pd.DataFrame(
            [
                {
                    "target_year": 2019,
                    "site_id": "P1",
                    "lead_day": 1,
                    "fit_sample_count": 10,
                    "temperature_center_additive_delta_c": 2.0,
                    "temperature_range_ratio": 2.0,
                }
            ]
        )
        corrected = apply_temperature_candidate(
            pairs, factors, center_alpha=0.5, range_alpha=0.0
        ).iloc[0]
        self.assertAlmostEqual(corrected["temperature_min_c"], 11.0)
        self.assertAlmostEqual(corrected["temperature_max_c"], 21.0)

    def test_selection_uses_2015_2018_and_confirms_2019(self) -> None:
        rows = []
        for year in range(2015, 2020):
            for variable in ("temperature_min_c", "temperature_max_c"):
                rows.extend(
                    [
                        {
                            "target_year": year,
                            "candidate_id": "temperature_center_a0_range_a0",
                            "center_alpha": 0.0,
                            "range_alpha": 0.0,
                            "variable": variable,
                            "sample_count": 10,
                            "bias": -1.0,
                            "mae": 2.0,
                            "rmse": 3.0,
                        },
                        {
                            "target_year": year,
                            "candidate_id": "temperature_center_a0.25_range_a0",
                            "center_alpha": 0.25,
                            "range_alpha": 0.0,
                            "variable": variable,
                            "sample_count": 10,
                            "bias": -0.7,
                            "mae": 1.8,
                            "rmse": 2.7,
                        },
                        {
                            "target_year": year,
                            "candidate_id": "temperature_center_a0.25_range_a0.25",
                            "center_alpha": 0.25,
                            "range_alpha": 0.25,
                            "variable": variable,
                            "sample_count": 10,
                            "bias": -0.8,
                            "mae": 1.9 if variable == "temperature_max_c" else 2.1,
                            "rmse": 2.8 if variable == "temperature_max_c" else 3.1,
                        },
                    ]
                )
        selected, audit = select_candidate(pd.DataFrame(rows))
        self.assertEqual(audit["selected_center_alpha"], 0.25)
        self.assertEqual(audit["selected_range_alpha"], 0.0)
        self.assertTrue(audit["selected_candidate_2019_confirmed"])
        self.assertTrue(selected["validation_all_metric_gates"].all())


if __name__ == "__main__":
    unittest.main()
