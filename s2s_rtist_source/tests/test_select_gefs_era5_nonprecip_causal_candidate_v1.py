from __future__ import annotations

import unittest

import pandas as pd

from scripts.diagnostics.select_gefs_era5_nonprecip_causal_candidate_v1 import (
    aggregate_metrics,
    select_candidates,
)


def metrics_fixture(validation_candidate_rmse: float = 0.8) -> pd.DataFrame:
    rows = []
    for year in range(2015, 2020):
        rows.extend(
            [
                {
                    "target_year": year,
                    "candidate_id": "raw_gefs",
                    "shrinkage_alpha": 0.0,
                    "variable": "wind_speed_m_s",
                    "sample_count": 10,
                    "bias_corrected_minus_era5": 1.0,
                    "mae": 1.0,
                    "rmse": 1.0,
                },
                {
                    "target_year": year,
                    "candidate_id": "hybrid_linear_scaling_shrink_a0.5",
                    "shrinkage_alpha": 0.5,
                    "variable": "wind_speed_m_s",
                    "sample_count": 10,
                    "bias_corrected_minus_era5": 0.2,
                    "mae": 0.7,
                    "rmse": validation_candidate_rmse if year == 2019 else 0.75,
                },
            ]
        )
    return pd.DataFrame(rows)


class GefsEra5NonprecipCandidateSelectionTests(unittest.TestCase):
    def test_weighted_metric_aggregation(self) -> None:
        aggregated = aggregate_metrics(metrics_fixture(), (2015, 2016, 2017, 2018))
        candidate = aggregated.loc[aggregated["shrinkage_alpha"] == 0.5].iloc[0]
        self.assertAlmostEqual(float(candidate["rmse"]), 0.75)
        self.assertEqual(int(candidate["sample_count"]), 40)

    def test_candidate_is_selected_and_confirmed(self) -> None:
        selected, audit = select_candidates(metrics_fixture())
        self.assertTrue(audit["all_variables_passed"])
        self.assertEqual(
            selected.iloc[0]["selection_status"],
            "selected_and_2019_confirmed",
        )

    def test_2019_failure_blocks_candidate(self) -> None:
        selected, audit = select_candidates(metrics_fixture(validation_candidate_rmse=1.2))
        self.assertFalse(audit["all_variables_passed"])
        self.assertEqual(
            selected.iloc[0]["selection_status"],
            "blocked_selected_candidate_failed_2019",
        )


if __name__ == "__main__":
    unittest.main()
