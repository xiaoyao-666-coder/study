from __future__ import annotations

import unittest

import pandas as pd

from scripts.diagnostics.select_gefs_era5_nonprecip_robust_candidate_v2 import (
    select_robust_candidates,
)


VARIABLES = [
    "actual_vapor_pressure_kpa",
    "solar_kj_m2_day",
    "temperature_max_c",
    "temperature_min_c",
    "wind_speed_m_s",
]


def metrics_fixture() -> pd.DataFrame:
    rows = []
    for year in range(2015, 2020):
        for variable in VARIABLES:
            rows.append(
                {
                    "target_year": year,
                    "candidate_id": "raw_gefs",
                    "shrinkage_alpha": 0.0,
                    "variable": variable,
                    "sample_count": 10,
                    "bias_corrected_minus_era5": 1.0,
                    "mae": 10.0,
                    "rmse": 12.0,
                }
            )
            for alpha, mae, rmse in ((0.25, 9.0, 10.0), (0.75, 9.5, 9.0)):
                bias = 0.8
                if variable == "solar_kj_m2_day" and year == 2019:
                    bias = 1.05 if alpha == 0.25 else 1.2
                rows.append(
                    {
                        "target_year": year,
                        "candidate_id": f"hybrid_affine_solar_shrink_a{alpha:g}",
                        "shrinkage_alpha": alpha,
                        "variable": variable,
                        "sample_count": 10,
                        "bias_corrected_minus_era5": bias,
                        "mae": mae,
                        "rmse": rmse,
                    }
                )
    return pd.DataFrame(rows)


class RobustCandidateSelectionTests(unittest.TestCase):
    def test_solar_uses_smallest_stable_nonzero_alpha(self) -> None:
        selected, _, audit = select_robust_candidates(metrics_fixture())
        solar = selected.loc[selected["variable"].eq("solar_kj_m2_day")].iloc[0]
        self.assertEqual(float(solar["shrinkage_alpha"]), 0.25)
        self.assertEqual(
            solar["selection_status"],
            "selected_for_five_member_raw_sensitivity_bias_tradeoff",
        )
        self.assertTrue(audit["five_member_sensitivity_allowed"])
        self.assertFalse(audit["strict_all_variables_passed"])
        self.assertFalse(audit["solar_2019_strict_bias_gate_passed"])

    def test_non_solar_variables_remain_strictly_confirmed(self) -> None:
        selected, _, _ = select_robust_candidates(metrics_fixture())
        statuses = selected.loc[
            ~selected["variable"].eq("solar_kj_m2_day"), "selection_status"
        ]
        self.assertTrue(statuses.eq("selected_and_2019_confirmed").all())


if __name__ == "__main__":
    unittest.main()
