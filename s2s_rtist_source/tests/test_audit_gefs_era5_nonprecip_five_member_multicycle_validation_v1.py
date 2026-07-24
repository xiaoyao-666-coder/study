from __future__ import annotations

import unittest

import pandas as pd

from scripts.diagnostics.audit_gefs_era5_nonprecip_five_member_application_smoke_v1 import (
    BRANCH_RAW,
    BRANCH_SELECTED_AFFINE_SOLAR,
    BRANCH_SELECTED_RAW_SOLAR,
)
from scripts.diagnostics.audit_gefs_era5_nonprecip_five_member_multicycle_validation_v1 import (
    aggregate_cycle_metrics,
    evaluate_performance_gates,
)


class MulticycleValidationTests(unittest.TestCase):
    def metrics_fixture(self, solar_bias: float = 0.4) -> pd.DataFrame:
        rows = []
        variables = [
            "temperature_min_c",
            "temperature_max_c",
            "actual_vapor_pressure_kpa",
            "wind_speed_m_s",
            "solar_kj_m2_day",
        ]
        for year in range(2015, 2020):
            date = f"{year}-07-01"
            for variable in variables:
                for branch, bias, mae, rmse in (
                    (BRANCH_RAW, 1.0, 2.0, 3.0),
                    (BRANCH_SELECTED_RAW_SOLAR, 0.5, 1.5, 2.5),
                    (BRANCH_SELECTED_AFFINE_SOLAR, 0.4, 1.4, 2.4),
                ):
                    if variable == "solar_kj_m2_day":
                        if branch == BRANCH_SELECTED_RAW_SOLAR:
                            bias, mae, rmse = 1.0, 2.0, 3.0
                        elif branch == BRANCH_SELECTED_AFFINE_SOLAR:
                            bias = solar_bias
                    rows.append(
                        {
                            "decision_date": date,
                            "branch_id": branch,
                            "variable": variable,
                            "sample_count": 35,
                            "bias_ensemble_mean_minus_era5": bias,
                            "mae": mae,
                            "rmse": rmse,
                        }
                    )
        return pd.DataFrame(rows)

    def test_all_variables_pass_and_affine_solar_is_recommended(self) -> None:
        cycle = self.metrics_fixture()
        pooled = aggregate_cycle_metrics(cycle)
        gates, policy = evaluate_performance_gates(cycle, pooled)
        self.assertTrue(gates["all_performance_gates_passed"].all())
        self.assertTrue(policy["non_solar_all_performance_gates_passed"])
        self.assertEqual(policy["recommended_solar_branch"], "affine_alpha_0.25")

    def test_solar_bias_failure_keeps_raw_fallback(self) -> None:
        cycle = self.metrics_fixture(solar_bias=1.2)
        pooled = aggregate_cycle_metrics(cycle)
        gates, policy = evaluate_performance_gates(cycle, pooled)
        solar = gates.loc[gates["variable"].eq("solar_kj_m2_day")].iloc[0]
        self.assertFalse(solar["all_performance_gates_passed"])
        self.assertEqual(policy["recommended_solar_branch"], "raw_solar_fallback")


if __name__ == "__main__":
    unittest.main()
