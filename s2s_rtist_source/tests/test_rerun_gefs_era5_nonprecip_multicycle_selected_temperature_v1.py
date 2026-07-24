from __future__ import annotations

import unittest

import pandas as pd

from scripts.diagnostics.audit_gefs_era5_nonprecip_five_member_application_smoke_v1 import (
    BRANCH_RAW,
    BRANCH_SELECTED_RAW_SOLAR,
)
from scripts.diagnostics.rerun_gefs_era5_nonprecip_multicycle_selected_temperature_v1 import (
    replace_selected_temperature,
    selected_temperature_policy,
)


class RerunSelectedTemperatureTests(unittest.TestCase):
    def test_unique_confirmed_temperature_policy_is_loaded(self) -> None:
        selection = pd.DataFrame(
            [
                {"center_alpha": 1.0, "range_alpha": 0.0, "validation_all_metric_gates": True},
                {"center_alpha": 1.0, "range_alpha": 0.0, "validation_all_metric_gates": True},
            ]
        )
        self.assertEqual(selected_temperature_policy(selection), (1.0, 0.0))

    def test_only_selected_branch_temperature_is_replaced(self) -> None:
        keys = {
            "decision_date": "2019-07-01",
            "site_id": "P1",
            "gefs_member": "c00",
            "local_date": "2019-07-01",
            "lead_day": 1,
        }
        weather = pd.DataFrame(
            [
                {
                    **keys,
                    "temperature_min_c": 10.0,
                    "temperature_max_c": 20.0,
                    "actual_vapor_pressure_kpa": 1.0,
                    "wind_speed_m_s": 2.0,
                    "solar_kj_m2_day": 20_000.0,
                }
            ]
        )
        reference = weather.drop(columns="gefs_member").copy()
        base_rows = []
        for branch in (BRANCH_RAW, BRANCH_SELECTED_RAW_SOLAR):
            base_rows.append(
                {
                    "branch_id": branch,
                    **keys,
                    "temperature_min_c": 10.0,
                    "temperature_max_c": 20.0,
                    "actual_vapor_pressure_kpa": 1.0,
                    "wind_speed_m_s": 2.0,
                    "solar_kj_m2_day": 20_000.0,
                }
            )
        factors = pd.DataFrame(
            [
                {
                    "validation_cycle": "2019-07-01",
                    "target_year": 2019,
                    "site_id": "P1",
                    "lead_day": 1,
                    "fit_sample_count": 10,
                    "temperature_center_additive_delta_c": 2.0,
                    "temperature_range_ratio": 1.5,
                }
            ]
        )
        updated = replace_selected_temperature(
            base_output=pd.DataFrame(base_rows),
            five_member_weather=weather,
            cycle_era5=reference,
            factors=factors,
            center_alpha=1.0,
            range_alpha=0.0,
        )
        raw = updated.loc[updated["branch_id"].eq(BRANCH_RAW)].iloc[0]
        selected = updated.loc[updated["branch_id"].eq(BRANCH_SELECTED_RAW_SOLAR)].iloc[0]
        self.assertEqual(raw["temperature_min_c"], 10.0)
        self.assertEqual(raw["temperature_max_c"], 20.0)
        self.assertEqual(selected["temperature_min_c"], 12.0)
        self.assertEqual(selected["temperature_max_c"], 22.0)


if __name__ == "__main__":
    unittest.main()
