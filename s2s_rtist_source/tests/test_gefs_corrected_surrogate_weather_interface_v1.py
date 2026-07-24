from __future__ import annotations

import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.data_preparation.build_gefs_corrected_surrogate_weather_smoke_v1 import (
    actual_vapor_pressure_kpa,
    load_contract,
    to_swap_member_weather,
)


class GefsCorrectedSurrogateWeatherInterfaceTests(unittest.TestCase):
    def test_contract_freezes_outputs_and_irrigation_range(self) -> None:
        contract = load_contract(
            Path(
                "site_general_surrogate_eval/gefs_corrected_surrogate_weather_interface_contract_v1.json"
            )
        )
        self.assertEqual(contract["irrigation_constraint_mm"], {"minimum": 0.0, "maximum": 60.0})
        self.assertIn("aet_7d_mm", contract["surrogate_primary_outputs"])
        self.assertEqual(contract["aet_definition"], "10*(Tact+Eact+Interc)_summed_over_days_1_to_7")

    def test_vpd_is_converted_to_actual_vapor_pressure(self) -> None:
        tmin = pd.Series([10.0])
        tmax = pd.Series([20.0])
        vpd = pd.Series([0.5])
        result = actual_vapor_pressure_kpa(tmin, tmax, vpd)
        saturation = 0.6108 * np.exp((17.27 * 15.0) / (15.0 + 237.3))
        self.assertAlmostEqual(float(result.iloc[0]), saturation - 0.5)

    def test_swap_mapping_converts_shortwave_and_keeps_both_precipitation_fields(self) -> None:
        frame = pd.DataFrame(
            {
                "site_id": ["P1"],
                "site_timezone": ["America/Chicago"],
                "forecast_init_utc": ["2024-07-16T00:00:00Z"],
                "decision_date": ["2024-07-16"],
                "local_date": ["2024-07-16"],
                "lead_day": [1],
                "gefs_member": ["gec00"],
                "temperature_min_c": [10.0],
                "temperature_max_c": [20.0],
                "vpd_kpa": [0.5],
                "wind_speed_m_s": [3.0],
                "shortwave_w_m2": [100.0],
                "precipitation_mm_raw": [4.0],
                "precipitation_mm_qm": [5.0],
                "ensemble_mean_raw_7d_mm": [20.0],
                "raw_ensemble_mean_7d_q90_mm": [30.0],
                "weekly_extreme_regime": [False],
                "weekly_linear_scaling_factor": [1.25],
                "factor_shrinkage_alpha": [0.75],
                "artifact_sha256": ["x" * 64],
            }
        )
        output = to_swap_member_weather(frame)
        self.assertAlmostEqual(float(output["solar_kj_m2_day"].iloc[0]), 8640.0)
        self.assertAlmostEqual(float(output["precipitation_mm_raw"].iloc[0]), 4.0)
        self.assertAlmostEqual(float(output["precipitation_mm_corrected"].iloc[0]), 5.0)


if __name__ == "__main__":
    unittest.main()
