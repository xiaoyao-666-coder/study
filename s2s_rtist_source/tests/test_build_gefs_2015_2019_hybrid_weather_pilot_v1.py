from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from scripts.data_preparation.build_gefs_2015_2019_hybrid_weather_pilot_v1 import (
    convert_era5_to_swap,
    splice_hybrid_weather,
)


class HybridWeatherPilotTests(unittest.TestCase):
    def test_era5_unit_conversion(self) -> None:
        source = pd.DataFrame(
            {
                "target_year": [2015],
                "site_id": ["P1"],
                "local_date": ["2015-01-01"],
                "temperature_mean_k": [283.15],
                "temperature_min_k": [278.15],
                "temperature_max_k": [288.15],
                "dewpoint_k": [280.15],
                "solar_j_m2_day": [10_000_000.0],
                "precipitation_m": [0.012],
                "potential_evaporation_m": [-0.003],
                "wind_u_m_s": [3.0],
                "wind_v_m_s": [4.0],
            }
        )
        result = convert_era5_to_swap(source).iloc[0]
        self.assertAlmostEqual(float(result["temperature_min_c"]), 5.0)
        self.assertAlmostEqual(float(result["temperature_max_c"]), 15.0)
        self.assertAlmostEqual(float(result["solar_kj_m2_day"]), 10000.0)
        self.assertAlmostEqual(float(result["precipitation_mm"]), 12.0)
        self.assertAlmostEqual(float(result["etref_mm"]), 3.0)
        self.assertAlmostEqual(float(result["wind_speed_m_s"]), 5.0)

    def test_splice_replaces_only_future_dates(self) -> None:
        dates = pd.date_range("2015-08-14", periods=9, freq="D")
        era5 = pd.DataFrame(
            {
                "target_year": 2015,
                "site_id": "P1",
                "local_date": dates.strftime("%Y-%m-%d"),
                "solar_kj_m2_day": 1.0,
                "temperature_min_c": 2.0,
                "temperature_max_c": 3.0,
                "actual_vapor_pressure_kpa": 4.0,
                "wind_speed_m_s": 5.0,
                "precipitation_mm": 6.0,
                "etref_mm": 7.0,
                "weather_source": "ERA5_Land_corresponding_year_predecision",
            }
        )
        future_dates = pd.date_range("2015-08-15", periods=7, freq="D")
        corrected = pd.DataFrame(
            {
                "target_year": 2015,
                "decision_date": "2015-08-15",
                "site_id": "P1",
                "local_date": future_dates.strftime("%Y-%m-%d"),
                "lead_day": range(1, 8),
                "solar_kj_m2_day_mean": 11.0,
                "temperature_min_c_mean": 12.0,
                "temperature_max_c_mean": 13.0,
                "actual_vapor_pressure_kpa_mean": 14.0,
                "wind_speed_m_s_mean": 15.0,
                "precipitation_mm_corrected_mean": 16.0,
            }
        )
        # Replicate to satisfy the formal 175-row future gate.
        era5_all = []
        corrected_all = []
        for year in range(2015, 2020):
            for site in ["P1", "P2", "P3", "P4", "P15"]:
                e = era5.copy()
                e["target_year"] = year
                e["site_id"] = site
                e["local_date"] = pd.date_range(
                    f"{year}-08-14", periods=9, freq="D"
                ).strftime("%Y-%m-%d")
                c = corrected.copy()
                c["target_year"] = year
                c["site_id"] = site
                c["decision_date"] = f"{year}-08-15"
                c["local_date"] = pd.date_range(
                    f"{year}-08-15", periods=7, freq="D"
                ).strftime("%Y-%m-%d")
                era5_all.append(e)
                corrected_all.append(c)
        output, audit = splice_hybrid_weather(
            pd.concat(era5_all, ignore_index=True),
            pd.concat(corrected_all, ignore_index=True),
        )
        target = output.loc[
            (output["target_year"] == 2015) & (output["site_id"] == "P1")
        ]
        self.assertEqual(float(target.iloc[0]["precipitation_mm"]), 6.0)
        self.assertTrue((target.iloc[1:8]["precipitation_mm"] == 16.0).all())
        self.assertEqual(float(target.iloc[-1]["precipitation_mm"]), 6.0)
        self.assertEqual(audit["predecision_gefs_rows"], 0)
        self.assertTrue(np.isclose(audit["maximum_absolute_future_splice_error"], 0.0))


if __name__ == "__main__":
    unittest.main()
