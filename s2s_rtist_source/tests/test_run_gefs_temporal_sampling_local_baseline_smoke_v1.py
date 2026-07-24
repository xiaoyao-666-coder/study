from __future__ import annotations

import unittest

import pandas as pd

from scripts.diagnostics.run_gefs_temporal_sampling_local_baseline_smoke_v1 import (
    aggregate_lean_state,
    build_lean_weather,
)


def state_points_fixture() -> pd.DataFrame:
    rows = []
    for end_hour in (6, 12, 18, 24):
        for short_name, value in (
            ("SPFH", 0.008),
            ("PRES", 100000.0),
            ("UGRD", 3.0),
            ("VGRD", 4.0),
        ):
            rows.append(
                {
                    "site": "P1",
                    "timezone": "UTC",
                    "cycle_init_utc": "2015-07-06T00:00:00Z",
                    "gefs_member": "c00",
                    "end_hour": end_hour,
                    "short_name": short_name,
                    "value": value,
                }
            )
    return pd.DataFrame(rows)


class LocalBaselineTemporalSamplingTests(unittest.TestCase):
    def test_aggregates_six_hour_humidity_pressure_and_wind(self) -> None:
        state = aggregate_lean_state(state_points_fixture())

        self.assertEqual(int(state["six_hour_sample_count"].sum()), 4)
        self.assertAlmostEqual(float(state["wind_speed_m_s"].iloc[0]), 5.0)
        self.assertGreater(float(state["actual_vapor_pressure_kpa"].iloc[0]), 0.0)

    def test_replaces_only_vapor_pressure_and_wind_in_baseline(self) -> None:
        baseline = pd.DataFrame(
            {
                "decision_date": ["2015-07-06"],
                "site_id": ["P1"],
                "gefs_member": ["c00"],
                "local_date": ["2015-07-06"],
                "lead_day": [1],
                "precipitation_mm_raw": [1.0],
                "temperature_min_c": [10.0],
                "temperature_max_c": [20.0],
                "actual_vapor_pressure_kpa": [1.0],
                "wind_speed_m_s": [2.0],
                "solar_kj_m2_day": [8000.0],
            }
        )
        state = pd.DataFrame(
            {
                "site_id": ["P1"],
                "site_timezone": ["UTC"],
                "gefs_member": ["c00"],
                "local_date": [pd.Timestamp("2015-07-06")],
                "actual_vapor_pressure_kpa": [1.2],
                "wind_speed_m_s": [2.5],
                "six_hour_sample_count": [4],
            }
        )
        lean = build_lean_weather(baseline, state)

        self.assertEqual(float(lean["precipitation_mm_raw"].iloc[0]), 1.0)
        self.assertEqual(float(lean["temperature_min_c"].iloc[0]), 10.0)
        self.assertEqual(float(lean["actual_vapor_pressure_kpa"].iloc[0]), 1.2)
        self.assertEqual(float(lean["wind_speed_m_s"].iloc[0]), 2.5)

    def test_rejects_missing_lean_state_rows(self) -> None:
        baseline = pd.DataFrame(
            {
                "site_id": ["P1"],
                "gefs_member": ["c00"],
                "local_date": ["2015-07-06"],
                "actual_vapor_pressure_kpa": [1.0],
                "wind_speed_m_s": [2.0],
            }
        )
        state = pd.DataFrame(
            columns=[
                "site_id",
                "gefs_member",
                "local_date",
                "actual_vapor_pressure_kpa",
                "wind_speed_m_s",
                "six_hour_sample_count",
            ]
        )

        with self.assertRaisesRegex(ValueError, "does not cover"):
            build_lean_weather(baseline, state)


if __name__ == "__main__":
    unittest.main()
