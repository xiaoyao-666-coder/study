from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import pandas as pd

from scripts.simulation.run_gefs_2015_2019_scenario_consistent_swap_pilot_v1 import (
    validate_crop_activity_gate,
    weather_extension,
    write_weather_file,
)
from scripts.simulation.run_gefs_corrected_swap_three_output_smoke_v1 import (
    parse_swap_weather_record,
)


class HistoricalSwapPilotTests(unittest.TestCase):
    def test_weather_extension(self) -> None:
        self.assertEqual(weather_extension(2015), ".015")
        self.assertEqual(weather_extension(2019), ".019")

    def test_weather_writer_retains_six_decimal_precipitation(self) -> None:
        weather = pd.DataFrame(
            {
                "local_date": ["2015-08-15"],
                "solar_kj_m2_day": [12345.678901],
                "temperature_min_c": [10.1],
                "temperature_max_c": [20.2],
                "actual_vapor_pressure_kpa": [1.234567],
                "wind_speed_m_s": [3.4],
                "precipitation_mm": [0.123456],
                "etref_mm": [2.0],
            }
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "weather.015"
            write_weather_file(path, weather)
            records = [
                parse_swap_weather_record(line)
                for line in path.read_text(encoding="utf-8").splitlines()
            ]
        records = [record for record in records if record is not None]
        self.assertEqual(len(records), 1)
        self.assertAlmostEqual(records[0]["precipitation_mm"], 0.123456)

    def test_formal_runner_rejects_provisional_crop_gate(self) -> None:
        contract = {
            "selected_cycles": [
                {
                    "target_year": 2015,
                    "decision_date": "2015-07-15",
                    "split": "training_oof",
                    "fit_first_year": 2000,
                    "fit_last_year": 2014,
                }
            ]
        }
        gate = pd.DataFrame(
            [
                {
                    "target_year": 2015,
                    "decision_date": "2015-07-15",
                    "all_five_sites_screening_eligible": True,
                    "uses_provisional_era5_future": True,
                }
            ]
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "gate.csv"
            gate.to_csv(path, index=False)
            with self.assertRaisesRegex(ValueError, "final crop recheck"):
                validate_crop_activity_gate(path, contract)

    def test_formal_runner_accepts_final_five_site_crop_gate(self) -> None:
        contract = {
            "selected_cycles": [
                {
                    "target_year": 2015,
                    "decision_date": "2015-07-15",
                    "split": "training_oof",
                    "fit_first_year": 2000,
                    "fit_last_year": 2014,
                }
            ]
        }
        gate = pd.DataFrame(
            [
                {
                    "target_year": 2015,
                    "decision_date": "2015-07-15",
                    "all_five_sites_screening_eligible": True,
                    "uses_provisional_era5_future": False,
                }
            ]
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "gate.csv"
            gate.to_csv(path, index=False)
            checked = validate_crop_activity_gate(path, contract)
        self.assertEqual(len(checked), 1)


if __name__ == "__main__":
    unittest.main()
