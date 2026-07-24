from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import pandas as pd

from scripts.simulation.run_gefs_corrected_swap_three_output_smoke_v1 import (
    load_contract,
    parse_swap_weather_record,
    patch_swap_weather_file,
)


CONTRACT = Path(
    "site_general_surrogate_eval/gefs_corrected_swap_three_output_smoke_contract_v1.json"
)


class GefsCorrectedSwapThreeOutputSmokeTests(unittest.TestCase):
    def test_contract_locks_causal_boundary_and_blocks_training(self) -> None:
        contract = load_contract(CONTRACT)
        self.assertEqual(contract["weather_replacement_policy"]["history_through"], "2024-07-15")
        self.assertEqual(contract["weather_replacement_policy"]["future_start"], "2024-07-16")
        self.assertEqual(contract["weather_replacement_policy"]["future_end"], "2024-07-22")
        self.assertTrue(contract["weather_replacement_policy"]["predecision_state_must_not_use_future_gefs"])
        self.assertFalse(contract["result_policy"]["training_eligible_if_passed"])
        self.assertEqual(
            contract["continuous_irrigation_constraint_mm"],
            {"minimum": 0.0, "maximum": 60.0},
        )

    def test_weather_patch_changes_only_requested_dates_and_preserves_etref(self) -> None:
        text = (
            "header\n"
            " 'Weather' 15 7 2024 100.0 10.0 20.0 1.0 2.0 3.0 4.0\n"
            " 'Weather' 16 7 2024 101.0 11.0 21.0 1.1 2.1 3.1 4.1\n"
            " 'Weather' 17 7 2024 102.0 12.0 22.0 1.2 2.2 3.2 4.2\n"
        )
        daily = pd.DataFrame(
            [
                {
                    "local_date": "2024-07-16",
                    "solar_kj_m2_day_mean": 200.0,
                    "temperature_min_c_mean": 15.0,
                    "temperature_max_c_mean": 25.0,
                    "actual_vapor_pressure_kpa_mean": 1.5,
                    "wind_speed_m_s_mean": 2.5,
                    "precipitation_mm_corrected_mean": 5.0,
                    "artifact_sha256": "x" * 64,
                },
                {
                    "local_date": "2024-07-17",
                    "solar_kj_m2_day_mean": 201.0,
                    "temperature_min_c_mean": 16.0,
                    "temperature_max_c_mean": 26.0,
                    "actual_vapor_pressure_kpa_mean": 1.6,
                    "wind_speed_m_s_mean": 2.6,
                    "precipitation_mm_corrected_mean": 6.0,
                    "artifact_sha256": "x" * 64,
                },
            ]
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "weather.024"
            path.write_text(text, encoding="utf-8")
            audit = patch_swap_weather_file(path, daily)
            lines = path.read_text(encoding="utf-8").splitlines()
        self.assertEqual(len(audit), 2)
        old = parse_swap_weather_record(lines[1])
        first = parse_swap_weather_record(lines[2])
        second = parse_swap_weather_record(lines[3])
        self.assertEqual(old["precipitation_mm"], 3.0)
        self.assertEqual(first["precipitation_mm"], 5.0)
        self.assertEqual(second["precipitation_mm"], 6.0)
        self.assertEqual(first["etref"], 4.1)
        self.assertEqual(second["etref"], 4.2)


if __name__ == "__main__":
    unittest.main()
