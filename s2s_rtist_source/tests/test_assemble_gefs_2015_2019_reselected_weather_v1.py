from __future__ import annotations

import unittest

import pandas as pd

from scripts.data_preparation.assemble_gefs_2015_2019_reselected_weather_v1 import (
    assemble_selected_weather,
    selected_cycle_dates,
)


class ReselectedWeatherAssemblyTests(unittest.TestCase):
    def setUp(self) -> None:
        self.contract = {
            "selected_cycles": [
                {"decision_date": f"{year}-07-15"} for year in range(2015, 2020)
            ]
        }

    def test_selected_dates_require_one_cycle_per_year(self) -> None:
        self.assertEqual(
            selected_cycle_dates(self.contract),
            [f"{year}-07-15" for year in range(2015, 2020)],
        )

    def test_assembly_keeps_only_contract_cycles(self) -> None:
        base = {
            "site_id": "P1",
            "gefs_member": "c00",
            "lead_day": 1,
            "local_date": "2018-07-15",
        }
        existing = pd.DataFrame(
            [
                {**base, "decision_date": "2018-07-15"},
                {**base, "decision_date": "2015-08-15"},
            ]
        )
        replacement = pd.DataFrame(
            [
                {
                    **base,
                    "decision_date": f"{year}-07-15",
                    "local_date": f"{year}-07-15",
                }
                for year in [2015, 2016, 2017, 2019]
            ]
        )
        result, counts = assemble_selected_weather(
            existing, replacement, self.contract
        )
        self.assertEqual(len(result), 5)
        self.assertNotIn("2015-08-15", set(result["decision_date"]))
        self.assertEqual(counts["existing_selected_rows"], 1)
        self.assertEqual(counts["replacement_selected_rows"], 4)


if __name__ == "__main__":
    unittest.main()
