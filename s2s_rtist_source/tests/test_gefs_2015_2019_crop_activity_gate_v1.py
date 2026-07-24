from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import pandas as pd

from scripts.simulation.audit_gefs_2015_2019_crop_activity_gate_v1 import (
    candidate_cycles,
    parse_last_crop_row,
    parse_restart_crop_flags,
    summarize_cycle_gate,
)


class CropActivityGateTests(unittest.TestCase):
    def test_candidate_cycles_expands_two_dates_for_five_years(self) -> None:
        contract = {
            "cycle_selection": {"candidate_month_days": ["07-15", "08-15"]},
            "selected_cycles": [
                {
                    "target_year": year,
                    "decision_date": f"{year}-08-15",
                    "split": "training_oof" if year < 2019 else "validation",
                    "fit_first_year": 2000,
                    "fit_last_year": year - 1,
                }
                for year in range(2015, 2020)
            ],
        }
        result = candidate_cycles(contract)
        self.assertEqual(len(result), 10)
        self.assertEqual(int(result["previously_selected"].sum()), 5)
        self.assertEqual(len(result.loc[result["previously_selected"]]), 5)

    def test_parses_harvested_predecision_state(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            crop = root / "result_forec.crp"
            crop.write_text(
                "header\n2015-08-09,221,161,1.99\n"
                "2015-08-10,222,162,2.00\n",
                encoding="utf-8",
            )
            end = root / "restart_initial.end"
            end.write_text(
                " swCropEmergence = 0\n swcropharvest = 1\n",
                encoding="utf-8",
            )
            crop_state = parse_last_crop_row(crop)
            flags = parse_restart_crop_flags(end)
        self.assertEqual(crop_state["last_crop_date"], "2015-08-10")
        self.assertEqual(crop_state["last_crop_dvs"], 2.0)
        self.assertEqual(flags["sw_crop_harvest"], 1)

    def test_cycle_requires_all_five_sites(self) -> None:
        frame = pd.DataFrame(
            {
                "target_year": [2015] * 5,
                "decision_date": ["2015-07-15"] * 5,
                "site_id": ["P1", "P2", "P3", "P4", "P15"],
                "previously_selected": [False] * 5,
                "screening_gate_passed": [True, True, True, True, False],
                "future_weather_role": ["era5_screening_only"] * 5,
            }
        )
        result = summarize_cycle_gate(frame, 5).iloc[0]
        self.assertFalse(bool(result["all_five_sites_screening_eligible"]))
        self.assertEqual(result["failed_sites"], "P15")
        self.assertTrue(bool(result["uses_provisional_era5_future"]))


if __name__ == "__main__":
    unittest.main()
