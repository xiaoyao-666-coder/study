from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from scripts.data_preparation.extract_gefs_era5_nonprecip_causal_history_v1 import (
    CANONICAL_COLUMNS,
    build_causal_cycles,
    resolve_causal_cycles,
    validate_history,
)


class GefsEra5NonprecipCausalHistoryTests(unittest.TestCase):
    def test_causal_schedule_has_sixteen_cycles_per_year(self) -> None:
        cycles = build_causal_cycles([2015, 2016])
        self.assertEqual(len(cycles), 32)
        self.assertEqual(cycles[0], "2015-01-15")
        self.assertEqual(cycles[15], "2015-08-13")
        self.assertEqual(cycles[-1], "2016-08-12")

    def test_explicit_cycles_support_minimum_2015_preseason_history(self) -> None:
        cycles = resolve_causal_cycles(
            years=None,
            explicit_cycles=[
                "2015-01-15",
                "2015-01-29",
                "2015-02-12",
                "2015-02-26",
                "2015-03-12",
                "2015-03-26",
                "2015-04-09",
                "2015-04-23",
            ],
        )
        self.assertEqual(len(cycles), 8)
        self.assertEqual(cycles[0], "2015-01-15")
        self.assertEqual(cycles[-1], "2015-04-23")

    def test_valid_history_fixture_passes(self) -> None:
        cycles = build_causal_cycles([2015])
        rows = []
        for cycle in cycles:
            for site in ["P1", "P2"]:
                for lead_day in range(1, 8):
                    rows.append(
                        {
                            "decision_date": cycle,
                            "site_id": site,
                            "gefs_member": "c00",
                            "lead_day": lead_day,
                            "temperature_min_c": 10.0,
                            "temperature_max_c": 20.0,
                            "actual_vapor_pressure_kpa": 1.0,
                            "wind_speed_m_s": 3.0,
                            "solar_kj_m2_day": 15_000.0,
                        }
                    )
        audit = validate_history(
            pd.DataFrame(rows), cycles=cycles, site_ids=["P1", "P2"], members=["c00"]
        )
        self.assertTrue(audit["mandatory_gate_passed"])
        self.assertEqual(audit["row_count"], 224)

    def test_nonfinite_value_fails(self) -> None:
        cycles = build_causal_cycles([2015])
        rows = []
        for cycle in cycles:
            for lead_day in range(1, 8):
                row = {
                    "decision_date": cycle,
                    "site_id": "P1",
                    "gefs_member": "c00",
                    "lead_day": lead_day,
                    **{column: 1.0 for column in CANONICAL_COLUMNS},
                }
                rows.append(row)
        rows[0]["wind_speed_m_s"] = np.nan
        audit = validate_history(
            pd.DataFrame(rows), cycles=cycles, site_ids=["P1"], members=["c00"]
        )
        self.assertFalse(audit["mandatory_gate_passed"])
        self.assertIn("missing_canonical_values", audit["gate_failures"])


if __name__ == "__main__":
    unittest.main()
