from __future__ import annotations

import unittest

import pandas as pd

from scripts.data_preparation.finalize_gefs_exact_schedule_2015_frozen_weather_v1 import (
    combine_causal_fit_gefs_history,
)


def rows(decisions: list[str], *, members: list[str]) -> pd.DataFrame:
    output = []
    for decision_text in decisions:
        decision = pd.Timestamp(decision_text)
        for member in members:
            for lead in range(1, 8):
                output.append(
                    {
                        "decision_date": decision.strftime("%Y-%m-%d"),
                        "site_id": "P1",
                        "gefs_member": member,
                        "local_date": (
                            decision + pd.Timedelta(days=lead - 1)
                        ).strftime("%Y-%m-%d"),
                        "lead_day": lead,
                        "temperature_min_c": 10.0,
                        "temperature_max_c": 20.0,
                        "actual_vapor_pressure_kpa": 1.5,
                        "wind_speed_m_s": 3.0,
                        "solar_kj_m2_day": 20_000.0,
                    }
                )
    return pd.DataFrame(output)


class FinalizeExactSchedule2015FrozenWeatherTests(unittest.TestCase):
    def test_combines_eight_preseason_cycles_with_target_c00_only(self) -> None:
        calibration_dates = pd.date_range(
            "2015-01-15", periods=8, freq="14D"
        ).strftime("%Y-%m-%d").tolist()
        calibration = rows(calibration_dates, members=["c00"])
        target = rows(["2015-05-11", "2015-05-18"], members=["c00", "p01"])
        combined, audit = combine_causal_fit_gefs_history(calibration, target)
        self.assertTrue(audit["mandatory_gate_passed"])
        self.assertEqual(audit["completed_preseason_cycle_count"], 8)
        self.assertEqual(len(combined), 70)
        self.assertEqual(set(combined["gefs_member"]), {"c00"})

    def test_rejects_fewer_than_eight_completed_preseason_cycles(self) -> None:
        calibration_dates = pd.date_range(
            "2015-01-15", periods=7, freq="14D"
        ).strftime("%Y-%m-%d").tolist()
        with self.assertRaisesRegex(ValueError, "fewer than eight"):
            combine_causal_fit_gefs_history(
                rows(calibration_dates, members=["c00"]),
                rows(["2015-05-11"], members=["c00"]),
            )


if __name__ == "__main__":
    unittest.main()
