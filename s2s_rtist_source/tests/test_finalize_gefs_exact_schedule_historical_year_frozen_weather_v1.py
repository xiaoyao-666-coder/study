from __future__ import annotations

import unittest

import pandas as pd

from scripts.data_preparation.finalize_gefs_exact_schedule_historical_year_frozen_weather_v1 import (
    prepare_strict_prior_year_history,
)


def weather(year: int, periods: int, members: tuple[str, ...]) -> pd.DataFrame:
    rows = []
    for decision in pd.date_range(f"{year}-01-15", periods=periods, freq="14D"):
        for lead in range(1, 8):
            for member in members:
                rows.append(
                    {
                        "decision_date": decision.strftime("%Y-%m-%d"),
                        "site_id": "P1",
                        "gefs_member": member,
                        "local_date": (decision + pd.Timedelta(days=lead - 1)).strftime(
                            "%Y-%m-%d"
                        ),
                        "lead_day": lead,
                        "temperature_min_c": 10.0,
                        "temperature_max_c": 20.0,
                        "actual_vapor_pressure_kpa": 1.5,
                        "wind_speed_m_s": 3.0,
                        "solar_kj_m2_day": 18_000.0,
                        "precipitation_mm_raw": 1.0,
                    }
                )
    return pd.DataFrame(rows)


class HistoricalYearFrozenWeatherTests(unittest.TestCase):
    def test_2016_uses_c00_from_strictly_prior_year(self) -> None:
        history, audit = prepare_strict_prior_year_history(
            [weather(2015, 8, ("c00", "p01"))],
            weather(2016, 2, ("c00", "p01", "p02", "p03", "p04")),
            minimum_samples=8,
        )
        self.assertTrue(audit["mandatory_gate_passed"])
        self.assertEqual(audit["target_year"], 2016)
        self.assertEqual(audit["history_last_year"], 2015)
        self.assertEqual(audit["minimum_fit_samples_per_site_lead"], 8)
        self.assertEqual(set(history["gefs_member"]), {"c00"})
        self.assertEqual(len(history), 56)

    def test_rejects_target_year_history(self) -> None:
        with self.assertRaisesRegex(ValueError, "target-year or future"):
            prepare_strict_prior_year_history(
                [weather(2016, 8, ("c00",))],
                weather(2016, 2, ("c00", "p01", "p02", "p03", "p04")),
                minimum_samples=8,
            )

    def test_rejects_insufficient_prior_year_history(self) -> None:
        with self.assertRaisesRegex(ValueError, "fewer than the minimum"):
            prepare_strict_prior_year_history(
                [weather(2015, 7, ("c00",))],
                weather(2016, 2, ("c00", "p01", "p02", "p03", "p04")),
                minimum_samples=8,
            )


if __name__ == "__main__":
    unittest.main()
