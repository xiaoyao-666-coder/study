from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from scripts.data_preparation.build_gefs_2015_2019_corrected_weather_pilot_v1 import (
    apply_frozen_factors,
    build_audit,
    build_ensemble_daily,
)


def member_fixture(raw_total: float = 40.0) -> pd.DataFrame:
    rows = []
    cycles = {
        2015: "2015-07-15",
        2016: "2016-07-15",
        2017: "2017-07-15",
        2018: "2018-07-15",
        2019: "2019-07-15",
    }
    for year, decision_date in cycles.items():
        for site_id in ["P1", "P2", "P3", "P4", "P15"]:
            start = pd.Timestamp(decision_date)
            for member in ["c00", "p01", "p02", "p03", "p04"]:
                for lead_day in range(1, 8):
                    rows.append(
                        {
                            "decision_date": decision_date,
                            "site_id": site_id,
                            "site_timezone": "America/Chicago",
                            "gefs_member": member,
                            "local_date": (start + pd.Timedelta(days=lead_day - 1)).strftime(
                                "%Y-%m-%d"
                            ),
                            "lead_day": lead_day,
                            "precipitation_mm_raw": raw_total / 7.0,
                            "temperature_min_c": 10.0,
                            "temperature_max_c": 20.0,
                            "actual_vapor_pressure_kpa": 1.0,
                            "wind_speed_m_s": 3.0,
                            "solar_kj_m2_day": 15000.0,
                        }
                    )
    return pd.DataFrame(rows)


def factor_fixture(q90: float = 44.0) -> pd.DataFrame:
    cycles = {
        2015: "2015-07-15",
        2016: "2016-07-15",
        2017: "2017-07-15",
        2018: "2018-07-15",
        2019: "2019-07-15",
    }
    return pd.DataFrame(
        {
            "target_year": year,
            "decision_date": decision_date,
            "site_id": site_id,
            "fit_first_year": 2000,
            "fit_last_year": year - 1,
            "raw_ensemble_mean_7d_q90_mm": q90,
            "overall_factor": 0.8,
            "final_extreme_factor": 0.6,
        }
        for year, decision_date in cycles.items()
        for site_id in ["P1", "P2", "P3", "P4", "P15"]
    )


class CorrectedWeatherPilotTests(unittest.TestCase):
    def test_general_regime_uses_shrunk_overall_factor(self) -> None:
        corrected, weekly = apply_frozen_factors(
            member_fixture(raw_total=40.0), factor_fixture(q90=44.0), 0.75
        )
        target = weekly.loc[
            (weekly["target_year"] == 2015) & (weekly["site_id"] == "P2")
        ].iloc[0]
        self.assertFalse(bool(target["weekly_extreme_regime"]))
        self.assertAlmostEqual(float(target["effective_factor"]), 0.85)
        self.assertTrue(
            np.allclose(
                corrected["precipitation_mm_corrected"],
                corrected["precipitation_mm_raw"] * 0.85,
            )
        )

    def test_extreme_regime_uses_shrunk_extreme_factor(self) -> None:
        _, weekly = apply_frozen_factors(
            member_fixture(raw_total=50.0), factor_fixture(q90=44.0), 0.75
        )
        self.assertTrue(weekly["weekly_extreme_regime"].all())
        self.assertTrue(np.allclose(weekly["effective_factor"], 0.7))

    def test_ensemble_daily_preserves_passthrough_means(self) -> None:
        corrected, _ = apply_frozen_factors(
            member_fixture(raw_total=40.0), factor_fixture(q90=44.0), 0.75
        )
        daily = build_ensemble_daily(corrected)
        self.assertEqual(len(daily), 175)
        self.assertTrue((daily["member_count"] == 5).all())
        self.assertTrue((daily["temperature_min_c_mean"] == 10.0).all())

    def test_duplicate_member_key_is_rejected(self) -> None:
        source = member_fixture(raw_total=40.0)
        source.iloc[1] = source.iloc[0]
        with self.assertRaisesRegex(ValueError, "875 unique"):
            apply_frozen_factors(source, factor_fixture(), 0.75)

    def test_old_evidence_is_compared_only_for_the_same_cycle(self) -> None:
        corrected, weekly = apply_frozen_factors(
            member_fixture(raw_total=40.0), factor_fixture(q90=44.0), 0.75
        )
        daily = build_ensemble_daily(corrected)
        old = weekly[
            [
                "target_year",
                "decision_date",
                "site_id",
                "raw_ensemble_mean_7d_mm",
                "effective_factor",
            ]
        ].copy()
        old.loc[old["target_year"] <= 2017, "decision_date"] = old.loc[
            old["target_year"] <= 2017, "target_year"
        ].astype(str) + "-08-15"
        audit, comparison = build_audit(corrected, daily, weekly, old)
        self.assertEqual(audit["old_utc_evidence_comparable_rows"], 10)
        self.assertEqual(audit["old_utc_evidence_factor_change_count"], 0)
        self.assertEqual(
            int(comparison["old_utc_evidence_comparison_available"].sum()), 10
        )


if __name__ == "__main__":
    unittest.main()
