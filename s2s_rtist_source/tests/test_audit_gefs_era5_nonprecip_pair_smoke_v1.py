from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from scripts.data_preparation.audit_gefs_era5_nonprecip_pair_smoke_v1 import (
    build_audit,
    build_gefs_ensemble_daily,
    build_metrics,
    normalize_era5_reference,
    pair_gefs_era5,
    saturation_vapor_pressure_kpa,
)


def legacy_era5_fixture() -> pd.DataFrame:
    dates = pd.date_range("2015-07-06", periods=7, freq="D")
    return pd.DataFrame(
        {
            "Date": dates.strftime("%m/%d/%Y"),
            "Year": 2015,
            "DOY": dates.dayofyear,
            "Solar": 20_000.0,
            "T-max": 20.0,
            "T-min": 10.0,
            "RelHum": 1.25,
            "Precip": 0.0,
            "ETref": 3.0,
            "WindSpeed": 3.0,
        }
    )


def gefs_member_fixture() -> pd.DataFrame:
    rows = []
    for member_index, member in enumerate(["c00", "p01", "p02", "p03", "p04"]):
        for lead_day in range(1, 8):
            rows.append(
                {
                    "site_id": "P1",
                    "decision_date": "2015-07-06",
                    "local_date": (
                        pd.Timestamp("2015-07-06")
                        + pd.Timedelta(days=lead_day - 1)
                    ).strftime("%Y-%m-%d"),
                    "lead_day": lead_day,
                    "gefs_member": member,
                    "temperature_min_c": 10.0 + member_index,
                    "temperature_max_c": 20.0 + member_index,
                    "actual_vapor_pressure_kpa": 1.0 + member_index / 10.0,
                    "wind_speed_m_s": 3.0 + member_index / 10.0,
                    "solar_kj_m2_day": 20_000.0 + member_index,
                }
            )
    return pd.DataFrame(rows)


class GefsEra5NonprecipPairSmokeTests(unittest.TestCase):
    def test_legacy_df_era_is_converted_to_canonical_units(self) -> None:
        normalized, metadata = normalize_era5_reference(
            legacy_era5_fixture(), "P1"
        )
        self.assertEqual(
            metadata["era5_input_schema"], "current_df_era_swap_weather"
        )
        self.assertEqual(
            metadata["era5_relative_humidity_scale"],
            "actual_vapor_pressure_kpa",
        )
        self.assertAlmostEqual(float(normalized.iloc[0]["solar_kj_m2_day"]), 20_000.0)
        self.assertAlmostEqual(
            float(normalized.iloc[0]["actual_vapor_pressure_kpa"]), 1.25
        )

    def test_member_weather_is_reduced_to_seven_five_member_days(self) -> None:
        daily = build_gefs_ensemble_daily(
            gefs_member_fixture(), "P1", "2015-07-06"
        )
        self.assertEqual(len(daily), 7)
        self.assertTrue((daily["member_count"] == 5).all())
        self.assertTrue(np.allclose(daily["temperature_min_c"], 12.0))

    def test_complete_pair_passes_audit(self) -> None:
        daily = build_gefs_ensemble_daily(
            gefs_member_fixture(), "P1", "2015-07-06"
        )
        era5, metadata = normalize_era5_reference(legacy_era5_fixture(), "P1")
        paired = pair_gefs_era5(daily, era5)
        metrics = build_metrics(paired)
        audit = build_audit(paired, metrics, metadata)
        self.assertEqual(len(paired), 7)
        self.assertEqual(len(metrics), 5)
        self.assertEqual(
            audit["status"], "gefs_era5_nonprecip_pair_smoke_passed"
        )
        self.assertEqual(audit["variable_pair_rows"], 35)

    def test_missing_era5_horizon_day_is_rejected(self) -> None:
        daily = build_gefs_ensemble_daily(
            gefs_member_fixture(), "P1", "2015-07-06"
        )
        era5, _ = normalize_era5_reference(legacy_era5_fixture().iloc[:-1], "P1")
        with self.assertRaisesRegex(ValueError, "complete GEFS horizon"):
            pair_gefs_era5(daily, era5)

    def test_invalid_relative_humidity_is_rejected(self) -> None:
        source = legacy_era5_fixture()
        source.loc[0, "RelHum"] = -0.1
        with self.assertRaisesRegex(ValueError, "finite and nonnegative"):
            normalize_era5_reference(source, "P1")

    def test_percent_relative_humidity_is_supported(self) -> None:
        source = legacy_era5_fixture()
        source = source.rename(columns={"ETref": "ET"})
        source["RelHum"] = 50.0
        normalized, metadata = normalize_era5_reference(source, "P1")
        self.assertEqual(
            metadata["era5_relative_humidity_scale"],
            "relative_humidity_percent_0_to_100",
        )
        expected = float(saturation_vapor_pressure_kpa(pd.Series([15.0])).iloc[0]) * 0.5
        self.assertAlmostEqual(
            float(normalized.iloc[0]["actual_vapor_pressure_kpa"]), expected
        )

    def test_scale_ratio_outlier_is_rejected(self) -> None:
        daily = build_gefs_ensemble_daily(
            gefs_member_fixture(), "P1", "2015-07-06"
        )
        era5, metadata = normalize_era5_reference(legacy_era5_fixture(), "P1")
        era5["solar_kj_m2_day"] = 20.0
        paired = pair_gefs_era5(daily, era5)
        metrics = build_metrics(paired)
        audit = build_audit(paired, metrics, metadata)
        self.assertFalse(audit["mandatory_gate_passed"])
        self.assertIn(
            "positive_variable_scale_ratio_outlier_count",
            audit["gate_failures"],
        )


if __name__ == "__main__":
    unittest.main()
