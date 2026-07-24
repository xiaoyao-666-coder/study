from __future__ import annotations

import unittest

import pandas as pd

from scripts.diagnostics.run_gefs_era5_nonprecip_causal_correction_v1 import (
    apply_candidate,
    fit_target_factors,
    prepare_pairs,
)


def pair_fixture() -> pd.DataFrame:
    gefs_rows = []
    era5_rows = []
    dates = list(pd.date_range("2015-01-15", periods=10, freq="14D")) + [
        pd.Timestamp("2016-05-15")
    ]
    for decision in dates:
        year = decision.year
        row = {
            "target_year": year,
            "decision_date": decision.strftime("%Y-%m-%d"),
            "site_id": "P1",
            "local_date": decision.strftime("%Y-%m-%d"),
            "lead_day": 1,
        }
        gefs_rows.append(
            {
                **row,
                "temperature_min_c": 10.0,
                "temperature_max_c": 20.0,
                "actual_vapor_pressure_kpa": 2.0,
                "wind_speed_m_s": 4.0,
                "solar_kj_m2_day": 20_000.0,
            }
        )
        era5_rows.append(
            {
                **row,
                "temperature_min_c": 12.0,
                "temperature_max_c": 22.0,
                "actual_vapor_pressure_kpa": 1.0,
                "wind_speed_m_s": 2.0,
                "solar_kj_m2_day": 10_000.0,
            }
        )
    return prepare_pairs(pd.DataFrame(gefs_rows), pd.DataFrame(era5_rows))


class GefsEra5NonprecipCausalCorrectionTests(unittest.TestCase):
    def test_2015_requires_eight_completed_prior_cycles(self) -> None:
        pairs = pair_fixture()
        eighth_target = "2015-04-23"
        ninth_target = "2015-05-07"
        self.assertIsNone(
            fit_target_factors(
                pairs, target_date=eighth_target, minimum_samples=8
            )
        )
        factors = fit_target_factors(
            pairs, target_date=ninth_target, minimum_samples=8
        )
        self.assertIsNotNone(factors)
        self.assertEqual(int(factors.iloc[0]["fit_sample_count"]), 8)

    def test_2016_fit_uses_only_2015(self) -> None:
        pairs = pair_fixture()
        factors = fit_target_factors(
            pairs, target_date="2016-05-15", minimum_samples=8
        )
        self.assertIsNotNone(factors)
        self.assertEqual(int(factors.iloc[0]["fit_first_year"]), 2015)
        self.assertEqual(int(factors.iloc[0]["fit_last_year"]), 2015)

    def test_linear_scaling_candidate_preserves_temperature_order(self) -> None:
        pairs = pair_fixture()
        target_date = "2016-05-15"
        factors = fit_target_factors(
            pairs, target_date=target_date, minimum_samples=8
        )
        target = pairs.loc[pairs["decision_date"] == target_date]
        corrected = apply_candidate(target, factors, alpha=1.0)
        row = corrected.iloc[0]
        self.assertAlmostEqual(float(row["temperature_min_c_corrected"]), 12.0)
        self.assertAlmostEqual(float(row["temperature_max_c_corrected"]), 22.0)
        self.assertAlmostEqual(float(row["actual_vapor_pressure_kpa_corrected"]), 1.0)
        self.assertAlmostEqual(float(row["solar_kj_m2_day_corrected"]), 10_000.0)
        self.assertLessEqual(
            float(row["temperature_min_c_corrected"]),
            float(row["temperature_max_c_corrected"]),
        )

    def test_solar_affine_fallback_handles_constant_training_predictor(self) -> None:
        pairs = pair_fixture()
        target_date = "2016-05-15"
        factors = fit_target_factors(
            pairs, target_date=target_date, minimum_samples=8
        )
        self.assertAlmostEqual(
            float(factors.iloc[0]["solar_kj_m2_day_affine_slope"]), 1.0
        )
        self.assertAlmostEqual(
            float(factors.iloc[0]["solar_kj_m2_day_affine_intercept"]), -10_000.0
        )

    def test_solar_affine_slope_is_clipped_at_zero(self) -> None:
        pairs = pair_fixture()
        training_mask = pairs["decision_date"] < "2016-05-15"
        training_indices = pairs.index[training_mask]
        pairs.loc[training_indices, "solar_kj_m2_day_gefs"] = range(
            10_000,
            10_000 + len(training_indices) * 1_000,
            1_000,
        )
        pairs.loc[training_indices, "solar_kj_m2_day_era5"] = range(
            20_000,
            20_000 - len(training_indices) * 1_000,
            -1_000,
        )
        factors = fit_target_factors(
            pairs,
            target_date="2016-05-15",
            minimum_samples=8,
        )
        self.assertIsNotNone(factors)
        self.assertAlmostEqual(
            float(factors.iloc[0]["solar_kj_m2_day_affine_slope"]), 0.0
        )


if __name__ == "__main__":
    unittest.main()
