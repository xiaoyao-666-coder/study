from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from scripts.data_preparation.build_gefs_2015_2019_frozen_all_variable_weather_v1 import (
    EXPECTED_POLICY,
    MEMBER_KEYS,
    NONPRECIP_BRANCH,
    build_frozen_weather,
    load_causal_precipitation_factors,
)


SITES = ["P1", "P2", "P3", "P4", "P15"]
MEMBERS = ["c00", "p01", "p02", "p03", "p04"]


def raw_weather_fixture() -> pd.DataFrame:
    rows = []
    for year in range(2015, 2020):
        decision = pd.Timestamp(f"{year}-07-06")
        for site in SITES:
            for member_index, member in enumerate(MEMBERS):
                for lead_day in range(1, 8):
                    rows.append(
                        {
                            "site_id": site,
                            "site_timezone": "America/Chicago",
                            "forecast_init_utc": decision.strftime("%Y-%m-%dT00:00:00Z"),
                            "decision_date": decision.strftime("%Y-%m-%d"),
                            "gefs_member": member,
                            "local_date": (
                                decision + pd.Timedelta(days=lead_day - 1)
                            ).strftime("%Y-%m-%d"),
                            "lead_day": lead_day,
                            "precipitation_mm_raw": 1.0 + member_index,
                            "temperature_min_c": 10.0 + member_index,
                            "temperature_max_c": 20.0 + member_index,
                            "actual_vapor_pressure_kpa": 1.0 + member_index * 0.1,
                            "wind_speed_m_s": 2.0 + member_index * 0.1,
                            "solar_kj_m2_day": 15000.0 + member_index * 100.0,
                        }
                    )
    return pd.DataFrame(rows)


def branch_fixture(raw: pd.DataFrame) -> pd.DataFrame:
    parts = []
    for branch_id in ["raw_all_nonprecip", NONPRECIP_BRANCH]:
        branch = raw[MEMBER_KEYS + [
            "temperature_min_c",
            "temperature_max_c",
            "actual_vapor_pressure_kpa",
            "wind_speed_m_s",
            "solar_kj_m2_day",
            "precipitation_mm_raw",
        ]].copy()
        if branch_id == NONPRECIP_BRANCH:
            branch["temperature_min_c"] += 1.0
            branch["temperature_max_c"] += 1.0
            branch["actual_vapor_pressure_kpa"] *= 0.9
            branch["wind_speed_m_s"] *= 0.8
            branch["solar_kj_m2_day"] *= 0.95
        branch.insert(0, "branch_id", branch_id)
        parts.append(branch)
    return pd.concat(parts, ignore_index=True)


def policy_fixture() -> dict[str, object]:
    return {
        **EXPECTED_POLICY,
        "policy_freeze_allowed": True,
        "recommended_solar_branch": "affine_alpha_0.25",
        "temperature_selection_uses_2019": False,
        "temperature_2019_confirmation_passed": True,
    }


def factor_inputs() -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    for year in range(2015, 2020):
        for site in SITES:
            rows.append(
                {
                    "candidate_id": "weekly_two_stage_linear_site_only",
                    "validation_year": year,
                    "site_id": site,
                    "fit_first_year": 2000,
                    "fit_last_year": year - 1,
                    "validation_rows_used_for_fit": 0,
                    "raw_ensemble_mean_7d_q90_mm": 1000.0,
                    "overall_factor": 0.8,
                    "final_extreme_factor": 0.5,
                }
            )
    frame = pd.DataFrame(rows)
    return frame.loc[frame["validation_year"] < 2019], frame.loc[
        frame["validation_year"] == 2019
    ]


class FrozenAllVariableWeatherTests(unittest.TestCase):
    def test_combines_frozen_nonprecipitation_and_causal_precipitation(self) -> None:
        raw = raw_weather_fixture()
        cv, validation = factor_inputs()
        factors = load_causal_precipitation_factors(cv, validation)
        output, weekly, audit = build_frozen_weather(
            raw_weather=raw,
            nonprecip_branches=branch_fixture(raw),
            nonprecip_policy=policy_fixture(),
            precipitation_factors=factors,
        )
        self.assertTrue(audit["mandatory_structural_gate_passed"])
        self.assertEqual(len(output), 875)
        self.assertEqual(len(weekly), 25)
        self.assertTrue(np.allclose(output["precipitation_mm"], output["precipitation_mm_raw"] * 0.85))
        self.assertTrue(np.allclose(output["temperature_min_c"], output["temperature_min_c_raw"] + 1.0))
        self.assertEqual(audit["precipitation_factor_fit_leakage_rows"], 0)
        self.assertEqual(audit["precipitation_member_order_inversion_count"], 0)

    def test_extreme_regime_uses_shrunk_extreme_factor(self) -> None:
        raw = raw_weather_fixture()
        cv, validation = factor_inputs()
        cv["raw_ensemble_mean_7d_q90_mm"] = 1.0
        validation["raw_ensemble_mean_7d_q90_mm"] = 1.0
        factors = load_causal_precipitation_factors(cv, validation)
        output, _, _ = build_frozen_weather(
            raw_weather=raw,
            nonprecip_branches=branch_fixture(raw),
            nonprecip_policy=policy_fixture(),
            precipitation_factors=factors,
        )
        self.assertTrue(np.allclose(output["precipitation_mm"], output["precipitation_mm_raw"] * 0.625))

    def test_accepts_one_formal_year_with_multiple_exact_schedule_cycles(self) -> None:
        raw = raw_weather_fixture().loc[
            lambda frame: frame["decision_date"].str.startswith("2015-")
        ].copy()
        second = raw.copy()
        second["decision_date"] = "2015-07-13"
        second["forecast_init_utc"] = "2015-07-13T00:00:00Z"
        second["local_date"] = (
            pd.to_datetime(second["local_date"]) + pd.Timedelta(days=7)
        ).dt.strftime("%Y-%m-%d")
        raw = pd.concat([raw, second], ignore_index=True)
        cv, validation = factor_inputs()
        output, weekly, audit = build_frozen_weather(
            raw_weather=raw,
            nonprecip_branches=branch_fixture(raw),
            nonprecip_policy=policy_fixture(),
            precipitation_factors=load_causal_precipitation_factors(cv, validation),
        )
        self.assertEqual(len(output), 350)
        self.assertEqual(len(weekly), 10)
        self.assertEqual(audit["cycle_count"], 2)
        self.assertTrue(audit["mandatory_structural_gate_passed"])

    def test_precipitation_factor_leakage_is_rejected(self) -> None:
        cv, validation = factor_inputs()
        cv.loc[cv["validation_year"].eq(2017), "fit_last_year"] = 2017
        with self.assertRaisesRegex(ValueError, "leaks"):
            load_causal_precipitation_factors(cv, validation)

    def test_nonfrozen_policy_is_rejected(self) -> None:
        raw = raw_weather_fixture()
        cv, validation = factor_inputs()
        policy = policy_fixture()
        policy["policy_freeze_allowed"] = False
        with self.assertRaisesRegex(ValueError, "not frozen"):
            build_frozen_weather(
                raw_weather=raw,
                nonprecip_branches=branch_fixture(raw),
                nonprecip_policy=policy,
                precipitation_factors=load_causal_precipitation_factors(cv, validation),
            )


if __name__ == "__main__":
    unittest.main()
