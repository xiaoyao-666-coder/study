from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from scripts.data_preparation.apply_gefs_exact_schedule_frozen_nonprecip_weather_v1 import (
    build_frozen_nonprecipitation,
)


MEMBERS = ["c00", "p01", "p02", "p03", "p04"]


def history_inputs(periods: int = 9) -> tuple[pd.DataFrame, pd.DataFrame]:
    gefs_rows = []
    era5_rows = []
    for decision in pd.date_range("2015-01-15", periods=periods, freq="14D"):
        for lead in range(1, 8):
            common = {
                "decision_date": decision.strftime("%Y-%m-%d"),
                "site_id": "P1",
                "local_date": (decision + pd.Timedelta(days=lead - 1)).strftime(
                    "%Y-%m-%d"
                ),
                "lead_day": lead,
            }
            gefs_rows.append(
                {
                    **common,
                    "gefs_member": "c00",
                    "temperature_min_c": 10.0 + lead,
                    "temperature_max_c": 20.0 + lead,
                    "actual_vapor_pressure_kpa": 2.0,
                    "wind_speed_m_s": 4.0,
                    "solar_kj_m2_day": 20_000.0 + lead * 100.0,
                }
            )
            era5_rows.append(
                {
                    **common,
                    "temperature_min_c": 11.0 + lead,
                    "temperature_max_c": 21.0 + lead,
                    "actual_vapor_pressure_kpa": 1.8,
                    "wind_speed_m_s": 3.0,
                    "solar_kj_m2_day": 19_000.0 + lead * 100.0,
                }
            )
    return pd.DataFrame(gefs_rows), pd.DataFrame(era5_rows)


def target_weather() -> pd.DataFrame:
    rows = []
    for decision in (pd.Timestamp("2015-05-11"), pd.Timestamp("2015-05-18")):
        for lead in range(1, 8):
            for member_index, member in enumerate(MEMBERS):
                shift = float(member_index - 2)
                rows.append(
                    {
                        "decision_date": decision.strftime("%Y-%m-%d"),
                        "site_id": "P1",
                        "gefs_member": member,
                        "local_date": (
                            decision + pd.Timedelta(days=lead - 1)
                        ).strftime("%Y-%m-%d"),
                        "lead_day": lead,
                        "temperature_min_c": 10.0 + lead + shift,
                        "temperature_max_c": 20.0 + lead + shift,
                        "actual_vapor_pressure_kpa": 2.0 + shift * 0.05,
                        "wind_speed_m_s": 4.0 + shift * 0.1,
                        "solar_kj_m2_day": 20_000.0 + lead * 100.0 + shift * 50.0,
                        "precipitation_mm_raw": max(0.0, 2.0 + shift * 0.1),
                    }
                )
    return pd.DataFrame(rows)


def policy() -> dict[str, object]:
    return {
        "actual_vapor_pressure_kpa_alpha": 0.75,
        "solar_kj_m2_day_alpha": 0.25,
        "temperature_center_alpha": 1.0,
        "temperature_range_alpha": 0.0,
        "wind_speed_m_s_alpha": 1.0,
        "policy_freeze_allowed": True,
        "recommended_solar_branch": "affine_alpha_0.25",
        "temperature_selection_uses_2019": False,
        "temperature_2019_confirmation_passed": True,
    }


class ExactScheduleFrozenNonprecipitationTests(unittest.TestCase):
    def test_applies_frozen_policy_without_target_era5(self) -> None:
        history_gefs, history_era5 = history_inputs()
        output, factors, audit = build_frozen_nonprecipitation(
            target_weather=target_weather(),
            history_gefs_c00=history_gefs,
            history_era5=history_era5,
            policy=policy(),
            minimum_samples=8,
        )
        self.assertTrue(audit["mandatory_gate_passed"])
        self.assertFalse(audit["target_era5_input_required"])
        self.assertEqual(audit["fit_leakage_rows"], 0)
        self.assertEqual(len(output), 70)
        self.assertEqual(len(factors), 14)
        self.assertEqual(audit["minimum_fit_samples_per_site_lead"], 8)
        self.assertEqual(audit["maximum_fit_samples_per_site_lead"], 9)

        raw = target_weather().sort_values(
            ["decision_date", "site_id", "gefs_member", "local_date", "lead_day"]
        ).reset_index(drop=True)
        corrected = output.sort_values(
            ["decision_date", "site_id", "gefs_member", "local_date", "lead_day"]
        ).reset_index(drop=True)
        raw_center = (raw["temperature_min_c"] + raw["temperature_max_c"]) / 2.0
        corrected_center = (
            corrected["temperature_min_c"] + corrected["temperature_max_c"]
        ) / 2.0
        raw_range = raw["temperature_max_c"] - raw["temperature_min_c"]
        corrected_range = (
            corrected["temperature_max_c"] - corrected["temperature_min_c"]
        )
        self.assertTrue(np.allclose(corrected_center, raw_center + 1.0))
        self.assertTrue(np.allclose(corrected_range, raw_range))
        self.assertTrue(
            np.allclose(corrected["actual_vapor_pressure_kpa"], raw["actual_vapor_pressure_kpa"] * 0.925)
        )
        self.assertTrue(
            np.allclose(corrected["wind_speed_m_s"], raw["wind_speed_m_s"] * 0.75)
        )
        self.assertTrue(
            np.allclose(corrected["solar_kj_m2_day"], raw["solar_kj_m2_day"] - 250.0)
        )

    def test_rejects_insufficient_strictly_causal_history(self) -> None:
        history_gefs, history_era5 = history_inputs(periods=7)
        with self.assertRaisesRegex(ValueError, "insufficient strictly causal"):
            build_frozen_nonprecipitation(
                target_weather=target_weather(),
                history_gefs_c00=history_gefs,
                history_era5=history_era5,
                policy=policy(),
                minimum_samples=8,
            )


if __name__ == "__main__":
    unittest.main()
