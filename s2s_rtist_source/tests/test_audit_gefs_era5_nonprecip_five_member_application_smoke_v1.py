from __future__ import annotations

import unittest

import pandas as pd

from scripts.diagnostics.audit_gefs_era5_nonprecip_five_member_application_smoke_v1 import (
    KEYS,
    add_temperature_structure,
    build_validation,
    count_order_inversions,
)


SITES = ["P1", "P2", "P3", "P4", "P15"]
MEMBERS = ["c00", "p01", "p02", "p03", "p04"]


def weather_row(
    decision: pd.Timestamp,
    site: str,
    lead: int,
    *,
    member: str | None = None,
    shift: float = 0.0,
) -> dict[str, object]:
    row: dict[str, object] = {
        "decision_date": decision.strftime("%Y-%m-%d"),
        "site_id": site,
        "local_date": (decision + pd.Timedelta(days=lead - 1)).strftime("%Y-%m-%d"),
        "lead_day": lead,
        "temperature_min_c": 10.0 + lead * 0.1 + shift,
        "temperature_max_c": 20.0 + lead * 0.1 + shift,
        "actual_vapor_pressure_kpa": 2.0 + shift * 0.02,
        "wind_speed_m_s": 4.0 + shift * 0.05,
        "solar_kj_m2_day": 20_000.0 + lead * 100.0 + shift * 200.0,
    }
    if member is not None:
        row["gefs_member"] = member
        row["precipitation_mm_raw"] = max(0.0, 2.0 + shift * 0.1)
    return row


def reference_row(decision: pd.Timestamp, site: str, lead: int) -> dict[str, object]:
    row = weather_row(decision, site, lead)
    row.update(
        {
            "temperature_min_c": float(row["temperature_min_c"]) + 1.0,
            "temperature_max_c": float(row["temperature_max_c"]) + 1.0,
            "actual_vapor_pressure_kpa": 1.8,
            "wind_speed_m_s": 3.0,
            "solar_kj_m2_day": float(row["solar_kj_m2_day"]) - 1_000.0,
        }
    )
    return row


def history_fixture() -> tuple[pd.DataFrame, pd.DataFrame]:
    decisions = pd.date_range("2015-01-15", periods=12, freq="14D")
    gefs = []
    era5 = []
    for decision in decisions:
        for site in SITES:
            for lead in range(1, 8):
                gefs.append(weather_row(decision, site, lead))
                era5.append(reference_row(decision, site, lead))
    return pd.DataFrame(gefs), pd.DataFrame(era5)


def cycle_fixture() -> tuple[pd.DataFrame, pd.DataFrame]:
    decision = pd.Timestamp("2015-07-06")
    weather = []
    reference = []
    for site in SITES:
        for lead in range(1, 8):
            reference.append(reference_row(decision, site, lead))
            for member_index, member in enumerate(MEMBERS):
                weather.append(
                    weather_row(
                        decision,
                        site,
                        lead,
                        member=member,
                        shift=float(member_index - 2),
                    )
                )
    return pd.DataFrame(weather), pd.DataFrame(reference)


def temperature_metrics_fixture() -> pd.DataFrame:
    rows = []
    for year in range(2015, 2020):
        for variable in ("temperature_min_c", "temperature_max_c"):
            rows.append(
                {
                    "target_year": year,
                    "candidate_id": "raw_gefs",
                    "shrinkage_alpha": 0.0,
                    "variable": variable,
                    "sample_count": 10,
                    "bias_corrected_minus_era5": -1.0,
                    "mae": 2.0,
                    "rmse": 3.0,
                }
            )
            for alpha in (0.25, 0.5):
                tmin_bad = variable == "temperature_min_c" and alpha == 0.5
                rows.append(
                    {
                        "target_year": year,
                        "candidate_id": f"hybrid_affine_solar_shrink_a{alpha:g}",
                        "shrinkage_alpha": alpha,
                        "variable": variable,
                        "sample_count": 10,
                        "bias_corrected_minus_era5": -0.8,
                        "mae": 2.1 if tmin_bad else 1.8,
                        "rmse": 3.1 if tmin_bad else 2.8,
                    }
                )
    return pd.DataFrame(rows)


def selection_fixture() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "variable": "actual_vapor_pressure_kpa",
                "shrinkage_alpha": 0.75,
                "raw_solar_sensitivity_required": False,
            },
            {
                "variable": "wind_speed_m_s",
                "shrinkage_alpha": 1.0,
                "raw_solar_sensitivity_required": False,
            },
            {
                "variable": "solar_kj_m2_day",
                "shrinkage_alpha": 0.25,
                "raw_solar_sensitivity_required": True,
            },
        ]
    )


class FiveMemberApplicationSmokeTests(unittest.TestCase):
    def test_complete_five_member_application_smoke(self) -> None:
        history_gefs, history_era5 = history_fixture()
        weather, cycle_era5 = cycle_fixture()
        output, metrics, factors, policy, audit = build_validation(
            five_member_weather=weather,
            history_gefs_c00=history_gefs,
            history_era5=history_era5,
            cycle_era5=cycle_era5,
            metrics=temperature_metrics_fixture(),
            robust_selection=selection_fixture(),
            minimum_samples=8,
        )
        self.assertTrue(audit["mandatory_gate_passed"])
        self.assertEqual(policy["temperature_joint_alpha"], 0.25)
        self.assertEqual(len(output), 525)
        self.assertEqual(len(metrics), 15)
        self.assertEqual(len(factors), 35)
        self.assertEqual(audit["fit_leakage_rows"], 0)
        self.assertEqual(
            audit["positive_variable_member_order_inversion_count"], 0
        )
        self.assertEqual(
            audit["temperature_structure_member_order_inversion_count"], 0
        )
        self.assertEqual(audit["temperature_order_error_count"], 0)

    def test_temperature_output_rank_change_does_not_break_structure_gate(self) -> None:
        shared = {
            "decision_date": "2015-07-06",
            "site_id": "P1",
            "local_date": "2015-07-06",
            "lead_day": 1,
        }
        raw = pd.DataFrame(
            [
                {**shared, "gefs_member": "c00", "temperature_min_c": 0.0, "temperature_max_c": 20.0},
                {**shared, "gefs_member": "p01", "temperature_min_c": 5.0, "temperature_max_c": 11.0},
            ]
        )
        corrected = pd.DataFrame(
            [
                {**shared, "gefs_member": "c00", "temperature_min_c": 7.5, "temperature_max_c": 12.5},
                {**shared, "gefs_member": "p01", "temperature_min_c": 7.25, "temperature_max_c": 8.75},
            ]
        )
        self.assertEqual(count_order_inversions(raw, corrected, "temperature_min_c"), 1)
        raw_structure = add_temperature_structure(raw)
        corrected_structure = add_temperature_structure(corrected)
        self.assertEqual(
            count_order_inversions(
                raw_structure, corrected_structure, "temperature_center_c"
            ),
            0,
        )
        self.assertEqual(
            count_order_inversions(
                raw_structure, corrected_structure, "temperature_range_c"
            ),
            0,
        )


if __name__ == "__main__":
    unittest.main()
