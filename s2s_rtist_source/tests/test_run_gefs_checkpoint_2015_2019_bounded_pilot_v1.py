from __future__ import annotations

import unittest

import pandas as pd

from scripts.simulation.run_gefs_checkpoint_2015_2019_bounded_pilot_v1 import (
    EXPECTED_YEARS,
    build_bounded_audit,
    selected_cycles,
)
from scripts.simulation.run_gefs_checkpoint_five_site_eight_ir_smoke_v1 import (
    SITE_ORDER,
)
from scripts.simulation.run_gefs_checkpoint_one_date_eight_ir_smoke_v1 import (
    IRRIGATION_OPTIONS_MM,
)


def weather_fixture() -> pd.DataFrame:
    rows = []
    for year in EXPECTED_YEARS:
        decision = pd.Timestamp(f"{year}-07-06")
        for site in SITE_ORDER:
            for member_index, member in enumerate(("c00", "p01", "p02", "p03", "p04")):
                for lead_day in range(1, 8):
                    rows.append(
                        {
                            "decision_date": decision.strftime("%Y-%m-%d"),
                            "site_id": site,
                            "gefs_member": member,
                            "local_date": (
                                decision + pd.Timedelta(days=lead_day - 1)
                            ).strftime("%Y-%m-%d"),
                            "lead_day": lead_day,
                            "temperature_min_c": 10.0 + member_index,
                            "temperature_max_c": 20.0 + member_index,
                            "actual_vapor_pressure_kpa": 1.0,
                            "wind_speed_m_s": 2.0,
                            "solar_kj_m2_day": 15000.0,
                            "precipitation_mm": float(lead_day),
                        }
                    )
    return pd.DataFrame(rows)


def candidate_fixture(*, responsive: bool = True) -> pd.DataFrame:
    rows = []
    for year in EXPECTED_YEARS:
        decision = pd.Timestamp(f"{year}-07-06")
        for site in SITE_ORDER:
            best_ir = 20.0 if responsive else 0.0
            for irrigation in IRRIGATION_OPTIONS_MM:
                rows.append(
                    {
                        "site": site,
                        "target_year": year,
                        "decision_date": decision.strftime("%Y-%m-%d"),
                        "date_t": decision.strftime("%d-%b-%Y"),
                        "ir": irrigation,
                        "is_best_ir": irrigation == best_ir,
                        "net_gain_7d": 10.0 if irrigation == best_ir else 0.0,
                        "cwdm_value": irrigation if responsive else 1.0,
                    }
                )
    return pd.DataFrame(rows)


def audit_fixture(cycles: pd.DataFrame) -> list[dict[str, object]]:
    rows = []
    for cycle in cycles.itertuples(index=False):
        for site in SITE_ORDER:
            rows.append(
                {
                    "site_id": site,
                    "target_year": int(cycle.target_year),
                    "decision_date": str(cycle.decision_date),
                    "mandatory_gate_passed": True,
                    "maximum_absolute_checkpoint_crop_state_error": 0.0,
                    "maximum_absolute_checkpoint_profile_state_error": 0.0,
                    "maximum_absolute_swap_rain_error_mm": 0.001,
                    "maximum_absolute_water_balance_residual_mm": 0.3,
                    "primary_output_missing_value_count": 0,
                    "prestate_swap_rerun_count": 0,
                }
            )
    return rows


class GefsCheckpointBoundedPilotTests(unittest.TestCase):
    def test_selected_cycles_require_one_complete_cycle_per_year(self) -> None:
        cycles = selected_cycles(weather_fixture())
        self.assertEqual(cycles["target_year"].tolist(), list(EXPECTED_YEARS))
        self.assertEqual(cycles["split"].tolist()[-1], "validation")

    def test_selected_cycles_reject_missing_year(self) -> None:
        weather = weather_fixture()
        weather = weather.loc[~weather["decision_date"].str.startswith("2019-")]
        with self.assertRaisesRegex(ValueError, "requires years"):
            selected_cycles(weather)

    def test_bounded_audit_passes_complete_response_coverage(self) -> None:
        cycles = selected_cycles(weather_fixture())
        audit = build_bounded_audit(
            candidates=candidate_fixture(),
            site_audits=audit_fixture(cycles),
            cycles=cycles,
        )
        self.assertTrue(audit["bounded_pilot_gate_passed"])
        self.assertEqual(audit["candidate_rows"], 200)
        self.assertEqual(audit["site_cycle_count"], 25)
        self.assertEqual(
            audit["next_gate"], "design_seasonal_date_density_without_training"
        )

    def test_response_failure_does_not_invalidate_mandatory_physics_gate(self) -> None:
        cycles = selected_cycles(weather_fixture())
        audit = build_bounded_audit(
            candidates=candidate_fixture(responsive=False),
            site_audits=audit_fixture(cycles),
            cycles=cycles,
        )
        self.assertTrue(audit["mandatory_gate_passed"])
        self.assertFalse(audit["response_coverage_gate_passed"])
        self.assertFalse(audit["bounded_pilot_gate_passed"])

    def test_string_best_flags_are_parsed_strictly(self) -> None:
        cycles = selected_cycles(weather_fixture())
        candidates = candidate_fixture()
        candidates["is_best_ir"] = candidates["is_best_ir"].map(
            {True: "True", False: "False"}
        )
        audit = build_bounded_audit(
            candidates=candidates,
            site_audits=audit_fixture(cycles),
            cycles=cycles,
        )
        self.assertTrue(audit["bounded_pilot_gate_passed"])


if __name__ == "__main__":
    unittest.main()
