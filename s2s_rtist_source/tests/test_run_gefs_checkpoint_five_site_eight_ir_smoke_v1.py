from __future__ import annotations

import unittest
import tempfile
from pathlib import Path

import pandas as pd

from scripts.simulation.run_gefs_checkpoint_five_site_eight_ir_smoke_v1 import (
    SITE_ORDER,
    build_five_site_audit,
    build_target_checkpoint_schedule,
    validate_source_workspace,
)
from scripts.simulation.run_gefs_checkpoint_one_date_eight_ir_smoke_v1 import (
    IRRIGATION_OPTIONS_MM,
)


def candidate_fixture() -> pd.DataFrame:
    rows = []
    for site_index, site in enumerate(SITE_ORDER):
        best_ir = IRRIGATION_OPTIONS_MM[(site_index + 1) % len(IRRIGATION_OPTIONS_MM)]
        for irrigation in IRRIGATION_OPTIONS_MM:
            rows.append(
                {
                    "site": site,
                    "date_t": "06-Jul-2015",
                    "ir": irrigation,
                    "is_best_ir": irrigation == best_ir,
                }
            )
    return pd.DataFrame(rows)


def audit_fixture() -> list[dict[str, object]]:
    return [
        {
            "site_id": site,
            "mandatory_gate_passed": True,
            "maximum_absolute_checkpoint_crop_state_error": 0.0,
            "maximum_absolute_checkpoint_profile_state_error": 0.0,
            "maximum_absolute_swap_rain_error_mm": 0.001,
            "maximum_absolute_water_balance_residual_mm": 0.3,
            "primary_output_missing_value_count": 0,
        }
        for site in SITE_ORDER
    ]


class GefsCheckpointFiveSiteSmokeTests(unittest.TestCase):
    def test_source_workspace_weather_gate_follows_target_year(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for name in (
                "Swap1.swp",
                "real_ir_update.py",
                "weather.019",
                "WeatherOriginal.019",
                "swap_test",
            ):
                (root / name).write_text("x", encoding="utf-8")
            validate_source_workspace(root, year=2019)
            with self.assertRaisesRegex(FileNotFoundError, "weather.015"):
                validate_source_workspace(root, year=2015)

    def test_target_schedule_requires_active_crop_and_complete_horizon(self) -> None:
        dates = pd.date_range("2015-07-05", periods=8, freq="D")
        crop = pd.DataFrame({"Date": dates, "DVS": [0.8] * len(dates)})
        result = build_target_checkpoint_schedule(
            crop,
            decision_date="2015-07-06",
        )
        self.assertEqual(result.loc[0, "state_checkpoint_date"], "2015-07-05")
        self.assertEqual(result.loc[0, "horizon_end_date"], "2015-07-12")
        self.assertEqual(int(result.loc[0, "decision_doy"]), 187)

    def test_target_schedule_rejects_harvested_crop(self) -> None:
        dates = pd.date_range("2015-07-05", periods=8, freq="D")
        crop = pd.DataFrame({"Date": dates, "DVS": [2.0] * len(dates)})
        with self.assertRaisesRegex(ValueError, "not crop-active"):
            build_target_checkpoint_schedule(crop, decision_date="2015-07-06")

    def test_five_site_audit_passes_complete_candidate_set(self) -> None:
        audit = build_five_site_audit(
            candidates=candidate_fixture(),
            site_audits=audit_fixture(),
        )
        self.assertTrue(audit["mandatory_gate_passed"])
        self.assertEqual(audit["candidate_rows"], 40)
        self.assertEqual(audit["site_count"], 5)
        self.assertEqual(
            audit["next_gate"],
            "review_five_site_response_before_bounded_2015_2019_label_generation",
        )

    def test_five_site_audit_rejects_missing_site(self) -> None:
        candidates = candidate_fixture()
        candidates = candidates.loc[~candidates["site"].eq("P15")].copy()
        audit = build_five_site_audit(
            candidates=candidates,
            site_audits=audit_fixture()[:-1],
        )
        self.assertFalse(audit["mandatory_gate_passed"])

    def test_five_site_audit_handles_empty_failure_result(self) -> None:
        audit = build_five_site_audit(
            candidates=pd.DataFrame(),
            site_audits=[],
        )
        self.assertFalse(audit["mandatory_gate_passed"])
        self.assertEqual(audit["candidate_rows"], 0)

    def test_five_site_audit_parses_string_false_strictly(self) -> None:
        candidates = candidate_fixture()
        candidates["is_best_ir"] = candidates["is_best_ir"].map(
            {True: "True", False: "False"}
        )
        audit = build_five_site_audit(
            candidates=candidates,
            site_audits=audit_fixture(),
        )
        self.assertTrue(audit["mandatory_gate_passed"])
        self.assertEqual(audit["best_row_count"], 5)


if __name__ == "__main__":
    unittest.main()
