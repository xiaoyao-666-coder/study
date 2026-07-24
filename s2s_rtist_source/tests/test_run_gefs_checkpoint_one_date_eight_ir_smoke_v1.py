from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import pandas as pd

from scripts.simulation.run_gefs_checkpoint_one_date_eight_ir_smoke_v1 import (
    IRRIGATION_OPTIONS_MM,
    build_audit,
    build_ensemble_mean_weather,
    parse_swap_weather_record,
    patch_swap_weather_file,
    swap_weather_filenames,
    validate_checkpoint,
)


def weather_fixture() -> pd.DataFrame:
    rows = []
    start = pd.Timestamp("2015-07-06")
    for site in ["P1", "P2"]:
        for member_index, member in enumerate(["c00", "p01", "p02", "p03", "p04"]):
            for lead_day in range(1, 8):
                rows.append(
                    {
                        "decision_date": "2015-07-06",
                        "site_id": site,
                        "gefs_member": member,
                        "local_date": (
                            start + pd.Timedelta(days=lead_day - 1)
                        ).strftime("%Y-%m-%d"),
                        "lead_day": lead_day,
                        "temperature_min_c": 10.0 + member_index,
                        "temperature_max_c": 20.0 + member_index,
                        "actual_vapor_pressure_kpa": 1.0 + member_index * 0.1,
                        "wind_speed_m_s": 2.0 + member_index * 0.1,
                        "solar_kj_m2_day": 15000.0 + member_index * 100.0,
                        "precipitation_mm": float(lead_day + member_index),
                    }
                )
    return pd.DataFrame(rows)


class GefsCheckpointBranchSmokeTests(unittest.TestCase):
    def test_weather_filenames_follow_target_year(self) -> None:
        self.assertEqual(
            swap_weather_filenames(2019),
            ("weather.019", "WeatherOriginal.019"),
        )

    def test_weather_patch_preserves_predecision_rows_and_etref(self) -> None:
        text = (
            "header\n"
            " 'Weather' 5 7 2015 100 10 20 1 2 3 4\n"
            " 'Weather' 6 7 2015 101 11 21 1.1 2.1 3.1 4.1\n"
        )
        daily = build_ensemble_mean_weather(
            weather_fixture(), site_id="P1", decision_date="2015-07-06"
        ).iloc[:1]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "weather.015"
            path.write_text(text, encoding="utf-8")
            audit = patch_swap_weather_file(path, daily)
            records = [
                parse_swap_weather_record(line)
                for line in path.read_text(encoding="utf-8").splitlines()
            ]
        records = [record for record in records if record is not None]
        self.assertEqual(len(audit), 1)
        self.assertEqual(records[0]["precipitation_mm"], 3.0)
        self.assertEqual(records[0]["etref"], 4.0)
        self.assertEqual(records[1]["etref"], 4.1)
        self.assertAlmostEqual(records[1]["precipitation_mm"], 3.0)

    def test_ensemble_mean_weather_selects_one_site_and_seven_days(self) -> None:
        daily = build_ensemble_mean_weather(
            weather_fixture(), site_id="P1", decision_date="2015-07-06"
        )
        self.assertEqual(len(daily), 7)
        self.assertTrue(daily["member_count"].eq(5).all())
        self.assertEqual(daily["lead_day"].tolist(), list(range(1, 8)))
        self.assertAlmostEqual(
            float(daily.iloc[0]["temperature_min_c_mean"]), 12.0
        )
        self.assertAlmostEqual(
            float(daily.iloc[0]["precipitation_mm_corrected_mean"]), 3.0
        )

    def test_checkpoint_must_be_previous_day_and_equivalent(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            checkpoint = root / "checkpoint"
            checkpoint.mkdir()
            for name in ("result_forec.end", "result_forec.crp", "result_forec.vap"):
                (checkpoint / name).write_text("x", encoding="utf-8")
            audit = pd.DataFrame(
                [
                    {
                        "decision_date": "2015-07-06",
                        "checkpoint_date": "2015-07-05",
                        "checkpoint_equivalence_passed": True,
                        "maximum_absolute_crop_state_error": 0.0,
                        "maximum_absolute_profile_state_error": 0.0,
                    }
                ]
            )
            audit_path = root / "audit.csv"
            audit.to_csv(audit_path, index=False)
            result = validate_checkpoint(checkpoint, audit_path, "2015-07-06")
        self.assertEqual(result["checkpoint_date"], "2015-07-05")

    def test_checkpoint_wrong_date_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            checkpoint = root / "checkpoint"
            checkpoint.mkdir()
            for name in ("result_forec.end", "result_forec.crp", "result_forec.vap"):
                (checkpoint / name).write_text("x", encoding="utf-8")
            audit_path = root / "audit.csv"
            pd.DataFrame(
                [
                    {
                        "decision_date": "2015-07-06",
                        "checkpoint_date": "2015-07-04",
                        "checkpoint_equivalence_passed": True,
                        "maximum_absolute_crop_state_error": 0.0,
                        "maximum_absolute_profile_state_error": 0.0,
                    }
                ]
            ).to_csv(audit_path, index=False)
            with self.assertRaisesRegex(ValueError, "minus one"):
                validate_checkpoint(checkpoint, audit_path, "2015-07-06")

    def test_audit_passes_complete_physical_candidate_set(self) -> None:
        candidates = pd.DataFrame(
            {
                "date_t": ["06-Jul-2015"] * 8,
                "ir": IRRIGATION_OPTIONS_MM,
                "rain_7d_mm": [35.0] * 8,
                "water_balance_residual_0_100cm_7d_mm": [0.1] * 8,
                **{field: [1.0] * 8 for field in [
                    "net_gain_7d",
                    "aet_7d_mm",
                    "soil_vwc_0_100cm_day01",
                    "soil_vwc_0_100cm_day02",
                    "soil_vwc_0_100cm_day03",
                    "soil_vwc_0_100cm_day04",
                    "soil_vwc_0_100cm_day05",
                    "soil_vwc_0_100cm_day06",
                    "soil_vwc_0_100cm_day07",
                ]},
            }
        )
        daily = pd.DataFrame(
            {
                "precipitation_mm_corrected_mean": [5.0] * 7,
                "member_count": [5] * 7,
            }
        )
        injection = pd.DataFrame({"local_date": list(range(14))})
        checkpoint = {
            "decision_date": "2015-07-06",
            "checkpoint_date": "2015-07-05",
            "maximum_absolute_crop_state_error": 0.0,
            "maximum_absolute_profile_state_error": 0.0,
        }
        audit = build_audit(
            candidates=candidates,
            daily=daily,
            injection=injection,
            checkpoint=checkpoint,
            site_id="P15",
            target_year=2015,
        )
        self.assertTrue(audit["mandatory_gate_passed"])
        self.assertEqual(audit["site_id"], "P15")
        self.assertEqual(audit["target_year"], 2015)
        self.assertEqual(audit["next_gate"], "expand_verified_checkpoint_branch_smoke_to_five_sites")


if __name__ == "__main__":
    unittest.main()
