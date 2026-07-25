from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from scripts.data_preparation.build_gefs_exact_schedule_surrogate_dataset_v1 import (
    aggregate_ensemble_mean_weather,
    build_learning_table,
    collect_checkpoint_states,
    parse_checkpoint_crop,
)


def crop_line(date: str = "2019-05-26") -> str:
    values = ["0"] * 33
    values[0] = date
    values[1] = "146"
    values[3] = "0.75"
    values[6] = "2.5"
    values[10] = "80.0"
    values[18] = "3500.0"
    values[20] = "100.0"
    return ",".join(values)


def weather_fixture() -> pd.DataFrame:
    rows = []
    for member_index, member in enumerate(("c00", "p01", "p02", "p03", "p04")):
        for lead in range(1, 8):
            rows.append(
                {
                    "target_year": 2019,
                    "site_id": "P2",
                    "decision_date": "2019-05-27",
                    "lead_day": lead,
                    "gefs_member": member,
                    "precipitation_mm": float(lead + member_index),
                    "temperature_min_c": 10.0 + member_index,
                    "temperature_max_c": 20.0 + member_index,
                    "actual_vapor_pressure_kpa": 1.0,
                    "wind_speed_m_s": 2.0,
                    "solar_kj_m2_day": 10000.0,
                }
            )
    return pd.DataFrame(rows)


def candidate_fixture() -> pd.DataFrame:
    rows = []
    irrigation_grid = (0.0, 10.0, 15.0, 20.0, 25.0, 30.0, 40.0, 60.0)
    rain = sum(float(lead + 2) for lead in range(1, 8))
    for irrigation in irrigation_grid:
        row = {
            "target_year": 2019,
            "site": "P2",
            "decision_date": "2019-05-27",
            "checkpoint_date": "2019-05-26",
            "ir": irrigation,
            "requested_ir_mm": irrigation,
            "simulated_ir_mm": irrigation,
            "numerical_irrigation_fallback": False,
            "numerical_irrigation_fallback_delta_mm": 0.0,
            "numerical_irrigation_fallback_attempt_count": 0,
            "predecision_root_depth_cm": 80.0,
            "predecision_soil_vwc_0_100cm": 0.30,
            "predecision_soil_storage_0_100cm_mm": 300.0,
            "rain_7d_mm": rain,
            "net_gain_7d": -irrigation,
            "aet_7d_mm": 20.0,
            "residual_flux_7d_mm": 10.0,
            "water_balance_residual_0_100cm_7d_mm": 0.01,
            "dvs": 1.2,
            "lai": 9.0 + irrigation,
            "rootd": 95.0,
            "cwdm_value": 9000.0,
            "cwso_value": 1000.0,
        }
        row.update(
            {
                f"soil_vwc_0_100cm_day{day:02d}": 0.30 + day / 1000.0
                for day in range(1, 8)
            }
        )
        rows.append(row)
    return pd.DataFrame(rows)


class ExactScheduleSurrogateDatasetTests(unittest.TestCase):
    def test_parse_checkpoint_crop_returns_predecision_state(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "result_forec.crp"
            path.write_text("header\n" + crop_line() + "\n", encoding="utf-8")
            state = parse_checkpoint_crop(path)
        self.assertEqual(state["checkpoint_crop_date"], "2019-05-26")
        self.assertEqual(state["predecision_dvs"], 0.75)
        self.assertEqual(state["predecision_lai"], 2.5)
        self.assertEqual(state["predecision_crop_root_depth_cm"], 80.0)
        self.assertEqual(state["predecision_cwdm_kg_ha"], 3500.0)

    def test_checkpoint_collection_uses_verified_continuous_trunk(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            formal = Path(directory)
            unit = formal / "units" / "Y2019" / "P2"
            workspace = unit / "workspace"
            checkpoint_root = unit / "all_checkpoints_v1"
            workspace.mkdir(parents=True)
            checkpoint_root.mkdir(parents=True)
            (unit / "setup_audit_v1.json").write_text(
                json.dumps({"mandatory_gate_passed": True}), encoding="utf-8"
            )
            (workspace / "trunk2019.crp").write_text(
                "header\n" + crop_line() + "\n", encoding="utf-8"
            )
            pd.DataFrame(
                [
                    {
                        "decision_date": "2019-05-27",
                        "checkpoint_date": "2019-05-26",
                        "checkpoint_equivalence_passed": True,
                        "maximum_absolute_crop_state_error": 0.0,
                        "maximum_absolute_profile_state_error": 0.0,
                    }
                ]
            ).to_csv(
                checkpoint_root / "swap_season_checkpoint_equivalence_v1.csv",
                index=False,
            )
            broken_checkpoint = (
                checkpoint_root / "checkpoints" / "20190527" / "result_forec.crp"
            )
            broken_checkpoint.parent.mkdir(parents=True)
            broken_checkpoint.write_text("header only\n", encoding="utf-8")
            plan = pd.DataFrame(
                [
                    {
                        "target_year": 2019,
                        "site_id": "P2",
                        "decision_date": "2019-05-27",
                        "checkpoint_date": "2019-05-26",
                    }
                ]
            )
            states = collect_checkpoint_states(plan, formal)

        self.assertEqual(len(states), 1)
        self.assertEqual(
            states["checkpoint_crop_source"].iloc[0],
            "verified_continuous_season_trunk",
        )
        self.assertEqual(states["predecision_dvs"].iloc[0], 0.75)

    def test_weather_aggregation_requires_five_members_and_returns_daily_mean(self) -> None:
        weather = weather_fixture()
        wide = aggregate_ensemble_mean_weather(weather)
        self.assertEqual(len(wide), 1)
        self.assertEqual(wide["weather_precipitation_mm_day01"].iloc[0], 3.0)
        with self.assertRaisesRegex(ValueError, "five GEFS members"):
            aggregate_ensemble_mean_weather(weather.iloc[:-1])
        wrong_members = weather.copy()
        wrong_members.loc[wrong_members["gefs_member"].eq("p04"), "gefs_member"] = "p05"
        with self.assertRaisesRegex(ValueError, "frozen five-member set"):
            aggregate_ensemble_mean_weather(wrong_members)

    def test_learning_table_uses_checkpoint_state_and_excludes_restart_end_state(self) -> None:
        weather = aggregate_ensemble_mean_weather(weather_fixture())
        checkpoint = pd.DataFrame(
            [
                {
                    "target_year": 2019,
                    "site_id": "P2",
                    "decision_date": "2019-05-27",
                    "checkpoint_date": "2019-05-26",
                    "checkpoint_crop_path": "/checkpoint/result_forec.crp",
                    "checkpoint_crop_date": "2019-05-26",
                    "predecision_dvs": 0.75,
                    "predecision_lai": 2.5,
                    "predecision_crop_root_depth_cm": 80.0,
                    "predecision_cwdm_kg_ha": 3500.0,
                    "predecision_cwso_kg_ha": 100.0,
                }
            ]
        )
        table, audit = build_learning_table(
            candidate_fixture(), weather, checkpoint
        )
        self.assertEqual(len(table), 8)
        self.assertTrue(table["predecision_lai"].eq(2.5).all())
        self.assertTrue(table["predecision_dvs"].eq(0.75).all())
        self.assertNotIn("lai", table.columns)
        self.assertNotIn("cwdm_value", table.columns)
        self.assertNotIn("simulated_ir_mm", audit["feature_columns"])
        self.assertIn("irrigation_mm", audit["feature_columns"])

    def test_learning_table_rejects_requested_irrigation_or_rain_mismatch(self) -> None:
        weather = aggregate_ensemble_mean_weather(weather_fixture())
        checkpoint = pd.DataFrame(
            [
                {
                    "target_year": 2019,
                    "site_id": "P2",
                    "decision_date": "2019-05-27",
                    "checkpoint_date": "2019-05-26",
                    "checkpoint_crop_path": "/checkpoint/result_forec.crp",
                    "checkpoint_crop_date": "2019-05-26",
                    "predecision_dvs": 0.75,
                    "predecision_lai": 2.5,
                    "predecision_crop_root_depth_cm": 80.0,
                    "predecision_cwdm_kg_ha": 3500.0,
                    "predecision_cwso_kg_ha": 100.0,
                }
            ]
        )
        requested_mismatch = candidate_fixture()
        requested_mismatch.loc[0, "requested_ir_mm"] = 0.1
        with self.assertRaisesRegex(ValueError, "requested irrigation differ"):
            build_learning_table(requested_mismatch, weather, checkpoint)

        rain_mismatch = candidate_fixture()
        rain_mismatch["rain_7d_mm"] += 0.02
        with self.assertRaisesRegex(ValueError, "rain differ"):
            build_learning_table(rain_mismatch, weather, checkpoint)


if __name__ == "__main__":
    unittest.main()
