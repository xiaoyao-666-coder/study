from __future__ import annotations

import argparse
import hashlib
import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import pandas as pd

import scripts.data_preparation.acquire_gefs_p3_2022_five_member_operational_v1 as acquisition


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def schedule_rows() -> list[dict[str, object]]:
    rows = []
    for index, decision in enumerate(pd.date_range("2022-06-02", periods=13, freq="7D")):
        rows.append(
            {
                "site_id": "P3",
                "target_year": 2022,
                "split": "independent_test",
                "schedule_index": index,
                "state_checkpoint_date": (decision - pd.Timedelta(days=1)).strftime(
                    "%Y-%m-%d"
                ),
                "state_dvs": 0.1 + 0.14 * index,
                "decision_date": decision.strftime("%Y-%m-%d"),
                "horizon_end_date": (decision + pd.Timedelta(days=6)).strftime(
                    "%Y-%m-%d"
                ),
                "horizon_days": 7,
            }
        )
    return rows


def write_trunk_fixture(root: Path) -> dict[str, str]:
    schedule_path = (
        root / "units/Y2022/P3/trunk/swap_season_decision_schedule_v1.csv"
    )
    checkpoint_dir = root / "units/Y2022/P3/all_checkpoints_v1"
    schedule_path.parent.mkdir(parents=True)
    checkpoint_dir.mkdir(parents=True)
    schedule = pd.DataFrame(schedule_rows())
    schedule.to_csv(schedule_path, index=False)

    checkpoint_rows = []
    restart_rows = []
    for row in schedule.itertuples(index=False):
        compact = pd.Timestamp(row.decision_date).strftime("%Y%m%d")
        restart = checkpoint_dir / "checkpoints" / compact / "result_forec.end"
        restart.parent.mkdir(parents=True)
        restart.write_bytes(f"restart-{compact}".encode("ascii"))
        checkpoint_rows.append(
            {
                "decision_date": row.decision_date,
                "checkpoint_equivalence_passed": True,
                "maximum_absolute_crop_state_error": 0.0,
                "maximum_absolute_profile_state_error": 0.0,
            }
        )
        restart_rows.append(
            {
                "role": "output_verified_restart_checkpoint",
                "path": f"{root.name}/{restart.relative_to(root).as_posix()}",
                "bytes": restart.stat().st_size,
                "sha256": sha256(restart),
            }
        )

    checkpoint_path = checkpoint_dir / "swap_season_checkpoint_equivalence_v1.csv"
    pd.DataFrame(checkpoint_rows).to_csv(checkpoint_path, index=False)
    checkpoint_audit_path = (
        checkpoint_dir / "swap_season_checkpoint_equivalence_audit_v1.json"
    )
    checkpoint_audit_path.write_text(
        json.dumps(
            {
                "status": "all_season_checkpoints_generated_and_verified",
                "target_year": 2022,
                "scheduled_checkpoint_count": 13,
                "tested_checkpoint_count": 13,
                "all_checkpoints_passed": True,
                "all_scheduled_checkpoints_saved": True,
                "formal_label_generation_allowed": False,
            }
        ),
        encoding="utf-8",
    )
    audit_path = root / "gefs_p3_2022_era5_trunk_schedule_freeze_audit_v1.json"
    audit_path.write_text(
        json.dumps(
            {
                "status": "p3_2022_era5_zero_irrigation_trunk_and_schedule_frozen_pending_operational_gefs_acquisition",
                "mandatory_gate_passed": True,
                "target_site": "P3",
                "target_year": 2022,
                "decision_rows": 13,
                "first_decision_date": "2022-06-02",
                "last_decision_date": "2022-08-25",
                "all_decisions_pre_maturity": True,
                "all_eligible_cycles_retained": True,
                "manual_date_selection_performed": False,
                "saved_checkpoint_count": 13,
                "all_checkpoint_equivalence_passed": True,
                "maximum_absolute_checkpoint_crop_state_error": 0.0,
                "maximum_absolute_checkpoint_profile_state_error": 0.0,
                "historical_parity_all_passed": True,
                "historical_conversion_parity": {
                    "exact_match_at_swap_file_precision": True
                },
                "GEFS_rows_read": 0,
                "GEFS_download_performed": False,
                "SWAP_candidate_label_rows_read": 0,
                "SWAP_candidate_label_generation_performed": False,
                "P3_2022_target_labels_read": 0,
                "P3_2023_rows_or_labels_read": 0,
                "P3_2024_rows_or_labels_read": 0,
                "target_columns_loaded": [],
                "model_checkpoint_files_loaded": 0,
                "model_inference_performed": False,
                "model_training_performed": False,
                "model_or_threshold_reselection_performed": False,
                "network_access_performed_by_this_stage": False,
                "schedule_sha256": sha256(schedule_path),
                "checkpoint_equivalence_sha256": sha256(checkpoint_path),
                "next_gate": "acquire_operational_GEFSv12_2022_for_frozen_decision_dates",
            }
        ),
        encoding="utf-8",
    )
    manifest_path = root / "gefs_p3_2022_era5_trunk_schedule_freeze_manifest_v1.csv"
    manifest_rows = [
        {
            "role": "output_schedule",
            "path": f"{root.name}/{schedule_path.relative_to(root).as_posix()}",
            "bytes": schedule_path.stat().st_size,
            "sha256": sha256(schedule_path),
        },
        {
            "role": "output_checkpoint_equivalence",
            "path": f"{root.name}/{checkpoint_path.relative_to(root).as_posix()}",
            "bytes": checkpoint_path.stat().st_size,
            "sha256": sha256(checkpoint_path),
        },
        {
            "role": "output_checkpoint_audit",
            "path": f"{root.name}/{checkpoint_audit_path.relative_to(root).as_posix()}",
            "bytes": checkpoint_audit_path.stat().st_size,
            "sha256": sha256(checkpoint_audit_path),
        },
        {
            "role": "output_stage_audit",
            "path": f"{root.name}/{audit_path.relative_to(root).as_posix()}",
            "bytes": audit_path.stat().st_size,
            "sha256": sha256(audit_path),
        },
        *restart_rows,
    ]
    pd.DataFrame(manifest_rows).to_csv(manifest_path, index=False)
    return {
        "audit": sha256(audit_path),
        "manifest": sha256(manifest_path),
        "schedule": sha256(schedule_path),
        "checkpoint_equivalence": sha256(checkpoint_path),
        "checkpoint_audit": sha256(checkpoint_audit_path),
    }


class P32022FiveMemberOperationalAcquisitionTests(unittest.TestCase):
    def test_formal_network_acquisition_rejects_server_environment(self) -> None:
        with patch.object(acquisition.os, "name", "posix"):
            with self.assertRaisesRegex(RuntimeError, "local Windows workstation"):
                acquisition.run(argparse.Namespace())

    def test_task_matrix_is_exactly_thirteen_by_five_by_sixty(self) -> None:
        dates = pd.date_range("2022-06-02", periods=13, freq="7D")
        tasks = acquisition.build_tasks(
            [value.strftime("%Y-%m-%d") for value in dates]
        )
        self.assertEqual(len(tasks), 13 * 5 * 60)
        self.assertEqual(
            {task.member for task in tasks}, set(acquisition.OPERATIONAL_MEMBERS)
        )
        self.assertEqual({task.lead_hour for task in tasks}, set(acquisition.LEAD_HOURS))
        with self.assertRaisesRegex(ValueError, "13 unique decision dates"):
            acquisition.build_tasks(
                [value.strftime("%Y-%m-%d") for value in dates[:-1]]
            )

    def test_weather_contract_rejects_nonfrozen_member(self) -> None:
        weather = {
            "archive": "NOAA_GEFS_operational_public_archive",
            "contract_id": "gefs-p3-2022-frozen-operational-weather-v1",
            "cycle_hour_utc": 0,
            "forecast_horizon": "local_decision_date_D_through_D_plus_6",
            "member_count": 5,
            "members": list(acquisition.OPERATIONAL_MEMBERS),
            "model_version": "GEFSv12",
            "network_access_performed": False,
            "product_grid": "atmos/pgrb2sp25_0p25_degree",
            "server_external_weather_download_allowed": False,
            "timezone_conversion": "America/Chicago_local_day",
            "derived_daily_variables": [
                "precipitation_mm",
                "temperature_min_c",
                "temperature_max_c",
                "actual_vapor_pressure_kpa",
                "wind_speed_m_s",
                "solar_kj_m2_day",
            ],
            "precipitation_correction": {
                "2022_reference_weather_used_for_fit": False,
                "fit_period": "2000-2019",
            },
            "nonprecipitation_correction": {
                "2022_reference_weather_used_for_fit": False,
            },
        }
        acquisition.validate_weather_contract(weather)
        weather["members"][-1] = "gep05"
        with self.assertRaisesRegex(ValueError, "operational weather contract"):
            acquisition.validate_weather_contract(weather)

    def test_trunk_validation_binds_all_thirteen_restarts(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory) / "trunk"
            root.mkdir()
            expected = write_trunk_fixture(root)
            with patch.object(acquisition, "EXPECTED_TRUNK_SHA256", expected):
                schedule, _ = acquisition.validate_schedule(root)
                self.assertEqual(len(schedule), 13)
                missing = (
                    root
                    / "units/Y2022/P3/all_checkpoints_v1/checkpoints/20220714/result_forec.end"
                )
                missing.unlink()
                with self.assertRaisesRegex(FileNotFoundError, "restart checkpoint"):
                    acquisition.validate_schedule(root)

    def test_daily_contract_requires_exactly_455_member_day_rows(self) -> None:
        rows = []
        for decision in pd.date_range("2022-06-02", periods=13, freq="7D"):
            for member in acquisition.OPERATIONAL_MEMBERS:
                for lead_day in range(1, 8):
                    rows.append(
                        {
                            "decision_date": decision.strftime("%Y-%m-%d"),
                            "local_date": (decision + pd.Timedelta(days=lead_day - 1)).strftime("%Y-%m-%d"),
                            "lead_day": lead_day,
                            "gefs_member": member,
                            "precipitation_mm": 1.0,
                            "temperature_min_c": 2.0,
                            "temperature_max_c": 3.0,
                            "shortwave_w_m2": 4.0,
                            "wind_speed_m_s": 5.0,
                            "vpd_kpa": 6.0,
                        }
                    )
        frame = pd.DataFrame(rows)
        with patch.object(
            acquisition.legacy,
            "aggregate_gefs_point_records",
            side_effect=[
                frame.loc[frame["gefs_member"].eq(member)].drop(
                    columns="gefs_member"
                )
                for member in acquisition.OPERATIONAL_MEMBERS
            ],
        ):
            points = pd.DataFrame(
                {"gefs_member": [member for member in acquisition.OPERATIONAL_MEMBERS]}
            )
            result = acquisition.assemble_daily_member_weather(points)
        self.assertEqual(len(result), 455)
        self.assertTrue(result.groupby("decision_date").size().eq(35).all())


if __name__ == "__main__":
    unittest.main()
