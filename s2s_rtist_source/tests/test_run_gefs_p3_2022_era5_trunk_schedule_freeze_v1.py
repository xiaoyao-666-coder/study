from __future__ import annotations

import tempfile
import unittest
import json
import csv
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

import scripts.simulation.run_gefs_p3_2022_era5_trunk_schedule_freeze_v1 as trunk2022

from scripts.simulation.run_gefs_p3_2022_era5_trunk_schedule_freeze_v1 import (
    convert_era5_legacy_original_to_swap,
    legacy_weather_header,
    read_legacy_swap_weather,
    rounded_swap_values,
    validate_source_workspace,
    validate_schedule,
    write_legacy_swap_weather,
)


def crop_fixture() -> pd.DataFrame:
    dates = pd.date_range("2022-04-26", "2022-06-30", freq="D")
    dvs = [
        min(1.5, max(0.0, (date - pd.Timestamp("2022-05-01")).days * 0.05))
        for date in dates
    ]
    return pd.DataFrame({"Date": dates, "DVS": dvs})


def schedule_fixture() -> pd.DataFrame:
    rows = []
    first = pd.Timestamp("2022-05-04")
    for index in range(8):
        decision = first + pd.Timedelta(days=7 * index)
        checkpoint = decision - pd.Timedelta(days=1)
        rows.append(
            {
                "site_id": "P3",
                "target_year": 2022,
                "split": "independent_test",
                "schedule_index": index,
                "state_checkpoint_date": checkpoint.strftime("%Y-%m-%d"),
                "state_dvs": min(
                    1.5,
                    max(0.0, (checkpoint - pd.Timestamp("2022-05-01")).days * 0.05),
                ),
                "decision_date": decision.strftime("%Y-%m-%d"),
                "horizon_end_date": (decision + pd.Timedelta(days=6)).strftime("%Y-%m-%d"),
                "harvest_date": "2022-06-30",
                "dvs_threshold": 0.1,
                "sampling_interval_days": 7,
                "horizon_days": 7,
            }
        )
    return pd.DataFrame(rows)


class P32022Era5TrunkScheduleFreezeTests(unittest.TestCase):
    def test_independent_protocol_binds_b2_and_source_hashes(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            protocol = {
                "status": "p3_2022_b2_independent_protocol_frozen_before_target_data_access",
                "target": {"site_id": "P3", "target_year": 2022},
                "schedule_rule": {
                    **trunk2022.SCHEDULE_RULE,
                    "trunk_weather_source": "ERA5_2022_observed_daily_weather",
                },
            }
            audit = {
                "status": "p3_2022_b2_independent_protocol_frozen_before_target_data_access",
                "mandatory_gate_passed": True,
            }
            (root / trunk2022.INDEPENDENT_PROTOCOL_NAMES["protocol"]).write_text(
                json.dumps(protocol), encoding="utf-8"
            )
            (root / trunk2022.INDEPENDENT_PROTOCOL_NAMES["audit"]).write_text(
                json.dumps(audit), encoding="utf-8"
            )
            pd.DataFrame(
                [
                    {
                        "component_id": "P1_full_history_composite",
                        "sha256": trunk2022.EXPECTED_COMPONENT_HASHES["P1"],
                    },
                    {
                        "component_id": "P15_full_history_composite",
                        "sha256": trunk2022.EXPECTED_COMPONENT_HASHES["P15"],
                    },
                    {
                        "component_id": "B2_regret_sensitive_router",
                        "sha256": trunk2022.EXPECTED_COMPONENT_HASHES["B2_gate"],
                    },
                    {
                        "component_id": "B2_regret_sensitive_router_payload",
                        "sha256": trunk2022.EXPECTED_COMPONENT_HASHES["B2_model_payload"],
                    },
                ]
            ).to_csv(root / trunk2022.INDEPENDENT_PROTOCOL_NAMES["components"], index=False)
            (root / trunk2022.INDEPENDENT_PROTOCOL_NAMES["stages"]).write_text(
                "stage_order,stage_id\n1,protocol_freeze\n", encoding="utf-8"
            )
            (root / trunk2022.INDEPENDENT_PROTOCOL_NAMES["manifest"]).write_text(
                "role,path,bytes,sha256\n", encoding="utf-8"
            )
            expected = {
                key: trunk2022.sha256_file(root / name)
                for key, name in trunk2022.INDEPENDENT_PROTOCOL_NAMES.items()
            }
            with patch.object(
                trunk2022, "EXPECTED_INDEPENDENT_PROTOCOL_SHA256", expected
            ):
                validated = trunk2022.validate_independent_protocol(root)
            self.assertEqual(validated["audit"]["mandatory_gate_passed"], True)

            components = pd.read_csv(
                root / trunk2022.INDEPENDENT_PROTOCOL_NAMES["components"]
            )
            components.loc[
                components["component_id"].eq("B2_regret_sensitive_router"), "sha256"
            ] = "0" * 64
            components.to_csv(
                root / trunk2022.INDEPENDENT_PROTOCOL_NAMES["components"], index=False
            )
            expected["components"] = trunk2022.sha256_file(
                root / trunk2022.INDEPENDENT_PROTOCOL_NAMES["components"]
            )
            with patch.object(
                trunk2022, "EXPECTED_INDEPENDENT_PROTOCOL_SHA256", expected
            ):
                with self.assertRaisesRegex(ValueError, "component hashes"):
                    trunk2022.validate_independent_protocol(root)

    def test_acquisition_protocol_manifest_is_fail_closed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            contract_path = root / "gefs_p3_2022_era5_local_acquisition_contract_v1.json"
            audit_path = root / "gefs_p3_2022_era5_local_acquisition_protocol_audit_v1.json"
            manifest_path = root / "gefs_p3_2022_era5_local_acquisition_protocol_manifest_v1.csv"
            contract = {
                "contract_id": "gefs-p3-2022-era5-local-acquisition-v1",
                "target_site": "P3",
                "target_year": 2022,
                "variables": list(trunk2022.ERA5_VARIABLES.values()),
                "source_product": {"expected_day_count": 364},
            }
            audit = {
                "mandatory_gate_passed": True,
                "P3_2022_weather_rows_read": 0,
                "P3_2022_network_requests": 0,
                "P3_2022_target_labels_read": 0,
            }
            contract_path.write_text(json.dumps(contract), encoding="utf-8")
            audit_path.write_text(json.dumps(audit), encoding="utf-8")
            pd.DataFrame(
                [
                    {"role": "output_contract", "sha256": trunk2022.sha256_file(contract_path)},
                    {"role": "output_audit", "sha256": trunk2022.sha256_file(audit_path)},
                ]
            ).to_csv(manifest_path, index=False)
            trunk2022.validate_acquisition_protocol(root)
            audit_path.write_text(json.dumps({**audit, "P3_2022_network_requests": 1}))
            with self.assertRaisesRegex(ValueError, "failed validation"):
                trunk2022.validate_acquisition_protocol(root)

    def test_download_archive_rejects_changed_file(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            raster = root / "temperature_2m" / "temperature_2m_0.tif"
            raster.parent.mkdir()
            raster.write_bytes(b"fixture")
            audit = {
                "status": "p3_2022_era5_local_acquisition_complete_pending_server_transfer",
                "mandatory_gate_passed": True,
                "historical_parity_rows": 1,
                "historical_parity_all_passed": True,
                "P3_2022_weather_rows_read": 1,
                "P3_2022_target_labels_read": 0,
                "P3_2023_rows_or_labels_read": 0,
                "P3_2024_rows_or_labels_read": 0,
                "raster_file_count": 1,
                "spatial_subset": "P3_exact_historical_export_grid_pixel",
                "model_training_performed": False,
                "decision_dates_resolved": False,
                "swap_simulation_performed": False,
            }
            (root / "gefs_p3_2022_era5_local_acquisition_audit_v1.json").write_text(
                json.dumps(audit), encoding="utf-8"
            )
            manifest_path = root / "gefs_p3_2022_era5_local_file_manifest_v1.csv"
            with manifest_path.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(
                    handle, fieldnames=["relative_path", "bytes", "sha256"]
                )
                writer.writeheader()
                writer.writerow(
                    {
                        "relative_path": "temperature_2m/temperature_2m_0.tif",
                        "bytes": raster.stat().st_size,
                        "sha256": trunk2022.sha256_file(raster),
                    }
                )
            with patch.object(trunk2022, "EXPECTED_DAYS", 1), patch.object(
                trunk2022, "EXPECTED_RASTERS", 1
            ), patch.object(trunk2022, "EXPECTED_PARITY_ROWS", 1):
                trunk2022.validate_download_archive(root)
                raster.write_bytes(b"changed")
                with self.assertRaisesRegex(ValueError, "changed"):
                    trunk2022.validate_download_archive(root)

    def test_schedule_accepts_exact_frozen_rule(self) -> None:
        summary = validate_schedule(schedule_fixture(), crop_fixture())
        self.assertEqual(summary["decision_rows"], 8)
        self.assertEqual(summary["first_decision_date"], "2022-05-04")
        self.assertTrue(summary["all_decisions_pre_maturity"])

    def test_schedule_rejects_manual_date_shift(self) -> None:
        schedule = schedule_fixture()
        schedule.loc[0, "decision_date"] = "2022-05-05"
        with self.assertRaisesRegex(ValueError, "frozen DVS rule"):
            validate_schedule(schedule, crop_fixture())

    def test_schedule_rejects_mature_checkpoint(self) -> None:
        schedule = schedule_fixture()
        schedule.loc[3, "state_dvs"] = 2.0
        with self.assertRaisesRegex(ValueError, "frozen DVS rule"):
            validate_schedule(schedule, crop_fixture())

    def test_legacy_weather_precision_matches_historical_swap_files(self) -> None:
        weather = pd.DataFrame(
            [
                {
                    "local_date": "2022-01-01",
                    "solar_kj_m2_day": 8045.1318359375,
                    "temperature_min_c": -7.8125305,
                    "temperature_max_c": -0.2331238,
                    "actual_vapor_pressure_kpa": 0.219379,
                    "wind_speed_m_s": 1.244629,
                    "precipitation_mm": 0.000852,
                    "etref_mm": 2.011217,
                }
            ]
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "weather.022"
            write_legacy_swap_weather(path, weather, ["header\n"])
            text = path.read_text(encoding="utf-8")
        self.assertIn("8045.1      -7.8      -0.2      0.22", text)
        self.assertIn("1.2      0.0      2.0", text)

    def test_header_comes_from_historical_2019_file(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            rows = [f"header {index}\n" for index in range(11)]
            (root / "WeatherOriginal.019").write_text("".join(rows), encoding="utf-8")
            self.assertEqual(legacy_weather_header(root), rows)

    def test_swap_weather_reader_preserves_file_precision(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "weather.019"
            path.write_text(
                "header\n 'Weather' 1 1 2019 8045.1 -7.8 -0.2 0.22 1.2 0.0 2.0\n",
                encoding="utf-8",
            )
            frame = read_legacy_swap_weather(path)
        self.assertEqual(frame.loc[0, "local_date"], "2019-01-01")
        self.assertEqual(frame.loc[0, "actual_vapor_pressure_kpa"], 0.22)

    def test_original_conversion_is_available_only_for_historical_parity(self) -> None:
        raw = pd.DataFrame(
            [
                {
                    "target_year": 2019,
                    "site_id": "P3",
                    "local_date": "2019-01-01",
                    "temperature_mean_k": 273.15,
                    "temperature_min_k": 272.15,
                    "temperature_max_k": 274.15,
                    "dewpoint_k": 271.15,
                    "solar_j_m2_day": 8_000_000.0,
                    "precipitation_m": 0.001,
                    "potential_evaporation_m": -0.002,
                    "wind_u_m_s": 3.0,
                    "wind_v_m_s": 4.0,
                }
            ]
        )
        converted = convert_era5_legacy_original_to_swap(raw)
        rounded = rounded_swap_values(converted)
        self.assertEqual(rounded.loc[0, "solar_kj_m2_day"], 8000.0)
        self.assertEqual(rounded.loc[0, "wind_speed_m_s"], 5.0)
        self.assertEqual(rounded.loc[0, "precipitation_mm"], 1.0)
        self.assertEqual(rounded.loc[0, "etref_mm"], 2.0)
        temperature = 0.0
        dewpoint = -2.0
        alpha = 17.27 * temperature / (237.7 + temperature) + np.log(
            temperature + 273.15
        )
        beta = 17.27 * dewpoint / (237.7 + dewpoint) + np.log(
            dewpoint + 273.15
        )
        expected_hum = 0.61078 * np.exp(
            temperature / (temperature + 238.3) * 17.2694
        ) * (1.0 - np.exp(beta - alpha))
        self.assertEqual(
            rounded.loc[0, "actual_vapor_pressure_kpa"],
            float(f"{expected_hum:.2f}"),
        )

    def test_formal_p3_setup_audit_overrides_stale_auxiliary_site_config(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            workspace = Path(directory) / "Y2019" / "P3" / "workspace"
            workspace.mkdir(parents=True)
            (workspace / "Swap1.pre_trunk_smoke.swp").write_text("SWINCO=1\n")
            (workspace / "real_ir_update.py").write_text("# helper\n")
            (workspace / "weather.019").write_text(
                " 'Weather' 1 1 2019 1 1 2 0.5 1 0 1\n"
            )
            (workspace / "Swap.exe").write_bytes(b"test")
            (workspace / "site_config.json").write_text(
                json.dumps({"paper_site_id": "stale_value"})
            )
            (workspace.parent / "setup_audit_v1.json").write_text(
                json.dumps(
                    {
                        "mandatory_gate_passed": True,
                        "site_id": "P3",
                        "target_year": 2019,
                    }
                )
            )
            result = validate_source_workspace(workspace)
        self.assertEqual(result["formal_setup_audit"].name, "setup_audit_v1.json")


if __name__ == "__main__":
    unittest.main()
