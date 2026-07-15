from __future__ import annotations

import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
from unittest.mock import patch

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from validate_three_output_smoke_v1 import (  # noqa: E402
    SmokeValidationError,
    validate_smoke_dataset,
    write_validation_outputs,
)
import run_confirmed_5site_restart_generation_smoke_v1 as smoke_runner  # noqa: E402


IRRIGATION_OPTIONS = [0.0, 10.0, 15.0, 20.0, 25.0, 30.0, 40.0, 60.0]


def valid_smoke_frame() -> pd.DataFrame:
    rows = []
    for irrigation in IRRIGATION_OPTIONS:
        row = {
            "site": "P1",
            "date_t": "16-Jul-2024",
            "ir": irrigation,
            "net_gain_7d": irrigation * 0.5,
            "horizon_days_actual": 7,
            "nprintday": 24,
            "flux_integration_method": "trapezoid_actual_subdaily_interval",
            "increment_grouping_method": "Dcum_1_to_horizon_days",
            "swap_version": "4.0.1",
            "water_depth_unit": "mm",
            "flux_rate_source_unit": "cm/day",
            "root_depth_unit": "cm",
            "soil_vwc_0_100cm_unit": "cm3/cm3",
            "control_volume_type": "fixed_0_100cm",
            "control_depth_cm": 100.0,
            "data_processing_spec_version": (
                "three_output_surrogate_data_processing_spec_v1_fixed_0_100cm"
            ),
            "raw_audit_preserved": int(irrigation in {0.0, 60.0}),
            "raw_audit_dir": (
                f"candidate_raw_audit/2024/16jul2024/ir_{irrigation:g}mm"
                if irrigation in {0.0, 60.0}
                else ""
            ),
            "rain_7d_mm": 4.0,
            "snow_7d_mm": 0.0,
            "irrigation_7d_mm": irrigation,
            "runon_7d_mm": 0.0,
            "aet_7d_mm": 14.0,
            "runoff_7d_mm": 0.0,
            "soil_drainage_0_100cm_7d_mm": 1.0,
            "soil_boundary_waterflux_100cm_signed_7d_mm": -2.0,
            "soil_boundary_outflow_100cm_7d_mm": 2.0,
            "residual_flux_7d_mm": 3.0,
            "predecision_soil_storage_0_100cm_mm": 250.0,
            "final_soil_storage_0_100cm_mm": 250.0 + irrigation - 13.0,
            "delta_soil_storage_0_100cm_7d_mm": irrigation - 13.0,
            "water_balance_residual_0_100cm_7d_mm": 0.0,
            "max_abs_soil_boundary_depth_error_cm": 0.0,
        }
        for day in range(1, 8):
            suffix = f"day{day:02d}"
            storage_mm = 250.0 + (irrigation - 13.0) * day / 7.0
            row[f"tact_{suffix}_mm"] = 1.0
            row[f"eact_{suffix}_mm"] = 0.5
            row[f"interc_{suffix}_mm"] = 0.5
            row[f"aet_{suffix}_mm"] = 2.0
            row[f"root_depth_{suffix}_cm"] = 60.0
            row[f"soil_vwc_0_100cm_{suffix}"] = storage_mm / 1000.0
            row[f"soil_storage_0_100cm_{suffix}_mm"] = storage_mm
            row[f"soil_drainage_0_100cm_{suffix}_mm"] = 1.0 / 7.0
            row[f"soil_boundary_waterflux_100cm_signed_{suffix}_mm"] = -2.0 / 7.0
            row[f"soil_boundary_outflow_100cm_{suffix}_mm"] = 2.0 / 7.0
            row[f"soil_boundary_depth_{suffix}_cm"] = 100.0
        rows.append(row)
    return pd.DataFrame(rows)


class SmokeValidationTests(unittest.TestCase):
    def test_valid_formal_smoke_dataset_passes(self) -> None:
        try:
            result = validate_smoke_dataset(valid_smoke_frame())
        except SmokeValidationError as exc:
            self.fail(f"fixed 0-100 cm formal dataset should pass: {exc}")

        self.assertTrue(result.passed)
        self.assertEqual(result.row_count, 8)
        self.assertEqual(result.site_count, 1)
        self.assertEqual(result.site_date_count, 1)
        self.assertAlmostEqual(result.max_abs_water_balance_residual_mm, 0.0)
        self.assertEqual(int(result.site_summary.loc[0, "raw_audit_rows"]), 2)
        self.assertNotIn(
            "max_abs_moving_root_boundary_term_mm", result.site_summary.columns
        )

    def test_rejects_nonformal_nprintday(self) -> None:
        frame = valid_smoke_frame()
        frame["nprintday"] = 4

        with self.assertRaisesRegex(SmokeValidationError, "nprintday"):
            validate_smoke_dataset(frame)

    def test_rejects_missing_daily_field(self) -> None:
        frame = valid_smoke_frame().drop(columns=["soil_vwc_0_100cm_day07"])

        with self.assertRaisesRegex(SmokeValidationError, "soil_vwc_0_100cm_day07"):
            validate_smoke_dataset(frame)

    def test_rejects_dynamic_control_volume(self) -> None:
        frame = valid_smoke_frame()
        frame["control_volume_type"] = "dynamic_rootzone"

        with self.assertRaisesRegex(SmokeValidationError, "control_volume_type"):
            validate_smoke_dataset(frame)

    def test_rejects_non_100cm_control_depth(self) -> None:
        frame = valid_smoke_frame()
        frame["control_depth_cm"] = 85.0

        with self.assertRaisesRegex(SmokeValidationError, "control_depth_cm"):
            validate_smoke_dataset(frame)

    def test_rejects_storage_inconsistent_with_fixed_100cm_vwc(self) -> None:
        frame = valid_smoke_frame()
        frame.loc[0, "soil_storage_0_100cm_day03_mm"] += 10.0

        with self.assertRaisesRegex(SmokeValidationError, "fixed storage identity"):
            validate_smoke_dataset(frame)

    def test_rejects_legacy_moving_boundary_fields(self) -> None:
        frame = valid_smoke_frame()
        frame["moving_root_boundary_term_7d_mm"] = 0.0

        with self.assertRaisesRegex(SmokeValidationError, "moving-boundary"):
            validate_smoke_dataset(frame)

    def test_rejects_inconsistent_aet_components(self) -> None:
        frame = valid_smoke_frame()
        frame.loc[0, "aet_day03_mm"] = 9.0

        with self.assertRaisesRegex(SmokeValidationError, "AET component"):
            validate_smoke_dataset(frame)

    def test_rejects_incomplete_irrigation_candidate_set(self) -> None:
        frame = valid_smoke_frame().iloc[:-1].copy()

        with self.assertRaisesRegex(SmokeValidationError, "irrigation candidates"):
            validate_smoke_dataset(frame)

    def test_rejects_wrong_unit_metadata(self) -> None:
        frame = valid_smoke_frame()
        frame["flux_rate_source_unit"] = "mm/day"

        with self.assertRaisesRegex(SmokeValidationError, "flux_rate_source_unit"):
            validate_smoke_dataset(frame)

    def test_rejects_missing_raw_endpoint_audit(self) -> None:
        frame = valid_smoke_frame()
        frame.loc[frame["ir"] == 60.0, "raw_audit_preserved"] = 0
        frame.loc[frame["ir"] == 60.0, "raw_audit_dir"] = ""

        with self.assertRaisesRegex(SmokeValidationError, "raw audit"):
            validate_smoke_dataset(frame)

    def test_confirmed_runner_writes_validation_outputs(self) -> None:
        source = (
            ROOT / "run_confirmed_5site_restart_generation_smoke_v1.py"
        ).read_text(encoding="utf-8")

        self.assertIn("validate_smoke_dataset", source)
        self.assertIn("write_validation_outputs", source)

    def test_validation_output_names_are_stable(self) -> None:
        result = validate_smoke_dataset(valid_smoke_frame())

        with tempfile.TemporaryDirectory() as tmp:
            summary_csv, report_md = write_validation_outputs(result, Path(tmp))

            self.assertEqual(
                summary_csv.name, "three_output_smoke_validation_summary_v1.csv"
            )
            self.assertEqual(report_md.name, "three_output_smoke_validation_v1.md")
            self.assertTrue(summary_csv.exists())
            self.assertTrue(report_md.exists())

    def test_formal_runner_rejects_missing_confirmed_workspace(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            confirmed = root / "confirmed"
            template = root / "Maize"
            confirmed.mkdir()
            template.mkdir()

            with (
                patch.object(smoke_runner, "CONFIRMED_WORKSPACES", confirmed),
                patch.object(smoke_runner, "DEFAULT_SOURCE_MAIZE", template),
            ):
                with self.assertRaisesRegex(FileNotFoundError, "confirmed workspace"):
                    smoke_runner.source_workspace("P1")

    def test_template_fallback_requires_explicit_debug_flag(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            confirmed = root / "confirmed"
            template = root / "Maize"
            confirmed.mkdir()
            template.mkdir()

            with (
                patch.object(smoke_runner, "CONFIRMED_WORKSPACES", confirmed),
                patch.object(smoke_runner, "DEFAULT_SOURCE_MAIZE", template),
            ):
                with redirect_stdout(StringIO()):
                    source = smoke_runner.source_workspace(
                        "P1", allow_template_fallback=True
                    )

            self.assertEqual(source, template)


if __name__ == "__main__":
    unittest.main()
