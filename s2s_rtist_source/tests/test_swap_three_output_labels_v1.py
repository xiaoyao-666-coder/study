from __future__ import annotations

import inspect
import json
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from swap_three_output_labels_v1 import (
    CRP_COLUMNS,
    extract_candidate_labels,
    flatten_candidate_labels,
    inclusive_horizon_end_doy,
    patch_nprintday_text,
)
from restart_raw_audit_v1 import (  # noqa: E402
    preserve_candidate_raw_outputs,
    should_preserve_raw_candidate,
)


AUDIT_ROOT = (
    ROOT
    / "site_general_surrogate_eval"
    / "three_output_balance_audit_p1_20240716_server_v1"
)


class HorizonTests(unittest.TestCase):
    def test_seven_day_inclusive_window_ends_on_day_six(self) -> None:
        self.assertEqual(inclusive_horizon_end_doy(198, 7), 204)


class CandidateLabelTests(unittest.TestCase):
    def extract(self, case: str):
        folder = AUDIT_ROOT / case
        return extract_candidate_labels(
            pre_crop_path=folder / "result_forec.crp",
            pre_profile_path=folder / "result_forec.vap",
            restart_crop_path=folder / "result_restart.crp",
            restart_profile_path=folder / "result_restart.vap",
            restart_increment_path=folder / "result_restart.inc",
            decision_date="2024-07-16",
            horizon_days=7,
            nprintday=1,
        )

    def test_ir0_uses_exactly_seven_dates(self) -> None:
        result = self.extract("P1_20240716_ir0")

        self.assertEqual(result.summary["horizon_days_actual"], 7)
        self.assertEqual(result.summary["horizon_start_date"], "2024-07-16")
        self.assertEqual(result.summary["horizon_end_date"], "2024-07-22")
        self.assertEqual(len(result.daily), 7)

    def test_ir0_actual_et_includes_interception(self) -> None:
        result = self.extract("P1_20240716_ir0")

        self.assertAlmostEqual(result.summary["tact_7d_mm"], 15.7695, places=6)
        self.assertAlmostEqual(result.summary["eact_7d_mm"], 1.7410, places=6)
        self.assertAlmostEqual(result.summary["interc_7d_mm"], 1.5040, places=6)
        self.assertAlmostEqual(result.summary["aet_7d_mm"], 19.0145, places=6)

    def test_ir0_uses_fixed_storage_signed_flux_and_direct_outflow(self) -> None:
        result = self.extract("P1_20240716_ir0")

        self.assertAlmostEqual(
            result.summary["predecision_soil_storage_0_100cm_mm"],
            80.39,
            places=6,
        )
        self.assertAlmostEqual(
            result.summary["delta_soil_storage_0_100cm_7d_mm"],
            -11.86,
            places=6,
        )
        self.assertAlmostEqual(
            result.summary["soil_boundary_waterflux_100cm_signed_7d_mm"],
            0.0716495,
            places=7,
        )
        self.assertAlmostEqual(
            result.summary["soil_boundary_outflow_100cm_7d_mm"],
            -0.0716495,
            places=7,
        )
        self.assertNotIn("moving_root_boundary_term_7d_mm", result.summary)
        self.assertAlmostEqual(
            result.summary["residual_flux_7d_mm"], -0.0716495, places=7
        )
        self.assertAlmostEqual(
            result.summary["water_balance_residual_0_100cm_7d_mm"],
            0.1171495,
            places=7,
        )

    def test_ir30_labels_match_audit(self) -> None:
        result = self.extract("P1_20240716_ir30")

        self.assertAlmostEqual(result.summary["irrigation_7d_mm"], 30.0, places=6)
        self.assertAlmostEqual(result.summary["aet_7d_mm"], 33.1791, places=6)
        self.assertAlmostEqual(
            result.summary["delta_soil_storage_0_100cm_7d_mm"],
            3.95,
            places=6,
        )
        self.assertAlmostEqual(
            result.summary["soil_boundary_waterflux_100cm_signed_7d_mm"],
            0.0427415,
            places=7,
        )
        self.assertAlmostEqual(
            result.summary["soil_boundary_outflow_100cm_7d_mm"],
            -0.0427415,
            places=7,
        )
        self.assertAlmostEqual(
            result.summary["water_balance_residual_0_100cm_7d_mm"],
            0.1136415,
            places=7,
        )

    def test_daily_output_contains_required_sequences(self) -> None:
        result = self.extract("P1_20240716_ir30")

        required = {
            "date",
            "root_depth_cm",
            "soil_vwc_0_100cm",
            "soil_storage_0_100cm_mm",
            "tact_mm",
            "eact_mm",
            "interc_mm",
            "aet_mm",
            "runoff_mm",
            "soil_drainage_0_100cm_mm",
            "soil_boundary_waterflux_100cm_signed_mm",
            "soil_boundary_outflow_100cm_mm",
            "soil_boundary_depth_cm",
        }
        self.assertTrue(required.issubset(result.daily.columns))
        self.assertNotIn("rootzone_vwc", result.daily.columns)
        self.assertNotIn("moving_root_boundary_term_mm", result.daily.columns)

    def test_flattened_labels_have_seven_numbered_daily_values(self) -> None:
        result = self.extract("P1_20240716_ir30")

        flat = flatten_candidate_labels(result)

        self.assertAlmostEqual(flat["aet_7d_mm"], 33.1791, places=6)
        self.assertIn("soil_vwc_0_100cm_day01", flat)
        self.assertIn("soil_vwc_0_100cm_day07", flat)
        self.assertIn("soil_boundary_waterflux_100cm_signed_day01_mm", flat)
        self.assertIn("soil_boundary_waterflux_100cm_signed_day07_mm", flat)
        self.assertIn("soil_boundary_outflow_100cm_day01_mm", flat)
        self.assertNotIn("moving_root_boundary_term_day07_mm", flat)
        self.assertNotIn("rootzone_vwc_day01", flat)
        self.assertNotIn("soil_vwc_0_100cm_day08", flat)

    def test_formal_labels_use_fixed_100cm_control_volume_when_roots_are_shallow(
        self,
    ) -> None:
        def crop_row(date: str, root_depth_cm: float) -> str:
            values = ["0"] * len(CRP_COLUMNS)
            values[CRP_COLUMNS.index("Date")] = date
            values[CRP_COLUMNS.index("Rootd")] = str(root_depth_cm)
            return ",".join(values)

        def profile_rows(date: str) -> list[str]:
            return [
                f"{date},25,0.2,0,0.01,0,0.00,0,-50",
                f"{date},75,0.4,0,0.01,0,-0.02,-50,-100",
                f"{date},125,0.5,0,0.01,0,-0.10,-100,-150",
            ]

        with tempfile.TemporaryDirectory() as temporary:
            folder = Path(temporary)
            dates = [f"2024-07-{day:02d}" for day in range(16, 23)]
            (folder / "result_forec.crp").write_text(
                crop_row("2024-07-15", 50.0) + "\n", encoding="utf-8"
            )
            (folder / "result_restart.crp").write_text(
                "\n".join(crop_row(date, 50.0) for date in dates) + "\n",
                encoding="utf-8",
            )
            profile_header = (
                "date,depth,wcontent,phead,drainage,rootext,waterflux,top,bottom"
            )
            (folder / "result_forec.vap").write_text(
                "\n".join(
                    [profile_header, *profile_rows("2024-07-15")]
                )
                + "\n",
                encoding="utf-8",
            )
            restart_profile_rows = [profile_header]
            for date in dates:
                restart_profile_rows.extend(profile_rows(date))
            (folder / "result_restart.vap").write_text(
                "\n".join(restart_profile_rows) + "\n", encoding="utf-8"
            )
            increment_rows = [
                "Date,Dcum,Rain,Snow,Irrig,Interc,Runon,Runoff,Tact,Eact"
            ]
            increment_rows.extend(
                f"{date},{index},0,0,0,0,0,0,0,0"
                for index, date in enumerate(dates, start=1)
            )
            (folder / "result_restart.inc").write_text(
                "\n".join(increment_rows) + "\n", encoding="utf-8"
            )

            result = extract_candidate_labels(
                pre_crop_path=folder / "result_forec.crp",
                pre_profile_path=folder / "result_forec.vap",
                restart_crop_path=folder / "result_restart.crp",
                restart_profile_path=folder / "result_restart.vap",
                restart_increment_path=folder / "result_restart.inc",
                decision_date="2024-07-16",
                horizon_days=7,
                nprintday=1,
            )

        self.assertIn("soil_vwc_0_100cm", result.daily.columns)
        self.assertTrue(result.daily["root_depth_cm"].eq(50.0).all())
        self.assertTrue(result.daily["soil_vwc_0_100cm"].eq(0.3).all())
        self.assertTrue(result.daily["soil_storage_0_100cm_mm"].eq(300.0).all())
        self.assertTrue(result.daily["soil_boundary_depth_cm"].eq(100.0).all())
        self.assertNotIn("moving_root_boundary_term_mm", result.daily.columns)
        self.assertEqual(result.summary["control_volume_type"], "fixed_0_100cm")
        self.assertEqual(result.summary["control_depth_cm"], 100.0)
        self.assertAlmostEqual(
            result.summary["predecision_soil_storage_0_100cm_mm"], 300.0
        )
        self.assertAlmostEqual(
            result.summary["delta_soil_storage_0_100cm_7d_mm"], 0.0
        )
        self.assertAlmostEqual(
            result.summary["soil_boundary_outflow_100cm_7d_mm"], 7.0
        )
        self.assertNotIn("moving_root_boundary_term_7d_mm", result.summary)


class SwapConfigTests(unittest.TestCase):
    def test_extractor_default_matches_formal_frequency(self) -> None:
        default = inspect.signature(extract_candidate_labels).parameters[
            "nprintday"
        ].default

        self.assertEqual(default, 24)

    def test_patch_nprintday_sets_formal_restart_frequency(self) -> None:
        text = "PERIOD = 1\nNPrintDay = 1 ! output times\n"

        patched = patch_nprintday_text(text, 24)

        self.assertIn("NPrintDay = 24", patched)

    def test_generator_wires_formal_restart_frequency_into_extraction(self) -> None:
        source = (ROOT / "generate_restart_decision_dataset.py").read_text(
            encoding="utf-8"
        )

        self.assertIn("RESTART_NPRINTDAY = 24", source)
        self.assertIn("patch_nprintday_text", source)
        self.assertIn("nprintday=RESTART_NPRINTDAY", source)
        self.assertIn('"swap_version": "4.0.1"', source)
        self.assertIn('"flux_rate_source_unit": "cm/day"', source)
        self.assertIn('"soil_vwc_0_100cm_unit": "cm3/cm3"', source)
        self.assertIn('"control_volume_type": "fixed_0_100cm"', source)
        self.assertIn('"control_depth_cm": 100.0', source)
        self.assertIn(
            '"data_processing_spec_version": '
            '"three_output_surrogate_data_processing_spec_v1_fixed_0_100cm"',
            source,
        )
        self.assertNotIn('"rootzone_vwc_unit":', source)

    def test_continuous_runner_copies_flux_diagnostic_dependency(self) -> None:
        source = (
            ROOT / "run_continuous_ir_12site_restart_generation_v1.py"
        ).read_text(encoding="utf-8")

        self.assertIn("rootzone_flux_frequency_diagnostic_v1.py", source)
        self.assertIn("restart_raw_audit_v1.py", source)

    def test_confirmed_smoke_runner_copies_formal_label_dependencies(self) -> None:
        source = (
            ROOT / "run_confirmed_5site_restart_generation_smoke_v1.py"
        ).read_text(encoding="utf-8")

        self.assertIn("swap_three_output_labels_v1.py", source)
        self.assertIn("rootzone_flux_frequency_diagnostic_v1.py", source)
        self.assertIn("restart_raw_audit_v1.py", source)

    def test_raw_audit_preserves_zero_and_maximum_irrigation_only(self) -> None:
        options = [0.0, 10.0, 30.0, 60.0]

        self.assertTrue(should_preserve_raw_candidate(0.0, options))
        self.assertFalse(should_preserve_raw_candidate(30.0, options))
        self.assertTrue(should_preserve_raw_candidate(60.0, options))

    def test_raw_audit_copies_reproducibility_files_and_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "source"
            audit_root = root / "audit"
            source.mkdir()
            for name in (
                "result_restart.inc",
                "result_restart.vap",
                "result_restart.crp",
                "result_restart.wba",
                "result_restart.end",
                "swap.swp",
            ):
                (source / name).write_text(name, encoding="utf-8")

            target = preserve_candidate_raw_outputs(
                date_t="16-Jul-2024",
                decision_doy=198,
                irrigation_mm=60.0,
                irrigation_options_mm=[0.0, 30.0, 60.0],
                source_dir=source,
                audit_root=audit_root,
            )

            self.assertIsNotNone(target)
            assert target is not None
            self.assertTrue((target / "result_restart.vap").exists())
            manifest = json.loads(
                (target / "raw_audit_manifest.json").read_text(encoding="utf-8")
            )
            self.assertEqual(manifest["nprintday"], 24)
            self.assertEqual(manifest["irrigation_mm"], 60.0)
            self.assertIn("result_restart.vap", manifest["files"])


if __name__ == "__main__":
    unittest.main()
