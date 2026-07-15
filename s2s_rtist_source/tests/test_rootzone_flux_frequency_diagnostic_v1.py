from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from rootzone_flux_frequency_diagnostic_v1 import (
    aggregate_increment_rows,
    analyze_case_outputs,
    assign_horizon_times,
    moving_boundary_term_mm,
    patch_nprintday_text,
    rootzone_snapshot_metrics,
    signed_flux_to_downward_outflow_mm,
    split_profile_snapshots,
    trapezoid_integral,
)
from run_rootzone_flux_frequency_validation_v1 import (
    case_directory_name,
    patch_workspace_nprintday,
    validation_cases,
)


class FluxIntegrationTests(unittest.TestCase):
    def test_trapezoid_integral_uses_subdaily_time_steps(self) -> None:
        integral = trapezoid_integral(
            times_days=[0.0, 0.5, 1.0],
            rates_cm_day=[0.0, 2.0, 0.0],
        )

        self.assertAlmostEqual(integral, 1.0, places=12)

    def test_native_upward_flux_converts_to_downward_outflow(self) -> None:
        self.assertAlmostEqual(
            signed_flux_to_downward_outflow_mm(-0.7),
            7.0,
            places=12,
        )


class MovingBoundaryTests(unittest.TestCase):
    def test_deepening_root_adds_water_from_new_slice(self) -> None:
        profile = pd.DataFrame(
            {
                "top": [0.0, -50.0, -60.0],
                "bottom": [-50.0, -60.0, -100.0],
                "wcontent": [0.10, 0.20, 0.30],
            }
        )

        term = moving_boundary_term_mm(
            previous_profile=profile,
            current_profile=profile,
            previous_root_depth_cm=50.0,
            current_root_depth_cm=60.0,
        )

        self.assertAlmostEqual(term, 20.0, places=12)

    def test_shrinking_root_removes_water_from_control_volume(self) -> None:
        profile = pd.DataFrame(
            {
                "top": [0.0, -50.0, -60.0],
                "bottom": [-50.0, -60.0, -100.0],
                "wcontent": [0.10, 0.20, 0.30],
            }
        )

        term = moving_boundary_term_mm(
            previous_profile=profile,
            current_profile=profile,
            previous_root_depth_cm=60.0,
            current_root_depth_cm=50.0,
        )

        self.assertAlmostEqual(term, -20.0, places=12)

    def test_nonoverlapping_nan_compartment_does_not_poison_slice(self) -> None:
        profile = pd.DataFrame(
            {
                "top": [0.0, -10.0],
                "bottom": [-10.0, -20.0],
                "wcontent": [0.20, float("nan")],
            }
        )

        term = moving_boundary_term_mm(
            previous_profile=profile,
            current_profile=profile,
            previous_root_depth_cm=0.0,
            current_root_depth_cm=10.0,
        )

        self.assertAlmostEqual(term, 20.0, places=12)


class ProfileSnapshotTests(unittest.TestCase):
    def test_subdaily_profiles_are_split_and_assigned_elapsed_times(self) -> None:
        frame = pd.DataFrame(
            {
                "date": [
                    "2024-07-15",
                    "2024-07-15",
                    "2024-07-16",
                    "2024-07-16",
                    "2024-07-16",
                    "2024-07-16",
                ],
                "top": [0.0, -10.0, 0.0, -10.0, 0.0, -10.0],
                "bottom": [-10.0, -20.0, -10.0, -20.0, -10.0, -20.0],
                "wcontent": [0.1] * 6,
                "drainage": [0.0] * 6,
                "waterflux": [0.0, -0.1, 0.0, -0.2, 0.0, -0.3],
            }
        )

        snapshots = split_profile_snapshots(frame)
        timeline = assign_horizon_times(
            snapshots,
            decision_date="2024-07-16",
            nprintday=2,
            horizon_days=1,
        )

        self.assertEqual(len(snapshots), 3)
        self.assertEqual([row[0] for row in timeline], [0.0, 0.5, 1.0])

    def test_root_boundary_flux_is_taken_at_compartment_top(self) -> None:
        profile = pd.DataFrame(
            {
                "top": [0.0, -10.0],
                "bottom": [-10.0, -20.0],
                "wcontent": [0.1, 0.2],
                "drainage": [0.0, 0.0],
                "waterflux": [0.5, -0.25],
            }
        )

        metrics = rootzone_snapshot_metrics(profile, root_depth_cm=10.0)

        self.assertAlmostEqual(metrics["root_boundary_waterflux_cm_day"], -0.25)
        self.assertAlmostEqual(metrics["root_boundary_depth_error_cm"], 0.0)
        self.assertAlmostEqual(metrics["rootzone_storage_mm"], 10.0)


class IncrementAggregationTests(unittest.TestCase):
    def test_subdaily_increment_rows_are_summed_by_date(self) -> None:
        increments = pd.DataFrame(
            {
                "Date": ["2024-07-16", "2024-07-16", "2024-07-17"],
                "Rain": [0.1, 0.2, 0.4],
                "Tact": [0.3, 0.4, 0.5],
            }
        )

        daily = aggregate_increment_rows(
            increments,
            dates=["2024-07-16", "2024-07-17"],
            numeric_columns=["Rain", "Tact"],
        )

        self.assertEqual(daily["Date"].tolist(), ["2024-07-16", "2024-07-17"])
        self.assertAlmostEqual(float(daily.iloc[0]["Rain"]), 0.3, places=12)
        self.assertAlmostEqual(float(daily.iloc[0]["Tact"]), 0.7, places=12)

    def test_midnight_row_is_grouped_by_simulation_day_not_calendar_date(self) -> None:
        increments = pd.DataFrame(
            {
                "Date": [
                    "2024-07-16",
                    "2024-07-16",
                    "2024-07-17",
                    "2024-07-17",
                ],
                "Dcum": [1, 1, 1, 2],
                "Rain": [0.1, 0.2, 0.3, 0.4],
                "Tact": [0.2, 0.3, 0.4, 0.5],
            }
        )

        daily = aggregate_increment_rows(
            increments,
            dates=["2024-07-16", "2024-07-17"],
            numeric_columns=["Rain", "Tact"],
        )

        self.assertAlmostEqual(float(daily.iloc[0]["Rain"]), 0.6, places=12)
        self.assertAlmostEqual(float(daily.iloc[0]["Tact"]), 0.9, places=12)
        self.assertAlmostEqual(float(daily.iloc[1]["Rain"]), 0.4, places=12)


class SwapConfigTests(unittest.TestCase):
    def test_patch_nprintday_changes_only_the_requested_value(self) -> None:
        original = "  PERIOD = 1\n  NPrintDay = 1     ! Number of output times during a day\n"

        patched = patch_nprintday_text(original, 24)

        self.assertIn("PERIOD = 1", patched)
        self.assertIn("NPrintDay = 24", patched)
        self.assertEqual(patched.count("NPrintDay"), 1)

    def test_patch_nprintday_rejects_out_of_range_values(self) -> None:
        with self.assertRaises(ValueError):
            patch_nprintday_text("NPrintDay = 1\n", 0)


class P1EndToEndDiagnosticTests(unittest.TestCase):
    def test_fixed_root_uses_native_sign_and_trapezoid_integration(self) -> None:
        case = (
            ROOT
            / "site_general_surrogate_eval"
            / "three_output_balance_audit_p1_20240716_server_v1"
            / "P1_20240716_ir0"
        )

        result = analyze_case_outputs(
            pre_crop_path=case / "result_forec.crp",
            pre_profile_path=case / "result_forec.vap",
            restart_crop_path=case / "result_restart.crp",
            restart_profile_path=case / "result_restart.vap",
            restart_increment_path=case / "result_restart.inc",
            decision_date="2024-07-16",
            nprintday=1,
            horizon_days=7,
        )

        self.assertAlmostEqual(
            result.summary["root_boundary_signed_integral_mm"],
            0.0716495,
            places=7,
        )
        self.assertAlmostEqual(
            result.summary["root_boundary_outflow_7d_mm"],
            -0.0716495,
            places=7,
        )
        self.assertAlmostEqual(
            result.summary["moving_root_boundary_term_7d_mm"],
            0.0,
            places=12,
        )
        self.assertAlmostEqual(
            result.summary["water_balance_residual_corrected_7d_mm"],
            0.1171495,
            places=7,
        )
        self.assertEqual(len(result.samples), 8)


class ValidationRunnerTests(unittest.TestCase):
    def test_plan_contains_exactly_nine_unique_cases(self) -> None:
        cases = validation_cases()
        names = [case_directory_name(case) for case in cases]

        self.assertEqual(len(cases), 9)
        self.assertEqual(len(set(names)), 9)
        self.assertEqual(
            {(case.site, case.irrigation_mm) for case in cases},
            {("code_C2", 30.0), ("code_C2", 60.0), ("code_N2", 30.0)},
        )
        self.assertEqual({case.nprintday for case in cases}, {1, 4, 24})

    def test_workspace_patch_updates_all_swap_config_files(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp)
            for name in ["SwapOriginal.swp", "Swap1.swp", "swap.swp"]:
                (workspace / name).write_text(
                    "PERIOD = 1\nNPrintDay = 1 ! output times\n",
                    encoding="utf-8",
                )

            patch_workspace_nprintday(workspace, 24)

            for name in ["SwapOriginal.swp", "Swap1.swp", "swap.swp"]:
                text = (workspace / name).read_text(encoding="utf-8")
                self.assertIn("NPrintDay = 24", text)


if __name__ == "__main__":
    unittest.main()
