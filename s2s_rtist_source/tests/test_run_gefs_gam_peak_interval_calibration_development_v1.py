from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

from scripts.training.run_gefs_gam_peak_interval_calibration_development_v1 import (
    FEATURE_COLUMNS,
    _completed_fold_outputs,
    _fit_calibration_fold,
    _write_fold_outputs,
    build_inner_holdout_units,
    evaluate_development_gate,
    validate_protocol,
)
from scripts.diagnostics.audit_gefs_gam_state_interaction_features_v1 import STATE_COLUMNS
from s2s_rtist.models.gefs_hierarchical_gam_bspline_v1 import (
    AET_TARGET,
    INITIAL_STORAGE,
    RAIN_COLUMNS,
    VWC_TARGETS,
)


def passing_frames() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    fold_ids = [f"fold_{index:02d}" for index in range(15)]
    rows = []
    for index in range(196):
        fold_id = fold_ids[index % 15]
        true_peak = 0.0 if index % 4 == 0 else 20.0
        rows.append(
            {
                "fold_id": fold_id,
                "site_id": ("P1", "P2", "P3", "P4", "P15")[index % 5],
                "decision_date": f"2018-05-{index + 1:03d}",
                "true_fixed_list_irrigation_mm": true_peak,
                "baseline_continuous_irrigation_mm": 10.0 if true_peak == 0.0 else 30.0,
                "calibrated_peak_mm": 0.0 if true_peak == 0.0 else 25.0,
                "baseline_peak_absolute_error_mm": 10.0,
                "calibrated_peak_absolute_error_mm": 0.0 if true_peak == 0.0 else 5.0,
                "true_oracle_is_zero": int(true_peak == 0.0),
                "true_oracle_is_positive": int(true_peak > 0.0),
                "baseline_zero_oracle_false_positive": int(true_peak == 0.0),
                "calibrated_zero_oracle_false_positive": 0,
                "interval_contains_true_peak": 1,
            }
        )
    cycles = pd.DataFrame(rows)
    folds = pd.DataFrame(
        {
            "fold_id": fold_ids,
            "baseline_peak_mae_mm": np.full(15, 10.0),
            "calibrated_peak_mae_mm": np.full(15, 5.0),
        }
    )
    intervals = cycles.loc[
        :,
        ["fold_id", "site_id", "decision_date", "interval_contains_true_peak"],
    ].copy()
    return cycles, folds, intervals


class ProtocolTests(unittest.TestCase):
    def test_frozen_protocol_validates(self) -> None:
        root = Path(__file__).resolve().parents[1]
        protocol = json.loads(
            (
                root
                / "docs/superpowers/specs/"
                "2026-08-06-teacher-guided-gam-peak-interval-calibration-development-v1.json"
            ).read_text(encoding="utf-8")
        )
        validate_protocol(protocol)

    def test_protocol_rejects_2019_access(self) -> None:
        root = Path(__file__).resolve().parents[1]
        protocol = json.loads(
            (
                root
                / "docs/superpowers/specs/"
                "2026-08-06-teacher-guided-gam-peak-interval-calibration-development-v1.json"
            ).read_text(encoding="utf-8")
        )
        protocol["forbidden"]["2019_rows_read"] = False
        with self.assertRaisesRegex(ValueError, "2019"):
            validate_protocol(protocol)


class InnerIsolationTests(unittest.TestCase):
    def test_inner_units_exclude_outer_target_and_future_years(self) -> None:
        rows = []
        for year in (2015, 2016, 2017):
            for site_id in ("P2", "P3", "P4", "P15"):
                rows.append({"target_year": year, "site_id": site_id, "decision_date": f"{year}-05-01"})
        outer_train = pd.DataFrame(rows)
        units = build_inner_holdout_units(outer_train, outer_target_site="P1")
        self.assertEqual(len(units), 12)
        for unit in units:
            self.assertNotIn("P1", set(unit.training["site_id"]))
            self.assertNotIn(unit.held_out_site, set(unit.training["site_id"]))
            self.assertTrue(unit.training["target_year"].le(unit.held_out_year).all())
            self.assertEqual(set(unit.validation["site_id"]), {unit.held_out_site})
            self.assertEqual(set(unit.validation["target_year"]), {unit.held_out_year})
        p2_2016 = next(
            unit for unit in units
            if unit.held_out_site == "P2" and unit.held_out_year == 2016
        )
        self.assertIn(2016, set(p2_2016.training["target_year"]))
        self.assertEqual(set(p2_2016.training.loc[p2_2016.training["target_year"].eq(2016), "site_id"]), {"P3", "P4", "P15"})

    def test_inner_units_reject_outer_target_leakage(self) -> None:
        frame = pd.DataFrame(
            [
                {"target_year": 2015, "site_id": "P1", "decision_date": "2015-05-01"},
                {"target_year": 2015, "site_id": "P2", "decision_date": "2015-05-01"},
            ]
        )
        with self.assertRaisesRegex(ValueError, "outer target"):
            build_inner_holdout_units(frame, outer_target_site="P1")


class GateTests(unittest.TestCase):
    def test_all_eight_conditions_pass_together(self) -> None:
        cycles, folds, intervals = passing_frames()
        gate = evaluate_development_gate(
            cycles,
            folds,
            intervals,
            execution_audit_passed=True,
            multi_output_hashes_unchanged=True,
        )
        self.assertTrue(gate["passed"])
        self.assertEqual(len(gate["conditions"]), 8)

    def test_each_condition_can_fail_the_total_gate(self) -> None:
        for condition in (
            "global_maximum_peak_distance_nonworse",
            "p95_peak_distance_nonworse",
            "positive_oracle_mean_peak_distance_nonworse",
            "overall_mean_peak_distance_nonworse",
            "zero_oracle_false_positive_count_nonworse",
            "at_least_10_of_15_fold_peak_distance_nonworse",
            "interval_coverage_at_least_0_90",
            "execution_isolation_finite_and_multi_output_hash_audits_passed",
        ):
            cycles, folds, intervals = passing_frames()
            execution = True
            hashes = True
            if condition == "global_maximum_peak_distance_nonworse":
                cycles.loc[0, "calibrated_peak_absolute_error_mm"] = 11.0
            elif condition == "p95_peak_distance_nonworse":
                cycles.loc[:19, "calibrated_peak_absolute_error_mm"] = 11.0
            elif condition == "positive_oracle_mean_peak_distance_nonworse":
                mask = cycles["true_oracle_is_positive"].eq(1)
                cycles.loc[mask, "calibrated_peak_absolute_error_mm"] = 11.0
            elif condition == "overall_mean_peak_distance_nonworse":
                cycles["calibrated_peak_absolute_error_mm"] = 11.0
            elif condition == "zero_oracle_false_positive_count_nonworse":
                zero = cycles.index[cycles["true_oracle_is_zero"].eq(1)]
                cycles.loc[zero, "calibrated_zero_oracle_false_positive"] = 1
                cycles.loc[zero, "baseline_zero_oracle_false_positive"] = 0
            elif condition == "at_least_10_of_15_fold_peak_distance_nonworse":
                folds.loc[:5, "calibrated_peak_mae_mm"] = 11.0
            elif condition == "interval_coverage_at_least_0_90":
                intervals.loc[:19, "interval_contains_true_peak"] = 0
            else:
                hashes = False
            gate = evaluate_development_gate(
                cycles,
                folds,
                intervals,
                execution_audit_passed=execution,
                multi_output_hashes_unchanged=hashes,
            )
            self.assertFalse(gate["passed"], condition)
            self.assertFalse(gate["conditions"][condition], condition)


class ResumeTests(unittest.TestCase):
    def test_completed_fold_rejects_file_hash_change(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            fold_dir = Path(temporary)
            frames = {
                "inner_samples": pd.DataFrame([{**{name: 0.0 for name in FEATURE_COLUMNS}, "group_id": "P2_2015"}]),
                "cycle_metrics": pd.DataFrame([{"fold_id": "f", "site_id": "P1", "decision_date": "2018-05-01"}]),
                "model_audit": pd.DataFrame([{"fold_id": "f", "selected_gamma": 0.0}]),
                "interval_audit": pd.DataFrame([{"fold_id": "f", "interval_contains_true_peak": 1}]),
            }
            source_paths = []
            source_hashes = []
            for index in range(5):
                source_path = fold_dir / f"source_{index}.npz"
                source_path.write_bytes(f"source-{index}".encode("ascii"))
                source_paths.append(str(source_path))
                import hashlib
                source_hashes.append(hashlib.sha256(source_path.read_bytes()).hexdigest())
            frames["model_audit"]["source_model_paths"] = "|".join(source_paths)
            frames["model_audit"]["source_model_sha256"] = "|".join(source_hashes)
            _write_fold_outputs(
                fold_dir,
                protocol_sha256="a" * 64,
                calibrator_arrays={"coefficients": np.asarray([1.0])},
                selection={"gamma": 0.0, "ridge_lambda": 1.0},
                **frames,
            )
            self.assertIsNotNone(
                _completed_fold_outputs(fold_dir, protocol_sha256="a" * 64)
            )
            with (fold_dir / "cycle_metrics.csv").open("a", encoding="utf-8") as handle:
                handle.write("changed\n")
            with self.assertRaisesRegex(ValueError, "hash"):
                _completed_fold_outputs(fold_dir, protocol_sha256="a" * 64)

    def test_completed_fold_rejects_source_model_hash_change(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            fold_dir = Path(temporary)
            source_paths = []
            source_hashes = []
            import hashlib
            for index in range(5):
                source_path = fold_dir / f"source_{index}.npz"
                source_path.write_bytes(f"source-{index}".encode("ascii"))
                source_paths.append(str(source_path))
                source_hashes.append(hashlib.sha256(source_path.read_bytes()).hexdigest())
            _write_fold_outputs(
                fold_dir,
                protocol_sha256="b" * 64,
                inner_samples=pd.DataFrame([{**{name: 0.0 for name in FEATURE_COLUMNS}, "group_id": "P2_2015"}]),
                cycle_metrics=pd.DataFrame([{"fold_id": "f", "site_id": "P1", "decision_date": "2018-05-01"}]),
                model_audit=pd.DataFrame([{
                    "fold_id": "f",
                    "selected_gamma": 0.0,
                    "source_model_paths": "|".join(source_paths),
                    "source_model_sha256": "|".join(source_hashes),
                }]),
                interval_audit=pd.DataFrame([{"fold_id": "f", "interval_contains_true_peak": 1}]),
                calibrator_arrays={"coefficients": np.asarray([1.0])},
                selection={"gamma": 0.0, "ridge_lambda": 1.0},
            )
            Path(source_paths[0]).write_bytes(b"changed-source")
            with self.assertRaisesRegex(ValueError, "source model hash"):
                _completed_fold_outputs(fold_dir, protocol_sha256="b" * 64)


class FakeBasis:
    knots = np.asarray([0.0, 0.0, 0.0, 0.0, 60.0, 60.0, 60.0, 60.0])


class FakeGam:
    irrigation_basis_ = FakeBasis()

    def predict_shared(self, frame: pd.DataFrame) -> pd.DataFrame:
        irrigation = frame["irrigation_mm"].to_numpy(dtype=np.float64)
        return pd.DataFrame(
            {"pred_target_net_gain_7d": 400.0 - (irrigation - 20.0) ** 2},
            index=frame.index,
        )

    def predict_net_gain_derivative(
        self,
        frame: pd.DataFrame,
        *,
        use_site_deviation: bool,
    ) -> pd.Series:
        del use_site_deviation
        irrigation = frame["irrigation_mm"].to_numpy(dtype=np.float64)
        return pd.Series(-2.0 * (irrigation - 20.0), index=frame.index)


def synthetic_cycle_frame(sites: tuple[str, ...], years: tuple[int, ...]) -> pd.DataFrame:
    rows = []
    irrigation_grid = (0.0, 10.0, 15.0, 20.0, 25.0, 30.0, 40.0, 60.0)
    for year in years:
        for site_index, site_id in enumerate(sites):
            for irrigation in irrigation_grid:
                row = {
                    "target_year": year,
                    "site_id": site_id,
                    "decision_date": f"{year}-05-01",
                    "irrigation_mm": irrigation,
                    "target_net_gain_7d": 400.0 - (irrigation - 20.0) ** 2,
                    AET_TARGET: 20.0 + 0.1 * irrigation,
                    INITIAL_STORAGE: 200.0,
                }
                for index, column in enumerate(STATE_COLUMNS):
                    row[column] = 0.1 + 0.01 * site_index + 0.001 * index
                for day, column in enumerate(VWC_TARGETS, start=1):
                    row[column] = 0.2 + 0.0001 * irrigation + 0.0001 * day
                for column in RAIN_COLUMNS:
                    row[column] = 1.0
                rows.append(row)
    return pd.DataFrame(rows)


class FoldIntegrationTests(unittest.TestCase):
    def test_single_fold_runs_nested_calibration_without_target_feature_access(self) -> None:
        root = Path(__file__).resolve().parents[1]
        protocol = json.loads(
            (
                root
                / "docs/superpowers/specs/"
                "2026-08-06-teacher-guided-gam-peak-interval-calibration-development-v1.json"
            ).read_text(encoding="utf-8")
        )
        source_protocol = json.loads(
            (
                root
                / "docs/superpowers/specs/"
                "2026-08-05-teacher-guided-hierarchical-gam-nested-shared-selection-development-v2.json"
            ).read_text(encoding="utf-8")
        )
        outer_train = synthetic_cycle_frame(("P2", "P3", "P4", "P15"), (2015, 2016))
        validation = synthetic_cycle_frame(("P1",), (2017,))
        fake_models = (FakeGam(), [FakeGam(), FakeGam(), FakeGam(), FakeGam()], {
            "source_model_mode": "synthetic",
            "source_model_reason": "unit_test",
            "source_model_paths": [str(root / f"fake_{index}.npz") for index in range(5)],
            "source_model_sha256": [str(index) * 64 for index in range(5)],
        })
        with tempfile.TemporaryDirectory() as temporary:
            with patch(
                "scripts.training.run_gefs_gam_peak_interval_calibration_development_v1._load_or_refit_outer_models",
                return_value=fake_models,
            ), patch(
                "scripts.training.run_gefs_gam_peak_interval_calibration_development_v1._fit_model",
                side_effect=lambda *args, **kwargs: FakeGam(),
            ):
                result = _fit_calibration_fold(
                    fold_id="holdout_P1_rolling_to_2017",
                    target_site="P1",
                    validation_year=2017,
                    outer_train=outer_train,
                    validation=validation,
                    parameters={"lambda_main": 0.1, "lambda_site": 1.0, "lambda_interaction": 1.0},
                    source_protocol=source_protocol,
                    robust_fold_dir=Path(temporary) / "robust",
                    calibration_fold_dir=Path(temporary) / "calibration",
                    protocol=protocol,
                )
        self.assertEqual(len(result["cycle_metrics"]), 1)
        self.assertEqual(result["inner_samples"]["group_id"].nunique(), 8)
        self.assertEqual(len(result["selection"]["search_rows"]), 9)
        self.assertEqual(int(result["model_audit"].iloc[0]["target_site_training_rows"]), 0)
        self.assertEqual(int(result["cycle_metrics"].iloc[0]["interval_contains_true_peak"]), 1)
        self.assertTrue(set(FEATURE_COLUMNS).issubset(result["cycle_metrics"].columns))
        self.assertTrue(
            np.isfinite(
                result["cycle_metrics"].select_dtypes(include=[np.number]).to_numpy(dtype=np.float64)
            ).all()
        )


if __name__ == "__main__":
    unittest.main()
