from __future__ import annotations

import copy
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.training.run_gefs_robust_envelope_gam_development_v1 import (
    _fit_fold,
    build_jackknife_training_sets,
    evaluate_development_gate,
    read_frozen_penalties,
    true_fixed_grid_peak,
    validate_protocol,
)
from scripts.diagnostics.audit_gefs_gam_state_interaction_features_v1 import STATE_COLUMNS
from s2s_rtist.models.gefs_hierarchical_gam_bspline_v1 import (
    AET_TARGET,
    INITIAL_STORAGE,
    NET_GAIN_TARGET,
    RAIN_COLUMNS,
    VWC_TARGETS,
)


class RobustEnvelopeProtocolTests(unittest.TestCase):
    def setUp(self) -> None:
        path = Path(__file__).resolve().parents[1] / (
            "docs/superpowers/specs/"
            "2026-08-06-teacher-guided-robust-envelope-gam-development-v1.json"
        )
        self.protocol = json.loads(path.read_text(encoding="utf-8"))

    def test_frozen_protocol_validates(self) -> None:
        validate_protocol(self.protocol)
        self.assertEqual(self.protocol["validation"]["strict_outer_fold_count"], 15)
        self.assertEqual(self.protocol["validation"]["validation_cycle_count"], 196)
        self.assertEqual(self.protocol["ensemble"]["jackknife_model_count_per_fold"], 4)

    def test_protocol_rejects_2019_training(self) -> None:
        changed = copy.deepcopy(self.protocol)
        changed["forbidden"]["2019_rows_read_for_training_or_selection"] = False
        with self.assertRaisesRegex(ValueError, "2019"):
            validate_protocol(changed)

    def test_protocol_rejects_new_penalty_search(self) -> None:
        changed = copy.deepcopy(self.protocol)
        changed["penalties"]["new_search_performed"] = True
        with self.assertRaisesRegex(ValueError, "penalty"):
            validate_protocol(changed)


def synthetic_outer_train() -> pd.DataFrame:
    rows = []
    for site_id in ("P2", "P3", "P4", "P15"):
        for year in (2015, 2016):
            for date_index in range(2):
                for irrigation in (0.0, 10.0, 15.0, 20.0, 25.0, 30.0, 40.0, 60.0):
                    rows.append(
                        {
                            "target_year": year,
                            "site_id": site_id,
                            "decision_date": f"{year}-05-{date_index + 1:02d}",
                            "irrigation_mm": irrigation,
                        }
                    )
    return pd.DataFrame(rows)


class RobustEnvelopeIsolationTests(unittest.TestCase):
    def test_jackknife_training_sets_exclude_target_and_deleted_source(self) -> None:
        splits = build_jackknife_training_sets(
            synthetic_outer_train(),
            target_site="P1",
            source_sites=("P2", "P3", "P4", "P15"),
        )
        self.assertEqual(len(splits), 4)
        for split in splits:
            self.assertNotIn("P1", set(split.frame["site_id"]))
            self.assertNotIn(split.deleted_source_site, set(split.frame["site_id"]))
            self.assertEqual(split.frame["site_id"].nunique(), 3)
            self.assertEqual(split.target_site_training_rows, 0)
            self.assertEqual(split.deleted_source_training_rows, 0)

    def test_jackknife_training_sets_reject_unknown_source_site(self) -> None:
        with self.assertRaisesRegex(ValueError, "source"):
            build_jackknife_training_sets(
                synthetic_outer_train(),
                target_site="P1",
                source_sites=("P2", "P3", "P4", "PX"),
            )

    def test_frozen_penalties_require_one_finite_gam_vc_row(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "summary.csv"
            pd.DataFrame(
                [
                    {
                        "fold_id": "holdout_P1_rolling_to_2016",
                        "variant": "GAM-VC",
                        "lambda_main": 0.1,
                        "lambda_site": 1.0,
                        "lambda_interaction": 10.0,
                    }
                ]
            ).to_csv(path, index=False)
            self.assertEqual(
                read_frozen_penalties(path, expected_fold_id="holdout_P1_rolling_to_2016"),
                {"lambda_main": 0.1, "lambda_site": 1.0, "lambda_interaction": 10.0},
            )

    def test_frozen_penalties_reject_duplicate_rows(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "summary.csv"
            row = {
                "fold_id": "holdout_P1_rolling_to_2016",
                "variant": "GAM-VC",
                "lambda_main": 0.1,
                "lambda_site": 1.0,
                "lambda_interaction": 10.0,
            }
            pd.DataFrame([row, row]).to_csv(path, index=False)
            with self.assertRaisesRegex(ValueError, "exactly one"):
                read_frozen_penalties(path, expected_fold_id="holdout_P1_rolling_to_2016")


def synthetic_gam_frame(sites: tuple[str, ...], year: int) -> pd.DataFrame:
    rows = []
    irrigation_grid = (0.0, 10.0, 15.0, 20.0, 25.0, 30.0, 40.0, 60.0)
    for site_index, site_id in enumerate(sites):
        for date_index in range(2):
            state = 0.15 + 0.03 * site_index + 0.02 * date_index
            for irrigation in irrigation_grid:
                row = {
                    "target_year": year,
                    "site_id": site_id,
                    "decision_date": f"{year}-05-{date_index + 1:02d}",
                    "irrigation_mm": irrigation,
                    NET_GAIN_TARGET: irrigation * (1.0 + state) - 0.03 * irrigation**2,
                    AET_TARGET: 20.0 + 0.08 * irrigation,
                    INITIAL_STORAGE: 200.0,
                }
                for state_number, column in enumerate(STATE_COLUMNS):
                    row[column] = state + 0.01 * state_number
                for day, column in enumerate(VWC_TARGETS, start=1):
                    row[column] = 0.2 + 0.0004 * irrigation + 0.0005 * day
                for column in RAIN_COLUMNS:
                    row[column] = 1.0
                rows.append(row)
    return pd.DataFrame(rows)


class RobustEnvelopeFoldIntegrationTests(unittest.TestCase):
    def test_fit_fold_trains_five_models_and_returns_finite_outputs(self) -> None:
        root = Path(__file__).resolve().parents[1]
        source_protocol = json.loads(
            (
                root
                / "docs/superpowers/specs/"
                "2026-08-05-teacher-guided-hierarchical-gam-nested-shared-selection-development-v2.json"
            ).read_text(encoding="utf-8")
        )
        outer_train = synthetic_gam_frame(("P2", "P3", "P4", "P15"), 2017)
        validation = synthetic_gam_frame(("P1",), 2018).iloc[:8].copy()
        with tempfile.TemporaryDirectory() as temporary:
            cycles, folds, audit, predictions = _fit_fold(
                fold_id="holdout_P1_rolling_to_2018",
                target_site="P1",
                validation_year=2018,
                outer_train=outer_train,
                validation=validation,
                parameters={"lambda_main": 0.1, "lambda_site": 1.0, "lambda_interaction": 1.0},
                source_protocol=source_protocol,
                fold_dir=Path(temporary),
                optimization={
                    "deployment_resolution_mm": 1.0e-6,
                    "dense_diagnostic_step_mm": 0.25,
                    "maximum_dense_gain_gap": 0.05,
                },
            )
        self.assertEqual(len(cycles), 1)
        self.assertEqual(len(folds), 1)
        self.assertEqual(len(audit), 5)
        self.assertEqual(audit["model_role"].eq("jackknife_submodel").sum(), 4)
        self.assertEqual(len(predictions), 2)
        self.assertTrue(np.isfinite(cycles.select_dtypes(include=[np.number])).all().all())


class RobustEnvelopeCycleMetricTests(unittest.TestCase):
    def test_true_fixed_grid_peak_breaks_target_tie_with_smaller_irrigation(self) -> None:
        frame = pd.DataFrame(
            {
                "irrigation_mm": [0.0, 10.0, 15.0, 20.0, 25.0, 30.0, 40.0, 60.0],
                "target_net_gain_7d": [0.0, 2.0, 5.0, 8.0, 8.0, 7.0, 4.0, -2.0],
            }
        )
        self.assertEqual(true_fixed_grid_peak(frame), (20.0, 8.0))


def passing_gate_frames() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    fold_ids = [f"fold_{index:02d}" for index in range(15)]
    cycle_rows = []
    for index in range(196):
        fold_id = fold_ids[index % 15]
        positive = index % 3 != 0
        cycle_rows.append(
            {
                "fold_id": fold_id,
                "site_id": ("P1", "P2", "P3", "P4", "P15")[index % 5],
                "true_oracle_is_positive": int(positive),
                "true_oracle_is_zero": int(not positive),
                "baseline_peak_absolute_error_mm": 10.0,
                "robust_peak_absolute_error_mm": 9.0,
                "baseline_zero_oracle_false_positive": int(not positive),
                "robust_zero_oracle_false_positive": 0,
            }
        )
    fold_rows = [
        {
            "fold_id": fold_id,
            "baseline_peak_mae_mm": 10.0,
            "robust_peak_mae_mm": 9.0,
            "cycle_count": 13 if index < 14 else 14,
        }
        for index, fold_id in enumerate(fold_ids)
    ]
    prediction_rows = []
    for fold_id in fold_ids:
        prediction_rows.extend(
            [
                {"fold_id": fold_id, "model_role": "four_source_baseline", "macro_composite": 0.6},
                {"fold_id": fold_id, "model_role": "jackknife_ensemble_mean", "macro_composite": 0.55},
            ]
        )
    return pd.DataFrame(cycle_rows), pd.DataFrame(fold_rows), pd.DataFrame(prediction_rows)


class RobustEnvelopeGateTests(unittest.TestCase):
    def test_all_seven_conditions_pass_together(self) -> None:
        cycles, folds, predictions = passing_gate_frames()
        gate = evaluate_development_gate(
            cycles,
            folds,
            predictions,
            execution_audit_passed=True,
        )
        self.assertTrue(gate["passed"])
        self.assertEqual(len(gate["conditions"]), 7)

    def test_gate_fails_when_one_condition_fails(self) -> None:
        cycles, folds, predictions = passing_gate_frames()
        predictions.loc[
            predictions["model_role"].eq("jackknife_ensemble_mean"), "macro_composite"
        ] = 0.7
        gate = evaluate_development_gate(
            cycles,
            folds,
            predictions,
            execution_audit_passed=True,
        )
        self.assertFalse(gate["passed"])
        self.assertFalse(gate["conditions"]["ensemble_mean_macro_composite_nonworse"])

    def test_gate_requires_exact_development_counts(self) -> None:
        cycles, folds, predictions = passing_gate_frames()
        with self.assertRaisesRegex(ValueError, "196"):
            evaluate_development_gate(
                cycles.iloc[:-1],
                folds,
                predictions,
                execution_audit_passed=True,
            )


if __name__ == "__main__":
    unittest.main()
