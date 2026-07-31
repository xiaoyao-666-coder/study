from __future__ import annotations

import unittest
import argparse
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import numpy as np
import pandas as pd

from scripts.training.run_gefs_p3_b2_regret_sensitive_loso_router_screen_v1 import (
    B2_POLICY_NAME,
    attach_regret_weights,
    b2_development_gate,
    development_folds,
    fit_regret_weighted_logistic_gate,
    load_b2_router_rows,
    model_sha256,
    run as run_b2,
)
from scripts.evaluation.freeze_gefs_p3_strict_loso_cross_year_gate_protocol_v1 import (
    DEVELOPMENT_FOLDS,
    gate_feature_contract,
)
from scripts.training.run_gefs_p3_strict_loso_cross_year_gate_screen_v1 import (
    expert_cycle_decisions,
)
from scripts.training.train_gefs_exact_schedule_no_tta_baseline_v1 import (
    CONTRACT_NAME,
    DATASET_NAME,
    EXPECTED_IRRIGATION_GRID,
    sha256_file,
)
from scripts.training.train_gefs_teacher_aligned_three_output_pretraining_v2 import (
    INITIAL_STORAGE,
)


FEATURES = ["feature_a", "feature_b"]
CONTINUOUS = [
    "irrigation_mm",
    "predecision_dvs",
    "predecision_crop_root_depth_cm",
    "predecision_soil_vwc_0_100cm",
    *[
        f"weather_{variable}_day{day:02d}"
        for variable in (
            "precipitation_mm",
            "temperature_min_c",
            "temperature_max_c",
            "actual_vapor_pressure_kpa",
            "wind_speed_m_s",
            "solar_kj_m2_day",
        )
        for day in range(1, 8)
    ],
]


def training_rows(count: int = 12) -> pd.DataFrame:
    rows = []
    for index in range(count):
        delta = float(index + 1) * (1.0 if index % 2 else -1.0)
        rows.append(
            {
                "target_year": 2015,
                "site_id": "P4",
                "decision_date": f"2015-07-{index + 1:02d}",
                "feature_a": float(index),
                "feature_b": float((index % 3) - 1),
                "selected_true_net_gain_7d_p1": delta,
                "selected_true_net_gain_7d_p15": 0.0,
                "recommendations_differ": True,
                "realized_gain_tie": False,
                "usable_for_gate": True,
                "gate_label_p1_wins": float(delta > 0.0),
            }
        )
    return pd.DataFrame(rows)


def gate_fixture() -> tuple[pd.DataFrame, pd.DataFrame]:
    metrics = []
    decisions = []
    for definition in development_folds():
        fold = definition["protocol_fold"]
        site = definition["held_out_site"]
        metrics.extend(
            [
                {
                    "protocol_fold": fold,
                    "held_out_site": site,
                    "policy": "always_P15",
                    "mean_regret_7d": 2.0,
                    "maximum_regret_7d": 5.0,
                    "false_positive_count": 1,
                    "predicted_60mm_count": 1,
                },
                {
                    "protocol_fold": fold,
                    "held_out_site": site,
                    "policy": B2_POLICY_NAME,
                    "mean_regret_7d": 1.0,
                    "maximum_regret_7d": 4.0,
                    "false_positive_count": 0,
                    "predicted_60mm_count": 0,
                },
            ]
        )
        decisions.append(
            {
                "protocol_fold": fold,
                "held_out_site": site,
                "policy": B2_POLICY_NAME,
                "selected_source_site_id": "P1",
                "recommendations_differ": True,
                "gate_fallback": False,
            }
        )
    return pd.DataFrame(metrics), pd.DataFrame(decisions)


def historical_cycles(site: str, year: int, count: int = 3) -> pd.DataFrame:
    rows = []
    optimum = 10.0 if site == "P2" else 25.0
    for cycle in range(count):
        for irrigation in EXPECTED_IRRIGATION_GRID:
            row = {
                "sample_id": f"{site}_{year}_{cycle}_{irrigation:g}",
                "target_year": year,
                "split": "train" if year < 2019 else "validation",
                "site_id": site,
                "decision_date": f"{year}-07-{cycle + 1:02d}",
                "irrigation_mm": irrigation,
                "target_net_gain_7d": -abs(float(irrigation) - optimum),
                "target_aet_7d_mm": 20.0 + float(irrigation) * 0.1,
                "predecision_dvs": 0.3 + cycle * 0.01,
                "predecision_crop_root_depth_cm": 50.0 + cycle,
                "predecision_soil_vwc_0_100cm": 0.2 + cycle * 0.001,
                INITIAL_STORAGE: 100.0,
            }
            for day in range(1, 8):
                row[f"target_soil_vwc_0_100cm_day{day:02d}"] = 0.2 + day * 0.001
                for variable in (
                    "precipitation_mm",
                    "temperature_min_c",
                    "temperature_max_c",
                    "actual_vapor_pressure_kpa",
                    "wind_speed_m_s",
                    "solar_kj_m2_day",
                ):
                    row[f"weather_{variable}_day{day:02d}"] = float(day + cycle)
            rows.append(row)
    return pd.DataFrame(rows)


class P3B2RegretSensitiveLosoRouterTests(unittest.TestCase):
    def test_development_folds_are_exact_rolling_leave_site_matrix(self) -> None:
        folds = development_folds()
        self.assertEqual(len(folds), 8)
        self.assertEqual(len({item["protocol_fold"] for item in folds}), 8)
        self.assertEqual(
            {(item["held_out_site"], item["validation_year"]) for item in folds},
            {(site, year) for year in range(2016, 2020) for site in ("P2", "P4")},
        )
        for item in folds:
            self.assertNotEqual(item["training_site"], item["held_out_site"])
            self.assertEqual(
                set((item["training_site"], item["held_out_site"])), {"P2", "P4"}
            )
            self.assertTrue(
                all(year < item["validation_year"] for year in item["train_years"])
            )
            self.assertEqual(
                tuple(item["train_years"]), tuple(range(2015, item["validation_year"]))
            )

    def test_loader_physically_skips_disallowed_rows_before_target_parsing(self) -> None:
        columns = [
            "sample_id",
            "target_year",
            "split",
            "site_id",
            "decision_date",
            "irrigation_mm",
            "target_net_gain_7d",
        ]
        rows = []
        for site, year in (("P2", 2015), ("P4", 2019)):
            for irrigation in EXPECTED_IRRIGATION_GRID:
                rows.append(
                    {
                        "sample_id": f"{site}_{year}_{irrigation:g}",
                        "target_year": year,
                        "split": "train",
                        "site_id": site,
                        "decision_date": f"{year}-07-01",
                        "irrigation_mm": irrigation,
                        "target_net_gain_7d": float(irrigation),
                    }
                )
        for site, year in (("P3", 2015), ("P2", 2020), ("P4", 2021), ("P3", 2024)):
            rows.append(
                {
                    "sample_id": f"excluded_{site}_{year}",
                    "target_year": year,
                    "split": "test",
                    "site_id": site,
                    "decision_date": f"{year}-07-01",
                    "irrigation_mm": "not_numeric",
                    "target_net_gain_7d": "must_not_be_parsed",
                }
            )
        with TemporaryDirectory() as directory:
            path = Path(directory) / "dataset.csv"
            pd.DataFrame(rows, columns=columns).to_csv(path, index=False)
            frame, audit = load_b2_router_rows(path, columns)
        self.assertEqual(set(frame["site_id"]), {"P2", "P4"})
        self.assertEqual(set(frame["target_year"]), {2015, 2019})
        self.assertEqual(audit["P3_rows_skipped_before_materialization"], 2)
        self.assertEqual(audit["year_2020_rows_skipped_before_materialization"], 1)
        self.assertEqual(audit["year_2021_rows_skipped_before_materialization"], 1)
        self.assertEqual(audit["year_2024_rows_skipped_before_materialization"], 1)
        self.assertEqual(audit["disallowed_rows_retained"], 0)

    def test_regret_weights_are_exact_and_fold_local(self) -> None:
        rows = training_rows(4)
        weighted = attach_regret_weights(rows)
        expected_raw = np.array([1.0, 2.0, 3.0, 4.0])
        np.testing.assert_allclose(weighted["realized_gain_delta"], [-1, 2, -3, 4])
        np.testing.assert_allclose(weighted["raw_regret_weight"], expected_raw)
        np.testing.assert_allclose(
            weighted["normalized_regret_weight"], expected_raw / expected_raw.mean()
        )
        self.assertAlmostEqual(weighted["normalized_regret_weight"].mean(), 1.0)

    def test_weighted_irls_is_deterministic_and_validation_independent(self) -> None:
        weighted = attach_regret_weights(training_rows())
        first = fit_regret_weighted_logistic_gate(weighted, FEATURES)
        second = fit_regret_weighted_logistic_gate(weighted.copy(), FEATURES)
        self.assertFalse(first["fallback"])
        self.assertEqual(model_sha256(first), model_sha256(second))
        validation = training_rows().copy()
        validation["selected_true_net_gain_7d_p1"] *= -1000.0
        self.assertEqual(model_sha256(first), model_sha256(second))

    def test_weighted_irls_fallbacks_fail_closed(self) -> None:
        insufficient = fit_regret_weighted_logistic_gate(
            attach_regret_weights(training_rows(7)), FEATURES
        )
        self.assertEqual(insufficient["fallback_reason"], "insufficient_actionable_cycles")
        one_class_rows = training_rows()
        one_class_rows["selected_true_net_gain_7d_p1"] = np.arange(1.0, 13.0)
        one_class = fit_regret_weighted_logistic_gate(
            attach_regret_weights(one_class_rows), FEATURES
        )
        self.assertEqual(one_class["fallback_reason"], "single_winner_class")
        negligible_rows = training_rows()
        negligible_rows["selected_true_net_gain_7d_p1"] = 0.0
        negligible_rows["selected_true_net_gain_7d_p15"] = 0.0
        negligible = fit_regret_weighted_logistic_gate(
            attach_regret_weights(negligible_rows), FEATURES
        )
        self.assertEqual(
            negligible["fallback_reason"], "negligible_mean_absolute_gain_delta"
        )
        nonfinite_rows = attach_regret_weights(training_rows())
        nonfinite_rows.loc[0, "feature_a"] = np.nan
        nonfinite = fit_regret_weighted_logistic_gate(nonfinite_rows, FEATURES)
        self.assertEqual(nonfinite["fallback_reason"], "nonfinite_gate_training_data")
        nonconverged = fit_regret_weighted_logistic_gate(
            attach_regret_weights(training_rows()), FEATURES, max_iter=0
        )
        self.assertEqual(nonconverged["fallback_reason"], "logistic_nonconvergence")
        with patch("numpy.linalg.solve", side_effect=np.linalg.LinAlgError):
            singular = fit_regret_weighted_logistic_gate(
                attach_regret_weights(training_rows()), FEATURES
            )
        self.assertEqual(singular["fallback_reason"], "singular_logistic_hessian")

    def test_development_gate_requires_all_nine_frozen_conditions(self) -> None:
        metrics, decisions = gate_fixture()
        passed = b2_development_gate(metrics, decisions)
        self.assertTrue(bool(passed.loc[0, "development_gate_passed"]))
        gate_columns = [
            column
            for column in passed.columns
            if column.endswith("_gate_passed") and column != "development_gate_passed"
        ]
        self.assertEqual(len(gate_columns), 9)

        def fail_macro(metrics: pd.DataFrame, decisions: pd.DataFrame) -> None:
            del decisions
            metrics.loc[
                metrics["policy"].eq(B2_POLICY_NAME), "mean_regret_7d"
            ] = 2.0

        def fail_site(
            metrics: pd.DataFrame, decisions: pd.DataFrame, site: str
        ) -> None:
            del decisions
            metrics.loc[
                metrics["policy"].eq(B2_POLICY_NAME)
                & metrics["held_out_site"].eq(site),
                "mean_regret_7d",
            ] = 3.0

        mutations = {
            "macro_mean_regret_gate_passed": fail_macro,
            "P2_site_mean_regret_gate_passed": lambda m, d: fail_site(m, d, "P2"),
            "P4_site_mean_regret_gate_passed": lambda m, d: fail_site(m, d, "P4"),
        }

        for expected_failure, mutate in mutations.items():
            changed_metrics, changed_decisions = metrics.copy(), decisions.copy()
            mutate(changed_metrics, changed_decisions)
            result = b2_development_gate(changed_metrics, changed_decisions)
            self.assertFalse(bool(result.loc[0, expected_failure]), expected_failure)
            self.assertFalse(bool(result.loc[0, "development_gate_passed"]))

        changed = metrics.copy()
        candidate = changed["policy"].eq(B2_POLICY_NAME)
        changed.loc[candidate, "maximum_regret_7d"] = 6.0
        self.assertFalse(
            bool(b2_development_gate(changed, decisions).loc[0, "worst_maximum_regret_gate_passed"])
        )
        changed = metrics.copy()
        changed.loc[candidate, "false_positive_count"] = 2
        self.assertFalse(
            bool(b2_development_gate(changed, decisions).loc[0, "false_positive_gate_passed"])
        )
        changed = metrics.copy()
        changed.loc[candidate, "predicted_60mm_count"] = 2
        self.assertFalse(
            bool(b2_development_gate(changed, decisions).loc[0, "predicted_60mm_gate_passed"])
        )
        changed_decisions = decisions.copy()
        changed_decisions["recommendations_differ"] = False
        self.assertFalse(
            bool(b2_development_gate(metrics, changed_decisions).loc[0, "P1_route_gate_passed"])
        )
        changed_decisions = decisions.copy()
        changed_decisions.loc[changed_decisions["held_out_site"].eq("P2"), "gate_fallback"] = True
        self.assertFalse(
            bool(
                b2_development_gate(metrics, changed_decisions).loc[
                    0, "each_site_nonfallback_gate_passed"
                ]
            )
        )
        changed = metrics.copy()
        candidate_indices = changed.index[changed["policy"].eq(B2_POLICY_NAME)][:3]
        changed.loc[candidate_indices, "mean_regret_7d"] = 3.0
        result = b2_development_gate(changed, decisions)
        self.assertFalse(bool(result.loc[0, "six_of_eight_nonworse_gate_passed"]))

    def test_negative_end_to_end_writes_audited_outputs_without_final_gate(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            dataset_dir = root / "dataset"
            screen_dir = root / "screen"
            final_dir = root / "final"
            output_dir = root / "output"
            dataset_dir.mkdir()
            screen_dir.mkdir()
            final_dir.mkdir()
            data = pd.concat(
                [
                    historical_cycles(site, year)
                    for year in range(2015, 2020)
                    for site in ("P2", "P4")
                ],
                ignore_index=True,
            )
            data.to_csv(dataset_dir / DATASET_NAME, index=False)
            (dataset_dir / CONTRACT_NAME).write_text(
                json.dumps({"continuous_feature_columns": CONTINUOUS}),
                encoding="utf-8",
            )
            protocol_manifest = root / "protocol_manifest.csv"
            protocol_manifest.write_text("role,path\n", encoding="utf-8")
            frozen = {
                "protocol": {
                    "B_source_only_router": {
                        "gate_feature_contract": gate_feature_contract(CONTINUOUS)
                    }
                },
                "paths": {"manifest": protocol_manifest},
            }
            checkpoints = {
                str(item["protocol_fold"]): {
                    "P1": {"source_site_id": "P1"},
                    "P15": {"source_site_id": "P15"},
                }
                for item in DEVELOPMENT_FOLDS
            }
            summary_path = (
                screen_dir / "gefs_p3_strict_loso_source_checkpoint_summary_v1.csv"
            )
            summary = pd.DataFrame(
                columns=["protocol_fold", "source_site_id", "checkpoint_path"]
            )
            summary.to_csv(summary_path, index=False)

            def fake_expert_decisions(checkpoint_set, frame, device):
                del device
                decisions = {}
                predictions = []
                for source, optimum in (("P1", 10.0), ("P15", 25.0)):
                    predicted = -(
                        frame["irrigation_mm"].to_numpy(dtype=float) - optimum
                    ) ** 2
                    decisions[source] = expert_cycle_decisions(
                        frame, predicted, source
                    )
                return decisions, pd.DataFrame(predictions)

            with patch(
                "scripts.training.run_gefs_p3_b2_regret_sensitive_loso_router_screen_v1.validate_dual_protocol",
                return_value=frozen,
            ), patch(
                "scripts.training.run_gefs_p3_b2_regret_sensitive_loso_router_screen_v1.load_development_checkpoints",
                return_value=(checkpoints, summary),
            ), patch(
                "scripts.training.run_gefs_p3_b2_regret_sensitive_loso_router_screen_v1.expert_decisions",
                side_effect=fake_expert_decisions,
            ), patch(
                "scripts.training.run_gefs_p3_b2_regret_sensitive_loso_router_screen_v1.load_final_checkpoints"
            ) as final_loader:
                outputs = run_b2(
                    argparse.Namespace(
                        dataset_dir=dataset_dir,
                        protocol_dir=root / "protocol",
                        development_screen_dir=screen_dir,
                        final_model_dir=final_dir,
                        output_dir=output_dir,
                        device="cpu",
                    )
                )
            final_loader.assert_not_called()
            self.assertNotIn("final_gate", outputs)
            audit = json.loads(outputs["audit"].read_text(encoding="utf-8"))
            self.assertEqual(
                audit["status"],
                "b2_regret_sensitive_loso_development_failed_frozen_negative",
            )
            self.assertEqual(audit["development_fold_count"], 8)
            self.assertEqual(audit["P3_target_values_used"], 0)
            self.assertEqual(len(pd.read_csv(outputs["validation_decisions"])), 96)
            self.assertEqual(len(pd.read_csv(outputs["per_fold_metrics"])), 32)
            manifest = pd.read_csv(outputs["manifest"])
            for record in manifest.loc[
                manifest["role"].str.startswith("output_")
            ].itertuples(index=False):
                self.assertEqual(
                    sha256_file(output_dir / str(record.path)), str(record.sha256)
                )

    def test_passing_end_to_end_writes_final_gate_only_after_development_pass(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            dataset_dir = root / "dataset"
            screen_dir = root / "screen"
            final_dir = root / "final"
            output_dir = root / "output"
            for path in (dataset_dir, screen_dir, final_dir):
                path.mkdir()
            pd.concat(
                [
                    historical_cycles(site, year)
                    for year in range(2015, 2020)
                    for site in ("P2", "P4")
                ],
                ignore_index=True,
            ).to_csv(dataset_dir / DATASET_NAME, index=False)
            (dataset_dir / CONTRACT_NAME).write_text(
                json.dumps({"continuous_feature_columns": CONTINUOUS}),
                encoding="utf-8",
            )
            protocol_manifest = root / "protocol_manifest.csv"
            protocol_manifest.write_text("role,path\n", encoding="utf-8")
            frozen = {
                "protocol": {
                    "B_source_only_router": {
                        "gate_feature_contract": gate_feature_contract(CONTINUOUS)
                    }
                },
                "paths": {"manifest": protocol_manifest},
            }
            checkpoints = {
                str(item["protocol_fold"]): {
                    "P1": {"source_site_id": "P1"},
                    "P15": {"source_site_id": "P15"},
                }
                for item in DEVELOPMENT_FOLDS
            }
            summary = pd.DataFrame(
                columns=["protocol_fold", "source_site_id", "checkpoint_path"]
            )
            summary.to_csv(
                screen_dir / "gefs_p3_strict_loso_source_checkpoint_summary_v1.csv",
                index=False,
            )
            final_policy = {
                "source_experts": {
                    "P1": {"checkpoint_path": "P1.pt"},
                    "P15": {"checkpoint_path": "P15.pt"},
                }
            }
            for source in ("P1", "P15"):
                (final_dir / f"{source}.pt").write_bytes(source.encode("ascii"))
            (final_dir / "gefs_p3_2021_final_full_history_policy_v1r2.json").write_text(
                json.dumps(final_policy), encoding="utf-8"
            )

            def fake_expert_decisions(checkpoint_set, frame, device):
                del checkpoint_set, device
                decisions = {}
                for source, optimum in (("P1", 10.0), ("P15", 25.0)):
                    predicted = -(
                        frame["irrigation_mm"].to_numpy(dtype=float) - optimum
                    ) ** 2
                    decisions[source] = expert_cycle_decisions(frame, predicted, source)
                return decisions, pd.DataFrame()

            def fake_model(rows, feature_names, **kwargs):
                del rows, kwargs
                return {
                    "model": "test_nonfallback_logistic",
                    "fallback": False,
                    "fallback_reason": "",
                    "feature_names": list(feature_names),
                    "feature_mean": [0.0] * len(feature_names),
                    "feature_std": [1.0] * len(feature_names),
                    "coefficients": [1.0] + [0.0] * len(feature_names),
                    "training_rows": 10,
                    "positive_rows": 5,
                    "negative_rows": 5,
                    "threshold": 0.5,
                    "tie_route": "P15",
                    "converged": True,
                }

            gate_record = {
                "macro_mean_regret_gate_passed": True,
                "P2_site_mean_regret_gate_passed": True,
                "P4_site_mean_regret_gate_passed": True,
                "six_of_eight_nonworse_gate_passed": True,
                "worst_maximum_regret_gate_passed": True,
                "false_positive_gate_passed": True,
                "predicted_60mm_gate_passed": True,
                "P1_route_gate_passed": True,
                "each_site_nonfallback_gate_passed": True,
                "development_gate_passed": True,
            }
            with patch(
                "scripts.training.run_gefs_p3_b2_regret_sensitive_loso_router_screen_v1.validate_dual_protocol",
                return_value=frozen,
            ), patch(
                "scripts.training.run_gefs_p3_b2_regret_sensitive_loso_router_screen_v1.load_development_checkpoints",
                return_value=(checkpoints, summary),
            ), patch(
                "scripts.training.run_gefs_p3_b2_regret_sensitive_loso_router_screen_v1.load_final_checkpoints",
                return_value=(checkpoints["rolling_to_2019"], final_policy),
            ), patch(
                "scripts.training.run_gefs_p3_b2_regret_sensitive_loso_router_screen_v1.expert_decisions",
                side_effect=fake_expert_decisions,
            ), patch(
                "scripts.training.run_gefs_p3_b2_regret_sensitive_loso_router_screen_v1.fit_regret_weighted_logistic_gate",
                side_effect=fake_model,
            ), patch(
                "scripts.training.run_gefs_p3_b2_regret_sensitive_loso_router_screen_v1.b2_development_gate",
                return_value=pd.DataFrame([gate_record]),
            ):
                outputs = run_b2(
                    argparse.Namespace(
                        dataset_dir=dataset_dir,
                        protocol_dir=root / "protocol",
                        development_screen_dir=screen_dir,
                        final_model_dir=final_dir,
                        output_dir=output_dir,
                        device="cpu",
                    )
                )
            self.assertIn("final_gate", outputs)
            audit = json.loads(outputs["audit"].read_text(encoding="utf-8"))
            self.assertTrue(audit["development_gate_passed"])
            self.assertTrue(audit["final_gate_written"])
            self.assertEqual(
                audit["status"],
                "b2_regret_sensitive_loso_development_passed_pending_new_independent_freeze",
            )
            final_gate = json.loads(outputs["final_gate"].read_text(encoding="utf-8"))
            self.assertEqual(final_gate["P3_rows_used"], 0)
            self.assertEqual(final_gate["policy_state"], "pending_new_independent_freeze")


if __name__ == "__main__":
    unittest.main()
