from __future__ import annotations

import importlib.util
import json
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

from s2s_rtist.weather.gefs_quantile_mapping import (
    CONTRACT_ID_TRAINING_CV,
    CONTRACT_VERSION_TRAINING_CV,
    GEFS_REFORECAST_MEMBERS,
    UPPER_TAIL_CONSTANT_ADDITIVE,
    UTC_DAY_BOUNDARY,
    apply_empirical_precipitation_qm,
    fit_empirical_precipitation_qm,
)


ROOT = Path(__file__).resolve().parents[1]
DIAGNOSTICS = ROOT / "scripts" / "diagnostics"


def load_runner():
    sys.path.insert(0, str(DIAGNOSTICS))
    path = DIAGNOSTICS / "run_gefs_qm_training_cv_v1.py"
    spec = importlib.util.spec_from_file_location("training_cv_runner_test", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def training_frame() -> pd.DataFrame:
    rows = []
    for year_index, year in enumerate((2015, 2016, 2017), start=1):
        for site_index, site_id in enumerate(("P1", "P2"), start=1):
            for lead_day in (1, 2):
                reference = float(year_index + site_index + lead_day)
                for member_index, member in enumerate(GEFS_REFORECAST_MEMBERS):
                    rows.append(
                        {
                            "site_id": site_id,
                            "decision_date": pd.Timestamp(f"{year}-06-01"),
                            "valid_date_utc": pd.Timestamp(f"{year}-06-0{lead_day}"),
                            "lead_day": lead_day,
                            "gefs_member": member,
                            "precipitation_mm_raw": reference
                            + 0.2 * member_index,
                            "precipitation_mm_reference": reference,
                        }
                    )
    return pd.DataFrame(rows)


def fit_candidate(group_keys: tuple[str, ...]):
    return fit_empirical_precipitation_qm(
        training_frame(),
        fit_years=(2015, 2016, 2017),
        contract_id=CONTRACT_ID_TRAINING_CV,
        contract_version=CONTRACT_VERSION_TRAINING_CV,
        aggregation_day_boundary=UTC_DAY_BOUNDARY,
        canonical_valid_date_column="valid_date_utc",
        upper_tail_policy=UPPER_TAIL_CONSTANT_ADDITIVE,
        group_keys=group_keys,
        artifact_context={
            "candidate_id": "synthetic",
            "fold_id": "F2018",
            "validation_year": 2018,
        },
    )


class TrainingCvContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.runner = load_runner()
        cls.contract = json.loads(
            (ROOT / "site_general_surrogate_eval" / "gefs_qm_training_cv_contract_v1.json").read_text(
                encoding="utf-8"
            )
        )

    def test_contract_has_four_disjoint_leave_one_year_out_folds(self) -> None:
        folds = self.contract["folds"]
        self.assertEqual(len(folds), 4)
        self.assertEqual(
            {int(fold["validation_year"]) for fold in folds},
            {2015, 2016, 2017, 2018},
        )
        for fold in folds:
            self.assertNotIn(int(fold["validation_year"]), fold["fit_years"])
            self.assertEqual(len(fold["fit_years"]), 3)
        self.assertFalse(self.contract["scope"]["new_network_download_allowed"])
        self.assertFalse(self.contract["scope"]["use_2019_allowed"])
        self.assertFalse(self.contract["scope"]["use_2024_allowed"])

    def test_fold_assignment_contains_each_prelocked_cycle_once(self) -> None:
        assignment = self.runner._fold_assignment(self.contract)
        self.assertEqual(len(assignment), 24)
        self.assertFalse(assignment["forecast_init_utc"].duplicated().any())
        self.assertEqual(set(assignment["validation_year"]), {2015, 2016, 2017, 2018})
        self.assertTrue(assignment["validation_rows_used_for_fit"].eq(0).all())

    def test_pareto_relationship_does_not_force_a_single_winner(self) -> None:
        pooled = pd.DataFrame(
            [
                {
                    "candidate_id": "a",
                    "candidate_seven_day_mae_mm": 1.0,
                    "candidate_mean_crps_mm": 2.0,
                    "candidate_mean_brier_score": 3.0,
                },
                {
                    "candidate_id": "b",
                    "candidate_seven_day_mae_mm": 2.0,
                    "candidate_mean_crps_mm": 1.0,
                    "candidate_mean_brier_score": 2.0,
                },
                {
                    "candidate_id": "c",
                    "candidate_seven_day_mae_mm": 3.0,
                    "candidate_mean_crps_mm": 3.0,
                    "candidate_mean_brier_score": 4.0,
                },
            ]
        )
        relation = self.runner._pareto_relationship(["a", "b", "c"], pooled)
        self.assertEqual(relation["a"], [])
        self.assertEqual(relation["b"], [])
        self.assertEqual(relation["c"], ["a", "b"])

    def test_bootstrap_requires_24_cycles_and_is_reproducible(self) -> None:
        differences = {
            name: pd.DataFrame(
                {"difference_qm_minus_raw": np.linspace(-1.0, 1.0, 24)}
            )
            for name in ("crps", "mean_brier", "seven_day_mae")
        }
        with patch.object(
            self.runner, "_cycle_metric_differences", return_value=differences
        ):
            first = self.runner._bootstrap(
                "candidate", pd.DataFrame(), expected_cycles=24, replicates=100, seed=7
            )
            second = self.runner._bootstrap(
                "candidate", pd.DataFrame(), expected_cycles=24, replicates=100, seed=7
            )
        pd.testing.assert_frame_equal(first, second)
        self.assertTrue(first["cycle_count"].eq(24).all())


class GenericGroupingTests(unittest.TestCase):
    def test_all_prelocked_grouping_levels_fit_expected_samples(self) -> None:
        expectations = {
            ("site_id", "lead_day"): (4, 15),
            ("site_id",): (2, 30),
            ("lead_day",): (2, 30),
            (): (1, 60),
        }
        for group_keys, (group_count, sample_count) in expectations.items():
            with self.subTest(group_keys=group_keys):
                artifact = fit_candidate(group_keys)
                self.assertEqual(len(artifact["groups"]), group_count)
                self.assertEqual(
                    set(artifact["group_sample_counts"].values()), {sample_count}
                )
                corrected = apply_empirical_precipitation_qm(
                    training_frame().iloc[:10], artifact, split="synthetic_oof"
                )
                self.assertTrue(np.isfinite(corrected["precipitation_mm_qm"]).all())
                self.assertTrue(corrected["precipitation_mm_qm"].ge(0.0).all())

    def test_default_site_lead_key_remains_v1_v2_compatible(self) -> None:
        artifact = fit_candidate(("site_id", "lead_day"))
        self.assertIn("P1|1", artifact["groups"])
        self.assertEqual(artifact["groups"]["P1|1"]["site_id"], "P1")
        self.assertEqual(artifact["groups"]["P1|1"]["lead_day"], 1)
        self.assertNotIn("group_values", artifact["groups"]["P1|1"])

    def test_generic_init_month_group_and_occurrence_ablation_are_explicit(self) -> None:
        frame = training_frame().copy()
        frame["init_month"] = frame["decision_date"].dt.month.astype(int)
        artifact = fit_empirical_precipitation_qm(
            frame,
            fit_years=(2015, 2016, 2017),
            contract_id=CONTRACT_ID_TRAINING_CV,
            contract_version=CONTRACT_VERSION_TRAINING_CV,
            aggregation_day_boundary=UTC_DAY_BOUNDARY,
            canonical_valid_date_column="valid_date_utc",
            upper_tail_policy=UPPER_TAIL_CONSTANT_ADDITIVE,
            group_keys=("init_month",),
            occurrence_correction=False,
            artifact_context={
                "candidate_id": "init-month-no-occurrence",
                "fold_id": "F2018",
                "validation_year": 2018,
            },
        )
        self.assertEqual(artifact["group_keys"], ["init_month"])
        self.assertFalse(artifact["occurrence_correction"])
        self.assertEqual(len(artifact["groups"]), 1)
        self.assertTrue(
            all(
                group["forecast_wet_threshold_mm"] == 0.0
                for group in artifact["groups"].values()
            )
        )

    def test_training_cv_artifact_rejects_validation_year_in_fit(self) -> None:
        with self.assertRaisesRegex(ValueError, "validation year leaked"):
            fit_empirical_precipitation_qm(
                training_frame(),
                fit_years=(2015, 2016, 2017),
                contract_id=CONTRACT_ID_TRAINING_CV,
                contract_version=CONTRACT_VERSION_TRAINING_CV,
                aggregation_day_boundary=UTC_DAY_BOUNDARY,
                canonical_valid_date_column="valid_date_utc",
                upper_tail_policy=UPPER_TAIL_CONSTANT_ADDITIVE,
                group_keys=("site_id",),
                artifact_context={
                    "candidate_id": "bad",
                    "fold_id": "F2017",
                    "validation_year": 2017,
                },
            )


if __name__ == "__main__":
    unittest.main()
