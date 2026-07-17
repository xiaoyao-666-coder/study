import importlib
import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from s2s_rtist.weather.gefs_gridmet_bias import build_gefs_product_url


class GefsMemberProductTests(unittest.TestCase):
    def test_lists_control_and_thirty_perturbed_members(self) -> None:
        module = importlib.import_module("s2s_rtist.weather.gefs_gridmet_bias")

        members = module.gefs_members()

        self.assertEqual(len(members), 31)
        self.assertEqual(members[0], "gec00")
        self.assertEqual(members[1], "gep01")
        self.assertEqual(members[-1], "gep30")
        self.assertEqual(len(set(members)), 31)

    def test_builds_official_control_and_perturbed_member_urls(self) -> None:
        control = build_gefs_product_url(
            "2024-07-16", cycle_hour=0, lead_hour=3, product="gec00"
        )
        perturbed = build_gefs_product_url(
            "2024-07-16", cycle_hour=0, lead_hour=180, product="gep27"
        )

        self.assertEqual(
            control,
            "https://noaa-gefs-pds.s3.amazonaws.com/gefs.20240716/00/atmos/"
            "pgrb2sp25/gec00.t00z.pgrb2s.0p25.f003",
        )
        self.assertEqual(
            perturbed,
            "https://noaa-gefs-pds.s3.amazonaws.com/gefs.20240716/00/atmos/"
            "pgrb2sp25/gep27.t00z.pgrb2s.0p25.f180",
        )

    def test_rejects_unknown_gefs_product_name(self) -> None:
        with self.assertRaisesRegex(ValueError, "unsupported GEFS product"):
            build_gefs_product_url(
                "2024-07-16", cycle_hour=0, lead_hour=3, product="gep31"
            )


class GefsEnsemblePolicyTests(unittest.TestCase):
    def policy_module(self):
        try:
            return importlib.import_module("s2s_rtist.weather.gefs_ensemble_policy")
        except ModuleNotFoundError:
            self.fail("GEFS ensemble policy module is missing")

    @staticmethod
    def prediction_rows() -> pd.DataFrame:
        return pd.DataFrame(
            [
                {"site_date_id": "P1_20240716", "candidate_ir": 0.0, "gefs_member": "gec00", "pred_net_gain_7d": 4.0},
                {"site_date_id": "P1_20240716", "candidate_ir": 0.0, "gefs_member": "gep01", "pred_net_gain_7d": 5.0},
                {"site_date_id": "P1_20240716", "candidate_ir": 0.0, "gefs_member": "gep02", "pred_net_gain_7d": 6.0},
                {"site_date_id": "P1_20240716", "candidate_ir": 30.0, "gefs_member": "gec00", "pred_net_gain_7d": 7.0},
                {"site_date_id": "P1_20240716", "candidate_ir": 30.0, "gefs_member": "gep01", "pred_net_gain_7d": 1.0},
                {"site_date_id": "P1_20240716", "candidate_ir": 30.0, "gefs_member": "gep02", "pred_net_gain_7d": 7.0},
            ]
        )

    def test_summarizes_member_predictions_for_each_irrigation_candidate(self) -> None:
        policy = self.policy_module()

        result = policy.summarize_member_predictions(
            self.prediction_rows(), expected_members=("gec00", "gep01", "gep02")
        )

        zero = result.loc[result["candidate_ir"].eq(0.0)].iloc[0]
        irrigated = result.loc[result["candidate_ir"].eq(30.0)].iloc[0]
        self.assertEqual(int(zero["member_count"]), 3)
        self.assertAlmostEqual(float(zero["mean_pred_net_gain_7d"]), 5.0)
        self.assertAlmostEqual(float(zero["median_pred_net_gain_7d"]), 5.0)
        self.assertAlmostEqual(float(irrigated["mean_pred_net_gain_7d"]), 5.0)
        self.assertAlmostEqual(float(irrigated["min_pred_net_gain_7d"]), 1.0)
        self.assertAlmostEqual(float(irrigated["max_pred_net_gain_7d"]), 7.0)

    def test_rejects_candidate_with_missing_ensemble_member(self) -> None:
        policy = self.policy_module()
        incomplete = self.prediction_rows().iloc[:-1].copy()

        with self.assertRaisesRegex(ValueError, "incomplete GEFS member set"):
            policy.summarize_member_predictions(
                incomplete, expected_members=("gec00", "gep01", "gep02")
            )

    def test_selects_smallest_irrigation_when_mean_profit_is_tied(self) -> None:
        policy = self.policy_module()

        decisions = policy.select_irrigation_by_mean_profit(
            self.prediction_rows(), expected_members=("gec00", "gep01", "gep02")
        )

        self.assertEqual(len(decisions), 1)
        self.assertEqual(float(decisions.iloc[0]["chosen_ir"]), 0.0)
        self.assertAlmostEqual(
            float(decisions.iloc[0]["chosen_mean_pred_net_gain_7d"]), 5.0
        )

    def test_summarizes_member_specific_optimum_irrigation_as_baseline(self) -> None:
        policy = self.policy_module()

        result = policy.summarize_member_optima(
            self.prediction_rows(), expected_members=("gec00", "gep01", "gep02")
        )

        self.assertEqual(len(result), 1)
        self.assertAlmostEqual(float(result.iloc[0]["mean_member_optimum_ir"]), 20.0)
        self.assertAlmostEqual(float(result.iloc[0]["median_member_optimum_ir"]), 30.0)
        self.assertEqual(float(result.iloc[0]["mode_member_optimum_ir"]), 30.0)


class GefsEnsemblePolicyRunnerTests(unittest.TestCase):
    def test_writes_candidate_decision_and_baseline_evidence(self) -> None:
        root = Path(__file__).resolve().parents[1]
        runner_path = (
            root
            / "scripts"
            / "evaluation"
            / "evaluate_gefs_member_ensemble_policy_v1.py"
        )
        if not runner_path.exists():
            self.fail("GEFS member ensemble policy runner is missing")
        spec = importlib.util.spec_from_file_location(
            "evaluate_gefs_member_ensemble_policy_v1", runner_path
        )
        assert spec is not None and spec.loader is not None
        runner = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = runner
        spec.loader.exec_module(runner)

        with tempfile.TemporaryDirectory() as directory:
            temp = Path(directory)
            input_path = temp / "member_predictions.csv"
            GefsEnsemblePolicyTests.prediction_rows().to_csv(input_path, index=False)

            outputs = runner.run_evaluation(
                input_path=input_path,
                output_dir=temp / "result",
                expected_members=("gec00", "gep01", "gep02"),
            )

            candidate = pd.read_csv(outputs["candidate_summary"])
            decision = pd.read_csv(outputs["decisions"])
            baseline = pd.read_csv(outputs["member_optima"])
            report = outputs["report"].read_text(encoding="utf-8")
            self.assertEqual(len(candidate), 2)
            self.assertEqual(float(decision.iloc[0]["chosen_ir"]), 0.0)
            self.assertAlmostEqual(
                float(baseline.iloc[0]["mean_member_optimum_ir"]), 20.0
            )
            self.assertIn("ensemble-mean predicted profit", report)
            self.assertIn("geavg", report)


if __name__ == "__main__":
    unittest.main()
