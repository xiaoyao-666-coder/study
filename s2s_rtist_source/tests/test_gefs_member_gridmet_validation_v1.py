import importlib
import importlib.util
import sys
import unittest
from pathlib import Path

import pandas as pd

from s2s_rtist.weather.gefs_gridmet_bias import forecast_daily_to_long


class ForecastMemberIdentityTests(unittest.TestCase):
    def test_daily_to_long_preserves_gefs_member(self) -> None:
        daily = pd.DataFrame(
            {
                "site": ["P1"],
                "gefs_member": ["gec00"],
                "local_date": [pd.Timestamp("2024-07-16")],
                "precipitation_mm": [1.0],
                "temperature_min_c": [10.0],
                "temperature_max_c": [20.0],
                "shortwave_w_m2": [200.0],
                "wind_speed_m_s": [3.0],
                "vpd_kpa": [1.2],
            }
        )

        result = forecast_daily_to_long(daily)

        self.assertIn("gefs_member", result.columns)
        self.assertEqual(set(result["gefs_member"]), {"gec00"})


class GefsProbabilisticValidationTests(unittest.TestCase):
    def validation_module(self):
        try:
            return importlib.import_module(
                "s2s_rtist.weather.gefs_ensemble_validation"
            )
        except ModuleNotFoundError:
            self.fail("GEFS ensemble validation module is missing")

    @staticmethod
    def paired_rows() -> pd.DataFrame:
        return pd.DataFrame(
            {
                "site": ["P1"] * 3,
                "decision_date": [pd.Timestamp("2024-07-16")] * 3,
                "local_date": [pd.Timestamp("2024-07-16")] * 3,
                "lead_day": [1] * 3,
                "variable": ["precipitation_mm"] * 3,
                "gefs_member": ["gec00", "gep01", "gep02"],
                "forecast_value": [0.0, 1.0, 2.0],
                "reference_value": [1.0, 1.0, 1.0],
            }
        )

    def test_summarizes_complete_member_distribution_and_crps(self) -> None:
        validation = self.validation_module()

        result = validation.summarize_ensemble_observations(
            self.paired_rows(), expected_members=("gec00", "gep01", "gep02")
        )

        self.assertEqual(len(result), 1)
        row = result.iloc[0]
        self.assertEqual(int(row["member_count"]), 3)
        self.assertAlmostEqual(float(row["ensemble_mean"]), 1.0)
        self.assertAlmostEqual(float(row["crps"]), 2.0 / 9.0)
        self.assertTrue(bool(row["covered_by_p10_p90"]))
        self.assertTrue(bool(row["covered_by_min_max"]))

    def test_rejects_an_incomplete_member_distribution(self) -> None:
        validation = self.validation_module()
        incomplete = self.paired_rows().iloc[:-1].copy()

        with self.assertRaisesRegex(ValueError, "incomplete GEFS member set"):
            validation.summarize_ensemble_observations(
                incomplete, expected_members=("gec00", "gep01", "gep02")
            )

    def test_aggregates_probabilistic_metrics_by_variable(self) -> None:
        validation = self.validation_module()
        observations = validation.summarize_ensemble_observations(
            self.paired_rows(), expected_members=("gec00", "gep01", "gep02")
        )

        result = validation.aggregate_probabilistic_metrics(
            observations, group_columns=("variable",)
        )

        row = result.iloc[0]
        self.assertEqual(int(row["n_observations"]), 1)
        self.assertAlmostEqual(float(row["ensemble_mean_bias"]), 0.0)
        self.assertAlmostEqual(float(row["mean_crps"]), 2.0 / 9.0)
        self.assertAlmostEqual(float(row["p10_p90_coverage"]), 1.0)

    def test_computes_precipitation_event_probability_and_brier_score(self) -> None:
        validation = self.validation_module()

        result = validation.compute_precipitation_probability_metrics(
            self.paired_rows(),
            thresholds_mm=(1.0, 5.0),
            expected_members=("gec00", "gep01", "gep02"),
        )

        one_mm = result.loc[result["threshold_mm"].eq(1.0)].iloc[0]
        five_mm = result.loc[result["threshold_mm"].eq(5.0)].iloc[0]
        self.assertAlmostEqual(float(one_mm["mean_forecast_probability"]), 2.0 / 3.0)
        self.assertAlmostEqual(float(one_mm["brier_score"]), 1.0 / 9.0)
        self.assertAlmostEqual(float(five_mm["brier_score"]), 0.0)


class GefsMemberValidationRunnerTests(unittest.TestCase):
    def test_validates_precipitation_only_member_daily_weather(self) -> None:
        root = Path(__file__).resolve().parents[1]
        runner_path = (
            root
            / "scripts"
            / "diagnostics"
            / "run_gefs_member_gridmet_validation_v1.py"
        )
        spec = importlib.util.spec_from_file_location(
            "run_gefs_member_gridmet_validation_precipitation_only", runner_path
        )
        assert spec is not None and spec.loader is not None
        runner = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = runner
        spec.loader.exec_module(runner)
        daily = pd.DataFrame(
            {
                "site": ["P1"],
                "decision_date": [pd.Timestamp("2024-07-16")],
                "local_date": [pd.Timestamp("2024-07-16")],
                "lead_day": [1],
                "gefs_member": ["gec00"],
                "precipitation_mm": [4.0],
            }
        )

        runner.validate_member_daily_weather(
            daily,
            decision_dates=("2024-07-16",),
            site_names=("P1",),
            expected_members=("gec00",),
            variables=("precipitation_mm",),
            horizon_days=1,
        )

    def test_analyzes_member_daily_weather_without_network_access(self) -> None:
        root = Path(__file__).resolve().parents[1]
        runner_path = (
            root
            / "scripts"
            / "diagnostics"
            / "run_gefs_member_gridmet_validation_v1.py"
        )
        if not runner_path.exists():
            self.fail("GEFS member validation runner is missing")
        spec = importlib.util.spec_from_file_location(
            "run_gefs_member_gridmet_validation_v1", runner_path
        )
        assert spec is not None and spec.loader is not None
        runner = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = runner
        spec.loader.exec_module(runner)

        members = ("gec00", "gep01", "gep02")
        daily_rows = []
        for index, member in enumerate(members):
            daily_rows.append(
                {
                    "site": "P1",
                    "timezone": "America/Chicago",
                    "cycle_init_utc": pd.Timestamp("2024-07-16T00:00:00Z"),
                    "decision_date": pd.Timestamp("2024-07-16"),
                    "local_date": pd.Timestamp("2024-07-16"),
                    "lead_day": 1,
                    "gefs_member": member,
                    "precipitation_mm": float(index),
                    "temperature_min_c": 10.0 + index,
                    "temperature_max_c": 20.0 + index,
                    "shortwave_w_m2": 200.0 + index,
                    "wind_speed_m_s": 3.0 + index,
                    "vpd_kpa": 1.0 + index,
                }
            )
        daily = pd.DataFrame(daily_rows)
        reference = pd.DataFrame(
            {
                "site": ["P1"] * 6,
                "local_date": [pd.Timestamp("2024-07-16")] * 6,
                "variable": [
                    "precipitation_mm",
                    "temperature_min_c",
                    "temperature_max_c",
                    "shortwave_w_m2",
                    "wind_speed_m_s",
                    "vpd_kpa",
                ],
                "reference_value": [1.0, 11.0, 21.0, 201.0, 4.0, 2.0],
            }
        )

        outputs = runner.analyze_member_daily_weather(
            daily, reference, expected_members=members
        )

        self.assertEqual(len(outputs["paired_members"]), 18)
        self.assertEqual(len(outputs["ensemble_observations"]), 6)
        self.assertEqual(set(outputs["ensemble_observations"]["member_count"]), {3})
        self.assertEqual(len(outputs["precipitation_probability_overall"]), 4)


if __name__ == "__main__":
    unittest.main()
