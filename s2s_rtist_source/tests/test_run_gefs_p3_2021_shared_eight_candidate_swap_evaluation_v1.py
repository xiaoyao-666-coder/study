from __future__ import annotations

import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import patch

import pandas as pd

from scripts.simulation.run_gefs_checkpoint_one_date_eight_ir_smoke_v1 import (
    sha256_file,
)
from scripts.simulation.run_gefs_p3_2021_shared_eight_candidate_swap_evaluation_v1 import (
    FIXED_IRRIGATION_MM,
    POLICY_ROLES,
    evaluate_frozen_policies,
    run,
)


def candidate_labels(decision_date: str) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "site": ["P3"] * 8,
            "date_t": [decision_date] * 8,
            "ir": list(FIXED_IRRIGATION_MM),
            "target_value": [
                -abs(value - 20.0) for value in FIXED_IRRIGATION_MM
            ],
            "water_balance_residual_0_100cm_7d_mm": [0.01] * 8,
            "requested_ir_mm": list(FIXED_IRRIGATION_MM),
            "simulated_ir_mm": list(FIXED_IRRIGATION_MM),
            "numerical_endpoint_fallback": [False] * 8,
            "numerical_endpoint_fallback_delta_mm": [0.0] * 8,
        }
    )


def policy_rows(dates: list[str]) -> pd.DataFrame:
    rows = []
    selected = {
        "B_source_only_router": ("P1", 20.0),
        "A_P3_historical_supervised_router": ("P1", 15.0),
        "always_P15": ("P15", 60.0),
        "always_P1": ("P1", 20.0),
    }
    for experiment_id in POLICY_ROLES:
        source, irrigation = selected[experiment_id]
        for decision_date in dates:
            rows.append(
                {
                    "experiment_id": experiment_id,
                    "site_id": "P3",
                    "decision_date": decision_date,
                    "gate_probability_p1": 1.0 if source == "P1" else 0.0,
                    "selected_source_site_id": source,
                    "selected_irrigation_mm": irrigation,
                    "selected_predicted_net_gain_7d": 1.0,
                }
            )
    return pd.DataFrame(rows)


class P32021SharedEightCandidateSwapTests(unittest.TestCase):
    def test_policy_evaluation_uses_discrete_regret_without_match_rate(self) -> None:
        date = "2021-06-17"
        labels = candidate_labels(date).assign(
            site_id="P3",
            decision_date=date,
            irrigation_mm=list(FIXED_IRRIGATION_MM),
            swap_gain_7d=[-abs(value - 20.0) for value in FIXED_IRRIGATION_MM],
        )
        decisions, summary = evaluate_frozen_policies(
            labels, policy_rows([date])
        )
        primary = decisions.loc[
            decisions["experiment_id"].eq("B_source_only_router")
        ].iloc[0]
        baseline = decisions.loc[
            decisions["experiment_id"].eq("always_P15")
        ].iloc[0]
        self.assertEqual(primary["regret_vs_fixed_eight_swap_7d"], 0.0)
        self.assertGreater(primary["swap_gain_delta_vs_always_P15_7d"], 0.0)
        self.assertGreater(baseline["regret_vs_fixed_eight_swap_7d"], 0.0)
        self.assertEqual(len(summary), 4)
        self.assertFalse(any("match" in column for column in decisions.columns))
        self.assertFalse(any("match" in column for column in summary.columns))

    def test_partial_run_builds_one_shared_table_for_all_four_policies(self) -> None:
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            recommendation_dir = root / "recommendations"
            trunk_dir = root / "trunk"
            weather_dir = root / "weather"
            output_dir = root / "output"
            recommendation_dir.mkdir()
            weather_dir.mkdir()
            dates = pd.date_range("2021-05-20", periods=12, freq="7D").strftime(
                "%Y-%m-%d"
            ).tolist()

            recommendations = policy_rows(dates)
            recommendation_path = (
                recommendation_dir
                / "gefs_p3_2021_A_B_policy_recommendations_v1.csv"
            )
            recommendations.to_csv(recommendation_path, index=False)
            (recommendation_dir / "gefs_p3_2021_A_B_recommendation_freeze_audit_v1.json").write_text(
                json.dumps(
                    {
                        "status": "p3_2021_A_B_recommendations_frozen_pending_shared_independent_SWAP_labels",
                        "mandatory_gate_passed": True,
                        "2021_SWAP_candidate_labels_read": 0,
                        "SWAP_simulation_performed": False,
                        "model_training_or_reselection_performed": False,
                        "formal_primary_experiment": "B_source_only_router",
                        "formal_secondary_experiment": "A_P3_historical_supervised_router",
                        "formal_baseline": "always_P15",
                    }
                ),
                encoding="utf-8",
            )
            pd.DataFrame(
                [
                    {
                        "role": "output_policy_recommendations",
                        "path": recommendation_path.name,
                        "sha256": sha256_file(recommendation_path),
                    }
                ]
            ).to_csv(
                recommendation_dir
                / "gefs_p3_2021_A_B_recommendation_freeze_manifest_v1.csv",
                index=False,
            )

            unit = trunk_dir / "units" / "Y2021" / "P3"
            (unit / "workspace").mkdir(parents=True)
            (unit / "all_checkpoints_v1" / "checkpoints").mkdir(parents=True)
            (unit / "trunk").mkdir(parents=True)
            (trunk_dir / "gefs_p3_2021_era5_trunk_schedule_freeze_audit_v1.json").write_text(
                json.dumps(
                    {
                        "mandatory_gate_passed": True,
                        "target_site": "P3",
                        "target_year": 2021,
                        "SWAP_candidate_label_rows_read": 0,
                        "SWAP_candidate_label_generation_performed": False,
                    }
                ),
                encoding="utf-8",
            )
            pd.DataFrame(
                {
                    "site_id": ["P3"] * 12,
                    "decision_date": dates,
                }
            ).to_csv(
                unit / "trunk" / "swap_season_decision_schedule_v1.csv",
                index=False,
            )
            pd.DataFrame(
                {
                    "decision_date": dates,
                    "checkpoint_equivalence_passed": [True] * 12,
                }
            ).to_csv(
                unit
                / "all_checkpoints_v1"
                / "swap_season_checkpoint_equivalence_v1.csv",
                index=False,
            )

            (weather_dir / "gefs_p3_2021_frozen_weather_correction_audit_v1.json").write_text(
                json.dumps(
                    {
                        "mandatory_gate_passed": True,
                        "target_site": "P3",
                        "target_year": 2021,
                        "2021_SWAP_candidate_labels_read": 0,
                        "model_inference_performed": False,
                    }
                ),
                encoding="utf-8",
            )
            weather_rows = []
            for decision_date in dates:
                for member in range(5):
                    for lead_day in range(1, 8):
                        weather_rows.append(
                            {
                                "site_id": "P3",
                                "decision_date": decision_date,
                                "gefs_member": f"m{member}",
                                "lead_day": lead_day,
                            }
                        )
            pd.DataFrame(weather_rows).to_csv(
                weather_dir
                / "gefs_p3_2021_five_member_frozen_corrected_weather_v1.csv",
                index=False,
            )

            def fake_stage(**kwargs):
                self.assertEqual(kwargs["target_year"], 2021)
                self.assertEqual(kwargs["sowing_month_day"], "04-26")
                return candidate_labels(kwargs["decision_date"]), kwargs["base_root"]

            with patch(
                "scripts.simulation.run_gefs_p3_2021_shared_eight_candidate_swap_evaluation_v1.run_swap_stage",
                side_effect=fake_stage,
            ):
                outputs = run(
                    SimpleNamespace(
                        recommendation_dir=recommendation_dir,
                        trunk_dir=trunk_dir,
                        weather_dir=weather_dir,
                        output_dir=output_dir,
                        restart_nprintday=48,
                        swap_dtmax_days=0.01,
                        max_cycles=2,
                        resume=False,
                    )
                )

            labels = pd.read_csv(outputs["labels"])
            decisions = pd.read_csv(outputs["decisions"])
            summary = pd.read_csv(outputs["summary"])
            audit = json.loads(outputs["audit"].read_text(encoding="utf-8"))
            self.assertEqual(len(labels), 16)
            self.assertEqual(len(decisions), 8)
            self.assertEqual(len(summary), 4)
            self.assertEqual(
                audit["status"],
                "p3_2021_shared_eight_candidate_swap_evaluation_partial_passed",
            )
            self.assertTrue(audit["mandatory_gate_passed"])
            self.assertTrue(audit["single_shared_label_table_used_for_all_policies"])
            self.assertEqual(audit["experiment_B_P3_rows_used_in_gate_fit"], 0)
            self.assertFalse(audit["exact_irrigation_match_metric_reported"])
            self.assertFalse(audit["interpolation_used_for_formal_metrics"])
            self.assertFalse(audit["model_training_or_reselection_performed"])


if __name__ == "__main__":
    unittest.main()
