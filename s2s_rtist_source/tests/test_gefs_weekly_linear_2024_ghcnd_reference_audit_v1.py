from __future__ import annotations

import unittest
from pathlib import Path

import pandas as pd

from scripts.diagnostics.audit_gefs_weekly_linear_2024_ghcnd_reference_v1 import (
    bootstrap_cycle_metrics,
    load_contract,
)


class WeeklyLinear2024GhcndReferenceAuditTests(unittest.TestCase):
    def test_contract_prohibits_refit_tuning_and_score_based_station_selection(self) -> None:
        contract = load_contract(
            Path(
                "site_general_surrogate_eval/gefs_weekly_linear_2024_ghcnd_reference_audit_contract_v1.json"
            )
        )
        self.assertFalse(contract["scope"]["artifact_refit_allowed"])
        self.assertFalse(contract["scope"]["hyperparameter_tuning_allowed"])
        self.assertFalse(
            contract["scope"]["station_selection_using_forecast_scores_allowed"]
        )

    def test_cycle_bootstrap_reports_all_three_metrics(self) -> None:
        frame = pd.DataFrame(
            {
                "reference_scope": ["scope"] * 5,
                "decision_date": [f"2024-07-{day:02d}" for day in range(1, 6)],
                "seven_day_mae_difference_candidate_minus_raw_mm": [-1, 0, 1, 0, -1],
                "crps_difference_candidate_minus_raw_mm": [-0.1, 0, 0.1, 0, -0.1],
                "mean_brier_difference_candidate_minus_raw": [-0.01, 0, 0.01, 0, -0.01],
            }
        )
        result = bootstrap_cycle_metrics(frame, replicates=100, seed=7)
        self.assertEqual(set(result["metric"]), {"seven_day_mae", "crps", "mean_brier"})
        self.assertTrue(result["cycle_count"].eq(5).all())
        self.assertTrue(result["bootstrap_replicates"].eq(100).all())


if __name__ == "__main__":
    unittest.main()
