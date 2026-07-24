from __future__ import annotations

import unittest
from pathlib import Path

from scripts.data_preparation.extract_gefs_weekly_linear_2024_six_cycle_precipitation_v1 import (
    expected_row_count,
    load_contract as load_extraction_contract,
)
from scripts.diagnostics.run_gefs_weekly_linear_2024_six_cycle_confirmation_v1 import (
    load_contract as load_scoring_contract,
    promotion_gate,
)


CONTRACT = Path(
    "site_general_surrogate_eval/gefs_weekly_linear_2024_six_cycle_confirmation_contract_v1.json"
)


class WeeklyLinear2024SixCycleConfirmationTests(unittest.TestCase):
    def test_contract_freezes_disjoint_cycles_and_expected_rows(self) -> None:
        contract = load_extraction_contract(CONTRACT)
        self.assertFalse(
            set(contract["decision_dates"]).intersection(
                contract["previously_scored_decision_dates"]
            )
        )
        self.assertEqual(expected_row_count(contract), 6510)
        self.assertEqual(contract["scope"]["network_download_location"], "local_workstation_only")

    def test_scoring_contract_prohibits_all_retuning(self) -> None:
        contract = load_scoring_contract(CONTRACT)
        self.assertFalse(contract["scope"]["artifact_refit_allowed"])
        self.assertFalse(contract["scope"]["hyperparameter_tuning_allowed"])
        self.assertFalse(contract["scope"]["station_reselection_allowed"])

    def test_promotion_gate_requires_every_prelocked_requirement(self) -> None:
        metric = {
            "candidate_ensemble_mean_mae": 1.0,
            "raw_ensemble_mean_mae": 1.0,
            "candidate_ensemble_mean_rmse": 2.0,
            "raw_ensemble_mean_rmse": 2.0,
            "seven_day_mae_not_worse": True,
            "crps_not_worse": True,
            "mean_brier_not_worse": True,
            "heavy_coverage_not_both_worse": True,
        }
        occurrence = {"occurrence_not_worse": True}
        numeric = {"negative_count": 0, "nonfinite_count": 0}
        self.assertTrue(all(promotion_gate(metric, occurrence, numeric).values()))
        metric["crps_not_worse"] = False
        self.assertFalse(all(promotion_gate(metric, occurrence, numeric).values()))


if __name__ == "__main__":
    unittest.main()
