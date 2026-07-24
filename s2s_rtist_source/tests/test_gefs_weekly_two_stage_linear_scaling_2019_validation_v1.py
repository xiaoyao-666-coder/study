from __future__ import annotations

import unittest
from pathlib import Path

from scripts.diagnostics.run_gefs_weekly_two_stage_linear_scaling_2019_validation_v1 import (
    CANDIDATE_ID,
    load_contract,
)


class WeeklyTwoStageLinear2019ContractTests(unittest.TestCase):
    def test_contract_freezes_site_candidate_and_prohibits_test_data(self) -> None:
        path = (
            Path(__file__).resolve().parents[1]
            / "site_general_surrogate_eval"
            / "gefs_weekly_two_stage_linear_scaling_2019_contract_v1.json"
        )
        contract = load_contract(path)
        self.assertEqual(contract["candidate_id"], CANDIDATE_ID)
        self.assertEqual(contract["group_keys"], ["site_id"])
        self.assertFalse(contract["scope"]["use_2019_reference_for_fit"])
        self.assertFalse(
            contract["scope"]["use_future_2019_cycles_for_factor_fit"]
        )
        self.assertFalse(contract["scope"]["use_2024_allowed"])


if __name__ == "__main__":
    unittest.main()
