from __future__ import annotations

import unittest
from pathlib import Path

from scripts.diagnostics.run_gefs_qdm_causal_current_cycle_2019_validation_v1 import (
    CANDIDATE_ID,
    load_contract,
)


class CausalCurrentCycle2019ContractTests(unittest.TestCase):
    def test_contract_freezes_causal_candidate_and_prohibits_future_data(self) -> None:
        path = (
            Path(__file__).resolve().parents[1]
            / "site_general_surrogate_eval"
            / "gefs_qdm_causal_current_cycle_2019_contract_v1.json"
        )
        contract = load_contract(path)
        self.assertEqual(contract["candidate_id"], CANDIDATE_ID)
        self.assertEqual(
            contract["target_cdf_mode"], "causal_current_cycle_global_batch"
        )
        self.assertFalse(contract["scope"]["use_2019_reference_for_fit"])
        self.assertFalse(
            contract["scope"]["use_future_2019_cycles_for_target_cdf"]
        )
        self.assertFalse(contract["scope"]["use_2024_allowed"])
        self.assertIsNone(contract["scale_cap"])


if __name__ == "__main__":
    unittest.main()
