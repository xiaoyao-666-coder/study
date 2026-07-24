from __future__ import annotations

import unittest
from pathlib import Path

from scripts.diagnostics.run_gefs_qdm_volume_preserving_2019_validation_v1 import (
    CANDIDATE_ID,
    load_contract,
)


class VolumePreserving2019ContractTests(unittest.TestCase):
    def test_contract_freezes_candidate_and_prohibits_2024(self) -> None:
        path = (
            Path(__file__).resolve().parents[1]
            / "site_general_surrogate_eval"
            / "gefs_qdm_volume_preserving_2019_contract_v1.json"
        )
        contract = load_contract(path)
        self.assertEqual(contract["candidate_id"], CANDIDATE_ID)
        self.assertFalse(contract["scope"]["use_2019_reference_for_fit"])
        self.assertFalse(contract["scope"]["use_2024_allowed"])
        self.assertIsNone(contract["scale_cap"])
        self.assertEqual(
            contract["target_cdf_mode"], "offline_complete_2019_gefs_batch"
        )


if __name__ == "__main__":
    unittest.main()
