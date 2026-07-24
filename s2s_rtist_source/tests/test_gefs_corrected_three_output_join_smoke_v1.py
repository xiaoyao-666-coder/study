from __future__ import annotations

import unittest
from pathlib import Path

import pandas as pd

from scripts.data_preparation.build_gefs_corrected_three_output_join_smoke_v1 import (
    audit_label_formulas,
    load_contract,
)


CONTRACT = Path(
    "site_general_surrogate_eval/gefs_corrected_three_output_join_smoke_contract_v1.json"
)


class GefsCorrectedThreeOutputJoinSmokeTests(unittest.TestCase):
    def test_contract_blocks_training_for_scenario_mismatch(self) -> None:
        contract = load_contract(CONTRACT)
        self.assertFalse(
            contract["scenario_provenance"]["weather_label_scenario_consistent"]
        )
        self.assertFalse(contract["scenario_provenance"]["training_eligible"])
        self.assertFalse(contract["scope"]["surrogate_training_allowed"])
        self.assertEqual(
            contract["continuous_irrigation_constraint_mm"],
            {"minimum": 0.0, "maximum": 60.0},
        )

    def test_formula_audit_accepts_exact_synthetic_labels(self) -> None:
        contract = load_contract(CONTRACT)
        rows = []
        for irrigation, cwdm in [(0.0, 1000.0), (10.0, 1100.0)]:
            target = (cwdm - 1000.0) * 0.2 - irrigation * 2.0 * 0.7
            row = {
                "site_id": "P1",
                "decision_date": "2024-07-16",
                "candidate_irrigation_mm": irrigation,
                "cwdm_value": cwdm,
                "target_value": target,
                "net_gain_7d": target,
                "tact_7d_mm": 7.0,
                "eact_7d_mm": 14.0,
                "interc_7d_mm": 0.7,
                "aet_7d_mm": 21.7,
            }
            for day in range(1, 8):
                row[f"aet_day{day:02d}_mm"] = 3.1
            rows.append(row)
        audit = audit_label_formulas(pd.DataFrame(rows), contract)
        for value in audit.values():
            self.assertAlmostEqual(value, 0.0)


if __name__ == "__main__":
    unittest.main()
