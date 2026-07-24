from __future__ import annotations

import unittest

import pandas as pd

from scripts.diagnostics.fit_gefs_weekly_linear_final_artifact_v1 import (
    artifact_hash,
    build_artifact,
    validate_selection_gate,
)


class WeeklyLinearFinalArtifactTests(unittest.TestCase):
    def setUp(self) -> None:
        self.contract = {
            "contract_id": "gefs-weekly-linear-final-fit-v1",
            "contract_version": 1,
            "candidate_id": "weekly_two_stage_linear_site_factor_shrink_a075",
            "base_candidate_id": "weekly_two_stage_linear_site_only",
            "group_keys": ["site_id"],
            "factor_shrinkage_alpha": 0.75,
            "fit_years": list(range(2000, 2020)),
            "required_selection_gate": {
                "selected_candidate": "weekly_two_stage_linear_site_factor_shrink_a075",
                "selected_alpha": 0.75,
                "2024_used": False,
            },
        }

    def test_artifact_shrinks_both_regime_factors_and_hashes(self) -> None:
        factors = pd.DataFrame(
            {
                "site_id": ["P1"],
                "fit_complete_cycle_count": [100],
                "fit_extreme_cycle_count": [10],
                "extreme_quantile": [0.9],
                "raw_ensemble_mean_7d_q90_mm": [50.0],
                "extreme_factor": [0.5],
                "overall_factor": [0.8],
                "final_extreme_factor": [0.4],
            }
        )
        artifact = build_artifact(
            factors=factors,
            contract=self.contract,
            selection_gate_sha256="a" * 64,
            input_hashes={"input": "b" * 64},
        )
        group = artifact["groups"][0]
        self.assertAlmostEqual(group["effective_overall_factor"], 0.85)
        self.assertAlmostEqual(group["effective_extreme_factor"], 0.55)
        self.assertEqual(artifact["artifact_sha256"], artifact_hash(artifact))
        self.assertFalse(artifact["2024_used_for_fit_or_selection"])

    def test_selection_gate_must_match_frozen_choice(self) -> None:
        gate = {
            "selected_candidate": "weekly_two_stage_linear_site_factor_shrink_a075",
            "selected_alpha": 0.75,
            "2024_used": False,
        }
        validate_selection_gate(gate, self.contract)
        gate["selected_alpha"] = 0.5
        with self.assertRaises(ValueError):
            validate_selection_gate(gate, self.contract)


if __name__ == "__main__":
    unittest.main()
