import json
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.training.run_gefs_gam_v8_unlimited_preserving_confidence_shrinkage_replay_v1 import (
    CONFIDENCE_REFERENCE_LCB_MM,
    apply_unlimited_preserving_confidence_shrinkage,
    validate_protocol,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROTOCOL_PATH = PROJECT_ROOT / (
    "docs/superpowers/specs/"
    "2026-08-08-teacher-guided-gam-v8-unlimited-preserving-"
    "confidence-shrinkage-replay-v1.json"
)


def equation_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "site_id": ["P1", "P2", "P3", "P4", "P15"],
            "target_year": [2016, 2017, 2018, 2018, 2016],
            "anchor_recommendation_mm": [10.0] * 5,
            "recommendation_mm": [12.5, 12.5, 7.5, 12.5, 10.0],
            "v5_movement_from_anchor_mm": [2.5, 2.5, -2.5, 2.5, 0.0],
            "selected_lcb_mm": [-1.0, 2.5, 5.0, 10.0, 4.0],
            "trust_region_limited": [False, True, True, True, False],
            "true_peak_mm": [13.0, 11.0, 8.0, 13.0, 10.0],
            "true_oracle_is_positive": [True, True, True, True, False],
        }
    )


class GamV8UnlimitedPreservingConfidenceShrinkageReplayTests(unittest.TestCase):
    def test_protocol_freezes_hybrid_rule_and_boundaries(self) -> None:
        protocol = json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))
        validate_protocol(protocol)

        self.assertEqual(CONFIDENCE_REFERENCE_LCB_MM, 5.0)
        self.assertEqual(
            protocol["selection_rule"]["confidence_reference_lcb_mm"], 5.0
        )
        self.assertEqual(protocol["selection_rule"]["limited_scale_clip"], [0.0, 1.0])
        self.assertEqual(protocol["evaluation_boundary"]["cycle_count"], 196)
        self.assertEqual(protocol["evaluation_boundary"]["outer_fold_count"], 15)
        self.assertEqual(protocol["evaluation_boundary"]["target_site_count"], 5)
        self.assertFalse(protocol["evaluation_boundary"]["2019_access"])
        self.assertFalse(protocol["evaluation_boundary"]["2024_access"])

        protocol["selection_rule"]["confidence_reference_lcb_mm"] = 4.0
        with self.assertRaisesRegex(ValueError, "confidence reference"):
            validate_protocol(protocol)

        protocol = json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))
        protocol["evaluation_boundary"]["2019_access"] = True
        with self.assertRaisesRegex(ValueError, "isolation"):
            validate_protocol(protocol)

    def test_unlimited_is_preserved_and_only_limited_actions_shrink(self) -> None:
        replay = apply_unlimited_preserving_confidence_shrinkage(equation_frame())

        np.testing.assert_allclose(
            replay["v8_confidence_scale"], [1.0, 0.5, 1.0, 1.0, 0.0]
        )
        np.testing.assert_allclose(
            replay["v8_recommendation_mm"], [12.5, 11.25, 7.5, 12.5, 10.0]
        )
        np.testing.assert_allclose(
            replay["recommendation_mm"], replay["v8_recommendation_mm"]
        )
        self.assertEqual(
            replay["irrigation_changed"].tolist(),
            [True, True, True, True, False],
        )

    def test_identity_and_truth_fields_cannot_change_recommendation(self) -> None:
        base = equation_frame()
        changed = base.copy()
        changed["site_id"] = ["X1", "X2", "X3", "X4", "X5"]
        changed["target_year"] = [2091, 2092, 2093, 2094, 2095]
        changed["true_peak_mm"] = [60.0, 0.0, 60.0, 0.0, 60.0]
        changed["true_oracle_is_positive"] = [False, False, False, False, True]

        np.testing.assert_allclose(
            apply_unlimited_preserving_confidence_shrinkage(base)[
                "v8_recommendation_mm"
            ],
            apply_unlimited_preserving_confidence_shrinkage(changed)[
                "v8_recommendation_mm"
            ],
        )

    def test_inconsistent_frozen_v5_movement_is_rejected(self) -> None:
        frame = equation_frame().iloc[[0]].copy()
        frame["v5_movement_from_anchor_mm"] = 99.0

        with self.assertRaisesRegex(ValueError, "v5 movement"):
            apply_unlimited_preserving_confidence_shrinkage(frame)


if __name__ == "__main__":
    unittest.main()
