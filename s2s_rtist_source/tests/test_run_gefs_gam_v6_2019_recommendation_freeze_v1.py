import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pandas as pd

from scripts.training.run_gefs_gam_v6_2019_recommendation_freeze_v1 import (
    EXPECTED_CYCLE_COUNT,
    LIMITED_LCB_THRESHOLD_MM,
    TRUST_REGION_RADIUS_MM,
    apply_frozen_v5_v6_rule,
    inherit_rolling_2018_penalties,
    load_feature_rows_without_targets,
    validate_protocol,
)


PROTOCOL_PATH = Path(
    "docs/superpowers/specs/2026-08-07-teacher-guided-gam-v6-2019-independent-confirmation-v1.json"
)


class GamV62019RecommendationFreezeTests(unittest.TestCase):
    def test_protocol_freezes_two_stage_boundary(self) -> None:
        protocol = json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))
        validate_protocol(protocol)
        self.assertEqual(EXPECTED_CYCLE_COUNT, 69)
        self.assertEqual(TRUST_REGION_RADIUS_MM, 2.5)
        self.assertEqual(LIMITED_LCB_THRESHOLD_MM, 5.0)
        self.assertEqual(protocol["data_boundary"]["fit_years"], [2015, 2016, 2017, 2018])
        self.assertEqual(protocol["data_boundary"]["confirmation_year"], 2019)
        self.assertEqual(protocol["expert_refit"]["penalty_source_fold_suffix"], "rolling_to_2018")
        self.assertFalse(protocol["selection_rule"]["threshold_search"])
        self.assertFalse(protocol["data_boundary"]["2024_access"])

    def test_protocol_rejects_threshold_or_2024_changes(self) -> None:
        protocol = json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))
        protocol["selection_rule"]["limited_candidate_required_lcb_mm"] = 4.0
        with self.assertRaisesRegex(ValueError, "LCB"):
            validate_protocol(protocol)

        protocol = json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))
        protocol["data_boundary"]["2024_access"] = True
        with self.assertRaisesRegex(ValueError, "2024"):
            validate_protocol(protocol)

    def test_target_blind_loader_never_parses_2019_targets(self) -> None:
        with TemporaryDirectory() as temporary:
            path = Path(temporary) / "dataset.csv"
            rows = []
            for year, site in ((2018, "P2"), (2019, "P1")):
                for irrigation in (0.0, 10.0):
                    rows.append({
                        "sample_id": f"{year}-{site}-{irrigation}",
                        "target_year": year,
                        "site_id": site,
                        "decision_date": f"{year}-06-01",
                        "irrigation_mm": irrigation,
                        "feature_a": irrigation + year,
                        "target_net_gain_7d": "SEALED",
                    })
            pd.DataFrame(rows).to_csv(path, index=False)
            index = pd.read_csv(path, usecols=["sample_id", "target_year", "site_id"])
            selected = index["target_year"].eq(2019)
            loaded = load_feature_rows_without_targets(
                path,
                index,
                selected,
                feature_columns=["feature_a"],
            )
        self.assertEqual(len(loaded), 2)
        self.assertNotIn("target_net_gain_7d", loaded.columns)
        self.assertEqual(loaded["target_year"].unique().tolist(), [2019])

    def test_penalties_are_inherited_from_exact_rolling_2018_fold(self) -> None:
        with TemporaryDirectory() as temporary:
            path = Path(temporary) / "folds.csv"
            pd.DataFrame([
                {"fold_id": "holdout_P1_rolling_to_2017", "variant": "GAM-VC", "lambda_main": 1.0, "lambda_site": 2.0, "lambda_interaction": 3.0},
                {"fold_id": "holdout_P1_rolling_to_2018", "variant": "GAM-VC", "lambda_main": 4.0, "lambda_site": 5.0, "lambda_interaction": 6.0},
            ]).to_csv(path, index=False)
            penalties = inherit_rolling_2018_penalties(path, "P1")
        self.assertEqual(
            penalties,
            {"lambda_interaction": 6.0, "lambda_main": 4.0, "lambda_site": 5.0},
        )

    def test_frozen_v5_v6_rule_covers_all_actions(self) -> None:
        frame = pd.DataFrame({
            "anchor_recommendation_mm": [10.0, 20.0, 30.0, 40.0],
            "gate_recommendation_mm": [10.0, 21.0, 36.0, 34.0],
            "selected_lcb_mm": [100.0, 0.1, 5.0, 4.999999],
        })
        selected = apply_frozen_v5_v6_rule(frame)
        np.testing.assert_allclose(
            selected["v5_recommendation_mm"],
            [10.0, 21.0, 32.5, 37.5],
        )
        self.assertEqual(selected["use_v6"].tolist(), [False, True, True, False])
        np.testing.assert_allclose(
            selected["selected_irrigation_mm"],
            [10.0, 21.0, 32.5, 40.0],
        )
        self.assertEqual(
            selected["selection_reason"].tolist(),
            [
                "gate_unchanged_anchor_endpoint",
                "v5_unlimited_endpoint",
                "v5_limited_lcb_passed",
                "v5_limited_lcb_failed_anchor_fallback",
            ],
        )

    def test_site_and_year_fields_cannot_change_frozen_rule(self) -> None:
        base = pd.DataFrame({
            "anchor_recommendation_mm": [10.0, 10.0],
            "gate_recommendation_mm": [16.0, 16.0],
            "selected_lcb_mm": [4.0, 6.0],
            "site_id": ["P1", "P2"],
            "target_year": [2019, 2019],
        })
        changed = base.copy()
        changed["site_id"] = ["P4", "P15"]
        changed["target_year"] = [2090, 2091]
        self.assertEqual(
            apply_frozen_v5_v6_rule(base)["use_v6"].tolist(),
            apply_frozen_v5_v6_rule(changed)["use_v6"].tolist(),
        )


if __name__ == "__main__":
    unittest.main()
