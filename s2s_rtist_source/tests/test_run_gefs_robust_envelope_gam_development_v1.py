from __future__ import annotations

import copy
import json
import unittest
from pathlib import Path

from scripts.training.run_gefs_robust_envelope_gam_development_v1 import validate_protocol


class RobustEnvelopeProtocolTests(unittest.TestCase):
    def setUp(self) -> None:
        path = Path(__file__).resolve().parents[1] / (
            "docs/superpowers/specs/"
            "2026-08-06-teacher-guided-robust-envelope-gam-development-v1.json"
        )
        self.protocol = json.loads(path.read_text(encoding="utf-8"))

    def test_frozen_protocol_validates(self) -> None:
        validate_protocol(self.protocol)
        self.assertEqual(self.protocol["validation"]["strict_outer_fold_count"], 15)
        self.assertEqual(self.protocol["validation"]["validation_cycle_count"], 196)
        self.assertEqual(self.protocol["ensemble"]["jackknife_model_count_per_fold"], 4)

    def test_protocol_rejects_2019_training(self) -> None:
        changed = copy.deepcopy(self.protocol)
        changed["forbidden"]["2019_rows_read_for_training_or_selection"] = False
        with self.assertRaisesRegex(ValueError, "2019"):
            validate_protocol(changed)

    def test_protocol_rejects_new_penalty_search(self) -> None:
        changed = copy.deepcopy(self.protocol)
        changed["penalties"]["new_search_performed"] = True
        with self.assertRaisesRegex(ValueError, "penalty"):
            validate_protocol(changed)


if __name__ == "__main__":
    unittest.main()
