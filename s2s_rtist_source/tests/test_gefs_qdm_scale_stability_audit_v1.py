from __future__ import annotations

import unittest

from scripts.diagnostics.audit_gefs_qdm_scale_stability_v1 import classify_scale


class ScaleStabilityAuditTests(unittest.TestCase):
    def test_scale_bins_have_locked_boundaries(self) -> None:
        self.assertEqual(classify_scale(float("nan"), True), "fallback_raw")
        self.assertEqual(classify_scale(2.0, False), "le_2")
        self.assertEqual(classify_scale(2.01, False), "gt_2_le_5")
        self.assertEqual(classify_scale(5.0, False), "gt_2_le_5")
        self.assertEqual(classify_scale(5.01, False), "gt_5_le_10")
        self.assertEqual(classify_scale(10.01, False), "gt_10_le_20")
        self.assertEqual(classify_scale(20.01, False), "gt_20")


if __name__ == "__main__":
    unittest.main()
