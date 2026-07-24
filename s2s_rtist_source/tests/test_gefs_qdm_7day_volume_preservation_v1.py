from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from scripts.diagnostics.diagnose_gefs_qdm_7day_volume_preservation_v1 import (
    volume_preserve_group,
)


class SevenDayVolumePreservationTests(unittest.TestCase):
    def test_positive_qdm_is_scaled_to_raw_total(self) -> None:
        frame = pd.DataFrame(
            {
                "valid_date_utc": pd.date_range("2018-06-01", periods=7),
                "precipitation_mm_raw": [0, 1, 2, 0, 3, 0, 4],
                "precipitation_mm_qm": [0, 2, 1, 0, 1, 0, 1],
            }
        )
        corrected, audit = volume_preserve_group(frame, tolerance_mm=1e-8)
        self.assertAlmostEqual(float(corrected.sum()), 10.0)
        self.assertAlmostEqual(audit["scale_factor"], 2.0)
        self.assertFalse(audit["fallback_to_raw"])
        self.assertTrue((corrected >= 0.0).all())

    def test_zero_qdm_with_positive_raw_falls_back_to_raw(self) -> None:
        raw = np.asarray([0, 1, 0, 2, 0, 3, 0], dtype=float)
        frame = pd.DataFrame(
            {
                "valid_date_utc": pd.date_range("2018-06-01", periods=7),
                "precipitation_mm_raw": raw,
                "precipitation_mm_qm": np.zeros(7),
            }
        )
        corrected, audit = volume_preserve_group(frame, tolerance_mm=1e-8)
        np.testing.assert_allclose(corrected, raw)
        self.assertTrue(audit["fallback_to_raw"])


if __name__ == "__main__":
    unittest.main()
