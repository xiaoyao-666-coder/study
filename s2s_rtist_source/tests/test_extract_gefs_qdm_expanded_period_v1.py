from __future__ import annotations

import unittest

from scripts.data_preparation.extract_gefs_qdm_expanded_period_v1 import cycle_dates


class ExpandedPeriodExtractionTests(unittest.TestCase):
    def test_three_year_smoke_has_expected_cycle_count(self) -> None:
        dates = cycle_dates(2000, 2002)
        self.assertEqual(len(dates), 18)
        self.assertEqual(dates[0], "2000-06-01")
        self.assertEqual(dates[-1], "2002-08-15")

    def test_invalid_year_range_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "must not exceed"):
            cycle_dates(2002, 2000)


if __name__ == "__main__":
    unittest.main()
