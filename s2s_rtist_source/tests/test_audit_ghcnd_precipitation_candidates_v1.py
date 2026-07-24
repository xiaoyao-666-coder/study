from __future__ import annotations

import unittest

import numpy as np

from scripts.diagnostics.audit_ghcnd_precipitation_candidates_v1 import (
    longest_false_run,
    required_dates,
)


class GhcndCandidateAuditTests(unittest.TestCase):
    def test_required_dates_are_42_per_year_without_overlap(self) -> None:
        dates = required_dates(2000, 2019)
        self.assertEqual(len(dates), 840)
        self.assertEqual(len(set(dates)), 840)

    def test_longest_false_run(self) -> None:
        values = np.asarray([True, False, False, True, False, False, False, True])
        self.assertEqual(longest_false_run(values), 3)


if __name__ == "__main__":
    unittest.main()
