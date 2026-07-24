from __future__ import annotations

import unittest

from scripts.diagnostics.run_gefs_qm_qdm_expanding_cv_2000_2018_v1 import (
    expanding_folds,
)


class ExpandingWindowCvTests(unittest.TestCase):
    def test_folds_are_temporally_causal(self) -> None:
        folds = expanding_folds()
        self.assertEqual([fold["validation_year"] for fold in folds], [2015, 2016, 2017, 2018])
        for fold in folds:
            self.assertEqual(min(fold["fit_years"]), 2000)
            self.assertEqual(max(fold["fit_years"]), fold["validation_year"] - 1)
            self.assertNotIn(fold["validation_year"], fold["fit_years"])


if __name__ == "__main__":
    unittest.main()
