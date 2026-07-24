from __future__ import annotations

import unittest

import pandas as pd

from scripts.diagnostics.analyze_gefs_2015_2019_pilot_response_coverage_v1 import (
    IRRIGATION_OPTIONS_MM,
    analyze_candidate_labels,
)


def candidate_fixture() -> pd.DataFrame:
    rows = []
    for year, site, responsive in (
        (2015, "P1", True),
        (2015, "P2", False),
    ):
        gains = {0.0: 0.0, 10.0: 5.0} if responsive else {0.0: 0.0}
        best_ir = 10.0 if responsive else 0.0
        best_gain = 5.0 if responsive else 0.0
        for irrigation in IRRIGATION_OPTIONS_MM:
            gain = gains.get(irrigation, -0.2 * irrigation)
            rows.append(
                {
                    "target_year": year,
                    "site": site,
                    "date_t": "15-Jul-2015",
                    "ir": irrigation,
                    "cwdm_value": 1000.0 + (irrigation if responsive else 0.0),
                    "net_gain_7d": gain,
                    "best_ir_for_date": best_ir,
                    "best_target_for_date": best_gain,
                    "aet_7d_mm": 20.0 + irrigation / 60.0,
                    "soil_vwc_0_100cm_day07": 0.2 + irrigation / 6000.0,
                    "water_balance_residual_0_100cm_7d_mm": 0.1,
                    "gefs_corrected_precipitation_7d_mm": 12.0,
                }
            )
    return pd.DataFrame(rows)


class PilotResponseCoverageTests(unittest.TestCase):
    def test_summarizes_response_without_selecting_density(self) -> None:
        result = analyze_candidate_labels(
            candidate_fixture(), expected_rows=16, expected_site_cycles=2
        )
        audit = result["audit"]
        self.assertEqual(audit["responsive_site_cycle_count"], 1)
        self.assertEqual(audit["profitable_nonzero_site_cycle_count"], 1)
        self.assertFalse(audit["within_season_density_identifiable_from_this_pilot"])
        self.assertFalse(audit["date_density_parameter_selected"])
        summary = result["site_cycle_summary"].set_index("site")
        self.assertEqual(float(summary.loc["P1", "best_ir_mm"]), 10.0)
        self.assertEqual(float(summary.loc["P2", "best_ir_mm"]), 0.0)

    def test_rejects_incomplete_irrigation_curve(self) -> None:
        data = candidate_fixture().iloc[:-1].copy()
        with self.assertRaisesRegex(ValueError, "irrigation candidates"):
            analyze_candidate_labels(
                data, expected_rows=15, expected_site_cycles=2
            )


if __name__ == "__main__":
    unittest.main()
