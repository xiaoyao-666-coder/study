from __future__ import annotations

import unittest

import pandas as pd

from scripts.data_preparation.extract_era5_nonprecip_reference_for_gefs_v1 import (
    convert_era5_reference,
    validate_reference_coverage,
)


class Era5NonprecipReferenceForGefsTests(unittest.TestCase):
    def test_unit_conversion(self) -> None:
        source = pd.DataFrame(
            {
                "target_year": [2015],
                "site_id": ["P1"],
                "local_date": ["2015-01-01"],
                "temperature_min_k": [278.15],
                "temperature_max_k": [288.15],
                "dewpoint_k": [280.15],
                "solar_j_m2_day": [20_000_000.0],
                "wind_u_m_s": [3.0],
                "wind_v_m_s": [4.0],
            }
        )
        converted = convert_era5_reference(source).iloc[0]
        self.assertAlmostEqual(float(converted["temperature_min_c"]), 5.0)
        self.assertAlmostEqual(float(converted["temperature_max_c"]), 15.0)
        self.assertAlmostEqual(float(converted["solar_kj_m2_day"]), 20_000.0)
        self.assertAlmostEqual(float(converted["wind_speed_m_s"]), 5.0)

    def test_exact_reference_coverage_passes(self) -> None:
        keys = pd.DataFrame(
            {
                "decision_date": ["2015-01-15"],
                "lead_day": [1],
                "target_year": [2015],
                "site_id": ["P1"],
                "local_date": ["2015-01-15"],
            }
        )
        reference = pd.DataFrame(
            {
                "target_year": [2015],
                "site_id": ["P1"],
                "local_date": ["2015-01-15"],
                "temperature_min_c": [1.0],
                "temperature_max_c": [2.0],
                "actual_vapor_pressure_kpa": [0.5],
                "wind_speed_m_s": [3.0],
                "solar_kj_m2_day": [10_000.0],
            }
        )
        output, audit = validate_reference_coverage(keys, reference)
        self.assertEqual(len(output), 1)
        self.assertTrue(audit["mandatory_gate_passed"])

    def test_missing_reference_is_recorded(self) -> None:
        keys = pd.DataFrame(
            {
                "decision_date": ["2015-01-15"],
                "lead_day": [1],
                "target_year": [2015],
                "site_id": ["P1"],
                "local_date": ["2015-01-15"],
            }
        )
        reference = pd.DataFrame(
            columns=[
                "target_year",
                "site_id",
                "local_date",
                "temperature_min_c",
                "temperature_max_c",
                "actual_vapor_pressure_kpa",
                "wind_speed_m_s",
                "solar_kj_m2_day",
            ]
        )
        _, audit = validate_reference_coverage(keys, reference)
        self.assertFalse(audit["mandatory_gate_passed"])
        self.assertEqual(audit["missing_canonical_value_count"], 5)


if __name__ == "__main__":
    unittest.main()
