from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from scripts.diagnostics.run_gefs_temporal_sampling_one_cycle_weather_smoke_v1 import (
    build_audit,
    compare_weather,
    select_smoke_cycle,
)
from s2s_rtist.weather.gefs_quantile_mapping import GEFS_REFORECAST_MEMBERS


def cycle_plan_fixture() -> pd.DataFrame:
    rows = []
    dates = pd.date_range("2015-05-01", periods=239, freq="D")
    for index, date in enumerate(dates):
        year = 2015 + min(index // 48, 4)
        shifted = date.replace(year=year)
        site_count = 2 if index in (1, 2) else 1
        rows.append(
            {
                "target_year": year,
                "decision_date": shifted.strftime("%Y-%m-%d"),
                "required_site_count": site_count,
                "required_sites": "P1,P2" if site_count == 2 else "P1",
                "expected_output_rows": site_count * len(GEFS_REFORECAST_MEMBERS) * 7,
                "selected_range_bytes": 90 if index == 2 else 100 + index,
            }
        )
    return pd.DataFrame(rows)


def weather_fixture(*, vapor_offset: float = 0.0, wind_offset: float = 0.0) -> pd.DataFrame:
    rows = []
    for lead_day in range(1, 8):
        rows.append(
            {
                "decision_date": "2015-05-03",
                "site_id": "P1",
                "gefs_member": "c00",
                "local_date": pd.Timestamp("2015-05-03") + pd.Timedelta(days=lead_day - 1),
                "lead_day": lead_day,
                "precipitation_mm_raw": 1.0,
                "temperature_min_c": 10.0,
                "temperature_max_c": 20.0,
                "actual_vapor_pressure_kpa": 1.0 + vapor_offset,
                "wind_speed_m_s": 2.0 + wind_offset,
                "solar_kj_m2_day": 8000.0,
            }
        )
    return pd.DataFrame(rows)


class TemporalSamplingOneCycleSmokeTests(unittest.TestCase):
    def test_selects_maximum_site_training_cycle_then_smallest_payload(self) -> None:
        chosen = select_smoke_cycle(cycle_plan_fixture())

        self.assertEqual(int(chosen["target_year"]), 2015)
        self.assertEqual(int(chosen["required_site_count"]), 2)
        self.assertEqual(int(chosen["selected_range_bytes"]), 90)

    def test_comparison_preserves_exact_variables_and_measures_state_errors(self) -> None:
        full = weather_fixture()
        lean = weather_fixture(vapor_offset=0.1, wind_offset=-0.2)
        comparison, metrics = compare_weather(full, lean)
        indexed = metrics.set_index("variable")

        self.assertEqual(len(comparison), 7)
        self.assertTrue(bool(indexed.loc["precipitation_mm_raw", "exact_match"]))
        self.assertAlmostEqual(
            float(indexed.loc["actual_vapor_pressure_kpa", "maximum_absolute_error"]),
            0.1,
        )
        self.assertAlmostEqual(
            float(indexed.loc["wind_speed_m_s", "mean_signed_error"]), -0.2
        )

    def test_rejects_mismatched_weather_keys(self) -> None:
        full = weather_fixture()
        lean = weather_fixture().iloc[:-1]

        with self.assertRaisesRegex(ValueError, "keys differ"):
            compare_weather(full, lean)

    def test_audit_requires_teacher_review_and_blocks_training(self) -> None:
        full = weather_fixture()
        lean = weather_fixture(vapor_offset=0.01)
        _, metrics = compare_weather(full, lean)
        chosen = pd.Series(
            {
                "target_year": 2015,
                "decision_date": "2015-05-03",
                "required_sites": "P1",
                "expected_output_rows": 7,
                "selected_range_bytes": 100,
            }
        )
        audit = build_audit(
            chosen=chosen,
            full_weather=full,
            lean_weather=lean,
            metrics=metrics,
            extraction_audit={
                "status": "full_weather_local_extraction_passed",
                "retained_grib_file_count": 0,
                "network_bytes_this_run": 100,
            },
        )

        self.assertTrue(audit["mandatory_structural_gate_passed"])
        self.assertTrue(audit["teacher_review_required"])
        self.assertFalse(audit["weather_equivalence_approved"])
        self.assertFalse(audit["training_eligible"])
        self.assertTrue(np.isfinite(metrics["maximum_absolute_error"]).all())


if __name__ == "__main__":
    unittest.main()
