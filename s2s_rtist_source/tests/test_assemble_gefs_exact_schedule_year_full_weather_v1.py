from __future__ import annotations

import argparse
import json
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from scripts.data_preparation.assemble_gefs_exact_schedule_year_full_weather_v1 import (
    run,
)
from s2s_rtist.weather.gefs_quantile_mapping import GEFS_REFORECAST_MEMBERS


def _write_fixture(root: Path, *, corrupt_key: bool = False) -> argparse.Namespace:
    batch_budget = root / "batch_budget.csv"
    cycle_plan = root / "cycle_plan.csv"
    output_root = root / "batches"
    output_dir = root / "year"
    pd.DataFrame(
        {
            "batch_id": ["Y2015_B01"],
            "target_year": [2015],
            "cycle_count": [1],
            "expected_output_rows": [35],
        }
    ).to_csv(batch_budget, index=False)
    pd.DataFrame(
        {
            "batch_id": ["Y2015_B01"],
            "target_year": [2015],
            "decision_date": ["2015-07-06"],
            "required_site_count": [1],
            "required_sites": ["P1"],
            "expected_member_count": [5],
            "expected_lead_day_count": [7],
            "expected_output_rows": [35],
        }
    ).to_csv(cycle_plan, index=False)
    batch_dir = output_root / "Y2015_B01"
    cycle_dir = batch_dir / "20150706"
    cycle_dir.mkdir(parents=True)
    batch_audit = {
        "status": "exact_schedule_full_weather_local_batch_passed",
        "mandatory_structural_gate_passed": True,
        "full_three_hourly_records": True,
        "all_required_weather_variables_retained": True,
        "member_count": 5,
        "temporary_grib_retained": False,
    }
    (batch_dir / "gefs_exact_schedule_batch_full_weather_audit_v1.json").write_text(
        json.dumps(batch_audit), encoding="utf-8"
    )
    cycle_audit = {
        "status": "full_weather_local_extraction_passed",
        "row_count": 35,
        "expected_row_count": 35,
        "member_count": 5,
        "canonical_missing_value_count": 0,
        "canonical_nonfinite_value_count": 0,
        "duplicate_sample_key_count": 0,
        "retained_grib_file_count": 0,
    }
    (cycle_dir / "gefs_2015_2019_full_weather_audit_v1.json").write_text(
        json.dumps(cycle_audit), encoding="utf-8"
    )
    decision = pd.Timestamp("2015-07-06")
    rows = []
    for member in GEFS_REFORECAST_MEMBERS:
        for lead in range(1, 8):
            rows.append(
                {
                    "decision_date": "2015-07-06",
                    "site_id": "P1",
                    "gefs_member": member,
                    "local_date": (decision + pd.Timedelta(days=lead - 1)).strftime("%Y-%m-%d"),
                    "lead_day": lead,
                    "precipitation_mm_raw": 1.0,
                    "temperature_min_c": 15.0,
                    "temperature_max_c": 25.0,
                    "actual_vapor_pressure_kpa": 1.2,
                    "wind_speed_m_s": 2.0,
                    "solar_kj_m2_day": 18000.0,
                }
            )
    if corrupt_key:
        rows[-1]["local_date"] = "2015-07-06"
    pd.DataFrame(rows).to_csv(
        cycle_dir / "gefs_2015_2019_full_weather_member_daily_v1.csv", index=False
    )
    return argparse.Namespace(
        batch_budget=batch_budget,
        cycle_plan=cycle_plan,
        output_root=output_root,
        target_year=2015,
        output_dir=output_dir,
        resume=False,
    )


class ExactScheduleYearAssemblyTests(unittest.TestCase):
    def test_merges_complete_year_and_writes_strict_audit(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            args = _write_fixture(Path(directory))
            outputs = run(args)
            audit = json.loads(outputs["audit"].read_text(encoding="utf-8"))
            self.assertEqual(audit["status"], "exact_schedule_year_raw_full_weather_passed")
            self.assertTrue(audit["mandatory_year_gate_passed"])
            self.assertEqual(audit["row_count"], 35)
            self.assertEqual(audit["weather_variable_count"], 6)
            self.assertFalse(audit["weather_correction_applied"])
            self.assertFalse(audit["label_generation_performed"])

    def test_rejects_cycle_with_wrong_exact_key_coverage(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            args = _write_fixture(Path(directory), corrupt_key=True)
            with self.assertRaisesRegex(ValueError, "exact sample-key coverage mismatch"):
                run(args)

    def test_ignores_temporary_grib_from_another_active_year(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            args = _write_fixture(Path(directory))
            other_year = args.output_root / "Y2016_B01" / "20160518"
            other_year.mkdir(parents=True)
            (other_year / "active.grib2").write_bytes(b"still downloading")
            outputs = run(args)
            audit = json.loads(outputs["audit"].read_text(encoding="utf-8"))
            self.assertTrue(audit["mandatory_year_gate_passed"])


if __name__ == "__main__":
    unittest.main()
