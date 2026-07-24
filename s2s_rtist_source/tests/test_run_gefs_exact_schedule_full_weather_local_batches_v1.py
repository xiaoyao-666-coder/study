from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import pandas as pd

from scripts.data_preparation.run_gefs_exact_schedule_full_weather_local_batches_v1 import (
    cycle_is_complete,
    read_batch_and_cycle_plan,
)


def plan_fixture() -> tuple[pd.DataFrame, pd.DataFrame]:
    batch = pd.DataFrame(
        {
            "batch_id": ["Y2015_B01"] * 61,
            "target_year": [2015] * 61,
            "cycle_count": [1] * 61,
            "first_decision_date": ["2015-07-06"] * 61,
            "last_decision_date": ["2015-07-06"] * 61,
        }
    )
    cycles = pd.DataFrame(
        {
            "batch_id": ["Y2015_B01"] * 239,
            "target_year": [2015] * 239,
            "decision_date": pd.date_range("2015-01-01", periods=239).strftime("%Y-%m-%d"),
            "required_site_count": [1] * 239,
            "required_sites": ["P1"] * 239,
            "expected_output_rows": [35] * 239,
        }
    )
    return batch, cycles


class FullWeatherBatchRunnerTests(unittest.TestCase):
    def test_rejects_invalid_global_plan_fixture(self) -> None:
        batch, cycles = plan_fixture()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            batch_path = root / "batch.csv"
            cycle_path = root / "cycle.csv"
            batch.to_csv(batch_path, index=False)
            cycles.to_csv(cycle_path, index=False)
            with self.assertRaisesRegex(ValueError, "expected one batch row"):
                read_batch_and_cycle_plan(batch_path, cycle_path, "Y2015_B01")

    def test_cycle_complete_requires_audited_rows_and_no_grib(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            audit = {
                "status": "full_weather_local_extraction_passed",
                "row_count": 35,
                "retained_grib_file_count": 0,
            }
            (root / "gefs_2015_2019_full_weather_audit_v1.json").write_text(
                __import__("json").dumps(audit), encoding="utf-8"
            )
            pd.DataFrame({"row": range(35)}).to_csv(
                root / "gefs_2015_2019_full_weather_member_daily_v1.csv", index=False
            )
            self.assertTrue(cycle_is_complete(root, 35))
            (root / "temporary.grib2").write_bytes(b"x")
            self.assertFalse(cycle_is_complete(root, 35))


if __name__ == "__main__":
    unittest.main()
