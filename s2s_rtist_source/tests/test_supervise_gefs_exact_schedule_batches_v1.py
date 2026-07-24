from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pandas as pd

from scripts.data_preparation.supervise_gefs_exact_schedule_batches_v1 import (
    active_batch_ids,
    batch_is_strictly_complete,
)


class GefsBatchSupervisorTests(unittest.TestCase):
    def test_reused_pid_is_not_treated_as_an_active_batch(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            run_root = Path(directory)
            pid_path = run_root / "Y2015_B01.pid"
            pid_path.write_text("1234\n", encoding="ascii")
            pid_file_time = pid_path.stat().st_mtime
            with patch(
                "scripts.data_preparation.supervise_gefs_exact_schedule_batches_v1.process_is_running",
                return_value=True,
            ), patch(
                "scripts.data_preparation.supervise_gefs_exact_schedule_batches_v1.process_creation_time",
                return_value=pid_file_time + 60.0,
            ):
                self.assertEqual(active_batch_ids(run_root, ["Y2015_B01"]), set())
            with patch(
                "scripts.data_preparation.supervise_gefs_exact_schedule_batches_v1.process_is_running",
                return_value=True,
            ), patch(
                "scripts.data_preparation.supervise_gefs_exact_schedule_batches_v1.process_creation_time",
                return_value=pid_file_time - 1.0,
            ):
                self.assertEqual(
                    active_batch_ids(run_root, ["Y2015_B01"]), {"Y2015_B01"}
                )

    def test_strict_completion_requires_clean_batch_and_cycle_audits(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output_root = root / "output"
            run_root = output_root / "_run"
            batch_dir = output_root / "Y2015_B01"
            cycle_dir = batch_dir / "20150706"
            cycle_dir.mkdir(parents=True)
            run_root.mkdir()
            batch = SimpleNamespace(
                batch_id="Y2015_B01", cycle_count=1, expected_output_rows=35
            )
            cycles = pd.DataFrame(
                {"decision_date": ["2015-07-06"], "expected_output_rows": [35]}
            )
            batch_audit = {
                "status": "exact_schedule_full_weather_local_batch_passed",
                "mandatory_structural_gate_passed": True,
                "full_three_hourly_records": True,
                "all_required_weather_variables_retained": True,
                "temporary_grib_retained": False,
                "member_count": 5,
                "cycle_count": 1,
                "completed_cycle_count": 1,
                "expected_rows": 35,
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
            (run_root / "Y2015_B01.workers8_ranges4.stderr.log").write_text(
                "", encoding="utf-8"
            )
            self.assertTrue(batch_is_strictly_complete(output_root, batch, cycles, run_root))
            (cycle_dir / "active.grib2").write_bytes(b"x")
            self.assertFalse(batch_is_strictly_complete(output_root, batch, cycles, run_root))


if __name__ == "__main__":
    unittest.main()
