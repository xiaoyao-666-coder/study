from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import pandas as pd

from scripts.data_preparation.plan_gefs_2015_2019_exact_schedule_batched_extraction_v1 import (
    assign_year_isolated_batches,
    build_audit,
    build_contract,
)


def cycle_budget_fixture() -> pd.DataFrame:
    rows = []
    for year, dates, sizes in (
        (2015, ("2015-05-01", "2015-05-08", "2015-05-15", "2015-05-22", "2015-05-29"), (10, 11, 12, 13, 14)),
        (2019, ("2019-05-01", "2019-05-08"), (15, 16)),
    ):
        for index, (date, size) in enumerate(zip(dates, sizes), start=1):
            rows.append(
                {
                    "target_year": year,
                    "decision_date": date,
                    "required_site_count": 1 + index % 2,
                    "required_sites": "P1,P2" if index % 2 else "P1",
                    "expected_output_rows": (1 + index % 2) * 35,
                    "selected_range_bytes": size,
                    "index_network_bytes_this_run": 1,
                }
            )
    return pd.DataFrame(rows)


class YearIsolatedBatchTests(unittest.TestCase):
    def test_keeps_years_and_cycles_separate(self) -> None:
        plan, budget = assign_year_isolated_batches(
            cycle_budget_fixture(), max_cycles_per_batch=4, max_batch_bytes=100
        )

        self.assertEqual(plan["batch_id"].tolist()[:5], [
            "Y2015_B01", "Y2015_B01", "Y2015_B01", "Y2015_B01", "Y2015_B02"
        ])
        self.assertEqual(plan["batch_id"].tolist()[-2:], ["Y2019_B01", "Y2019_B01"])
        self.assertTrue(budget["cycle_count"].le(4).all())
        self.assertEqual(set(budget["target_year"]), {2015, 2019})

    def test_starts_new_batch_when_byte_limit_would_be_exceeded(self) -> None:
        plan, budget = assign_year_isolated_batches(
            cycle_budget_fixture(), max_cycles_per_batch=4, max_batch_bytes=25
        )

        self.assertEqual(plan.loc[0, "batch_id"], "Y2015_B01")
        self.assertEqual(plan.loc[2, "batch_id"], "Y2015_B02")
        self.assertTrue(budget["selected_range_bytes"].le(25).all())

    def test_rejects_a_cycle_larger_than_batch_limit(self) -> None:
        with self.assertRaisesRegex(ValueError, "single cycle"):
            assign_year_isolated_batches(
                cycle_budget_fixture(), max_cycles_per_batch=4, max_batch_bytes=9
            )

    def test_audit_reports_plan_only_gate(self) -> None:
        cycles = []
        for year, count in zip((2015, 2016, 2017, 2018, 2019), (46, 52, 43, 54, 44)):
            for number, date in enumerate(pd.date_range(f"{year}-05-01", periods=count, freq="7D"), start=1):
                cycles.append(
                    {
                        "target_year": year,
                        "decision_date": date.strftime("%Y-%m-%d"),
                        "required_site_count": 1,
                        "required_sites": "P1",
                        "expected_output_rows": 35,
                        "selected_range_bytes": 10,
                        "index_network_bytes_this_run": 1,
                    }
                )
        cycle_plan, batch_budget = assign_year_isolated_batches(
            pd.DataFrame(cycles), max_cycles_per_batch=4, max_batch_bytes=100
        )
        contract = {
            "contract_id": "test",
            "max_cycles_per_batch": 4,
            "max_payload_bytes_per_batch": 100,
            "max_payload_gib_per_batch": 100 / 1024**3,
        }
        audit = build_audit(
            cycle_plan=cycle_plan, batch_budget=batch_budget, contract=contract
        )

        self.assertTrue(audit["mandatory_structural_gate_passed"])
        self.assertFalse(audit["payload_download_started"])
        self.assertEqual(
            audit["next_gate"], "explicit_approval_required_before_batched_gefs_payload_download"
        )

    def test_contract_requires_input_hashes_and_one_worker(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            weather = root / "weather.csv"
            budget = root / "budget.csv"
            audit = root / "audit.json"
            weather.write_text("weather\n", encoding="utf-8")
            budget.write_text("budget\n", encoding="utf-8")
            audit.write_text("{}\n", encoding="utf-8")
            contract = build_contract(
                weather_plan_path=weather,
                cycle_budget_path=budget,
                preflight_audit_path=audit,
                max_cycles_per_batch=4,
                max_batch_bytes=100,
            )

        self.assertEqual(contract["payload_download_workers_required"], 1)
        self.assertTrue(contract["year_isolated"])
        self.assertEqual(set(contract["input_sha256"]), {"weather_plan", "cycle_budget", "preflight_audit"})


if __name__ == "__main__":
    unittest.main()
