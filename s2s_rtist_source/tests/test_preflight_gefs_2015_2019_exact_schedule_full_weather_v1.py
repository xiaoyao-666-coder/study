from __future__ import annotations

import unittest

import pandas as pd

from scripts.data_preparation.preflight_gefs_2015_2019_exact_schedule_full_weather_v1 import (
    build_audit,
    build_budget_tables,
    build_cycle_plan,
    validate_weather_plan,
)


def plan_fixture() -> pd.DataFrame:
    rows = []
    for site, decision in (("P1", "2015-05-18"), ("P2", "2015-05-18"), ("P1", "2015-05-25")):
        date = pd.Timestamp(decision)
        rows.append(
            {
                "target_year": 2015,
                "formal_split": "training",
                "site_id": site,
                "decision_date": date,
                "state_checkpoint_date": date - pd.Timedelta(days=1),
                "state_dvs": 0.5,
                "horizon_end_date": date + pd.Timedelta(days=6),
                "harvest_date": "2015-08-31",
                "precipitation_fit_last_year": 2014,
                "is_mature_checkpoint_dvs_ge_2": False,
                "expected_gefs_member_day_rows": 35,
            }
        )
    return pd.DataFrame(rows)


def preflight_fixture() -> pd.DataFrame:
    rows = []
    for cycle in ("2015-05-18", "2015-05-25"):
        for member in ("c00", "p01"):
            for product in ("apcp_sfc", "tmp_2m"):
                rows.append(
                    {
                        "cycle_date": cycle,
                        "gefs_member": member,
                        "product_id": product,
                        "selected_range_bytes": 100,
                        "index_network_bytes_this_run": 10,
                    }
                )
    return pd.DataFrame(rows)


class ExactScheduleFullWeatherPreflightTests(unittest.TestCase):
    def test_validates_causal_exact_schedule_plan(self) -> None:
        result = validate_weather_plan(
            plan_fixture(), expected_site_cycle_rows=3, expected_unique_cycles=2,
            expected_years=(2015,), expected_sites=("P1", "P2")
        )
        self.assertEqual(len(result), 3)

    def test_rejects_noncausal_precipitation_fit_boundary(self) -> None:
        plan = plan_fixture()
        plan.loc[0, "precipitation_fit_last_year"] = 2015
        with self.assertRaisesRegex(ValueError, "not causal"):
            validate_weather_plan(
                plan, expected_site_cycle_rows=3, expected_unique_cycles=2,
                expected_years=(2015,), expected_sites=("P1", "P2")
            )

    def test_cycle_plan_deduplicates_shared_site_dates(self) -> None:
        plan = validate_weather_plan(
            plan_fixture(), expected_site_cycle_rows=3, expected_unique_cycles=2,
            expected_years=(2015,), expected_sites=("P1", "P2")
        )
        cycles = build_cycle_plan(plan)
        self.assertEqual(len(cycles), 2)
        self.assertEqual(int(cycles.iloc[0]["required_site_count"]), 2)
        self.assertEqual(int(cycles.iloc[0]["expected_output_rows"]), 70)

    def test_budget_tables_aggregate_cycle_and_year_bytes(self) -> None:
        plan = validate_weather_plan(
            plan_fixture(), expected_site_cycle_rows=3, expected_unique_cycles=2,
            expected_years=(2015,), expected_sites=("P1", "P2")
        )
        cycle_plan = build_cycle_plan(plan)
        cycle, year = build_budget_tables(preflight_fixture(), cycle_plan)
        self.assertEqual(cycle["selected_range_bytes"].tolist(), [400, 400])
        self.assertEqual(int(year.iloc[0]["selected_range_bytes"]), 800)

    def test_audit_reports_preflight_without_payload_download(self) -> None:
        plan = validate_weather_plan(
            plan_fixture(), expected_site_cycle_rows=3, expected_unique_cycles=2,
            expected_years=(2015,), expected_sites=("P1", "P2")
        )
        cycle_plan = build_cycle_plan(plan)
        preflight = preflight_fixture()
        cycle_budget, _ = build_budget_tables(preflight, cycle_plan)
        inventory = pd.DataFrame(
            {
                "cycle_date": ["2015-05-18"] * 2 + ["2015-05-25"] * 2,
                "gefs_member": ["c00", "p01", "c00", "p01"],
                "network_bytes_this_run": [5, 5, 5, 5],
            }
        )
        audit = build_audit(
            plan=plan,
            cycle_plan=cycle_plan,
            preflight=preflight,
            inventory=inventory,
            cycle_budget=cycle_budget,
        )
        self.assertFalse(audit["product_payload_download_started"])
        self.assertEqual(audit["selected_range_bytes"], 800)


if __name__ == "__main__":
    unittest.main()
