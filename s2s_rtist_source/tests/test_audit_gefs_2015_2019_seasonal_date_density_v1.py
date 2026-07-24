from __future__ import annotations

import unittest

import pandas as pd

from scripts.data_preparation.audit_gefs_2015_2019_seasonal_date_density_v1 import (
    EXPECTED_SITES,
    EXPECTED_YEARS,
    build_audit,
    build_summary,
    validate_schedule,
)


def schedule_fixture(*, mature: bool = False) -> pd.DataFrame:
    decisions = pd.date_range("2015-05-18", periods=3, freq="7D")
    state_dvs = [0.1, 0.8, 2.0 if mature else 1.5]
    return pd.DataFrame(
        {
            "site_id": "P1",
            "target_year": 2015,
            "schedule_index": range(3),
            "state_checkpoint_date": decisions - pd.Timedelta(days=1),
            "state_dvs": state_dvs,
            "decision_date": decisions,
            "decision_doy": decisions.dayofyear,
            "horizon_start_date": decisions,
            "horizon_end_date": decisions + pd.Timedelta(days=6),
            "horizon_days": 7,
            "harvest_date": "2015-06-30",
            "dvs_threshold": 0.1,
            "sampling_interval_days": 7,
        }
    )


def complete_design(*, mature: bool = False) -> tuple[pd.DataFrame, pd.DataFrame]:
    schedules = []
    sources = []
    for year in EXPECTED_YEARS:
        for site in EXPECTED_SITES:
            fixture = schedule_fixture(mature=mature)
            fixture["site_id"] = site
            fixture["target_year"] = year
            for column in (
                "state_checkpoint_date",
                "decision_date",
                "horizon_start_date",
                "horizon_end_date",
                "harvest_date",
            ):
                values = pd.to_datetime(fixture[column])
                fixture[column] = values.map(lambda value: value.replace(year=year))
            fixture["decision_doy"] = pd.to_datetime(fixture["decision_date"]).dt.dayofyear
            schedules.append(validate_schedule(fixture, site_id=site, target_year=year))
            sources.append({"target_year": year, "site_id": site})
    return pd.concat(schedules, ignore_index=True), pd.DataFrame(sources)


class SeasonalDateDensityAuditTests(unittest.TestCase):
    def test_validates_teacher_schedule_rules_and_overrides_split(self) -> None:
        result = validate_schedule(schedule_fixture(), site_id="P1", target_year=2015)
        self.assertEqual(result["formal_split"].unique().tolist(), ["training"])
        self.assertEqual(result["precipitation_fit_last_year"].unique().tolist(), [2014])
        self.assertFalse(result["is_mature_checkpoint_dvs_ge_2"].any())

    def test_rejects_nonseven_day_spacing(self) -> None:
        schedule = schedule_fixture()
        schedule.loc[1, "decision_date"] = pd.Timestamp("2015-05-26")
        with self.assertRaisesRegex(ValueError, "spaced by seven days"):
            validate_schedule(schedule, site_id="P1", target_year=2015)

    def test_rejects_horizon_past_harvest(self) -> None:
        schedule = schedule_fixture()
        schedule.loc[2, "harvest_date"] = pd.Timestamp("2015-06-02")
        with self.assertRaisesRegex(ValueError, "incomplete harvest horizons"):
            validate_schedule(schedule, site_id="P1", target_year=2015)

    def test_mature_checkpoints_are_reported_and_block_weather_expansion(self) -> None:
        schedule, sources = complete_design(mature=True)
        audit = build_audit(schedule, sources)
        self.assertTrue(audit["structural_gate_passed"])
        self.assertEqual(audit["mature_checkpoint_dvs_ge_2_row_count"], 25)
        self.assertFalse(audit["weather_expansion_allowed"])
        self.assertEqual(
            audit["next_gate"],
            "confirm_whether_dvs_ge_2_checkpoints_remain_in_formal_schedule",
        )

    def test_pre_maturity_design_reports_label_and_weather_budget(self) -> None:
        schedule, sources = complete_design(mature=False)
        audit = build_audit(schedule, sources)
        self.assertTrue(audit["weather_expansion_allowed"])
        self.assertEqual(audit["decision_rows"], 75)
        self.assertEqual(audit["expected_candidate_label_rows_all_teacher_rule_dates"], 600)
        self.assertEqual(audit["expected_gefs_member_day_rows_all_teacher_rule_dates"], 2625)
        summary = build_summary(schedule)
        self.assertEqual(len(summary), 25)


if __name__ == "__main__":
    unittest.main()
