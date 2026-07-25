from __future__ import annotations

import unittest
import tempfile
import json
from pathlib import Path

import pandas as pd

from scripts.simulation.run_gefs_exact_schedule_formal_label_generation_v1 import (
    FORMAL_NUMERICAL_ENDPOINT_FALLBACKS_MM,
    FORMAL_RESTART_NPRINTDAY,
    FORMAL_SWAP_DTMAX_DAYS,
    archive_nonreusable_branch,
    archive_stale_aggregate_outputs,
    build_formal_audit,
    build_formal_plan,
    ensure_formal_endpoint_metadata,
    import_verified_site_year_seed,
    load_passed_branch,
    validate_schedule_match,
    write_aggregate_outputs,
)
from scripts.simulation.run_gefs_checkpoint_five_site_eight_ir_smoke_v1 import (
    SITE_ORDER,
)
from scripts.simulation.run_gefs_checkpoint_one_date_eight_ir_smoke_v1 import (
    IRRIGATION_OPTIONS_MM,
)


SITE_DATES = {
    "P1": "2015-07-06",
    "P2": "2015-07-06",
    "P3": "2015-07-05",
    "P4": "2015-07-04",
    "P15": "2015-07-06",
}


def weather_fixture() -> pd.DataFrame:
    rows = []
    for site, date_text in SITE_DATES.items():
        decision = pd.Timestamp(date_text)
        for member in ("c00", "p01", "p02", "p03", "p04"):
            for lead in range(1, 8):
                rows.append(
                    {
                        "target_year": 2015,
                        "site_id": site,
                        "decision_date": date_text,
                        "gefs_member": member,
                        "local_date": (decision + pd.Timedelta(days=lead - 1)).strftime("%Y-%m-%d"),
                        "lead_day": lead,
                        "temperature_min_c": 10.0,
                        "temperature_max_c": 20.0,
                        "actual_vapor_pressure_kpa": 1.0,
                        "wind_speed_m_s": 2.0,
                        "solar_kj_m2_day": 10000.0,
                        "precipitation_mm": 1.0,
                    }
                )
    return pd.DataFrame(rows)


def candidate_fixture(plan: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for item in plan.itertuples(index=False):
        for irrigation in IRRIGATION_OPTIONS_MM:
            rows.append(
                {
                    "target_year": item.target_year,
                    "site": item.site_id,
                    "date_t": pd.Timestamp(item.decision_date).strftime("%d-%b-%Y"),
                    "decision_date": item.decision_date,
                    "ir": irrigation,
                    "is_best_ir": irrigation == 20.0,
                    "net_gain_7d": 10.0 if irrigation == 20.0 else 0.0,
                    "cwdm_value": irrigation,
                    "swap_dtmax_days": FORMAL_SWAP_DTMAX_DAYS,
                }
            )
    return pd.DataFrame(rows)


def audit_fixture(plan: pd.DataFrame) -> list[dict[str, object]]:
    return [
        {
            "target_year": int(item.target_year),
            "site_id": str(item.site_id),
            "decision_date": str(item.decision_date),
            "mandatory_gate_passed": True,
            "maximum_absolute_checkpoint_crop_state_error": 0.0,
            "maximum_absolute_checkpoint_profile_state_error": 0.0,
            "maximum_absolute_swap_rain_error_mm": 0.001,
            "maximum_absolute_water_balance_residual_mm": 0.3,
            "primary_output_missing_value_count": 0,
            "prestate_swap_rerun_count": 0,
            "restart_nprintday": FORMAL_RESTART_NPRINTDAY,
            "swap_dtmax_days": FORMAL_SWAP_DTMAX_DAYS,
        }
        for item in plan.itertuples(index=False)
    ]


class FormalExactScheduleGenerationTests(unittest.TestCase):
    def test_plan_preserves_site_specific_dates(self) -> None:
        plan = build_formal_plan(
            weather_fixture(),
            expected_years=(2015,),
            expected_sites=SITE_ORDER,
            expected_site_cycles=5,
        )
        self.assertEqual(plan.set_index("site_id")["decision_date"].to_dict(), SITE_DATES)
        self.assertTrue(plan["member_day_rows"].eq(35).all())

    def test_plan_rejects_incomplete_member_day_cycle(self) -> None:
        weather = weather_fixture().iloc[:-1].copy()
        with self.assertRaisesRegex(ValueError, "35 unique member-day rows"):
            build_formal_plan(
                weather,
                expected_years=(2015,),
                expected_sites=SITE_ORDER,
                expected_site_cycles=5,
            )

    def test_schedule_match_accepts_exact_site_plan(self) -> None:
        plan = build_formal_plan(
            weather_fixture(),
            expected_years=(2015,),
            expected_sites=SITE_ORDER,
            expected_site_cycles=5,
        )
        unit = plan.loc[plan["site_id"].eq("P3")]
        schedule = unit.rename(columns={"checkpoint_date": "state_checkpoint_date"})
        validate_schedule_match(unit, schedule)

    def test_schedule_match_rejects_shifted_date(self) -> None:
        plan = build_formal_plan(
            weather_fixture(),
            expected_years=(2015,),
            expected_sites=SITE_ORDER,
            expected_site_cycles=5,
        )
        unit = plan.loc[plan["site_id"].eq("P4")]
        schedule = unit.rename(columns={"checkpoint_date": "state_checkpoint_date"}).copy()
        schedule["decision_date"] = "2015-07-05"
        with self.assertRaisesRegex(ValueError, "differs"):
            validate_schedule_match(unit, schedule)

    def test_partial_audit_passes_completed_subset(self) -> None:
        plan = build_formal_plan(
            weather_fixture(),
            expected_years=(2015,),
            expected_sites=SITE_ORDER,
            expected_site_cycles=5,
        )
        completed = plan.iloc[:1].copy()
        audit = build_formal_audit(
            plan=plan,
            candidates=candidate_fixture(completed),
            branch_audits=audit_fixture(completed),
        )
        self.assertTrue(audit["completed_output_gate_passed"])
        self.assertFalse(audit["full_generation_gate_passed"])
        self.assertEqual(audit["candidate_rows"], 8)
        self.assertEqual(audit["restart_nprintday"], 48)
        self.assertEqual(audit["observed_restart_nprintday_values"], [48])
        self.assertEqual(audit["swap_dtmax_days"], 0.01)
        self.assertEqual(audit["observed_swap_dtmax_days_values"], [0.01])

    def test_full_audit_passes_all_site_cycles(self) -> None:
        plan = build_formal_plan(
            weather_fixture(),
            expected_years=(2015,),
            expected_sites=SITE_ORDER,
            expected_site_cycles=5,
        )
        audit = build_formal_audit(
            plan=plan,
            candidates=candidate_fixture(plan),
            branch_audits=audit_fixture(plan),
        )
        self.assertTrue(audit["full_generation_gate_passed"])
        self.assertEqual(audit["candidate_rows"], 40)

    def test_formal_audit_counts_explicit_endpoint_fallback(self) -> None:
        plan = build_formal_plan(
            weather_fixture(),
            expected_years=(2015,),
            expected_sites=SITE_ORDER,
            expected_site_cycles=5,
        )
        candidates = ensure_formal_endpoint_metadata(candidate_fixture(plan))
        fallback_index = candidates.index[
            candidates["site"].eq("P4") & candidates["ir"].eq(20.0)
        ][0]
        candidates.loc[fallback_index, "simulated_ir_mm"] = 19.9
        candidates.loc[fallback_index, "numerical_endpoint_fallback"] = True
        candidates.loc[
            fallback_index, "numerical_endpoint_fallback_delta_mm"
        ] = -0.1
        audit = build_formal_audit(
            plan=plan,
            candidates=candidates,
            branch_audits=audit_fixture(plan),
        )
        self.assertTrue(audit["full_generation_gate_passed"])
        self.assertEqual(audit["numerical_endpoint_fallback_count"], 1)
        self.assertEqual(audit["numerical_irrigation_fallback_count"], 1)
        self.assertEqual(
            audit["numerical_irrigation_fallback_policy"],
            {
                f"{requested:g}": simulated
                for requested, simulated in FORMAL_NUMERICAL_ENDPOINT_FALLBACKS_MM.items()
            },
        )
        self.assertEqual(
            audit["next_gate"],
            "review_complete_formal_labels_and_numerical_irrigation_fallbacks_before_training",
        )

    def test_formal_fallback_policy_covers_positive_grid_but_excludes_zero(self) -> None:
        self.assertEqual(
            FORMAL_NUMERICAL_ENDPOINT_FALLBACKS_MM,
            {
                10.0: (9.9, 9.8),
                15.0: (14.9, 14.8),
                20.0: (19.9, 19.8),
                25.0: (24.9, 24.8),
                30.0: (29.9, 29.8),
                40.0: (39.9, 39.8),
                60.0: (59.9, 59.8),
            },
        )

    def test_audit_rejects_duplicate_candidate(self) -> None:
        plan = build_formal_plan(
            weather_fixture(),
            expected_years=(2015,),
            expected_sites=SITE_ORDER,
            expected_site_cycles=5,
        )
        candidates = candidate_fixture(plan)
        candidates = pd.concat([candidates, candidates.iloc[[0]]], ignore_index=True)
        audit = build_formal_audit(
            plan=plan,
            candidates=candidates,
            branch_audits=audit_fixture(plan),
        )
        self.assertFalse(audit["completed_output_gate_passed"])

    def test_plan_only_audit_does_not_claim_label_generation(self) -> None:
        plan = build_formal_plan(
            weather_fixture(),
            expected_years=(2015,),
            expected_sites=SITE_ORDER,
            expected_site_cycles=5,
        )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            weather_path = root / "weather.csv"
            weather_fixture().to_csv(weather_path, index=False)
            outputs = write_aggregate_outputs(
                root,
                plan=plan,
                candidates=pd.DataFrame(),
                branch_audits=[],
                weather_path=weather_path,
                plan_only=True,
            )
            audit = json.loads(outputs["audit"].read_text(encoding="utf-8"))
        self.assertEqual(
            audit["status"],
            "exact_schedule_2015_2019_formal_label_plan_passed",
        )
        self.assertTrue(audit["completed_output_gate_passed"])
        self.assertFalse(audit["label_generation_performed"])
        self.assertEqual(audit["planned_candidate_rows"], 40)
        self.assertEqual(audit["restart_nprintday"], 48)
        self.assertEqual(audit["swap_dtmax_days"], 0.01)

    def test_passed_branch_requires_formal_restart_frequency(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            candidates = candidate_fixture(
                build_formal_plan(
                    weather_fixture(),
                    expected_years=(2015,),
                    expected_sites=SITE_ORDER,
                    expected_site_cycles=5,
                ).iloc[:1]
            )
            candidates["nprintday"] = 24
            candidates.to_csv(
                root / "gefs_checkpoint_one_date_eight_ir_candidates_v1.csv",
                index=False,
            )
            (root / "gefs_checkpoint_one_date_eight_ir_audit_v1.json").write_text(
                json.dumps(
                    {
                        "mandatory_gate_passed": True,
                        "site_id": "P1",
                        "target_year": 2015,
                        "decision_date": "2015-07-06",
                        "restart_nprintday": 24,
                        "swap_dtmax_days": FORMAL_SWAP_DTMAX_DAYS,
                    }
                ),
                encoding="utf-8",
            )
            (root / "gefs_checkpoint_one_date_eight_ir_manifest_v1.json").write_text(
                json.dumps({"inputs": {
                    "all_variable_weather_sha256": "weather",
                    "swap_dtmax_days": FORMAL_SWAP_DTMAX_DAYS,
                }}),
                encoding="utf-8",
            )

            loaded = load_passed_branch(
                root,
                site="P1",
                year=2015,
                decision_date="2015-07-06",
                weather_sha256="weather",
            )

        self.assertIsNone(loaded)

    def test_passed_branch_reuses_formal_restart_frequency(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            plan = build_formal_plan(
                weather_fixture(),
                expected_years=(2015,),
                expected_sites=SITE_ORDER,
                expected_site_cycles=5,
            ).iloc[:1]
            candidates = candidate_fixture(plan)
            candidates["nprintday"] = FORMAL_RESTART_NPRINTDAY
            candidates.to_csv(
                root / "gefs_checkpoint_one_date_eight_ir_candidates_v1.csv",
                index=False,
            )
            (root / "gefs_checkpoint_one_date_eight_ir_audit_v1.json").write_text(
                json.dumps(
                    {
                        "mandatory_gate_passed": True,
                        "site_id": "P1",
                        "target_year": 2015,
                        "decision_date": "2015-07-06",
                        "restart_nprintday": FORMAL_RESTART_NPRINTDAY,
                        "swap_dtmax_days": FORMAL_SWAP_DTMAX_DAYS,
                    }
                ),
                encoding="utf-8",
            )
            (root / "gefs_checkpoint_one_date_eight_ir_manifest_v1.json").write_text(
                json.dumps({"inputs": {
                    "all_variable_weather_sha256": "weather",
                    "swap_dtmax_days": FORMAL_SWAP_DTMAX_DAYS,
                }}),
                encoding="utf-8",
            )

            loaded = load_passed_branch(
                root,
                site="P1",
                year=2015,
                decision_date="2015-07-06",
                weather_sha256="weather",
            )

        self.assertIsNotNone(loaded)
        assert loaded is not None
        self.assertEqual(len(loaded[0]), 8)

    def test_passed_branch_rejects_nonformal_dtmax(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            plan = build_formal_plan(
                weather_fixture(),
                expected_years=(2015,),
                expected_sites=SITE_ORDER,
                expected_site_cycles=5,
            ).iloc[:1]
            candidates = candidate_fixture(plan)
            candidates["nprintday"] = FORMAL_RESTART_NPRINTDAY
            candidates["swap_dtmax_days"] = 0.02
            candidates.to_csv(
                root / "gefs_checkpoint_one_date_eight_ir_candidates_v1.csv",
                index=False,
            )
            (root / "gefs_checkpoint_one_date_eight_ir_audit_v1.json").write_text(
                json.dumps(
                    {
                        "mandatory_gate_passed": True,
                        "site_id": "P1",
                        "target_year": 2015,
                        "decision_date": "2015-07-06",
                        "restart_nprintday": FORMAL_RESTART_NPRINTDAY,
                        "swap_dtmax_days": 0.02,
                    }
                ),
                encoding="utf-8",
            )
            (root / "gefs_checkpoint_one_date_eight_ir_manifest_v1.json").write_text(
                json.dumps({"inputs": {
                    "all_variable_weather_sha256": "weather",
                    "swap_dtmax_days": 0.02,
                }}),
                encoding="utf-8",
            )

            loaded = load_passed_branch(
                root,
                site="P1",
                year=2015,
                decision_date="2015-07-06",
                weather_sha256="weather",
            )

        self.assertIsNone(loaded)

    def test_imports_complete_verified_site_year_seed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output_dir = root / "formal"
            seed_root = root / "seed"
            plan = build_formal_plan(
                weather_fixture(),
                expected_years=(2015,),
                expected_sites=SITE_ORDER,
                expected_site_cycles=5,
            ).iloc[:1].copy()
            decision_date = str(plan.iloc[0]["decision_date"])
            branch = seed_root / "branches" / pd.Timestamp(decision_date).strftime("%Y%m%d")
            branch.mkdir(parents=True)
            candidates = candidate_fixture(plan)
            candidates["nprintday"] = FORMAL_RESTART_NPRINTDAY
            candidates.to_csv(
                branch / "gefs_checkpoint_one_date_eight_ir_candidates_v1.csv",
                index=False,
            )
            (branch / "gefs_checkpoint_one_date_eight_ir_audit_v1.json").write_text(
                json.dumps(
                    {
                        "mandatory_gate_passed": True,
                        "site_id": "P1",
                        "target_year": 2015,
                        "decision_date": decision_date,
                        "restart_nprintday": FORMAL_RESTART_NPRINTDAY,
                        "swap_dtmax_days": FORMAL_SWAP_DTMAX_DAYS,
                    }
                ),
                encoding="utf-8",
            )
            (branch / "gefs_checkpoint_one_date_eight_ir_manifest_v1.json").write_text(
                json.dumps(
                    {
                        "inputs": {
                            "all_variable_weather_sha256": "weather",
                            "swap_dtmax_days": FORMAL_SWAP_DTMAX_DAYS,
                        }
                    }
                ),
                encoding="utf-8",
            )
            (seed_root / "gefs_site_year_dtmax_stability_audit_v1.json").write_text(
                json.dumps(
                    {
                        "mandatory_gate_passed": True,
                        "formal_output_modified": False,
                        "site_id": "P1",
                        "target_year": 2015,
                        "restart_nprintday": FORMAL_RESTART_NPRINTDAY,
                        "swap_dtmax_days": FORMAL_SWAP_DTMAX_DAYS,
                        "completed_decision_dates": [decision_date],
                        "completed_site_cycle_count": 1,
                        "candidate_rows": 8,
                    }
                ),
                encoding="utf-8",
            )

            import_audit_path = import_verified_site_year_seed(
                seed_root=seed_root,
                output_dir=output_dir,
                plan=plan,
                weather_sha256="weather",
            )
            imported = json.loads(import_audit_path.read_text(encoding="utf-8"))
            destination = (
                output_dir
                / "units"
                / "Y2015"
                / "P1"
                / "branches"
                / pd.Timestamp(decision_date).strftime("%Y%m%d")
            )
            destination_exists = destination.is_dir()

        self.assertEqual(imported["status"], "verified_site_year_seed_import_passed")
        self.assertEqual(imported["imported_site_cycle_count"], 1)
        self.assertTrue(destination_exists)

    def test_archives_nonreusable_branch_without_deleting_it(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output_dir = Path(directory)
            source = output_dir / "units/Y2015/P1/branches/20150706"
            source.mkdir(parents=True)
            pd.DataFrame({"nprintday": [24]}).to_csv(
                source / "gefs_checkpoint_one_date_eight_ir_candidates_v1.csv",
                index=False,
            )
            (source / "kept.txt").write_text("old result", encoding="utf-8")

            archived = archive_nonreusable_branch(
                source,
                output_dir=output_dir,
                year=2015,
                site="P1",
                decision_date="2015-07-06",
                required_restart_nprintday=48,
            )

            self.assertFalse(source.exists())
            self.assertEqual(archived.parts[-5:], (
                "archived_nonreusable_branches_v1",
                "restart_nprintday_24_swap_dtmax_unknown_or_incomplete",
                "Y2015",
                "P1",
                "20150706",
            ))
            self.assertEqual((archived / "kept.txt").read_text(encoding="utf-8"), "old result")

    def test_archives_stale_aggregate_outputs(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output_dir = Path(directory)
            pd.DataFrame({"nprintday": [24]}).to_csv(
                output_dir / "gefs_exact_schedule_formal_candidates_v1.csv",
                index=False,
            )
            (output_dir / "gefs_exact_schedule_formal_label_generation_audit_v1.json").write_text(
                "{}", encoding="utf-8"
            )

            archived = archive_stale_aggregate_outputs(
                output_dir,
                required_restart_nprintday=48,
            )

            self.assertIsNotNone(archived)
            assert archived is not None
            self.assertTrue(
                (archived / "gefs_exact_schedule_formal_candidates_v1.csv").is_file()
            )
            self.assertFalse(
                (output_dir / "gefs_exact_schedule_formal_candidates_v1.csv").exists()
            )


if __name__ == "__main__":
    unittest.main()
