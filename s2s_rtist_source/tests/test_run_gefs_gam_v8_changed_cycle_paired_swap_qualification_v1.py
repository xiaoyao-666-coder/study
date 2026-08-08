import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.simulation.run_gefs_gam_v8_changed_cycle_paired_swap_qualification_v1 import (
    ANCHOR_ROLE,
    EXPECTED_CHANGED_CYCLES,
    EXPECTED_FULL_CYCLES,
    EXPECTED_PLAN_ROWS,
    V8_ROLE,
    build_changed_plan,
    build_changed_comparison,
    build_full_ledger,
    evaluate_gate,
    load_prior_candidate_bundle,
    split_exact_reuse,
    validate_protocol,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROTOCOL_PATH = PROJECT_ROOT / (
    "docs/superpowers/specs/"
    "2026-08-08-teacher-guided-gam-v8-changed-cycle-"
    "paired-swap-qualification-v1.json"
)


def sample_stage_a_decisions() -> pd.DataFrame:
    full_counts = {"P1": 38, "P2": 38, "P3": 41, "P4": 39, "P15": 40}
    changed_counts = {"P1": 3, "P2": 2, "P3": 1, "P4": 6, "P15": 0}
    rows = []
    for site_id, count in full_counts.items():
        for index in range(count):
            year = 2016 + index % 3
            date = pd.Timestamp(year, 5, 1) + pd.Timedelta(
                days=7 * (index // 3)
            )
            changed = index < changed_counts[site_id]
            anchor = 10.0 + float(index % 10)
            if not changed:
                v5 = anchor
                v8 = anchor
                lcb = 0.0
                limited = False
            else:
                v5 = anchor - 2.5
                if site_id == "P1" and index == 0:
                    v8, lcb, limited = v5, 7.0, True
                elif site_id == "P4" and index in (0, 1):
                    v8, lcb, limited = v5, 6.0, True
                elif site_id == "P4" and index == 2:
                    v8, lcb, limited = v5, 0.0, False
                else:
                    lcb, limited = 2.0, True
                    v8 = anchor - 1.0
            truth = v8 if changed else anchor
            rows.append(
                {
                    "fold_id": f"holdout_{site_id}_rolling_to_{year}",
                    "target_site": site_id,
                    "validation_year": year,
                    "target_year": year,
                    "site_id": site_id,
                    "decision_date": date.strftime("%Y-%m-%d"),
                    "anchor_recommendation_mm": anchor,
                    "recommendation_mm": v8,
                    "v8_recommendation_mm": v8,
                    "v5_recommendation_mm": v5,
                    "v5_movement_from_anchor_mm": v5 - anchor,
                    "selected_lcb_mm": lcb,
                    "trust_region_limited": limited,
                    "irrigation_changed": changed,
                    "true_peak_mm": truth,
                    "true_oracle_is_positive": truth > 0.0,
                }
            )
    return pd.DataFrame(rows)


class GamV8ChangedCyclePairedSwapQualificationTests(unittest.TestCase):
    def test_protocol_freezes_v8_plan_and_execution_boundaries(self) -> None:
        protocol = json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))
        validate_protocol(protocol)
        self.assertEqual(EXPECTED_CHANGED_CYCLES, 12)
        self.assertEqual(EXPECTED_PLAN_ROWS, 24)
        self.assertEqual(EXPECTED_FULL_CYCLES, 196)
        self.assertEqual(protocol["candidate_reuse"]["maximum_new_exact_swap_runs"], 8)
        self.assertFalse(protocol["evidence_boundary"]["2019_access"])
        self.assertFalse(protocol["evidence_boundary"]["2024_access"])

        protocol["candidate_reuse"]["maximum_new_exact_swap_runs"] = 9
        with self.assertRaisesRegex(ValueError, "candidate reuse"):
            validate_protocol(protocol)

    def test_build_plan_contains_anchor_and_v8_for_each_changed_cycle(self) -> None:
        plan = build_changed_plan(sample_stage_a_decisions())

        self.assertEqual(len(plan), EXPECTED_PLAN_ROWS)
        self.assertEqual(set(plan["model_role"]), {ANCHOR_ROLE, V8_ROLE})
        self.assertTrue(
            plan.groupby(["site_id", "decision_date"])["model_role"]
            .nunique()
            .eq(2)
            .all()
        )
        self.assertEqual(plan["site_id"].nunique(), 4)

    def test_split_exact_reuse_rejects_nearby_endpoint(self) -> None:
        prior = pd.DataFrame({"ir": [8.0, 10.0]})
        reusable, missing = split_exact_reuse(prior, [8.000002, 10.0])
        self.assertEqual(reusable, [10.0])
        self.assertEqual(missing, [8.000002])

    def test_real_plan_has_four_reusable_v8_endpoints_and_eight_missing(self) -> None:
        plan = build_changed_plan(sample_stage_a_decisions())
        reusable_count = 0
        missing_count = 0
        for cycle in plan.drop_duplicates(["site_id", "decision_date"]).itertuples(
            index=False
        ):
            requested = plan.loc[
                plan["site_id"].eq(cycle.site_id)
                & plan["decision_date"].eq(cycle.decision_date),
                "continuous_irrigation_mm",
            ].astype(float).tolist()
            v8_value = requested[1]
            if (cycle.site_id, cycle.decision_date) in {
                ("P1", "2016-05-01"),
                ("P4", "2016-05-01"),
                ("P4", "2016-05-08"),
                ("P4", "2018-05-01"),
            }:
                prior = pd.DataFrame({"ir": requested})
            else:
                prior = pd.DataFrame({"ir": [requested[0]]})
            reusable, missing = split_exact_reuse(prior, [requested[0], v8_value])
            reusable_count += len(reusable)
            missing_count += len(missing)

        self.assertEqual(reusable_count, 16)
        self.assertEqual(missing_count, 8)

    def test_full_ledger_fills_unchanged_cycles_with_exact_zero(self) -> None:
        inventory = sample_stage_a_decisions()
        inventory.loc[0, "true_oracle_is_positive"] = False
        plan = build_changed_plan(inventory)
        decisions = plan[[
            "model_id",
            "model_role",
            "site_id",
            "decision_date",
            "continuous_irrigation_mm",
        ]].copy()
        is_v8 = decisions["model_role"].eq(V8_ROLE)
        decisions["continuous_recommendation_swap_gain_7d"] = np.where(
            is_v8, 1.0, 0.0
        )
        decisions["regret_vs_fixed_eight_swap_7d"] = np.where(
            is_v8, 0.0, 1.0
        )
        decisions["regret_vs_adaptive_swap_oracle_7d"] = np.where(
            is_v8, 0.0, 1.0
        )

        comparison = build_changed_comparison(decisions, plan)
        ledger = build_full_ledger(inventory, comparison)
        unchanged = ~ledger["irrigation_changed"].astype(bool)

        self.assertEqual(len(ledger), EXPECTED_FULL_CYCLES)
        self.assertEqual(int(unchanged.sum()), 184)
        self.assertTrue(
            ledger.loc[
                unchanged,
                [
                    "v8_minus_anchor_gain_7d",
                    "v8_minus_anchor_regret_fixed_7d",
                    "v8_minus_anchor_regret_adaptive_7d",
                ],
            ].eq(0.0).all().all()
        )
        _, _, gate = evaluate_gate(
            ledger, comparison, execution_complete=True
        )
        self.assertTrue(gate["passed"])

    def test_gate_fails_when_exact_execution_is_incomplete(self) -> None:
        inventory = sample_stage_a_decisions()
        plan = build_changed_plan(inventory)
        decisions = plan[[
            "model_id",
            "model_role",
            "site_id",
            "decision_date",
            "continuous_irrigation_mm",
        ]].copy()
        decisions["continuous_recommendation_swap_gain_7d"] = 0.0
        decisions["regret_vs_fixed_eight_swap_7d"] = 0.0
        decisions["regret_vs_adaptive_swap_oracle_7d"] = 0.0
        comparison = build_changed_comparison(decisions, plan)
        ledger = build_full_ledger(inventory, comparison)

        _, _, gate = evaluate_gate(
            ledger, comparison, execution_complete=False
        )

        self.assertFalse(gate["passed"])
        self.assertFalse(
            gate["conditions"][
                "all_reused_and_new_swap_execution_audits_passed"
            ]
        )

    def test_prior_exact_branch_must_reproduce_official_v5_decision(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            aggregate = root / "aggregate.csv"
            decisions = root / "decisions.csv"
            branch = root / "exact_pair_branches" / "P1" / "20180501"
            branch.mkdir(parents=True)
            pd.DataFrame(
                {
                    "evaluation_cycle_id": ["P1_2018-05-01"],
                    "ir": [0.0],
                    "target_value": [0.0],
                    "evaluation_stage": ["global_coarse"],
                }
            ).to_csv(aggregate, index=False)
            pd.DataFrame(
                {
                    "site_id": ["P1"],
                    "decision_date": ["2018-05-01"],
                    "continuous_irrigation_mm": [8.25],
                    "continuous_recommendation_swap_gain_7d": [3.5],
                }
            ).to_csv(decisions, index=False)
            branch_path = branch / "swap_candidates.csv"
            pd.DataFrame(
                {
                    "site": ["P1"],
                    "decision_date": ["2018-05-01"],
                    "ir": [8.25],
                    "target_value": [3.5],
                }
            ).to_csv(branch_path, index=False)

            bundle, branch_paths = load_prior_candidate_bundle(
                root, aggregate, decisions
            )
            self.assertEqual(branch_paths, [branch_path])
            self.assertIn(8.25, bundle["ir"].astype(float).tolist())

            bad = pd.read_csv(branch_path)
            bad["target_value"] = 3.4
            bad.to_csv(branch_path, index=False)
            with self.assertRaisesRegex(ValueError, "official v5 decision"):
                load_prior_candidate_bundle(root, aggregate, decisions)


if __name__ == "__main__":
    unittest.main()
