import json
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.simulation.run_gefs_gam_v6_2019_paired_swap_qualification_v1 import (
    build_changed_plan,
    build_full_ledger,
    evaluate_gate,
    score_peak_inventory,
    validate_protocol,
)


PROTOCOL_PATH = Path(
    "docs/superpowers/specs/2026-08-07-teacher-guided-gam-v6-2019-independent-confirmation-v1.json"
)


def sample_inventory() -> pd.DataFrame:
    counts = {"P1": 13, "P2": 13, "P3": 15, "P4": 14, "P15": 14}
    rows = []
    changed = {("P1", 0), ("P4", 0)}
    for site, count in counts.items():
        for index in range(count):
            anchor = 10.0 + index
            use_v6 = (site, index) in changed
            rows.append({
                "fold_id": f"holdout_{site}_rolling_to_2019",
                "target_site": site,
                "target_year": 2019,
                "site_id": site,
                "decision_date": (pd.Timestamp("2019-05-01") + pd.Timedelta(days=7 * index)).strftime("%Y-%m-%d"),
                "anchor_recommendation_mm": anchor,
                "selected_irrigation_mm": anchor + 2.0 if use_v6 else anchor,
                "use_v6": use_v6,
            })
    return pd.DataFrame(rows)


def sample_truth(inventory: pd.DataFrame) -> pd.DataFrame:
    truth = inventory[["target_year", "site_id", "decision_date"]].copy()
    truth["true_peak_mm"] = np.where(truth["site_id"].eq("P15"), 0.0, 20.0)
    truth["true_fixed_list_net_gain_7d"] = 100.0
    truth["true_oracle_is_positive"] = truth["true_peak_mm"] > 0.000001
    return truth


def sample_comparison(inventory: pd.DataFrame) -> pd.DataFrame:
    changed = inventory[inventory["use_v6"]].copy()
    changed["anchor_gain_7d"] = 10.0
    changed["selected_gain_7d"] = 11.0
    changed["anchor_regret_fixed_7d"] = 5.0
    changed["selected_regret_fixed_7d"] = 4.0
    changed["anchor_regret_adaptive_7d"] = 6.0
    changed["selected_regret_adaptive_7d"] = 5.0
    return changed


class GamV62019PairedSwapTests(unittest.TestCase):
    def test_protocol_freezes_joint_confirmation_gate(self) -> None:
        protocol = json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))
        validate_protocol(protocol)
        gate = protocol["performance_gate"]
        self.assertEqual(gate["minimum_changed_cycles"], 2)
        self.assertEqual(gate["minimum_changed_sites"], 2)
        self.assertEqual(gate["minimum_nonworse_sites"], 4)
        self.assertTrue(gate["positive_oracle_mean_swap_gain_nonworse"])
        self.assertTrue(gate["global_maximum_peak_distance_nonworse"])
        self.assertFalse(protocol["swap"]["interpolation"])

    def test_score_peak_inventory_is_exact_and_complete(self) -> None:
        inventory = sample_inventory()
        scored = score_peak_inventory(inventory, sample_truth(inventory))
        self.assertEqual(len(scored), 69)
        self.assertIn("anchor_peak_distance_mm", scored)
        self.assertIn("selected_peak_distance_mm", scored)
        self.assertTrue(scored["true_oracle_is_positive"].isin([True, False]).all())

    def test_changed_plan_contains_two_exact_endpoints_per_changed_cycle(self) -> None:
        inventory = sample_inventory()
        plan = build_changed_plan(inventory)
        self.assertEqual(len(plan), 4)
        self.assertEqual(plan.groupby(["site_id", "decision_date"])["model_role"].nunique().tolist(), [2, 2])

    def test_full_ledger_expands_unchanged_cycles_to_exact_zero(self) -> None:
        inventory = score_peak_inventory(sample_inventory(), sample_truth(sample_inventory()))
        comparison = sample_comparison(inventory)
        ledger = build_full_ledger(inventory, comparison)
        self.assertEqual(len(ledger), 69)
        unchanged = ledger[~ledger["use_v6"]]
        np.testing.assert_array_equal(unchanged["selected_minus_anchor_gain_7d"], 0.0)

    def test_joint_gate_passes_complete_nonworse_fixture(self) -> None:
        inventory = score_peak_inventory(sample_inventory(), sample_truth(sample_inventory()))
        comparison = sample_comparison(inventory)
        ledger = build_full_ledger(inventory, comparison)
        site_summary, peak_summary, gate = evaluate_gate(
            ledger,
            comparison,
            execution_complete=True,
        )
        self.assertEqual(len(site_summary), 5)
        self.assertTrue(gate["passed"])
        self.assertTrue(gate["conditions"]["at_least_2_irrigation_decisions_changed"])
        self.assertIn("overall", set(peak_summary["stratum"]))

    def test_positive_loss_or_peak_maximum_regression_fails_gate(self) -> None:
        inventory = score_peak_inventory(sample_inventory(), sample_truth(sample_inventory()))
        comparison = sample_comparison(inventory)
        comparison.loc[comparison.index[0], "selected_gain_7d"] = -20.0
        ledger = build_full_ledger(inventory, comparison)
        _, _, gate = evaluate_gate(ledger, comparison, execution_complete=True)
        self.assertFalse(gate["conditions"]["positive_oracle_mean_swap_gain_nonworse"])
        self.assertFalse(gate["passed"])

        inventory.loc[inventory.index[0], "selected_peak_distance_mm"] = 60.0
        comparison = sample_comparison(inventory)
        ledger = build_full_ledger(inventory, comparison)
        _, _, gate = evaluate_gate(ledger, comparison, execution_complete=True)
        self.assertFalse(gate["conditions"]["global_maximum_peak_distance_nonworse"])
        self.assertFalse(gate["passed"])

    def test_zero_changed_cycles_produces_auditable_nontriviality_failure(self) -> None:
        inventory = sample_inventory()
        inventory["use_v6"] = False
        inventory["selected_irrigation_mm"] = inventory["anchor_recommendation_mm"]
        scored = score_peak_inventory(inventory, sample_truth(inventory))
        comparison = pd.DataFrame(
            columns=[
                "fold_id",
                "target_site",
                "site_id",
                "decision_date",
                "anchor_gain_7d",
                "selected_gain_7d",
                "anchor_regret_fixed_7d",
                "selected_regret_fixed_7d",
                "anchor_regret_adaptive_7d",
                "selected_regret_adaptive_7d",
            ]
        )
        self.assertTrue(build_changed_plan(scored).empty)
        ledger = build_full_ledger(scored, comparison)
        _, _, gate = evaluate_gate(ledger, comparison, execution_complete=True)
        self.assertFalse(gate["conditions"]["at_least_2_irrigation_decisions_changed"])
        self.assertFalse(gate["passed"])


if __name__ == "__main__":
    unittest.main()
