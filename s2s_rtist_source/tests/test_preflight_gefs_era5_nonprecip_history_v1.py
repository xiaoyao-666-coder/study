from __future__ import annotations

import unittest

import pandas as pd

from scripts.data_preparation.preflight_gefs_era5_nonprecip_history_v1 import (
    NONPRECIP_PRODUCT_SPECS,
    build_audit,
    build_cycles,
)


class GefsEra5NonprecipHistoryPreflightTests(unittest.TestCase):
    def test_default_design_has_four_cycles_per_year(self) -> None:
        cycles = build_cycles([2000, 2001], ["05-15", "06-15", "07-15", "08-15"])
        self.assertEqual(len(cycles), 8)
        self.assertEqual(cycles[0], "2000-05-15")
        self.assertEqual(cycles[-1], "2001-08-15")

    def test_duplicate_cycles_are_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "not unique"):
            build_cycles([2000], ["05-15", "05-15"])

    def test_audit_projects_five_member_network_bytes(self) -> None:
        cycles = ["2000-05-15", "2000-06-15"]
        rows = []
        for cycle in cycles:
            for spec in NONPRECIP_PRODUCT_SPECS:
                rows.append(
                    {
                        "cycle_date": cycle,
                        "gefs_member": "c00",
                        "product_id": spec.product_id,
                        "selected_range_bytes": 100,
                    }
                )
        manifest = pd.DataFrame(rows)
        inventory = pd.DataFrame(
            {
                "cycle_date": cycles,
                "gefs_member": ["c00", "c00"],
            }
        )
        audit = build_audit(
            manifest,
            inventory,
            cycles=cycles,
            members=["c00"],
            maximum_selected_bytes=10_000,
        )
        self.assertTrue(audit["mandatory_gate_passed"])
        self.assertEqual(audit["selected_range_bytes"], 1200)
        self.assertEqual(audit["projected_five_member_bytes"], 6000)

    def test_contract_limit_failure_is_recorded(self) -> None:
        cycles = ["2000-05-15"]
        manifest = pd.DataFrame(
            {
                "cycle_date": cycles * len(NONPRECIP_PRODUCT_SPECS),
                "gefs_member": ["c00"] * len(NONPRECIP_PRODUCT_SPECS),
                "product_id": [spec.product_id for spec in NONPRECIP_PRODUCT_SPECS],
                "selected_range_bytes": [100] * len(NONPRECIP_PRODUCT_SPECS),
            }
        )
        inventory = pd.DataFrame({"cycle_date": cycles, "gefs_member": ["c00"]})
        audit = build_audit(
            manifest,
            inventory,
            cycles=cycles,
            members=["c00"],
            maximum_selected_bytes=500,
        )
        self.assertFalse(audit["mandatory_gate_passed"])
        self.assertIn("selected_bytes_exceed_contract", audit["gate_failures"])


if __name__ == "__main__":
    unittest.main()
