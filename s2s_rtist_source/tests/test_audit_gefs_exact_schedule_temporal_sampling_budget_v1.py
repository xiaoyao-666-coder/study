from __future__ import annotations

import unittest

import pandas as pd

from scripts.data_preparation.audit_gefs_exact_schedule_temporal_sampling_budget_v1 import (
    BALANCED_DIAGNOSTIC,
    CONSERVATIVE_CANDIDATE,
    FULL_REFERENCE,
    build_audit,
    build_budget_tables,
    product_cadence_hours,
    select_records_for_variant,
    selected_range_bytes,
)
from s2s_rtist.weather.gefs_gridmet_bias import GribIndexRecord, StepWindow


def records_fixture() -> list[GribIndexRecord]:
    records = []
    for number, end_hour in enumerate(range(3, 175, 3), start=1):
        records.append(
            GribIndexRecord(
                message_number=number,
                offset=(number - 1) * 100,
                init_text="d=2015010100",
                short_name="SPFH",
                level="2 m above ground",
                step_text=f"{end_hour} hour fcst",
                ensemble_text="ENS=test",
                step=StepWindow(end_hour, end_hour, "instant"),
                range_end=number * 100 - 1,
            )
        )
    return records


class TemporalSamplingBudgetTests(unittest.TestCase):
    def test_conservative_variant_keeps_temperature_and_fluxes_three_hourly(self) -> None:
        self.assertEqual(
            product_cadence_hours(product_id="tmp_2m", variant_id=CONSERVATIVE_CANDIDATE),
            3,
        )
        self.assertEqual(
            product_cadence_hours(product_id="apcp_sfc", variant_id=CONSERVATIVE_CANDIDATE),
            3,
        )
        self.assertEqual(
            product_cadence_hours(product_id="spfh_2m", variant_id=CONSERVATIVE_CANDIDATE),
            6,
        )

    def test_six_hour_variant_selects_half_the_state_records_and_reaches_174h(self) -> None:
        selected = select_records_for_variant(
            records_fixture(), product_id="spfh_2m", variant_id=CONSERVATIVE_CANDIDATE
        )

        self.assertEqual(len(selected), 29)
        self.assertEqual(selected[0].step.end_hour, 6)
        self.assertEqual(selected[-1].step.end_hour, 174)

    def test_flux_product_cannot_be_thinned(self) -> None:
        selected = select_records_for_variant(
            records_fixture(), product_id="apcp_sfc", variant_id=BALANCED_DIAGNOSTIC
        )

        self.assertEqual(len(selected), 58)

    def test_byte_budget_uses_selected_grib_ranges(self) -> None:
        self.assertEqual(selected_range_bytes(records_fixture()), 5800)
        thinned = select_records_for_variant(
            records_fixture(), product_id="pres_sfc", variant_id=CONSERVATIVE_CANDIDATE
        )
        self.assertEqual(selected_range_bytes(thinned), 2900)

    def test_budget_and_audit_preserve_reference_and_block_training(self) -> None:
        rows = []
        for variant_id, byte_count in (
            (FULL_REFERENCE, 100),
            (CONSERVATIVE_CANDIDATE, 70),
            (BALANCED_DIAGNOSTIC, 60),
        ):
            for product_id, cadence in (("apcp_sfc", 3), ("dswrf_sfc", 3)):
                rows.append(
                    {
                        "variant_id": variant_id,
                        "target_year": 2015,
                        "cycle_date": "2015-05-01",
                        "gefs_member": "c00",
                        "product_id": product_id,
                        "short_name": product_id,
                        "cadence_hours": cadence,
                        "selected_message_count": 58,
                        "selected_range_bytes": byte_count,
                    }
                )
        detail = pd.DataFrame(rows)
        variants, _, _ = build_budget_tables(detail)
        audit = build_audit(
            detail=detail,
            variant_budget=variants,
            preflight_reference_bytes=200,
            expected_product_rows_per_variant=2,
        )

        self.assertTrue(audit["mandatory_structural_gate_passed"])
        self.assertTrue(audit["full_reference_reconciled"])
        self.assertFalse(audit["product_payload_download_started"])
        self.assertFalse(audit["training_eligible"])


if __name__ == "__main__":
    unittest.main()
