from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from s2s_rtist.weather.gefs_quantile_delta_mapping import (
    apply_current_cycle_precipitation_qdm,
    fit_offline_precipitation_qdm,
)


def fit_frame() -> pd.DataFrame:
    rows = []
    for member, offset in (("c00", 0.0), ("p01", 0.5)):
        for lead, (raw, reference) in enumerate(((0.0, 0.0), (1.0, 2.0), (4.0, 5.0)), start=1):
            rows.append(
                {
                    "site_id": "P1",
                    "decision_date": "2010-06-01",
                    "valid_date_utc": f"2010-06-0{lead}",
                    "lead_day": lead,
                    "gefs_member": member,
                    "precipitation_mm_raw": raw + offset,
                    "precipitation_mm_reference": reference,
                }
            )
    return pd.DataFrame(rows)


def target_cycle(decision_date: str, raw_values: list[float]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "site_id": ["P1"] * len(raw_values),
            "decision_date": [decision_date] * len(raw_values),
            "valid_date_utc": pd.date_range(decision_date, periods=len(raw_values)),
            "lead_day": range(1, len(raw_values) + 1),
            "gefs_member": ["c00"] * len(raw_values),
            "precipitation_mm_raw": raw_values,
        }
    )


class CausalCurrentCycleQdmTests(unittest.TestCase):
    def test_future_cycle_does_not_change_earlier_cycle(self) -> None:
        artifact = fit_offline_precipitation_qdm(fit_frame(), fit_years=(2010,))
        first = target_cycle("2015-06-01", [0.0, 1.0, 3.0])
        future = target_cycle("2015-06-15", [20.0, 40.0, 80.0])
        alone = apply_current_cycle_precipitation_qdm(
            first, artifact, split="test"
        )
        combined = apply_current_cycle_precipitation_qdm(
            pd.concat([first, future], ignore_index=True), artifact, split="test"
        )
        earlier = combined.loc[
            combined["decision_date"].eq(pd.Timestamp("2015-06-01"))
        ]
        np.testing.assert_allclose(
            alone["precipitation_mm_qdm"], earlier["precipitation_mm_qdm"]
        )
        self.assertTrue(earlier["qdm_target_cdf_sample_count"].eq(3).all())

    def test_expected_cycle_size_is_enforced(self) -> None:
        artifact = fit_offline_precipitation_qdm(fit_frame(), fit_years=(2010,))
        with self.assertRaisesRegex(ValueError, "expected=4"):
            apply_current_cycle_precipitation_qdm(
                target_cycle("2015-06-01", [0.0, 1.0, 3.0]),
                artifact,
                split="test",
                expected_rows_per_cycle=4,
            )


if __name__ == "__main__":
    unittest.main()
