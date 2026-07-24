from __future__ import annotations

import copy
import unittest

import numpy as np
import pandas as pd

from s2s_rtist.weather.gefs_quantile_delta_mapping import (
    apply_offline_precipitation_qdm,
    fit_offline_precipitation_qdm,
    verify_qdm_artifact,
)


def frame_for_year(year: int, *, scale: float = 1.0) -> pd.DataFrame:
    rows = []
    members = ("c00", "p01", "p02", "p03", "p04")
    raw_values = (0.0, 1.0, 2.0, 4.0, 8.0)
    for site_id in ("P1", "P2"):
        for day, reference in enumerate((0.0, 2.0, 6.0), start=1):
            for member, raw in zip(members, raw_values, strict=True):
                rows.append(
                    {
                        "site_id": site_id,
                        "decision_date": f"{year}-06-01",
                        "forecast_init_utc": f"{year}-06-01T00:00:00Z",
                        "valid_date_utc": f"{year}-06-0{day}",
                        "lead_day": day,
                        "gefs_member": member,
                        "precipitation_mm_raw": raw * scale + day - 1,
                        "precipitation_mm_reference": reference,
                    }
                )
    return pd.DataFrame(rows)


class OfflinePrecipitationQdmTests(unittest.TestCase):
    def test_fit_and_apply_are_deterministic_and_nonnegative(self) -> None:
        historical = frame_for_year(2018)
        target = frame_for_year(2019, scale=1.5).drop(
            columns=["precipitation_mm_reference"]
        )
        artifact = fit_offline_precipitation_qdm(
            historical, fit_years=(2018,), group_keys=("site_id",)
        )
        first = apply_offline_precipitation_qdm(target, artifact, split="validation")
        second = apply_offline_precipitation_qdm(target, artifact, split="validation")
        np.testing.assert_allclose(
            first["precipitation_mm_qdm"], second["precipitation_mm_qdm"]
        )
        self.assertTrue(np.isfinite(first["precipitation_mm_qdm"]).all())
        self.assertTrue((first["precipitation_mm_qdm"] >= 0.0).all())
        self.assertEqual(set(first["qdm_group"]), {"site_id=P1", "site_id=P2"})

    def test_multiplicative_change_is_preserved_for_identical_bias_factor(self) -> None:
        historical = frame_for_year(2018)
        target = historical.copy()
        target["decision_date"] = "2020-06-01"
        target["forecast_init_utc"] = "2020-06-01T00:00:00Z"
        target["valid_date_utc"] = pd.to_datetime(target["valid_date_utc"]) + pd.DateOffset(years=2)
        target["precipitation_mm_raw"] *= 2.0
        target = target.drop(columns=["precipitation_mm_reference"])
        artifact = fit_offline_precipitation_qdm(
            historical, fit_years=(2018,), group_keys=()
        )
        corrected = apply_offline_precipitation_qdm(target, artifact, split="synthetic")
        wet = corrected["precipitation_mm_qdm"] > 0.0
        self.assertTrue((corrected.loc[wet, "qdm_relative_quantile_change"] > 1.0).all())

    def test_validation_and_test_years_cannot_be_fit(self) -> None:
        with self.assertRaisesRegex(ValueError, "2019 or 2024"):
            fit_offline_precipitation_qdm(frame_for_year(2019), fit_years=(2019,))

    def test_only_teacher_selected_groupings_are_supported(self) -> None:
        with self.assertRaisesRegex(ValueError, "global or site_id"):
            fit_offline_precipitation_qdm(
                frame_for_year(2018),
                fit_years=(2018,),
                group_keys=("lead_day",),
            )

    def test_artifact_hash_detects_tampering(self) -> None:
        artifact = fit_offline_precipitation_qdm(
            frame_for_year(2018), fit_years=(2018,)
        )
        tampered = copy.deepcopy(artifact)
        tampered["groups"]["global"]["historical_model_sorted_mm"][0] += 0.1
        with self.assertRaisesRegex(ValueError, "hash mismatch"):
            verify_qdm_artifact(tampered)


if __name__ == "__main__":
    unittest.main()
