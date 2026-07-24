from __future__ import annotations

import copy
import importlib.util
import json
import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from s2s_rtist.weather.gefs_quantile_mapping import (
    CONTRACT_ID_V2,
    CONTRACT_VERSION_V2,
    GEFS_REFORECAST_MEMBERS,
    UPPER_TAIL_CONSTANT_ADDITIVE,
    UTC_DAY_BOUNDARY,
    aggregate_reforecast_member_daily_utc,
    apply_empirical_precipitation_qm,
    fit_empirical_precipitation_qm,
    verify_quantile_mapping_artifact,
)


def v2_training_frame() -> pd.DataFrame:
    rows = []
    observations = [
        ("2015-06-01", 0.0, [0.0, 0.0, 0.1, 0.2, 0.3]),
        ("2016-06-01", 0.0, [0.0, 0.1, 0.2, 0.3, 0.4]),
        ("2017-06-01", 5.0, [1.0, 2.0, 3.0, 4.0, 5.0]),
        ("2018-06-01", 10.0, [2.0, 4.0, 6.0, 8.0, 10.0]),
    ]
    for decision_date, reference, forecasts in observations:
        for member, forecast in zip(GEFS_REFORECAST_MEMBERS, forecasts):
            rows.append(
                {
                    "site_id": "P1",
                    "decision_date": pd.Timestamp(decision_date),
                    "valid_date_utc": pd.Timestamp(decision_date),
                    "lead_day": 1,
                    "gefs_member": member,
                    "precipitation_mm_raw": forecast,
                    "precipitation_mm_reference": reference,
                }
            )
    return pd.DataFrame(rows)


def fit_v2_artifact() -> dict[str, object]:
    return fit_empirical_precipitation_qm(
        v2_training_frame(),
        contract_id=CONTRACT_ID_V2,
        contract_version=CONTRACT_VERSION_V2,
        aggregation_day_boundary=UTC_DAY_BOUNDARY,
        canonical_valid_date_column="valid_date_utc",
        upper_tail_policy=UPPER_TAIL_CONSTANT_ADDITIVE,
    )


class QuantileMappingV2Tests(unittest.TestCase):
    def test_contract_locks_utc_day_and_constant_additive_tail(self) -> None:
        root = Path(__file__).resolve().parents[1]
        path = (
            root
            / "site_general_surrogate_eval"
            / "gefs_quantile_mapping_data_contract_v2.json"
        )
        contract = json.loads(path.read_text(encoding="utf-8"))

        self.assertEqual(contract["contract_id"], CONTRACT_ID_V2)
        self.assertEqual(contract["contract_version"], CONTRACT_VERSION_V2)
        self.assertEqual(
            contract["time_mapping"]["aggregation_day_boundary"],
            UTC_DAY_BOUNDARY,
        )
        self.assertEqual(
            contract["time_mapping"]["canonical_valid_date_column"],
            "valid_date_utc",
        )
        self.assertEqual(
            contract["quantile_mapping"]["upper_tail_policy"],
            UPPER_TAIL_CONSTANT_ADDITIVE,
        )
        self.assertFalse(contract["scope"]["model_training_allowed"])
        self.assertFalse(contract["scope"]["apply_to_2024_allowed"])

    def test_constant_additive_tail_is_continuous_and_auditable(self) -> None:
        artifact = fit_v2_artifact()
        group = artifact["groups"]["P1|1"]
        forecast_maximum = float(group["training_forecast_maximum_mm"])
        reference_maximum = float(group["training_reference_maximum_mm"])
        offset = float(group["upper_tail_additive_offset_mm"])
        epsilon = 1.0e-6
        application = pd.DataFrame(
            {
                "site_id": ["P1", "P1"],
                "lead_day": [1, 1],
                "precipitation_mm_raw": [
                    forecast_maximum,
                    forecast_maximum + epsilon,
                ],
                "precipitation_mm_reference": [reference_maximum] * 2,
            }
        )

        corrected = apply_empirical_precipitation_qm(
            application, artifact, split="validation_2019_v2"
        )

        self.assertAlmostEqual(
            float(corrected.iloc[0]["precipitation_mm_qm"]), reference_maximum
        )
        self.assertAlmostEqual(
            float(corrected.iloc[1]["precipitation_mm_qm"]),
            forecast_maximum + epsilon + offset,
        )
        self.assertAlmostEqual(
            float(corrected.iloc[1]["precipitation_mm_qm"])
            - float(corrected.iloc[0]["precipitation_mm_qm"]),
            epsilon,
        )
        self.assertFalse(bool(corrected.iloc[0]["qm_extrapolated_upper"]))
        self.assertTrue(bool(corrected.iloc[1]["qm_extrapolated_upper"]))
        self.assertEqual(
            corrected.iloc[1]["qm_upper_tail_policy"],
            UPPER_TAIL_CONSTANT_ADDITIVE,
        )
        self.assertAlmostEqual(
            float(corrected.iloc[1]["qm_upper_tail_offset_mm"]), offset
        )

    def test_v1_artifact_cannot_be_loaded_as_v2(self) -> None:
        v1_frame = v2_training_frame().rename(
            columns={"valid_date_utc": "local_date"}
        )
        artifact = fit_empirical_precipitation_qm(v1_frame)

        with self.assertRaisesRegex(ValueError, "expected contract"):
            verify_quantile_mapping_artifact(
                artifact, expected_contract_id=CONTRACT_ID_V2
            )

        tampered = copy.deepcopy(fit_v2_artifact())
        tampered["aggregation_day_boundary"] = "SITE_LOCAL_00_to_24"
        with self.assertRaisesRegex(ValueError, "hash mismatch"):
            verify_quantile_mapping_artifact(tampered)

    def test_utc_aggregation_uses_valid_date_utc_not_local_date(self) -> None:
        rows = []
        cumulative = 0.0
        for end_hour in range(3, 25, 3):
            start_hour = ((end_hour - 1) // 6) * 6
            if end_hour % 6 == 3:
                cumulative = 1.0
            else:
                cumulative = 2.0
            rows.append(
                {
                    "site": "P1",
                    "timezone": "America/Chicago",
                    "cycle_init_utc": pd.Timestamp("2015-06-01T00:00:00Z"),
                    "gefs_member": "c00",
                    "lead_hour": end_hour,
                    "short_name": "APCP",
                    "value": cumulative,
                    "start_hour": start_hour,
                    "end_hour": end_hour,
                    "kind": "acc",
                }
            )
        points = pd.DataFrame(rows)
        manifest = pd.DataFrame(
            {
                "forecast_init_utc": ["2015-06-01T00:00:00+00:00"],
                "gefs_member": ["c00"],
                "source_key": ["source/c00"],
                "source_etag": ["etag-c00"],
            }
        )

        daily = aggregate_reforecast_member_daily_utc(points, manifest=manifest)

        self.assertEqual(len(daily), 1)
        self.assertIn("valid_date_utc", daily.columns)
        self.assertNotIn("local_date", daily.columns)
        self.assertEqual(daily.iloc[0]["aggregation_day_boundary"], UTC_DAY_BOUNDARY)
        self.assertEqual(daily.iloc[0]["site_timezone"], "America/Chicago")
        self.assertEqual(pd.Timestamp(daily.iloc[0]["valid_date_utc"]), pd.Timestamp("2015-06-01"))
        self.assertAlmostEqual(float(daily.iloc[0]["precipitation_mm_raw"]), 8.0)

    def test_upper_tail_audit_reports_residual_error_without_hiding_it(self) -> None:
        root = Path(__file__).resolve().parents[1]
        diagnostics = root / "scripts" / "diagnostics"
        sys.path.insert(0, str(diagnostics))
        try:
            path = diagnostics / "run_gefs_quantile_mapping_validation_v2.py"
            spec = importlib.util.spec_from_file_location(
                "run_gefs_quantile_mapping_validation_v2_test", path
            )
            assert spec is not None and spec.loader is not None
            module = importlib.util.module_from_spec(spec)
            sys.modules[spec.name] = module
            spec.loader.exec_module(module)
        finally:
            sys.path.remove(str(diagnostics))
        corrected = pd.DataFrame(
            {
                "qm_extrapolated_upper": [True, True],
                "precipitation_mm_raw": [84.4, 45.5],
                "precipitation_mm_qm": [62.3, 36.9],
                "precipitation_mm_reference": [0.0, 36.3],
            }
        )

        events, summary = module._upper_tail_audit(corrected)

        self.assertEqual(summary["event_count"], 2)
        self.assertTrue(summary["numeric_audit_passed"])
        self.assertEqual(summary["worsened_count"], 0)
        self.assertGreater(float(events.iloc[0]["qm_absolute_error_mm"]), 60.0)


if __name__ == "__main__":
    unittest.main()
