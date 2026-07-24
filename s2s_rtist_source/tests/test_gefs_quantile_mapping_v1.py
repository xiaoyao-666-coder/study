from __future__ import annotations

import copy
import http.client
import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

from s2s_rtist.weather import gefs_quantile_mapping as qm_module
from s2s_rtist.weather.gefs_quantile_mapping import (
    GEFS_REFORECAST_MEMBERS,
    apply_empirical_precipitation_qm,
    build_reforecast_precipitation_url,
    fit_empirical_precipitation_qm,
    normalize_era5_precipitation_m,
    read_quantile_mapping_artifact,
    select_reforecast_precipitation_records,
    validate_member_daily_precipitation,
    validate_reference_daily_precipitation,
    verify_quantile_mapping_artifact,
    write_quantile_mapping_artifact,
)


class GefsReforecastArchiveTests(unittest.TestCase):
    def test_builds_official_reforecast_precipitation_urls(self) -> None:
        product = build_reforecast_precipitation_url("2015-06-01", "c00")
        index = build_reforecast_precipitation_url(
            "2015-06-01", "p04", index=True
        )

        self.assertEqual(
            product,
            "https://noaa-gefs-retrospective.s3.amazonaws.com/GEFSv12/"
            "reforecast/2015/2015060100/c00/Days:1-10/"
            "apcp_sfc_2015060100_c00.grib2",
        )
        self.assertTrue(index.endswith("apcp_sfc_2015060100_p04.grib2.idx"))

    def test_selects_complete_three_hour_coverage_through_hour_174(self) -> None:
        lines = []
        for message, end_hour in enumerate(range(3, 181, 3), start=1):
            start_hour = ((end_hour - 1) // 6) * 6
            lines.append(
                f"{message}:{(message - 1) * 100}:d=2015060100:APCP:surface:"
                f"{start_hour}-{end_hour} hour acc fcst:ENS=test"
            )

        selected = select_reforecast_precipitation_records("\n".join(lines))

        self.assertEqual(len(selected), 58)
        self.assertEqual(selected[0].step.end_hour, 3)
        self.assertEqual(selected[-1].step.end_hour, 174)
        self.assertEqual(selected[-1].range_end, 5799)

    def test_retries_an_incomplete_range_response(self) -> None:
        class FakeResponse:
            def __init__(self, payload: bytes | None):
                self.payload = payload
                self.headers = {}

            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc, traceback):
                return False

            def read(self) -> bytes:
                if self.payload is None:
                    raise http.client.IncompleteRead(b"partial", 5)
                return self.payload

        with patch.object(
            qm_module.urllib.request,
            "urlopen",
            side_effect=[FakeResponse(None), FakeResponse(b"complete")],
        ) as urlopen:
            payload, _ = qm_module._request(
                "https://example.test/range", retries=2, timeout=1
            )

        self.assertEqual(payload, b"complete")
        self.assertEqual(urlopen.call_count, 2)

    def test_clips_only_float32_scale_negative_era5_roundoff(self) -> None:
        normalized, clipped = normalize_era5_precipitation_m(-1.1920918e-8)

        self.assertEqual(normalized, 0.0)
        self.assertTrue(clipped)
        with self.assertRaisesRegex(ValueError, "exceeds tolerance"):
            normalize_era5_precipitation_m(-1.0e-6)


def quantile_mapping_training_frame() -> pd.DataFrame:
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
                    "local_date": pd.Timestamp(decision_date),
                    "lead_day": 1,
                    "gefs_member": member,
                    "precipitation_mm_raw": forecast,
                    "precipitation_mm_reference": reference,
                }
            )
    return pd.DataFrame(rows)


class EmpiricalQuantileMappingTests(unittest.TestCase):
    def test_fits_monotone_mapping_and_corrects_wet_day_frequency(self) -> None:
        artifact = fit_empirical_precipitation_qm(
            quantile_mapping_training_frame()
        )
        group = artifact["groups"]["P1|1"]

        self.assertEqual(group["sample_count"], 20)
        self.assertEqual(group["reference_wet_sample_count"], 10)
        self.assertGreaterEqual(group["forecast_wet_threshold_mm"], 0.0)
        self.assertTrue(
            np.all(np.diff(group["forecast_quantile_nodes"]) > 0.0)
        )
        self.assertTrue(
            np.all(np.diff(group["reference_quantile_nodes"]) >= 0.0)
        )

    def test_zeroes_lower_tail_and_flags_multiplicative_upper_extrapolation(self) -> None:
        artifact = fit_empirical_precipitation_qm(
            quantile_mapping_training_frame()
        )
        group = artifact["groups"]["P1|1"]
        maximum = float(group["training_forecast_maximum_mm"])
        threshold = float(group["forecast_wet_threshold_mm"])
        application = pd.DataFrame(
            {
                "site_id": ["P1", "P1", "P1"],
                "lead_day": [1, 1, 1],
                "precipitation_mm_raw": [threshold, maximum, maximum * 2.0],
                "precipitation_mm_reference": [0.0, 10.0, 20.0],
            }
        )

        corrected = apply_empirical_precipitation_qm(
            application, artifact, split="validation"
        )

        self.assertEqual(float(corrected.iloc[0]["precipitation_mm_qm"]), 0.0)
        self.assertFalse(bool(corrected.iloc[1]["qm_extrapolated_upper"]))
        self.assertTrue(bool(corrected.iloc[2]["qm_extrapolated_upper"]))
        self.assertAlmostEqual(
            float(corrected.iloc[2]["precipitation_mm_qm"]),
            float(group["training_reference_maximum_mm"]) * 2.0,
        )

    def test_rejects_validation_year_in_fit_data(self) -> None:
        contaminated = quantile_mapping_training_frame().copy()
        contaminated.loc[0, "decision_date"] = pd.Timestamp("2019-06-01")

        with self.assertRaisesRegex(ValueError, "non-fit years"):
            fit_empirical_precipitation_qm(contaminated)

    def test_artifact_round_trip_and_tamper_detection(self) -> None:
        artifact = fit_empirical_precipitation_qm(
            quantile_mapping_training_frame()
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "qm.json"
            write_quantile_mapping_artifact(path, artifact)
            loaded = read_quantile_mapping_artifact(path)
        self.assertEqual(loaded["artifact_sha256"], artifact["artifact_sha256"])

        tampered = copy.deepcopy(artifact)
        tampered["groups"]["P1|1"]["forecast_wet_threshold_mm"] += 1.0
        with self.assertRaisesRegex(ValueError, "hash mismatch"):
            verify_quantile_mapping_artifact(tampered)


class QuantileMappingContractTests(unittest.TestCase):
    def test_validates_exact_35_member_rows_and_7_reference_rows(self) -> None:
        member_rows = []
        for member in GEFS_REFORECAST_MEMBERS:
            for offset in range(7):
                member_rows.append(
                    {
                        "site_id": "P1",
                        "site_timezone": "America/Chicago",
                        "forecast_init_utc": pd.Timestamp(
                            "2015-06-01T00:00:00Z"
                        ),
                        "decision_date": pd.Timestamp("2015-06-01"),
                        "gefs_member": member,
                        "local_date": pd.Timestamp("2015-06-01")
                        + pd.Timedelta(days=offset),
                        "lead_day": offset + 1,
                        "precipitation_mm_raw": float(offset),
                        "source_key": f"source/{member}",
                        "source_etag": f"etag-{member}",
                        "source_start_step": offset * 24,
                        "source_end_step": (offset + 1) * 24,
                    }
                )
        member = pd.DataFrame(member_rows)
        reference = pd.DataFrame(
            {
                "site_id": ["P1"] * 7,
                "local_date": pd.date_range("2015-06-01", periods=7),
                "precipitation_mm_reference": np.arange(7, dtype=float),
                "reference_dataset": ["ERA5"] * 7,
                "reference_source_path": ["source.tif"] * 7,
                "reference_unit_conversion": ["m * 1000 = mm"] * 7,
            }
        )

        validate_member_daily_precipitation(
            member,
            expected_sites=("P1",),
            expected_members=GEFS_REFORECAST_MEMBERS,
            expected_cycles=("2015-06-01",),
        )
        validate_reference_daily_precipitation(
            reference,
            expected_sites=("P1",),
            expected_dates=reference["local_date"].tolist(),
        )


class QuantileMappingValidationRunnerTests(unittest.TestCase):
    @staticmethod
    def runner_module():
        root = Path(__file__).resolve().parents[1]
        path = (
            root
            / "scripts"
            / "diagnostics"
            / "run_gefs_quantile_mapping_validation_v1.py"
        )
        spec = importlib.util.spec_from_file_location(
            "run_gefs_quantile_mapping_validation_v1_test", path
        )
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        return module

    def test_pilot_schedule_has_24_fit_and_6_validation_cycles(self) -> None:
        runner = self.runner_module()
        dates = runner.pilot_cycle_dates()

        self.assertEqual(len(dates), 30)
        self.assertEqual(sum(value.startswith("2019-") for value in dates), 6)
        self.assertEqual(sum(not value.startswith("2019-") for value in dates), 24)

    def test_offline_metrics_and_promotion_gate_are_consistent(self) -> None:
        runner = self.runner_module()
        rows = []
        for lead_day in range(1, 8):
            reference = float(lead_day - 1)
            for member_index, member in enumerate(GEFS_REFORECAST_MEMBERS):
                rows.append(
                    {
                        "site_id": "P1",
                        "decision_date": pd.Timestamp("2019-06-01"),
                        "local_date": pd.Timestamp("2019-06-01")
                        + pd.Timedelta(days=lead_day - 1),
                        "lead_day": lead_day,
                        "gefs_member": member,
                        "precipitation_mm_raw": reference
                        + 0.5
                        + member_index * 0.1,
                        "precipitation_mm_qm": reference,
                        "precipitation_mm_reference": reference,
                        "qm_extrapolated_upper": False,
                    }
                )
        paired = pd.DataFrame(rows)

        observations, probabilistic, probabilities = runner._probabilistic_metrics(
            paired, members=GEFS_REFORECAST_MEMBERS
        )
        seven_day = runner._seven_day_metrics(paired)
        gate = runner._promotion_gate(
            observations=observations,
            probabilities=probabilities,
            seven_day=seven_day,
            paired=paired,
        )

        self.assertEqual(len(observations), 14)
        self.assertFalse(probabilistic.empty)
        self.assertEqual(len(seven_day), 2)
        self.assertTrue(gate["automatic_requirements_passed"])
        self.assertEqual(
            gate["promotion_status"], "passed_for_fuller_2019_validation"
        )
        self.assertEqual(gate["qm_seven_day_mae_mm"], 0.0)


if __name__ == "__main__":
    unittest.main()
