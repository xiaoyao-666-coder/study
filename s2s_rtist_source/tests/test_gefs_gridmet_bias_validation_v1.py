from __future__ import annotations

import http.client
import importlib.util
import inspect
import sys
import unittest
import warnings
from datetime import datetime, timezone
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import h5py
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from s2s_rtist.weather.gefs_gridmet_bias import (
    ByteRange,
    add_reference_condition,
    aggregate_gefs_point_records,
    allocate_interval_to_local_days,
    build_gefs_product_url,
    compute_bias_metrics,
    compute_precipitation_event_metrics,
    convert_gridmet_reference_units,
    decumulate_reset_intervals,
    merge_contiguous_ranges,
    fetch_selected_byte_ranges,
    parse_gefs_index,
    parse_step_window,
    pair_forecast_and_reference,
    packing_resolution,
    read_gridmet_variable_points,
    required_valid_dates,
    forecast_daily_to_long,
    select_gefs_messages,
    validate_reference_coverage,
    vapor_pressure_deficit_kpa,
)


RUNNER_NAME = "run_gefs_gridmet_bias_validation_v1"
RUNNER_PATH = ROOT / "scripts" / "diagnostics" / f"{RUNNER_NAME}.py"
RUNNER_SPEC = importlib.util.spec_from_file_location(RUNNER_NAME, RUNNER_PATH)
assert RUNNER_SPEC is not None and RUNNER_SPEC.loader is not None
RUNNER = importlib.util.module_from_spec(RUNNER_SPEC)
sys.modules[RUNNER_NAME] = RUNNER
RUNNER_SPEC.loader.exec_module(RUNNER)
_request_bytes = RUNNER._request_bytes


INDEX_TEXT = """\
1:0:d=2024071600:VIS:surface:3 hour fcst:ens mean
10:3820703:d=2024071600:TMP:2 m above ground:3 hour fcst:ens mean
11:4242857:d=2024071600:DPT:2 m above ground:3 hour fcst:ens mean
12:4678088:d=2024071600:RH:2 m above ground:3 hour fcst:ens mean
13:5328427:d=2024071600:TMAX:2 m above ground:0-3 hour max fcst:ens mean
14:5755904:d=2024071600:TMIN:2 m above ground:0-3 hour min fcst:ens mean
15:6184575:d=2024071600:UGRD:10 m above ground:3 hour fcst:ens mean
16:7014648:d=2024071600:VGRD:10 m above ground:3 hour fcst:ens mean
17:7824430:d=2024071600:CPOFP:surface:3 hour fcst:ens mean
18:8641665:d=2024071600:APCP:surface:0-3 hour acc fcst:ens mean
19:8879531:d=2024071600:CSNOW:surface:0-3 hour ave fcst:ens mean
30:13507961:d=2024071600:DSWRF:surface:0-3 hour ave fcst:ens mean
31:13893566:d=2024071600:DLWRF:surface:0-3 hour ave fcst:ens mean
"""


class RunnerPathTests(unittest.TestCase):
    def test_migrated_runner_resolves_the_project_root(self) -> None:
        self.assertEqual(RUNNER.PROJECT_ROOT, ROOT)

    def test_point_record_builder_accepts_a_gefs_product(self) -> None:
        parameters = inspect.signature(RUNNER.build_gefs_point_records).parameters

        self.assertIn("product", parameters)

    def test_point_record_builder_accepts_required_messages(self) -> None:
        parameters = inspect.signature(RUNNER.build_gefs_point_records).parameters

        self.assertIn("required_messages", parameters)
        self.assertEqual(parameters["product"].default, "geavg")


class DownloadRetryTests(unittest.TestCase):
    def test_retries_when_response_body_is_incomplete(self) -> None:
        class FakeResponse:
            def __init__(
                self,
                payload: bytes | None = None,
                error: Exception | None = None,
            ) -> None:
                self.payload = payload
                self.error = error

            def __enter__(self) -> "FakeResponse":
                return self

            def __exit__(self, *args: object) -> None:
                return None

            def read(self) -> bytes:
                if self.error is not None:
                    raise self.error
                assert self.payload is not None
                return self.payload

        responses = [
            FakeResponse(error=http.client.IncompleteRead(b"partial", 3)),
            FakeResponse(payload=b"complete"),
        ]
        with patch(
            "run_gefs_gridmet_bias_validation_v1.urllib.request.urlopen",
            side_effect=responses,
        ) as urlopen, patch(
            "run_gefs_gridmet_bias_validation_v1.time.sleep"
        ) as sleep:
            payload = _request_bytes(
                "https://example.test/range",
                timeout=1,
                retries=2,
            )

        self.assertEqual(payload, b"complete")
        self.assertEqual(urlopen.call_count, 2)
        sleep.assert_called_once_with(1)


class GefsIndexTests(unittest.TestCase):
    def test_parses_index_records_and_step_windows(self) -> None:
        records = parse_gefs_index(INDEX_TEXT)

        self.assertEqual(records[1].short_name, "TMP")
        self.assertEqual(records[1].offset, 3820703)
        self.assertEqual(records[1].step.start_hour, 3)
        self.assertEqual(records[1].step.end_hour, 3)
        self.assertEqual(records[1].step.kind, "instant")

        accumulated = parse_step_window("0-6 hour acc fcst")
        self.assertEqual(
            (accumulated.start_hour, accumulated.end_hour, accumulated.kind),
            (0, 6, "acc"),
        )

    def test_selects_exact_surface_and_height_messages(self) -> None:
        selected = select_gefs_messages(parse_gefs_index(INDEX_TEXT))

        self.assertEqual(
            [record.short_name for record in selected],
            ["TMP", "DPT", "TMAX", "TMIN", "UGRD", "VGRD", "APCP", "DSWRF"],
        )
        apcp = next(record for record in selected if record.short_name == "APCP")
        self.assertEqual(apcp.range_end, 8879530)

    def test_derives_packing_resolution_from_grib_scaling(self) -> None:
        self.assertAlmostEqual(
            packing_resolution(binary_scale_factor=0, decimal_scale_factor=1),
            0.1,
        )


class AccumulationTests(unittest.TestCase):
    def test_reconstructs_nonoverlapping_precipitation_across_resets(self) -> None:
        frame = pd.DataFrame(
            {
                "start_hour": [0, 0, 6, 6],
                "end_hour": [3, 6, 9, 12],
                "value": [1.0, 3.0, 4.0, 9.0],
            }
        )

        result = decumulate_reset_intervals(frame, kind="acc")

        self.assertEqual(result["interval_start_hour"].tolist(), [0, 3, 6, 9])
        self.assertEqual(result["interval_end_hour"].tolist(), [3, 6, 9, 12])
        self.assertEqual(result["interval_value"].tolist(), [1.0, 2.0, 4.0, 5.0])

    def test_reconstructs_interval_mean_from_cumulative_average(self) -> None:
        frame = pd.DataFrame(
            {
                "start_hour": [0, 0],
                "end_hour": [3, 6],
                "value": [100.0, 150.0],
            }
        )

        result = decumulate_reset_intervals(frame, kind="ave")

        self.assertEqual(result["interval_value"].tolist(), [100.0, 200.0])

    def test_rejects_materially_negative_reconstructed_precipitation(self) -> None:
        frame = pd.DataFrame(
            {
                "start_hour": [0, 0],
                "end_hour": [3, 6],
                "value": [2.0, 1.0],
            }
        )

        with self.assertRaisesRegex(ValueError, "negative interval"):
            decumulate_reset_intervals(frame, kind="acc", negative_tolerance=0.01)


class LocalDayMappingTests(unittest.TestCase):
    def test_splits_accumulated_interval_at_local_midnight(self) -> None:
        result = allocate_interval_to_local_days(
            start_utc=datetime(2024, 7, 16, 3, tzinfo=timezone.utc),
            end_utc=datetime(2024, 7, 16, 6, tzinfo=timezone.utc),
            value=3.0,
            kind="acc",
            timezone_name="America/Chicago",
        )

        self.assertEqual(result["local_date"].astype(str).tolist(), ["2024-07-15", "2024-07-16"])
        self.assertEqual(result["overlap_hours"].tolist(), [2.0, 1.0])
        self.assertEqual(result["allocated_value"].tolist(), [2.0, 1.0])

    def test_preserves_interval_mean_as_overlap_weighted_value(self) -> None:
        result = allocate_interval_to_local_days(
            start_utc=datetime(2024, 7, 16, 3, tzinfo=timezone.utc),
            end_utc=datetime(2024, 7, 16, 6, tzinfo=timezone.utc),
            value=120.0,
            kind="ave",
            timezone_name="America/Chicago",
        )

        self.assertEqual(result["weighted_value_hours"].tolist(), [240.0, 120.0])


class GridmetTests(unittest.TestCase):
    def test_reads_scaled_nearest_grid_point(self) -> None:
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "pr_2024.nc"
            with h5py.File(path, "w") as handle:
                handle.create_dataset("lon", data=np.array([-100.0, -99.0]))
                handle.create_dataset("lat", data=np.array([41.0, 42.0]))
                day = handle.create_dataset("day", data=np.array([45472.0, 45473.0]))
                day.attrs["units"] = "days since 1900-01-01 00:00:00"
                values = handle.create_dataset(
                    "precipitation_amount",
                    data=np.array(
                        [
                            [[1, 2], [3, 4]],
                            [[5, 6], [7, 32767]],
                        ],
                        dtype=np.uint16,
                    ),
                )
                values.attrs["scale_factor"] = np.array([0.1])
                values.attrs["add_offset"] = np.array([0.0])
                values.attrs["_FillValue"] = np.array([32767], dtype=np.uint16)

            sites = pd.DataFrame(
                {"site": ["P1"], "latitude": [41.9], "longitude": [-99.1]}
            )
            result = read_gridmet_variable_points(
                path,
                sites=sites,
                dates=["2024-07-01", "2024-07-02"],
                output_variable="precipitation_mm",
            )

            self.assertAlmostEqual(result.iloc[0]["reference_value"], 0.4)
            self.assertTrue(np.isnan(result.iloc[1]["reference_value"]))
            self.assertAlmostEqual(result.iloc[0]["grid_latitude"], 42.0)
            self.assertAlmostEqual(result.iloc[0]["grid_longitude"], -99.0)

    def test_reports_missing_reference_dates(self) -> None:
        reference = pd.DataFrame(
            {
                "site": ["P1"],
                "local_date": [pd.Timestamp("2024-07-16")],
                "variable": ["precipitation_mm"],
                "reference_value": [1.0],
            }
        )

        with self.assertRaisesRegex(ValueError, "missing reference coverage"):
            validate_reference_coverage(
                reference,
                sites=["P1"],
                variables=["precipitation_mm"],
                dates=["2024-07-16", "2024-07-17"],
            )

    def test_converts_gridmet_temperature_from_kelvin_to_celsius(self) -> None:
        frame = pd.DataFrame(
            {
                "variable": ["temperature_min_c", "precipitation_mm"],
                "reference_value": [293.15, 4.0],
            }
        )

        result = convert_gridmet_reference_units(frame)

        self.assertAlmostEqual(result.iloc[0]["reference_value"], 20.0)
        self.assertAlmostEqual(result.iloc[1]["reference_value"], 4.0)


class BiasMetricTests(unittest.TestCase):
    def test_computes_vpd_from_temperature_and_dewpoint(self) -> None:
        value = vapor_pressure_deficit_kpa(293.15, 283.15)

        self.assertAlmostEqual(value, 1.110, places=3)

    def test_computes_signed_bias_and_absolute_errors(self) -> None:
        frame = pd.DataFrame(
            {
                "variable": ["x", "x", "x"],
                "forecast_value": [2.0, 4.0, 9.0],
                "reference_value": [1.0, 5.0, 6.0],
            }
        )

        result = compute_bias_metrics(frame, group_columns=["variable"])

        self.assertEqual(int(result.iloc[0]["n"]), 3)
        self.assertAlmostEqual(result.iloc[0]["bias"], 1.0)
        self.assertAlmostEqual(result.iloc[0]["mae"], 5.0 / 3.0)
        self.assertAlmostEqual(result.iloc[0]["rmse"], (11.0 / 3.0) ** 0.5)
        self.assertAlmostEqual(result.iloc[0]["correlation"], 0.8386279, places=6)

    def test_constant_series_has_undefined_correlation_without_warning(self) -> None:
        frame = pd.DataFrame(
            {
                "variable": ["x", "x"],
                "forecast_value": [1.0, 2.0],
                "reference_value": [4.0, 4.0],
            }
        )

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            result = compute_bias_metrics(frame, group_columns=["variable"])

        self.assertTrue(np.isnan(result.iloc[0]["correlation"]))
        self.assertEqual(caught, [])

    def test_builds_union_of_required_seven_day_dates(self) -> None:
        dates = required_valid_dates(["2024-07-16", "2024-07-20"], horizon_days=7)

        self.assertEqual(dates[0], "2024-07-16")
        self.assertEqual(dates[-1], "2024-07-26")
        self.assertEqual(len(dates), 11)

    def test_classifies_reference_precipitation_intensity(self) -> None:
        frame = pd.DataFrame(
            {
                "variable": ["precipitation_mm"] * 4,
                "reference_value": [0.0, 2.0, 10.0, 25.0],
            }
        )

        result = add_reference_condition(frame)

        self.assertEqual(
            result["reference_condition"].tolist(),
            ["dry", "light", "moderate", "heavy"],
        )

    def test_computes_precipitation_hits_misses_and_false_alarms(self) -> None:
        frame = pd.DataFrame(
            {
                "forecast_value": [0.0, 2.0, 0.0, 5.0],
                "reference_value": [0.0, 0.0, 4.0, 6.0],
            }
        )

        result = compute_precipitation_event_metrics(frame, thresholds_mm=[1.0])

        self.assertEqual(int(result.iloc[0]["hits"]), 1)
        self.assertEqual(int(result.iloc[0]["misses"]), 1)
        self.assertEqual(int(result.iloc[0]["false_alarms"]), 1)
        self.assertEqual(int(result.iloc[0]["correct_negatives"]), 1)
        self.assertAlmostEqual(result.iloc[0]["probability_of_detection"], 0.5)
        self.assertAlmostEqual(result.iloc[0]["false_alarm_ratio"], 0.5)


class ProductAndDailyAggregationTests(unittest.TestCase):
    def test_daily_to_long_accepts_precipitation_only_input(self) -> None:
        daily = pd.DataFrame(
            {
                "site": ["P1"],
                "local_date": [pd.Timestamp("2024-07-16")],
                "precipitation_mm": [4.0],
            }
        )

        result = forecast_daily_to_long(
            daily, variables=("precipitation_mm",)
        )

        self.assertEqual(result["variable"].tolist(), ["precipitation_mm"])
        self.assertEqual(result["forecast_value"].tolist(), [4.0])

    def test_fetches_and_concatenates_selected_ranges(self) -> None:
        calls: list[tuple[str, int, int]] = []

        def fetcher(url: str, start: int, end: int) -> bytes:
            calls.append((url, start, end))
            return bytes([start]) * (end - start + 1)

        payload = fetch_selected_byte_ranges(
            "https://example.test/product",
            [
                ByteRange(start=1, end=3, short_names=("A",)),
                ByteRange(start=8, end=9, short_names=("B",)),
            ],
            fetcher=fetcher,
            workers=2,
        )

        self.assertEqual(payload, b"\x01\x01\x01\x08\x08")
        self.assertEqual(
            calls,
            [
                ("https://example.test/product", 1, 3),
                ("https://example.test/product", 8, 9),
            ],
        )

    def test_builds_official_ensemble_mean_product_url(self) -> None:
        url = build_gefs_product_url("2024-07-16", cycle_hour=0, lead_hour=3)

        self.assertEqual(
            url,
            "https://noaa-gefs-pds.s3.amazonaws.com/gefs.20240716/00/atmos/"
            "pgrb2sp25/geavg.t00z.pgrb2s.0p25.f003",
        )

    def test_merges_only_adjacent_selected_byte_ranges(self) -> None:
        selected = select_gefs_messages(parse_gefs_index(INDEX_TEXT))

        ranges = merge_contiguous_ranges(selected)

        self.assertEqual(
            [(item.start, item.end) for item in ranges],
            [
                (3820703, 4678087),
                (5328427, 7824429),
                (8641665, 8879530),
                (13507961, 13893565),
            ],
        )

    def test_aggregates_point_records_to_local_daily_weather(self) -> None:
        cycle = pd.Timestamp("2024-07-16T00:00:00Z")
        records = pd.DataFrame(
            [
                {"site": "P1", "timezone": "America/Chicago", "cycle_init_utc": cycle, "lead_hour": 6, "short_name": "TMP", "value": 293.15, "start_hour": 6, "end_hour": 6, "kind": "instant"},
                {"site": "P1", "timezone": "America/Chicago", "cycle_init_utc": cycle, "lead_hour": 9, "short_name": "TMP", "value": 295.15, "start_hour": 9, "end_hour": 9, "kind": "instant"},
                {"site": "P1", "timezone": "America/Chicago", "cycle_init_utc": cycle, "lead_hour": 6, "short_name": "DPT", "value": 283.15, "start_hour": 6, "end_hour": 6, "kind": "instant"},
                {"site": "P1", "timezone": "America/Chicago", "cycle_init_utc": cycle, "lead_hour": 9, "short_name": "DPT", "value": 285.15, "start_hour": 9, "end_hour": 9, "kind": "instant"},
                {"site": "P1", "timezone": "America/Chicago", "cycle_init_utc": cycle, "lead_hour": 6, "short_name": "UGRD", "value": 3.0, "start_hour": 6, "end_hour": 6, "kind": "instant"},
                {"site": "P1", "timezone": "America/Chicago", "cycle_init_utc": cycle, "lead_hour": 9, "short_name": "UGRD", "value": 0.0, "start_hour": 9, "end_hour": 9, "kind": "instant"},
                {"site": "P1", "timezone": "America/Chicago", "cycle_init_utc": cycle, "lead_hour": 6, "short_name": "VGRD", "value": 4.0, "start_hour": 6, "end_hour": 6, "kind": "instant"},
                {"site": "P1", "timezone": "America/Chicago", "cycle_init_utc": cycle, "lead_hour": 9, "short_name": "VGRD", "value": 0.0, "start_hour": 9, "end_hour": 9, "kind": "instant"},
                {"site": "P1", "timezone": "America/Chicago", "cycle_init_utc": cycle, "lead_hour": 9, "short_name": "TMAX", "value": 298.15, "start_hour": 6, "end_hour": 9, "kind": "max"},
                {"site": "P1", "timezone": "America/Chicago", "cycle_init_utc": cycle, "lead_hour": 9, "short_name": "TMIN", "value": 290.15, "start_hour": 6, "end_hour": 9, "kind": "min"},
                {"site": "P1", "timezone": "America/Chicago", "cycle_init_utc": cycle, "lead_hour": 3, "short_name": "APCP", "value": 3.0, "start_hour": 0, "end_hour": 3, "kind": "acc"},
                {"site": "P1", "timezone": "America/Chicago", "cycle_init_utc": cycle, "lead_hour": 6, "short_name": "APCP", "value": 6.0, "start_hour": 0, "end_hour": 6, "kind": "acc"},
                {"site": "P1", "timezone": "America/Chicago", "cycle_init_utc": cycle, "lead_hour": 9, "short_name": "APCP", "value": 3.0, "start_hour": 6, "end_hour": 9, "kind": "acc"},
            ]
        )

        result = aggregate_gefs_point_records(records)
        day = result.loc[result["local_date"].eq(pd.Timestamp("2024-07-16"))].iloc[0]

        self.assertAlmostEqual(day["precipitation_mm"], 4.0)
        self.assertAlmostEqual(day["temperature_min_c"], 17.0)
        self.assertAlmostEqual(day["temperature_max_c"], 25.0)
        self.assertAlmostEqual(day["wind_speed_m_s"], 2.5)
        self.assertGreater(day["vpd_kpa"], 1.0)

    def test_pairs_long_forecasts_with_unique_reference_values(self) -> None:
        daily = pd.DataFrame(
            {
                "site": ["P1"],
                "timezone": ["America/Chicago"],
                "cycle_init_utc": [pd.Timestamp("2024-07-16T00:00:00Z")],
                "decision_date": [pd.Timestamp("2024-07-16")],
                "local_date": [pd.Timestamp("2024-07-16")],
                "lead_day": [1],
                "precipitation_mm": [2.0],
                "temperature_min_c": [18.0],
                "temperature_max_c": [30.0],
                "shortwave_w_m2": [250.0],
                "wind_speed_m_s": [3.0],
                "vpd_kpa": [1.2],
            }
        )
        forecast = forecast_daily_to_long(daily)
        reference = forecast[["site", "local_date", "variable"]].copy()
        reference["reference_value"] = forecast["forecast_value"] - 1.0

        paired = pair_forecast_and_reference(forecast, reference)

        self.assertEqual(len(paired), 6)
        self.assertTrue(paired["reference_value"].notna().all())
        self.assertTrue(np.allclose(paired["error"], 1.0))

    def test_rejects_duplicate_reference_values(self) -> None:
        forecast = pd.DataFrame(
            {
                "site": ["P1"],
                "local_date": [pd.Timestamp("2024-07-16")],
                "variable": ["precipitation_mm"],
                "forecast_value": [2.0],
            }
        )
        reference = pd.DataFrame(
            {
                "site": ["P1", "P1"],
                "local_date": [pd.Timestamp("2024-07-16")] * 2,
                "variable": ["precipitation_mm"] * 2,
                "reference_value": [1.0, 1.5],
            }
        )

        with self.assertRaisesRegex(ValueError, "duplicate reference"):
            pair_forecast_and_reference(forecast, reference)


if __name__ == "__main__":
    unittest.main()
