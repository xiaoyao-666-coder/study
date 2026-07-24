from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pandas as pd

from scripts.data_preparation import (
    extract_gefs_2015_2019_full_weather_pilot_v1 as runner,
)
from s2s_rtist.weather.gefs_quantile_mapping import GEFS_REFORECAST_MEMBERS
from s2s_rtist.weather.gefs_reforecast_full_weather import (
    REQUIRED_PRODUCT_SPECS,
    ReforecastObject,
    aggregate_member_weather,
    parse_reforecast_inventory,
    reforecast_inventory_url,
    select_product_objects,
    select_product_records,
    specific_humidity_to_vapor_pressure_kpa,
    validate_full_weather,
)


class ReforecastInventoryTests(unittest.TestCase):
    def test_builds_encoded_official_inventory_url(self) -> None:
        url = reforecast_inventory_url("2015-08-15", "c00")

        self.assertTrue(url.startswith("https://noaa-gefs-retrospective.s3.amazonaws.com/?"))
        self.assertIn("2015081500%2Fc00%2FDays%3A1-10%2F", url)

    def test_parses_inventory_and_selects_required_products(self) -> None:
        prefix = "GEFSv12/reforecast/2015/2015081500/c00/Days:1-10/"
        contents = []
        expected_size = 100
        for spec in REQUIRED_PRODUCT_SPECS:
            filename = f"{spec.product_id}_2015081500_c00.grib2"
            for suffix, size in (("", expected_size), (".idx", 10)):
                contents.append(
                    "<Contents>"
                    f"<Key>{prefix}{filename}{suffix}</Key>"
                    "<LastModified>2020-01-01T00:00:00.000Z</LastModified>"
                    f"<ETag>&quot;etag-{spec.product_id}{suffix}&quot;</ETag>"
                    f"<Size>{size}</Size>"
                    "</Contents>"
                )
        xml = (
            '<ListBucketResult xmlns="http://s3.amazonaws.com/doc/2006-03-01/">'
            "<IsTruncated>false</IsTruncated>"
            + "".join(contents)
            + "</ListBucketResult>"
        )

        objects = parse_reforecast_inventory(xml)
        selected = select_product_objects(objects)

        self.assertEqual(len(selected), len(REQUIRED_PRODUCT_SPECS))
        self.assertEqual(
            {item.spec.product_id for item in selected},
            {item.product_id for item in REQUIRED_PRODUCT_SPECS},
        )
        self.assertTrue(all(item.product.size == expected_size for item in selected))

    def test_rejects_missing_required_product(self) -> None:
        objects = [
            ReforecastObject(
                key="prefix/apcp_sfc_2015081500_c00.grib2",
                size=100,
                etag="etag",
                last_modified="",
            ),
            ReforecastObject(
                key="prefix/apcp_sfc_2015081500_c00.grib2.idx",
                size=10,
                etag="idx-etag",
                last_modified="",
            ),
        ]

        with self.assertRaisesRegex(ValueError, "missing required reforecast products"):
            select_product_objects(objects)


class ReforecastRecordSelectionTests(unittest.TestCase):
    def test_selects_complete_three_hour_temperature_records(self) -> None:
        spec = next(item for item in REQUIRED_PRODUCT_SPECS if item.short_name == "TMP")
        lines = []
        for message, end_hour in enumerate(range(3, 181, 3), start=1):
            lines.append(
                f"{message}:{(message - 1) * 100}:d=2015081500:TMP:"
                f"2 m above ground:{end_hour} hour fcst:ENS=test"
            )

        selected = select_product_records("\n".join(lines), spec=spec)

        self.assertEqual(len(selected), 58)
        self.assertEqual(selected[0].step.end_hour, 3)
        self.assertEqual(selected[-1].step.end_hour, 174)

    def test_rejects_incomplete_three_hour_coverage(self) -> None:
        spec = next(item for item in REQUIRED_PRODUCT_SPECS if item.short_name == "TMP")
        lines = [
            "1:0:d=2015081500:TMP:2 m above ground:3 hour fcst:ENS=test",
            "2:100:d=2015081500:TMP:2 m above ground:9 hour fcst:ENS=test",
            "3:200:d=2015081500:TMP:2 m above ground:174 hour fcst:ENS=test",
            "4:300:d=2015081500:TMP:2 m above ground:177 hour fcst:ENS=test",
        ]

        with self.assertRaisesRegex(ValueError, "complete three-hour coverage"):
            select_product_records("\n".join(lines), spec=spec)


def synthetic_point_records() -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    cycle = pd.Timestamp("2015-08-15T00:00:00Z")
    for end_hour in range(3, 175, 3):
        for short_name, value, kind in (
            ("TMP", 293.15 + (end_hour % 12) / 6.0, "instant"),
            ("SPFH", 0.008, "instant"),
            ("PRES", 100000.0, "instant"),
            ("UGRD", 3.0, "instant"),
            ("VGRD", 4.0, "instant"),
            ("APCP", 1.0, "acc"),
            ("DSWRF", 100.0, "ave"),
        ):
            start_hour = end_hour if kind == "instant" else end_hour - 3
            rows.append(
                {
                    "site": "P1",
                    "timezone": "America/Chicago",
                    "cycle_init_utc": cycle,
                    "lead_hour": 0,
                    "short_name": short_name,
                    "value": value,
                    "start_hour": start_hour,
                    "end_hour": end_hour,
                    "kind": kind,
                    "packing_resolution": 0.01,
                }
            )
    return pd.DataFrame(rows)


class FullWeatherAggregationTests(unittest.TestCase):
    def test_aggregates_canonical_weather_and_converts_units(self) -> None:
        manifest = pd.DataFrame(
            {
                "source_key": [f"key-{index}" for index in range(6)],
                "source_etag": [f"etag-{index}" for index in range(6)],
            }
        )

        daily = aggregate_member_weather(
            synthetic_point_records(),
            member="c00",
            product_manifest=manifest,
        )

        self.assertEqual(len(daily), 7)
        self.assertEqual(daily["lead_day"].tolist(), list(range(1, 8)))
        self.assertTrue(np.allclose(daily["wind_speed_m_s"], 5.0))
        self.assertTrue(np.allclose(daily["shortwave_w_m2"], 100.0))
        self.assertTrue(np.allclose(daily["solar_kj_m2_day"], 8640.0))
        self.assertTrue(daily["actual_vapor_pressure_kpa"].gt(0.0).all())
        self.assertTrue(
            (daily["temperature_min_c"] <= daily["temperature_max_c"]).all()
        )

    def test_converts_specific_humidity_and_surface_pressure(self) -> None:
        result = specific_humidity_to_vapor_pressure_kpa(
            pd.Series([0.008]), pd.Series([100000.0])
        )

        self.assertAlmostEqual(float(result.iloc[0]), 1.279951, places=5)


def full_contract_frame() -> pd.DataFrame:
    cycles = (
        "2015-07-15",
        "2016-07-15",
        "2017-07-15",
        "2018-07-15",
        "2019-07-15",
    )
    sites = ("P1", "P2", "P3", "P4", "P15")
    rows = []
    for cycle in cycles:
        for site in sites:
            for member in GEFS_REFORECAST_MEMBERS:
                for lead_day in range(1, 8):
                    rows.append(
                        {
                            "decision_date": cycle,
                            "site_id": site,
                            "gefs_member": member,
                            "local_date": pd.Timestamp(cycle)
                            + pd.Timedelta(days=lead_day - 1),
                            "lead_day": lead_day,
                            "precipitation_mm_raw": 1.0,
                            "temperature_min_c": 10.0,
                            "temperature_max_c": 20.0,
                            "actual_vapor_pressure_kpa": 1.0,
                            "wind_speed_m_s": 2.0,
                            "solar_kj_m2_day": 8000.0,
                        }
                    )
    return pd.DataFrame(rows)


class FullWeatherContractTests(unittest.TestCase):
    def test_accepts_exact_875_row_frozen_pilot(self) -> None:
        cycles = (
            "2015-07-15",
            "2016-07-15",
            "2017-07-15",
            "2018-07-15",
            "2019-07-15",
        )
        sites = ("P1", "P2", "P3", "P4", "P15")

        audit = validate_full_weather(
            full_contract_frame(),
            expected_cycles=cycles,
            expected_sites=sites,
            expected_members=GEFS_REFORECAST_MEMBERS,
        )

        self.assertEqual(audit["row_count"], 875)
        self.assertEqual(audit["duplicate_sample_key_count"], 0)
        self.assertFalse(audit["contains_2024"])

    def test_rejects_2024(self) -> None:
        frame = full_contract_frame()
        frame.loc[frame["decision_date"].eq("2019-07-15"), "decision_date"] = "2024-07-15"

        with self.assertRaisesRegex(ValueError, "2024 is forbidden"):
            validate_full_weather(
                frame,
                expected_cycles=(
                    "2015-07-15",
                    "2016-07-15",
                    "2017-07-15",
                    "2018-07-15",
                    "2024-07-15",
                ),
                expected_sites=("P1", "P2", "P3", "P4", "P15"),
                expected_members=GEFS_REFORECAST_MEMBERS,
            )


class FullWeatherRunnerTests(unittest.TestCase):
    def test_writes_audited_outputs_without_retaining_grib(self) -> None:
        daily = pd.DataFrame(
            {
                "site_id": ["P1"] * 7,
                "site_timezone": ["America/Chicago"] * 7,
                "forecast_init_utc": [pd.Timestamp("2015-08-15T00:00:00Z")] * 7,
                "decision_date": [pd.Timestamp("2015-08-15")] * 7,
                "gefs_member": ["c00"] * 7,
                "local_date": pd.date_range("2015-08-15", periods=7),
                "lead_day": list(range(1, 8)),
                "precipitation_mm_raw": [1.0] * 7,
                "temperature_min_c": [10.0] * 7,
                "temperature_max_c": [20.0] * 7,
                "actual_vapor_pressure_kpa": [1.0] * 7,
                "wind_speed_m_s": [2.0] * 7,
                "solar_kj_m2_day": [8000.0] * 7,
            }
        )
        product_rows = [
            {
                "cycle_date": "2015-08-15",
                "gefs_member": "c00",
                "product_id": "apcp_sfc",
                "selected_range_bytes": 100,
                "network_bytes_this_run": 110,
            }
        ]
        inventory = {
            "cycle_date": "2015-08-15",
            "gefs_member": "c00",
            "network_bytes_this_run": 20,
        }
        preflight = pd.DataFrame(
            {
                "cycle_date": ["2015-08-15"],
                "gefs_member": ["c00"],
                "product_id": ["apcp_sfc"],
                "selected_range_bytes": [100],
                "index_network_bytes_this_run": [5],
            }
        )
        with tempfile.TemporaryDirectory() as directory, patch.object(
            runner,
            "preflight_extraction",
            return_value=({("2015-08-15", "c00"): []}, preflight, pd.DataFrame([inventory])),
        ), patch.object(
            runner,
            "_extract_member",
            return_value=(daily, product_rows),
        ):
            outputs = runner.run_extraction(
                cycles=("2015-08-15",),
                site_ids=("P1",),
                members=("c00",),
                output_dir=Path(directory),
                timeout=1,
                retries=1,
                workers=1,
            )
            audit = pd.read_json(outputs["audit"], typ="series")

            self.assertEqual(int(audit["row_count"]), 7)
            self.assertEqual(int(audit["network_bytes_this_run"]), 135)
            self.assertEqual(int(audit["retained_grib_file_count"]), 0)
            self.assertTrue(outputs["manifest"].exists())
            self.assertFalse(list(Path(directory).rglob("*.grib2")))

    def test_product_parallel_mode_downloads_each_product_independently(self) -> None:
        daily = pd.DataFrame(
            {
                "site_id": ["P1"] * 7,
                "site_timezone": ["America/Chicago"] * 7,
                "forecast_init_utc": [pd.Timestamp("2015-08-15T00:00:00Z")] * 7,
                "decision_date": [pd.Timestamp("2015-08-15")] * 7,
                "gefs_member": ["c00"] * 7,
                "local_date": pd.date_range("2015-08-15", periods=7),
                "lead_day": list(range(1, 8)),
                "precipitation_mm_raw": [1.0] * 7,
                "temperature_min_c": [10.0] * 7,
                "temperature_max_c": [20.0] * 7,
                "actual_vapor_pressure_kpa": [1.0] * 7,
                "wind_speed_m_s": [2.0] * 7,
                "solar_kj_m2_day": [8000.0] * 7,
            }
        )
        pairs = [
            SimpleNamespace(spec=SimpleNamespace(product_id=f"product_{index}"))
            for index in range(7)
        ]
        preflight = pd.DataFrame(
            {
                "cycle_date": ["2015-08-15"] * 7,
                "gefs_member": ["c00"] * 7,
                "product_id": [f"product_{index}" for index in range(7)],
                "selected_range_bytes": [100] * 7,
                "index_network_bytes_this_run": [0] * 7,
            }
        )
        inventory = pd.DataFrame(
            [
                {
                    "cycle_date": "2015-08-15",
                    "gefs_member": "c00",
                    "network_bytes_this_run": 0,
                }
            ]
        )

        def downloaded_product(**kwargs):
            product_id = kwargs["pair"].spec.product_id
            return pd.DataFrame({"product_id": [product_id]}), {
                "cycle_date": "2015-08-15",
                "gefs_member": "c00",
                "product_id": product_id,
                "selected_range_bytes": 100,
                "network_bytes_this_run": 100,
            }

        with tempfile.TemporaryDirectory() as directory, patch.object(
            runner,
            "preflight_extraction",
            return_value=(
                {("2015-08-15", "c00"): pairs},
                preflight,
                inventory,
            ),
        ), patch.object(
            runner,
            "download_product_points",
            side_effect=downloaded_product,
        ) as download_mock, patch.object(
            runner,
            "aggregate_member_weather",
            return_value=daily,
        ) as aggregate_mock, patch.object(
            runner,
            "_extract_member",
        ) as member_mock:
            outputs = runner.run_extraction(
                cycles=("2015-08-15",),
                site_ids=("P1",),
                members=("c00",),
                output_dir=Path(directory),
                timeout=1,
                retries=1,
                workers=1,
                product_workers=8,
                product_range_workers=4,
            )

            self.assertEqual(download_mock.call_count, 7)
            self.assertTrue(
                all(call.kwargs["range_workers"] == 4 for call in download_mock.call_args_list)
            )
            self.assertEqual(aggregate_mock.call_count, 1)
            member_mock.assert_not_called()
            self.assertTrue(outputs["audit"].exists())


if __name__ == "__main__":
    unittest.main()
