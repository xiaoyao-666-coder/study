from __future__ import annotations

import argparse
import csv
import json
import unittest
from datetime import datetime, timezone
from pathlib import Path
from tempfile import TemporaryDirectory

from scripts.data_preparation.download_gefs_p3_2022_era5_local_v1 import (
    load_contract,
    validate_target_rows,
)
from scripts.evaluation.freeze_gefs_p3_2022_era5_local_acquisition_v1 import (
    INDEPENDENT_STATUS,
    OUTPUT_NAMES,
    SUCCESS_STATUS,
    VARIABLES,
    build_acquisition_contract,
    sha256_file,
    validate_independent_protocol,
    write_json,
)


def sample_archive() -> dict[str, object]:
    return {
        "historical_year_counts": {"2015": 364, "2019": 364},
        "grid": {"crs": "EPSG:4326", "dtype": "float32"},
        "p3_export_pixel": {
            "row": 75,
            "col": 311,
            "bounds": [-96.95, 46.30, -96.86, 46.39],
            "center": [-96.91, 46.35],
            "one_pixel_transform": [0.09, 0.0, -96.95, 0.0, -0.09, 46.39],
        },
        "representative_files": [],
    }


def raw_weather_row(day: str, value: float) -> dict[str, object]:
    stamp = datetime.fromisoformat(day).replace(tzinfo=timezone.utc)
    row: dict[str, object] = {
        "id": day.replace("-", ""),
        "time": int(stamp.timestamp() * 1000),
        "date": day,
    }
    row.update({variable: value + index for index, variable in enumerate(VARIABLES)})
    return row


class P32022Era5LocalAcquisitionTests(unittest.TestCase):
    def test_contract_is_bound_to_2022_point_pixel_without_labels(self) -> None:
        independent = {
            "audit": {"status": INDEPENDENT_STATUS},
            "protocol": {"target": {"site_id": "P3", "target_year": 2022}},
        }
        contract = build_acquisition_contract(independent, sample_archive())
        source = contract["source_product"]
        self.assertEqual(contract["target_year"], 2022)
        self.assertEqual(source["collection_start_inclusive"], "2022-01-01")
        self.assertEqual(source["collection_end_exclusive"], "2022-12-31")
        self.assertEqual(source["expected_day_count"], 364)
        self.assertEqual(contract["variables"], list(VARIABLES))
        self.assertFalse(contract["spatial_contract"]["full_CONUS_raster_download_required"])
        self.assertFalse(contract["method_invariants"]["target_label_access_allowed"])
        self.assertFalse(
            contract["method_invariants"]["decision_schedule_resolution_allowed_during_download"]
        )

    def test_target_rows_require_exact_2022_sequence(self) -> None:
        rows = [raw_weather_row("2022-01-01", 1.0), raw_weather_row("2022-01-02", 2.0)]
        validated = validate_target_rows(rows, "2022-01-01", 2)
        self.assertEqual([row["day_index"] for row in validated], [0, 1])
        rows[1]["date"] = "2022-01-03"
        with self.assertRaisesRegex(ValueError, "date sequence"):
            validate_target_rows(rows, "2022-01-01", 2)

    def test_download_contract_manifest_is_verified(self) -> None:
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            contract_path = root / OUTPUT_NAMES["contract"]
            audit_path = root / OUTPUT_NAMES["audit"]
            manifest_path = root / OUTPUT_NAMES["manifest"]
            contract = build_acquisition_contract(
                {"audit": {"status": INDEPENDENT_STATUS}}, sample_archive()
            )
            write_json(contract_path, contract)
            audit = {
                "status": SUCCESS_STATUS,
                "mandatory_gate_passed": True,
                "P3_2022_weather_rows_read": 0,
                "P3_2022_network_requests": 0,
                "P3_2022_target_labels_read": 0,
                "P3_2023_rows_or_labels_read": 0,
                "P3_2024_rows_or_labels_read": 0,
            }
            write_json(audit_path, audit)
            with manifest_path.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(
                    handle, fieldnames=["role", "path", "bytes", "sha256"]
                )
                writer.writeheader()
                for key, path in (("contract", contract_path), ("audit", audit_path)):
                    writer.writerow(
                        {
                            "role": f"output_{key}",
                            "path": path.name,
                            "bytes": path.stat().st_size,
                            "sha256": sha256_file(path),
                        }
                    )
            loaded = load_contract(root)
            self.assertEqual(loaded["target_year"], 2022)
            audit["P3_2022_network_requests"] = 1
            write_json(audit_path, audit)
            with self.assertRaisesRegex(ValueError, "not zero"):
                load_contract(root)

    def test_returned_independent_protocol_is_exactly_verified(self) -> None:
        root = (
            Path(__file__).resolve().parents[1]
            / "site_general_surrogate_eval"
            / "gefs_p3_2022_b2_independent_protocol_v1"
        )
        validated = validate_independent_protocol(root)
        self.assertEqual(validated["audit"]["status"], INDEPENDENT_STATUS)


if __name__ == "__main__":
    unittest.main()
