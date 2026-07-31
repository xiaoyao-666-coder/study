#!/usr/bin/env python3
"""Freeze the local P3 2022 ERA5-Land acquisition before network access."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Any, Sequence

from scripts.evaluation.freeze_gefs_p3_2021_era5_local_acquisition_v1 import (
    EARTH_ENGINE_ASSET,
    TARGET_LATITUDE,
    TARGET_LONGITUDE,
    VARIABLES,
    inspect_historical_archive,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INDEPENDENT_PROTOCOL_DIR = (
    PROJECT_ROOT
    / "site_general_surrogate_eval"
    / "gefs_p3_2022_b2_independent_protocol_v1"
)
DEFAULT_HISTORICAL_ERA5_ROOT = PROJECT_ROOT / "model3_opt_sto_upload" / "data"
DEFAULT_OUTPUT_DIR = (
    PROJECT_ROOT
    / "site_general_surrogate_eval"
    / "gefs_p3_2022_era5_local_acquisition_protocol_v1"
)

INDEPENDENT_PROTOCOL_NAMES = {
    "protocol": "gefs_p3_2022_b2_independent_protocol_v1.json",
    "components": "gefs_p3_2022_b2_component_registry_v1.csv",
    "stages": "gefs_p3_2022_b2_stage_registry_v1.csv",
    "audit": "gefs_p3_2022_b2_independent_protocol_audit_v1.json",
    "manifest": "gefs_p3_2022_b2_independent_protocol_manifest_v1.csv",
}
EXPECTED_INDEPENDENT_PROTOCOL_SHA256 = {
    "protocol": "eadecbcc25863642ce355a853b5d390fb7534f72deae1bc30a671f7f2102a255",
    "components": "c7a7f06bbccef0c51eae690c8f27443b01574b4ba1f4bfa2edbb3218f23b8911",
    "stages": "9ce72a98aa8557adf1574d2cc7098c8cdb2489a2325bfb524db322fa5164a853",
    "audit": "1f021c5b80ccf48339c47d0cc1b9ad9ec8ec5e7fcbbfa14835ed92d4bf4013fb",
    "manifest": "25e731ab750e3b19d0d7ce835d0977aa435c3d494cd184f05d0b25abb9162d34",
}
OUTPUT_NAMES = {
    "contract": "gefs_p3_2022_era5_local_acquisition_contract_v1.json",
    "audit": "gefs_p3_2022_era5_local_acquisition_protocol_audit_v1.json",
    "manifest": "gefs_p3_2022_era5_local_acquisition_protocol_manifest_v1.csv",
}
INDEPENDENT_STATUS = (
    "p3_2022_b2_independent_protocol_frozen_before_target_data_access"
)
SUCCESS_STATUS = (
    "p3_2022_era5_local_acquisition_protocol_frozen_before_network_access"
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object: {path}")
    return payload


def write_json(path: Path, payload: object) -> None:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def validate_independent_protocol(protocol_dir: Path) -> dict[str, Any]:
    paths = {key: protocol_dir / name for key, name in INDEPENDENT_PROTOCOL_NAMES.items()}
    for key, path in paths.items():
        if not path.is_file():
            raise FileNotFoundError(f"independent protocol {key} is missing: {path}")
        actual = sha256_file(path)
        expected = EXPECTED_INDEPENDENT_PROTOCOL_SHA256[key]
        if actual != expected:
            raise ValueError(
                f"independent protocol {key} SHA256 changed: expected {expected}, received {actual}"
            )

    protocol = read_json(paths["protocol"])
    audit = read_json(paths["audit"])
    if protocol.get("status") != INDEPENDENT_STATUS:
        raise ValueError("independent protocol status changed")
    if audit.get("status") != INDEPENDENT_STATUS:
        raise ValueError("independent protocol audit status changed")
    if audit.get("mandatory_gate_passed") is not True:
        raise ValueError("independent protocol mandatory gate did not pass")
    if protocol.get("target", {}).get("site_id") != "P3":
        raise ValueError("independent protocol target site changed")
    if protocol.get("target", {}).get("target_year") != 2022:
        raise ValueError("independent protocol target year changed")
    if protocol.get("schedule_rule", {}).get("trunk_weather_source") != (
        "ERA5_2022_observed_daily_weather"
    ):
        raise ValueError("independent protocol ERA5 schedule source changed")
    zero_fields = [
        key
        for key in audit
        if key.startswith("P3_2022_")
        or key.startswith("P3_2023_")
        or key.startswith("P3_2024_")
    ]
    if not zero_fields or any(audit[key] != 0 for key in zero_fields):
        raise ValueError("independent protocol target-year zero-access gate failed")

    with paths["stages"].open(encoding="utf-8", newline="") as handle:
        stages = list(csv.DictReader(handle))
    if [int(row["stage_order"]) for row in stages] != list(range(1, 9)):
        raise ValueError("independent protocol stage order changed")
    if stages[0]["stage_id"] != "protocol_freeze":
        raise ValueError("independent protocol stage 1 changed")
    if stages[1]["stage_id"] != "local_ERA5_2022_acquisition":
        raise ValueError("independent protocol stage 2 changed")
    if stages[0]["eligible_now"].lower() != "true":
        raise ValueError("independent protocol stage 1 was not eligible")

    with paths["manifest"].open(encoding="utf-8", newline="") as handle:
        manifest = list(csv.DictReader(handle))
    output_rows = {
        row["role"].removeprefix("output_"): row
        for row in manifest
        if row["role"].startswith("output_")
    }
    for key in ("protocol", "components", "stages", "audit"):
        row = output_rows.get(key)
        if row is None or row["sha256"] != EXPECTED_INDEPENDENT_PROTOCOL_SHA256[key]:
            raise ValueError(f"independent manifest binding failed for {key}")
    return {"protocol": protocol, "audit": audit, "paths": paths}


def build_acquisition_contract(
    independent: dict[str, Any], archive: dict[str, Any]
) -> dict[str, Any]:
    return {
        "contract_id": "gefs-p3-2022-era5-local-acquisition-v1",
        "status": "frozen_before_2022_weather_or_network_access",
        "predecessor_protocol": {
            "status": independent["audit"]["status"],
            "files_sha256": dict(EXPECTED_INDEPENDENT_PROTOCOL_SHA256),
        },
        "target_site": "P3",
        "target_coordinate": {
            "longitude": TARGET_LONGITUDE,
            "latitude": TARGET_LATITUDE,
        },
        "target_year": 2022,
        "source_product": {
            "provider": "Google Earth Engine daily aggregate of ECMWF ERA5-Land",
            "asset_id": EARTH_ENGINE_ASSET,
            "collection_start_inclusive": "2022-01-01",
            "collection_end_exclusive": "2022-12-31",
            "expected_day_count": 364,
            "day_index_rule": "zero_based_days_since_2022_01_01",
            "utc_day_boundary": True,
            "flow_band_rule": "sum_for_calendar_day_using_following_00_UTC_accumulation",
            "nonflow_band_rule": "mean_or_extreme_over_calendar_day_as_named",
        },
        "variables": list(VARIABLES),
        "spatial_contract": {
            "mode": "P3_only_exact_historical_export_grid_pixel",
            "sampling": "nearest_source_value_at_historical_export_pixel_center",
            "full_CONUS_raster_download_required": False,
            "scientific_equivalence": (
                "the historical workflow samples only this P3 pixel from each raster"
            ),
            **archive["p3_export_pixel"],
        },
        "historical_archive_contract": {
            "counts": archive["historical_year_counts"],
            "grid": archive["grid"],
            "representative_files": archive["representative_files"],
        },
        "mandatory_historical_parity_gate": {
            "year": 2019,
            "day_indices": [0, 181, 363],
            "dates": ["2019-01-01", "2019-07-01", "2019-12-30"],
            "variables": list(VARIABLES),
            "relative_tolerance": 1.0e-6,
            "absolute_tolerance": 1.0e-7,
            "scope": "all_2019_days_all_nine_variables",
            "must_pass_before_2022_value_request": True,
        },
        "output_contract": {
            "directory_name": "era5_2022",
            "layout": "era5_2022/<variable>/<variable>_<day_index>.tif",
            "raster_shape": [1, 1],
            "raster_crs": archive["grid"]["crs"],
            "raster_dtype": "float32",
            "expected_raster_file_count": 364 * len(VARIABLES),
            "raw_point_values_and_sha256_manifest_required": True,
        },
        "method_invariants": {
            "historical_and_2022_trunk_source": "ERA5-Land Daily Aggregated",
            "future_D_through_D_plus_6_source": "frozen_corrected_GEFS",
            "model_training_allowed": False,
            "model_or_threshold_reselection_allowed": False,
            "target_label_access_allowed": False,
            "decision_schedule_resolution_allowed_during_download": False,
            "P3_2023_or_2024_access_allowed": False,
        },
    }


def run(args: argparse.Namespace) -> dict[str, Path]:
    output_dir = Path(args.output_dir).resolve()
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite acquisition protocol: {output_dir}")
    independent = validate_independent_protocol(Path(args.protocol_dir).resolve())
    archive = inspect_historical_archive(Path(args.historical_era5_root).resolve())
    contract = build_acquisition_contract(independent, archive)

    output_dir.mkdir(parents=True)
    outputs = {key: output_dir / name for key, name in OUTPUT_NAMES.items()}
    write_json(outputs["contract"], contract)
    audit = {
        "status": SUCCESS_STATUS,
        "mandatory_gate_passed": True,
        "predecessor_protocol_verified": True,
        "historical_archive_verified": True,
        "historical_parity_gate_frozen": True,
        "target_site": "P3",
        "target_year": 2022,
        "P3_2022_weather_rows_read": 0,
        "P3_2022_network_requests": 0,
        "P3_2022_target_labels_read": 0,
        "P3_2023_rows_or_labels_read": 0,
        "P3_2024_rows_or_labels_read": 0,
        "ERA5_download_performed": False,
        "model_training_performed": False,
        "model_or_threshold_reselection_performed": False,
        "decision_dates_resolved": False,
        "next_gate": "run_historical_parity_then_download_P3_2022_ERA5_point_pixel",
    }
    write_json(outputs["audit"], audit)

    rows = []
    for key, path in independent["paths"].items():
        rows.append(
            {
                "role": f"input_independent_{key}",
                "path": path.name,
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
    code_path = Path(__file__).resolve()
    rows.append(
        {
            "role": "code",
            "path": code_path.relative_to(PROJECT_ROOT).as_posix(),
            "bytes": code_path.stat().st_size,
            "sha256": sha256_file(code_path),
        }
    )
    for key in ("contract", "audit"):
        path = outputs[key]
        rows.append(
            {
                "role": f"output_{key}",
                "path": path.name,
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
    with outputs["manifest"].open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["role", "path", "bytes", "sha256"])
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps(audit, indent=2, sort_keys=True), flush=True)
    return outputs


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol-dir", type=Path, default=DEFAULT_INDEPENDENT_PROTOCOL_DIR)
    parser.add_argument(
        "--historical-era5-root", type=Path, default=DEFAULT_HISTORICAL_ERA5_ROOT
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args(argv)


if __name__ == "__main__":
    generated = run(parse_args())
    for name, path in generated.items():
        print(f"{name}: {path}", flush=True)
