#!/usr/bin/env python3
"""Download the frozen P3 2022 ERA5-Land point pixel on Windows."""

from __future__ import annotations

import argparse
import csv
import json
import math
from datetime import date, timedelta
from pathlib import Path
from typing import Any, Sequence

from scripts.data_preparation.download_gefs_p3_2021_era5_local_v1 import (
    _fetch_region_rows,
    _rows_from_get_region,
    verify_historical_parity,
    write_csv,
    write_point_archive,
)
from scripts.evaluation.freeze_gefs_p3_2022_era5_local_acquisition_v1 import (
    DEFAULT_HISTORICAL_ERA5_ROOT,
    DEFAULT_OUTPUT_DIR as DEFAULT_ACQUISITION_PROTOCOL_DIR,
    EARTH_ENGINE_ASSET,
    OUTPUT_NAMES as PROTOCOL_OUTPUT_NAMES,
    SUCCESS_STATUS as PROTOCOL_SUCCESS_STATUS,
    VARIABLES,
    read_json,
    sha256_file,
    write_json,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT_DIR = DEFAULT_HISTORICAL_ERA5_ROOT / "era5_2022"


def load_contract(protocol_dir: Path) -> dict[str, Any]:
    paths = {key: protocol_dir / name for key, name in PROTOCOL_OUTPUT_NAMES.items()}
    for path in paths.values():
        if not path.is_file():
            raise FileNotFoundError(f"ERA5 2022 acquisition protocol file is missing: {path}")
    contract = read_json(paths["contract"])
    audit = read_json(paths["audit"])
    if contract.get("contract_id") != "gefs-p3-2022-era5-local-acquisition-v1":
        raise ValueError("ERA5 2022 acquisition contract id changed")
    if contract.get("target_year") != 2022 or contract.get("target_site") != "P3":
        raise ValueError("ERA5 2022 acquisition target changed")
    if contract.get("source_product", {}).get("asset_id") != EARTH_ENGINE_ASSET:
        raise ValueError("ERA5 2022 source asset changed")
    if contract.get("variables") != list(VARIABLES):
        raise ValueError("ERA5 2022 variable contract changed")
    if audit.get("status") != PROTOCOL_SUCCESS_STATUS:
        raise ValueError("ERA5 2022 acquisition protocol status changed")
    if audit.get("mandatory_gate_passed") is not True:
        raise ValueError("ERA5 2022 acquisition protocol gate failed")
    for key in (
        "P3_2022_weather_rows_read",
        "P3_2022_network_requests",
        "P3_2022_target_labels_read",
        "P3_2023_rows_or_labels_read",
        "P3_2024_rows_or_labels_read",
    ):
        if audit.get(key) != 0:
            raise ValueError(f"ERA5 2022 acquisition protocol {key} is not zero")

    with paths["manifest"].open(encoding="utf-8", newline="") as handle:
        manifest = list(csv.DictReader(handle))
    outputs = {row["role"]: row for row in manifest}
    for key in ("contract", "audit"):
        role = f"output_{key}"
        row = outputs.get(role)
        path = paths[key]
        if (
            row is None
            or int(row["bytes"]) != path.stat().st_size
            or row["sha256"] != sha256_file(path)
        ):
            raise ValueError(f"ERA5 2022 acquisition manifest failed for {key}")
    return contract


def validate_target_rows(
    rows: list[dict[str, Any]], start: str, expected_count: int
) -> list[dict[str, Any]]:
    start_date = date.fromisoformat(start)
    expected_dates = [
        (start_date + timedelta(days=index)).isoformat()
        for index in range(expected_count)
    ]
    observed_dates = [str(row["date"]) for row in rows]
    if observed_dates != expected_dates:
        raise ValueError(
            "2022 ERA5 date sequence differs from contract: "
            f"count={len(observed_dates)} first={observed_dates[:2]} "
            f"last={observed_dates[-2:]}"
        )
    for index, row in enumerate(rows):
        row["day_index"] = index
        for variable in VARIABLES:
            value = float(row[variable])
            if not math.isfinite(value):
                raise ValueError(
                    f"non-finite 2022 ERA5 value date={row['date']} variable={variable}"
                )
    return rows


def run(args: argparse.Namespace) -> dict[str, Path]:
    if not args.ee_project.strip():
        raise ValueError("--ee-project must be an authorized Google Cloud project ID")
    protocol_dir = Path(args.protocol_dir).resolve()
    historical_root = Path(args.historical_era5_root).resolve()
    output_dir = Path(args.output_dir).resolve()
    contract = load_contract(protocol_dir)
    if output_dir.exists() and not args.resume:
        raise FileExistsError(f"ERA5 2022 output already exists; use --resume: {output_dir}")

    import ee

    ee.Initialize(project=args.ee_project)
    parity = verify_historical_parity(ee, contract, historical_root)
    output_dir.mkdir(parents=True, exist_ok=True)
    parity_path = output_dir / "gefs_p3_2022_era5_historical_parity_v1.csv"
    write_csv(
        parity_path,
        parity,
        fields=[
            "date",
            "day_index",
            "variable",
            "archived_value",
            "earth_engine_value",
            "absolute_difference",
            "passed",
        ],
    )

    raw_path = output_dir / "gefs_p3_2022_era5_land_get_region_raw_v1.json"
    source = contract["source_product"]
    if raw_path.is_file() and args.resume:
        raw = json.loads(raw_path.read_text(encoding="utf-8"))
        rows = _rows_from_get_region(raw)
        network_fetched_2022 = False
    else:
        rows = _fetch_region_rows(
            ee,
            source["collection_start_inclusive"],
            source["collection_end_exclusive"],
            contract["spatial_contract"]["center"],
        )
        header = ["id", "longitude", "latitude", "time", *VARIABLES]
        raw = [
            header,
            *[
                [
                    row.get("id"),
                    row.get("longitude"),
                    row.get("latitude"),
                    row.get("time"),
                    *[row[variable] for variable in VARIABLES],
                ]
                for row in rows
            ],
        ]
        write_json(raw_path, raw)
        network_fetched_2022 = True
    rows = validate_target_rows(
        rows,
        source["collection_start_inclusive"],
        int(source["expected_day_count"]),
    )
    values_path = output_dir / "gefs_p3_2022_era5_land_daily_point_values_v1.csv"
    write_csv(values_path, rows)
    raster_paths = write_point_archive(rows, output_dir, contract)

    audit_path = output_dir / "gefs_p3_2022_era5_local_acquisition_audit_v1.json"
    audit = {
        "status": "p3_2022_era5_local_acquisition_complete_pending_server_transfer",
        "mandatory_gate_passed": True,
        "earth_engine_project": args.ee_project,
        "source_asset": EARTH_ENGINE_ASSET,
        "historical_parity_rows": len(parity),
        "historical_parity_day_count": len(parity) // len(VARIABLES),
        "historical_parity_scope": "all_2019_days_all_nine_variables",
        "historical_parity_all_passed": all(bool(row["passed"]) for row in parity),
        "P3_2022_weather_rows_read": len(rows),
        "P3_2022_target_labels_read": 0,
        "P3_2022_network_request_performed_this_invocation": network_fetched_2022,
        "P3_2023_rows_or_labels_read": 0,
        "P3_2024_rows_or_labels_read": 0,
        "network_access_performed": True,
        "ERA5_download_performed": True,
        "raster_file_count": len(raster_paths),
        "expected_raster_file_count": int(source["expected_day_count"])
        * len(VARIABLES),
        "spatial_subset": "P3_exact_historical_export_grid_pixel",
        "full_CONUS_raster_downloaded": False,
        "model_training_performed": False,
        "model_or_threshold_reselection_performed": False,
        "decision_dates_resolved": False,
        "swap_simulation_performed": False,
        "next_gate": "transfer_era5_2022_to_server_then_generate_zero_irrigation_trunk_and_freeze_schedule",
    }
    write_json(audit_path, audit)
    manifest_path = output_dir / "gefs_p3_2022_era5_local_file_manifest_v1.csv"
    with manifest_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["relative_path", "bytes", "sha256"])
        writer.writeheader()
        for path in [audit_path, raw_path, values_path, parity_path, *raster_paths]:
            writer.writerow(
                {
                    "relative_path": path.relative_to(output_dir).as_posix(),
                    "bytes": path.stat().st_size,
                    "sha256": sha256_file(path),
                }
            )
    print(json.dumps(audit, indent=2, sort_keys=True), flush=True)
    return {
        "audit": audit_path,
        "values": values_path,
        "parity": parity_path,
        "manifest": manifest_path,
        "raw": raw_path,
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ee-project", required=True)
    parser.add_argument(
        "--protocol-dir", type=Path, default=DEFAULT_ACQUISITION_PROTOCOL_DIR
    )
    parser.add_argument(
        "--historical-era5-root", type=Path, default=DEFAULT_HISTORICAL_ERA5_ROOT
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args(argv)


if __name__ == "__main__":
    generated = run(parse_args())
    for name, path in generated.items():
        print(f"{name}: {path}", flush=True)
