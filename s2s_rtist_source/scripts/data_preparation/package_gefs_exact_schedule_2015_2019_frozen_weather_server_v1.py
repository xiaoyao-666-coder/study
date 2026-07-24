#!/usr/bin/env python3
"""Package audited 2015-2019 frozen GEFS weather for the server."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import tarfile
import tempfile
from pathlib import Path, PurePosixPath


EXPECTED_YEARS = {2015, 2016, 2017, 2018, 2019}
REVIEW_AUDIT_NAME = "gefs_exact_schedule_2015_2019_packaging_review_audit_v1.json"
SOURCE_MANIFEST_NAME = "gefs_exact_schedule_2015_2019_packaging_source_sha256_v1.csv"
PACKAGE_CONTENTS_NAME = "gefs_exact_schedule_2015_2019_server_package_contents_v1.json"
PACKAGE_SHA256_NAME = "gefs_exact_schedule_2015_2019_server_package_SHA256SUMS_v1.txt"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_relative_path(value: str) -> PurePosixPath:
    path = PurePosixPath(value.replace("\\", "/"))
    if path.is_absolute() or ".." in path.parts or not path.parts:
        raise ValueError(f"Unsafe relative path: {value!r}")
    return path


def load_payload(
    project_root: Path,
    review_root: Path,
) -> list[dict[str, object]]:
    audit_path = review_root / REVIEW_AUDIT_NAME
    manifest_path = review_root / SOURCE_MANIFEST_NAME
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    if audit.get("mandatory_packaging_review_gate_passed") is not True:
        raise RuntimeError(f"Packaging review gate is not passed: {audit_path}")
    if audit.get("next_gate") != "ready_to_package_2015_2019_frozen_weather_for_server":
        raise RuntimeError(f"Unexpected next gate in {audit_path}")

    payload: list[dict[str, object]] = []
    seen_archive_paths: set[str] = set()
    with manifest_path.open("r", encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    years = {int(row["target_year"]) for row in rows}
    if years != EXPECTED_YEARS:
        raise RuntimeError(f"Unexpected manifest year set: {sorted(years)}")
    if len(rows) != 75:
        raise RuntimeError(f"Expected 75 audited source artifacts, found {len(rows)}")

    for row in rows:
        year = int(row["target_year"])
        source_relative = validate_relative_path(row["relative_path"])
        source = (
            project_root
            / "site_general_surrogate_eval"
            / f"gefs_exact_schedule_{year}_frozen_weather_v1"
            / Path(*source_relative.parts)
        )
        archive_path = PurePosixPath(
            "site_general_surrogate_eval",
            f"gefs_exact_schedule_{year}_frozen_weather_v1",
            *source_relative.parts,
        ).as_posix()
        if archive_path in seen_archive_paths:
            raise RuntimeError(f"Duplicate archive path: {archive_path}")
        if not source.is_file():
            raise FileNotFoundError(source)
        expected_size = int(row["size_bytes"])
        expected_sha256 = row["sha256"].lower()
        actual_size = source.stat().st_size
        actual_sha256 = sha256_file(source)
        if actual_size != expected_size or actual_sha256 != expected_sha256:
            raise RuntimeError(f"Audited source changed after review: {source}")
        seen_archive_paths.add(archive_path)
        payload.append(
            {
                "source": source,
                "archive_path": archive_path,
                "size_bytes": actual_size,
                "sha256": actual_sha256,
                "target_year": year,
                "artifact_stage": row["artifact_stage"],
            }
        )

    for source in (audit_path, manifest_path):
        archive_path = source.relative_to(project_root).as_posix()
        if archive_path in seen_archive_paths:
            raise RuntimeError(f"Duplicate archive path: {archive_path}")
        seen_archive_paths.add(archive_path)
        payload.append(
            {
                "source": source,
                "archive_path": archive_path,
                "size_bytes": source.stat().st_size,
                "sha256": sha256_file(source),
                "target_year": None,
                "artifact_stage": "unified_packaging_review",
            }
        )
    return payload


def verify_archive(
    archive_path: Path,
    expected_hashes: dict[str, str],
    checksum_archive_path: str,
) -> None:
    with tarfile.open(archive_path, "r:gz") as archive:
        file_members = [member for member in archive.getmembers() if member.isfile()]
        member_names = [member.name for member in file_members]
        if len(member_names) != len(set(member_names)):
            raise RuntimeError("Archive contains duplicate member names")
        expected_names = set(expected_hashes) | {checksum_archive_path}
        if set(member_names) != expected_names:
            missing = sorted(expected_names - set(member_names))
            extra = sorted(set(member_names) - expected_names)
            raise RuntimeError(f"Archive member mismatch; missing={missing}, extra={extra}")
        for member in file_members:
            if member.name == checksum_archive_path:
                continue
            extracted = archive.extractfile(member)
            if extracted is None:
                raise RuntimeError(f"Cannot read archive member: {member.name}")
            digest = hashlib.sha256()
            for chunk in iter(lambda: extracted.read(1024 * 1024), b""):
                digest.update(chunk)
            if digest.hexdigest() != expected_hashes[member.name]:
                raise RuntimeError(f"Archive payload hash mismatch: {member.name}")


def run(args: argparse.Namespace) -> dict[str, object]:
    project_root = args.project_root.resolve()
    review_root = args.review_root.resolve()
    output = args.output.resolve()
    if output.exists() and not args.overwrite:
        raise FileExistsError(f"Output already exists: {output}")
    if output.parent != project_root:
        raise RuntimeError("Server archive must be written to the project root")

    payload = load_payload(project_root, review_root)
    review_archive_root = PurePosixPath(
        "site_general_surrogate_eval",
        review_root.name,
    )
    contents_archive_path = (review_archive_root / PACKAGE_CONTENTS_NAME).as_posix()
    checksum_archive_path = (review_archive_root / PACKAGE_SHA256_NAME).as_posix()
    payload_records = [
        {
            key: item[key]
            for key in (
                "archive_path",
                "size_bytes",
                "sha256",
                "target_year",
                "artifact_stage",
            )
        }
        for item in payload
    ]
    package_contents = {
        "status": "gefs_exact_schedule_2015_2019_frozen_weather_server_package_ready",
        "target_years": sorted(EXPECTED_YEARS),
        "payload_file_count": len(payload_records),
        "payload_total_bytes": sum(int(item["size_bytes"]) for item in payload_records),
        "archive_member_file_count": len(payload_records) + 2,
        "extraction_root": "server_project_root",
        "weather_variables": [
            "precipitation_mm",
            "temperature_min_c",
            "temperature_max_c",
            "actual_vapor_pressure_kpa",
            "wind_speed_m_s",
            "solar_kj_m2_day",
        ],
        "gefs_members": ["c00", "p01", "p02", "p03", "p04"],
        "lead_days": list(range(1, 8)),
        "packaging_review_gate_passed": True,
        "swap_simulation_performed": False,
        "label_generation_performed": False,
        "surrogate_training_performed": False,
        "tta_performed": False,
        "payload": payload_records,
    }

    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="gefs_frozen_weather_package_") as temp_name:
        temp = Path(temp_name)
        contents_path = temp / PACKAGE_CONTENTS_NAME
        contents_path.write_text(
            json.dumps(package_contents, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        expected_hashes = {
            str(item["archive_path"]): str(item["sha256"])
            for item in payload
        }
        expected_hashes[contents_archive_path] = sha256_file(contents_path)
        checksums_path = temp / PACKAGE_SHA256_NAME
        checksums_path.write_text(
            "".join(
                f"{digest}  {archive_path}\n"
                for archive_path, digest in sorted(expected_hashes.items())
            ),
            encoding="utf-8",
        )

        with tarfile.open(output, "w:gz", compresslevel=9) as archive:
            for item in sorted(payload, key=lambda value: str(value["archive_path"])):
                archive.add(
                    Path(item["source"]),
                    arcname=str(item["archive_path"]),
                    recursive=False,
                )
            archive.add(contents_path, arcname=contents_archive_path, recursive=False)
            archive.add(checksums_path, arcname=checksum_archive_path, recursive=False)
        verify_archive(output, expected_hashes, checksum_archive_path)

    return {
        "archive": str(output),
        "archive_size_bytes": output.stat().st_size,
        "archive_sha256": sha256_file(output),
        "payload_file_count": len(payload),
        "archive_member_file_count": len(payload) + 2,
        "payload_total_bytes": sum(int(item["size_bytes"]) for item in payload),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--review-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    print(json.dumps(run(parse_args()), indent=2, sort_keys=True))
