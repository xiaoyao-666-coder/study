from __future__ import annotations

import argparse
import csv
import hashlib
import json
import tarfile
import tempfile
import unittest
from pathlib import Path

from scripts.data_preparation.package_gefs_exact_schedule_2015_2019_frozen_weather_server_v1 import (
    REVIEW_AUDIT_NAME,
    SOURCE_MANIFEST_NAME,
    run,
    validate_relative_path,
)


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class FrozenWeatherServerPackageTests(unittest.TestCase):
    def test_rejects_unsafe_relative_paths(self) -> None:
        with self.assertRaises(ValueError):
            validate_relative_path("../secret.txt")
        with self.assertRaises(ValueError):
            validate_relative_path("/absolute.txt")
        self.assertEqual(validate_relative_path("04_stage/file.csv").as_posix(), "04_stage/file.csv")

    def test_packages_exactly_the_audited_payload(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            project = Path(temporary) / "project"
            review = (
                project
                / "site_general_surrogate_eval"
                / "gefs_exact_schedule_2015_2019_frozen_weather_packaging_review_v1"
            )
            review.mkdir(parents=True)
            (review / REVIEW_AUDIT_NAME).write_text(
                json.dumps(
                    {
                        "mandatory_packaging_review_gate_passed": True,
                        "next_gate": "ready_to_package_2015_2019_frozen_weather_for_server",
                    }
                ),
                encoding="utf-8",
            )
            rows = []
            for year in range(2015, 2020):
                root = (
                    project
                    / "site_general_surrogate_eval"
                    / f"gefs_exact_schedule_{year}_frozen_weather_v1"
                )
                for index in range(15):
                    relative = Path("04_frozen_all_variable_weather") / f"artifact_{index:02d}.txt"
                    source = root / relative
                    source.parent.mkdir(parents=True, exist_ok=True)
                    source.write_text(f"{year}:{index}\n", encoding="utf-8")
                    rows.append(
                        {
                            "target_year": year,
                            "relative_path": relative.as_posix(),
                            "size_bytes": source.stat().st_size,
                            "sha256": digest(source),
                            "artifact_stage": "04_frozen_all_variable_weather",
                        }
                    )
            with (review / SOURCE_MANIFEST_NAME).open("w", encoding="utf-8", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)

            output = project / "package.tar.gz"
            result = run(
                argparse.Namespace(
                    project_root=project,
                    review_root=review,
                    output=output,
                    overwrite=False,
                )
            )

            self.assertEqual(result["payload_file_count"], 77)
            self.assertEqual(result["archive_member_file_count"], 79)
            self.assertEqual(result["archive_sha256"], digest(output))
            with tarfile.open(output, "r:gz") as archive:
                members = [member for member in archive.getmembers() if member.isfile()]
            self.assertEqual(len(members), 79)


if __name__ == "__main__":
    unittest.main()
