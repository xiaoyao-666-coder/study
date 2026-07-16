"""Final project layout, catalog coverage, and integrity checks."""

from __future__ import annotations

import csv
import unittest
from pathlib import Path

from s2s_rtist.catalog import Catalog, sha256_file

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_CATALOG = PROJECT_ROOT / "scripts" / "script_catalog.csv"
DOCUMENT_CATALOG = PROJECT_ROOT / "docs" / "document_catalog.csv"
SCRIPT_INVENTORY = PROJECT_ROOT / "docs" / "migration" / "root_python_inventory_20260715.csv"
DOCUMENT_INVENTORY = PROJECT_ROOT / "docs" / "migration" / "root_document_inventory_20260715.csv"

REQUIRED_READMES = (
    "scripts/README.md",
    "scripts/data_preparation/README.md",
    "scripts/simulation/README.md",
    "scripts/diagnostics/README.md",
    "scripts/training/README.md",
    "scripts/evaluation/README.md",
    "scripts/visualization/README.md",
    "scripts/archive/README.md",
    "scripts/archive/VERSIONS.md",
    "docs/README.md",
    "docs/operations/server/README.md",
    "docs/research/reproduction/README.md",
    "docs/archive/README.md",
)


class ProjectLayoutTests(unittest.TestCase):
    def test_root_contains_only_project_cli_python_file(self) -> None:
        root_scripts = sorted(path.name for path in PROJECT_ROOT.glob("*.py"))
        self.assertEqual(root_scripts, ["project_cli.py"])

    def test_no_original_markdown_or_text_files_remain_at_root(self) -> None:
        remaining = sorted(
            path.name
            for path in PROJECT_ROOT.iterdir()
            if path.is_file() and path.suffix.lower() in {".md", ".txt"}
        )
        self.assertEqual(remaining, [])

    def test_script_catalog_covers_all_140_original_scripts(self) -> None:
        with SCRIPT_INVENTORY.open(encoding="utf-8-sig", newline="") as handle:
            inventory = list(csv.DictReader(handle))
        catalog = Catalog.from_csv(SCRIPT_CATALOG)
        original_paths = {row["original_path"] for row in inventory}
        catalog_originals = {
            record.original_path
            for record in catalog.records
            if record.original_path.endswith(".py")
        }
        self.assertEqual(len(original_paths), 140)
        self.assertEqual(catalog_originals, original_paths)

    def test_document_catalog_covers_all_nine_original_files(self) -> None:
        with DOCUMENT_INVENTORY.open(encoding="utf-8-sig", newline="") as handle:
            inventory = list(csv.DictReader(handle))
        catalog = Catalog.from_csv(DOCUMENT_CATALOG)
        self.assertEqual(len(inventory), 9)
        self.assertEqual(
            {row["original_path"] for row in inventory},
            {record.original_path for record in catalog.records},
        )

    def test_unchanged_files_keep_source_hash(self) -> None:
        catalog = Catalog.from_csv(SCRIPT_CATALOG)
        for record in catalog.records:
            current = sha256_file(PROJECT_ROOT / record.current_path)
            self.assertEqual(current, record.current_sha256, record.current_path)
            if record.status in {"historical", "superseded", "legacy_unreviewed"}:
                self.assertEqual(
                    record.source_sha256,
                    record.current_sha256,
                    record.current_path,
                )

    def test_script_catalog_paths_exist(self) -> None:
        catalog = Catalog.from_csv(SCRIPT_CATALOG)
        catalog.validate_paths(PROJECT_ROOT)

    def test_document_catalog_paths_exist(self) -> None:
        catalog = Catalog.from_csv(DOCUMENT_CATALOG)
        catalog.validate_paths(PROJECT_ROOT)

    def test_required_navigation_files_exist(self) -> None:
        self.assertEqual(
            [path for path in REQUIRED_READMES if not (PROJECT_ROOT / path).is_file()],
            [],
        )


if __name__ == "__main__":
    unittest.main()
