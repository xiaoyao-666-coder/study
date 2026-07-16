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

MOVED_FORMAL_NAMES = (
    "gefs_gridmet_bias_validation_v1.py",
    "generate_restart_decision_dataset.py",
    "rootzone_flux_frequency_diagnostic_v1.py",
    "swap_three_output_labels_v1.py",
    "validate_three_output_smoke_v1.py",
    "run_confirmed_5site_restart_generation_smoke_v1.py",
    "run_continuous_ir_12site_restart_generation_v1.py",
    "run_gefs_gridmet_bias_validation_v1.py",
    "run_rootzone_flux_frequency_validation_v1.py",
)

FORMAL_DOCUMENTS = (
    PROJECT_ROOT / "docs" / "operations" / "server" / "fixed_0_100cm_5site_smoke_server_run_20260715.md",
    PROJECT_ROOT / "docs" / "operations" / "server" / "formal_npd24_5site_smoke_server_run_20260714.md",
    PROJECT_ROOT / "docs" / "operations" / "server" / "gefs_gridmet_bias_validation_server_run_20260715.md",
    PROJECT_ROOT / "site_general_surrogate_eval" / "three_output_rootzone_flux_frequency_validation_results_2026-07-14.md",
    PROJECT_ROOT / "site_general_surrogate_eval" / "three_output_rootzone_water_balance_audit_2026-07-13.md",
    PROJECT_ROOT / "site_general_surrogate_eval" / "three_output_surrogate_data_processing_spec_v1.md",
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

    def test_formal_docs_do_not_use_old_root_commands(self) -> None:
        failures = []
        for path in FORMAL_DOCUMENTS:
            text = path.read_text(encoding="utf-8")
            for name in MOVED_FORMAL_NAMES:
                if f"python3 {name}" in text or f"python {name}" in text:
                    failures.append(f"{path.relative_to(PROJECT_ROOT)}:{name}")
                if f"python3 /media/" in text and name in text:
                    # Absolute server paths that still invoke old root filenames.
                    if f"/{name}" in text.replace("\\", "/"):
                        failures.append(f"{path.relative_to(PROJECT_ROOT)}:abs:{name}")
        self.assertEqual(failures, [])


if __name__ == "__main__":
    unittest.main()
