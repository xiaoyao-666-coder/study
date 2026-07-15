import csv
import tempfile
import unittest
from pathlib import Path

from s2s_rtist.catalog import Catalog, CatalogRecord, sha256_file


def make_record(record_id="r1", **overrides):
    values = {
        "id": record_id,
        "record_type": "script",
        "original_path": "Root/Build_Model.py",
        "current_path": "src/build_model.py",
        "status": "active",
        "purpose": "Build the model",
    }
    values.update(overrides)
    return CatalogRecord(**values)


class ProjectCatalogTests(unittest.TestCase):
    def test_catalog_rejects_duplicate_ids(self):
        with self.assertRaisesRegex(ValueError, "duplicate catalog id"):
            Catalog([make_record("same"), make_record("same", current_path="other.py")])

    def test_catalog_rejects_duplicate_current_paths(self):
        with self.assertRaisesRegex(ValueError, "duplicate current_path"):
            Catalog([make_record("one"), make_record("two")])

    def test_find_searches_case_insensitively_across_path_and_purpose(self):
        catalog = Catalog(
            [
                make_record("path-hit"),
                make_record(
                    "purpose-hit",
                    original_path="other.py",
                    current_path="other/current.py",
                    purpose="Analyze Soil Moisture",
                ),
            ]
        )

        self.assertEqual([record.id for record in catalog.find("BUILD_MODEL")], ["path-hit"])
        self.assertEqual([record.id for record in catalog.find("soil moisture")], ["purpose-hit"])

    def test_sha256_file_streams_file_contents(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "abc.bin"
            path.write_bytes(b"abc")
            self.assertEqual(
                sha256_file(path),
                "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad",
            )

    def test_from_csv_handles_utf8_bom_and_absent_optional_columns(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "catalog.csv"
            with path.open("w", encoding="utf-8-sig", newline="") as handle:
                writer = csv.DictWriter(
                    handle,
                    fieldnames=[
                        "id",
                        "record_type",
                        "original_path",
                        "current_path",
                        "status",
                        "purpose",
                    ],
                )
                writer.writeheader()
                writer.writerow(
                    {
                        "id": "r1",
                        "record_type": "script",
                        "original_path": "old.py",
                        "current_path": "new.py",
                        "status": "active",
                        "purpose": "Run it",
                    }
                )

            record = Catalog.from_csv(path).get("r1")

        self.assertEqual(record.category, "")
        self.assertEqual(record.document_type, "")
        self.assertEqual(record.formal_reference, "false")
        self.assertEqual(record.runnable, "false")
        self.assertEqual(record.source_sha256, "")

    def test_validate_paths_accepts_files_and_reports_missing_paths(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "present.py").write_text("pass", encoding="utf-8")
            Catalog([make_record("present", current_path="present.py")]).validate_paths(root)

            with self.assertRaisesRegex(ValueError, "missing catalog paths") as context:
                Catalog([make_record("missing", current_path="does/not-exist.py")]).validate_paths(root)
            self.assertIn("does/not-exist.py", str(context.exception))

    def test_get_returns_record_or_raises_key_error(self):
        catalog = Catalog([make_record("known")])
        self.assertEqual(catalog.get("known").id, "known")
        with self.assertRaises(KeyError):
            catalog.get("unknown")


if __name__ == "__main__":
    unittest.main()
