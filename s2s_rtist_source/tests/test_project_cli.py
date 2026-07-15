import csv
import io
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from dataclasses import fields
from pathlib import Path

from s2s_rtist.catalog import CatalogRecord
from s2s_rtist.cli import main


class ProjectCliTests(unittest.TestCase):
    def setUp(self):
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.project_root = Path(self.temporary_directory.name)
        (self.project_root / "scripts" / "utilities").mkdir(parents=True)
        (self.project_root / "scripts" / "rootzone").mkdir(parents=True)
        (self.project_root / "scripts" / "setup").mkdir(parents=True)
        (self.project_root / "docs" / "reports").mkdir(parents=True)

        echo_path = self.project_root / "scripts" / "utilities" / "echo_args.py"
        echo_path.write_text(
            "import sys\n"
            "print('ARGS:' + '|'.join(sys.argv[1:]))\n"
            "raise SystemExit(7)\n",
            encoding="utf-8",
        )
        (self.project_root / "scripts" / "rootzone" / "rootzone_frequency.py").write_text(
            "pass\n", encoding="utf-8"
        )
        (self.project_root / "scripts" / "setup" / "prepare.py").write_text(
            "pass\n", encoding="utf-8"
        )
        (self.project_root / "docs" / "reports" / "rootzone_report.md").write_text(
            "# Rootzone report\n", encoding="utf-8"
        )

        self._write_catalog(
            self.project_root / "scripts" / "script_catalog.csv",
            [
                self._record(
                    "echo-args",
                    current_path="scripts/utilities/echo_args.py",
                    purpose="Echo forwarded arguments",
                    category="utilities",
                    runnable="true",
                ),
                self._record(
                    "rootzone-frequency",
                    current_path="scripts/rootzone/rootzone_frequency.py",
                    purpose="Analyze rootzone frequency",
                    category="rootzone",
                    runnable="true",
                ),
                self._record(
                    "prepare-only",
                    current_path="scripts/setup/prepare.py",
                    purpose="Prepare inputs",
                    category="setup",
                    runnable="false",
                ),
            ],
        )
        self._write_catalog(
            self.project_root / "docs" / "document_catalog.csv",
            [
                self._record(
                    "rootzone-report",
                    record_type="document",
                    current_path="docs/reports/rootzone_report.md",
                    purpose="Rootzone analysis report",
                    document_type="report",
                )
            ],
        )

    def tearDown(self):
        self.temporary_directory.cleanup()

    @staticmethod
    def _record(record_id, **overrides):
        values = {
            "id": record_id,
            "record_type": "script",
            "original_path": f"legacy/{record_id}",
            "current_path": f"scripts/{record_id}.py",
            "status": "active",
            "purpose": record_id,
            "category": "",
            "document_type": "",
            "replaced_by": "",
            "related_script_ids": "",
            "formal_reference": "false",
            "runnable": "false",
            "source_sha256": "source-digest",
            "current_sha256": "current-digest",
        }
        values.update(overrides)
        return values

    @staticmethod
    def _write_catalog(path, records):
        fieldnames = [field.name for field in fields(CatalogRecord)]
        with path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(records)

    def invoke(self, *argv):
        stdout = io.StringIO()
        stderr = io.StringIO()
        with redirect_stdout(stdout), redirect_stderr(stderr):
            return_code = main(project_root=self.project_root, argv=list(argv))
        return return_code, stdout.getvalue(), stderr.getvalue()

    def test_list_defaults_to_scripts(self):
        return_code, stdout, stderr = self.invoke("list")

        self.assertEqual(return_code, 0)
        self.assertIn("echo-args", stdout)
        self.assertIn("rootzone-frequency", stdout)
        self.assertNotIn("rootzone-report", stdout)
        self.assertEqual(stderr, "")

    def test_list_docs_shows_documents_only(self):
        return_code, stdout, _ = self.invoke("list", "--type", "docs")

        self.assertEqual(return_code, 0)
        self.assertIn("rootzone-report", stdout)
        self.assertNotIn("rootzone-frequency", stdout)

    def test_list_all_shows_scripts_and_documents(self):
        return_code, stdout, _ = self.invoke("list", "--type", "all")

        self.assertEqual(return_code, 0)
        self.assertIn("rootzone-frequency", stdout)
        self.assertIn("rootzone-report", stdout)

    def test_find_searches_both_catalogs_case_insensitively(self):
        return_code, stdout, _ = self.invoke("find", "ROOTZONE")

        self.assertEqual(return_code, 0)
        self.assertIn("script\trootzone-frequency", stdout)
        self.assertIn("document\trootzone-report", stdout)

    def test_show_exact_id_includes_id_and_current_path(self):
        return_code, stdout, _ = self.invoke("show", "rootzone-frequency")

        self.assertEqual(return_code, 0)
        self.assertIn("id: rootzone-frequency", stdout)
        self.assertIn("current_path: scripts/rootzone/rootzone_frequency.py", stdout)

    def test_unknown_id_reports_close_suggestion(self):
        return_code, _, stderr = self.invoke("show", "rootzone-frequncy")

        self.assertNotEqual(return_code, 0)
        self.assertIn("unknown ID", stderr)
        self.assertIn("rootzone-frequency", stderr)

    def test_run_forwards_arguments_captures_output_and_returns_child_code(self):
        return_code, stdout, stderr = self.invoke(
            "run", "echo-args", "--", "alpha", "--beta"
        )

        self.assertEqual(return_code, 7)
        self.assertIn("ARGS:alpha|--beta", stdout)
        self.assertEqual(stderr, "")

    def test_run_rejects_document(self):
        return_code, _, stderr = self.invoke("run", "rootzone-report")

        self.assertNotEqual(return_code, 0)
        self.assertIn("not runnable", stderr)

    def test_run_rejects_non_runnable_script(self):
        return_code, _, stderr = self.invoke("run", "prepare-only")

        self.assertNotEqual(return_code, 0)
        self.assertIn("not runnable", stderr)

    def test_missing_catalog_is_a_concise_error(self):
        (self.project_root / "docs" / "document_catalog.csv").unlink()

        return_code, _, stderr = self.invoke("list")

        self.assertNotEqual(return_code, 0)
        self.assertIn("catalog", stderr.casefold())
        self.assertIn("document_catalog.csv", stderr)
        self.assertNotIn("Traceback", stderr)


if __name__ == "__main__":
    unittest.main()
