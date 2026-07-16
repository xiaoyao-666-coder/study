import csv
import io
import os
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from dataclasses import fields
from pathlib import Path
from unittest.mock import patch

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

    def _update_catalog_record(self, path, record_id, **updates):
        with path.open("r", encoding="utf-8", newline="") as handle:
            records = list(csv.DictReader(handle))
        for record in records:
            if record["id"] == record_id:
                record.update(updates)
                break
        else:
            self.fail(f"catalog record not found: {record_id}")
        self._write_catalog(path, records)

    def invoke(self, *argv, project_root=None):
        stdout = io.StringIO()
        stderr = io.StringIO()
        with redirect_stdout(stdout), redirect_stderr(stderr):
            return_code = main(
                project_root=self.project_root if project_root is None else project_root,
                argv=list(argv),
            )
        return return_code, stdout.getvalue(), stderr.getvalue()

    def _replace_echo_script(self, source):
        (self.project_root / "scripts" / "utilities" / "echo_args.py").write_text(
            source, encoding="utf-8"
        )

    @staticmethod
    def _path_key(path):
        return os.path.normcase(str(Path(path).resolve()))

    def test_list_defaults_to_scripts(self):
        return_code, stdout, stderr = self.invoke("list")

        self.assertEqual(return_code, 0)
        self.assertIn("echo-args", stdout)
        self.assertIn("rootzone-frequency", stdout)
        self.assertNotIn("rootzone-report", stdout)
        self.assertEqual(stderr, "")

    def test_list_summary_includes_category_and_status(self):
        return_code, stdout, stderr = self.invoke("list")

        self.assertEqual(return_code, 0)
        self.assertIn(
            "script\techo-args\tutilities\tactive\t"
            "scripts/utilities/echo_args.py\tEcho forwarded arguments",
            stdout,
        )
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

    def test_find_smoke_returns_script_and_document_matches(self):
        self._write_catalog(
            self.project_root / "scripts" / "script_catalog.csv",
            [
                self._record(
                    "confirmed-5site-smoke",
                    current_path="scripts/rootzone/rootzone_frequency.py",
                    purpose="Confirmed five site smoke runner",
                    category="simulation",
                    runnable="true",
                ),
            ],
        )
        self._write_catalog(
            self.project_root / "docs" / "document_catalog.csv",
            [
                self._record(
                    "fixed-smoke-server-run",
                    record_type="document",
                    current_path="docs/reports/rootzone_report.md",
                    purpose="Fixed 0-100cm 5site smoke server run guide",
                    document_type="server_guide",
                    status="formal",
                ),
            ],
        )

        return_code, stdout, _ = self.invoke("find", "smoke")

        self.assertEqual(return_code, 0)
        self.assertIn("script\tconfirmed-5site-smoke", stdout)
        self.assertIn("document\tfixed-smoke-server-run", stdout)

    def test_find_smoke_returns_script_and_document_matches(self):
        self._write_catalog(
            self.project_root / "scripts" / "script_catalog.csv",
            [
                self._record(
                    "confirmed-5site-smoke",
                    current_path="scripts/rootzone/rootzone_frequency.py",
                    purpose="Confirmed five site smoke runner",
                    category="simulation",
                    runnable="true",
                ),
            ],
        )
        self._write_catalog(
            self.project_root / "docs" / "document_catalog.csv",
            [
                self._record(
                    "fixed-smoke-server-run",
                    record_type="document",
                    current_path="docs/reports/rootzone_report.md",
                    purpose="Fixed 0-100cm 5site smoke server run guide",
                    document_type="server_guide",
                    status="formal",
                ),
            ],
        )

        return_code, stdout, _ = self.invoke("find", "smoke")

        self.assertEqual(return_code, 0)
        self.assertIn("script\tconfirmed-5site-smoke", stdout)
        self.assertIn("document\tfixed-smoke-server-run", stdout)

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

    def test_run_resolves_relative_project_root_before_starting_child(self):
        original_cwd = Path.cwd()
        try:
            os.chdir(self.project_root.parent)
            return_code, stdout, stderr = self.invoke(
                "run",
                "echo-args",
                "--",
                "relative",
                project_root=Path(self.project_root.name),
            )
        finally:
            os.chdir(original_cwd)

        self.assertEqual(return_code, 7)
        self.assertIn("ARGS:relative", stdout)
        self.assertEqual(stderr, "")

    def test_run_child_cwd_is_resolved_project_root(self):
        self._replace_echo_script(
            "from pathlib import Path\n"
            "print('CWD:' + str(Path.cwd().resolve()))\n"
        )

        return_code, stdout, stderr = self.invoke("run", "echo-args")

        self.assertEqual(return_code, 0)
        child_cwd = stdout.removeprefix("CWD:").strip()
        self.assertEqual(self._path_key(child_cwd), self._path_key(self.project_root))
        self.assertEqual(stderr, "")

    def test_run_propagates_child_stderr(self):
        self._replace_echo_script(
            "import sys\n"
            "print('CHILD-ERROR', file=sys.stderr)\n"
        )

        return_code, stdout, stderr = self.invoke("run", "echo-args")

        self.assertEqual(return_code, 0)
        self.assertEqual(stdout, "")
        self.assertIn("CHILD-ERROR", stderr)

    def test_run_builds_ordered_deduplicated_child_pythonpath(self):
        sentinel = self.project_root / "sentinel"
        sentinel.mkdir()
        self._replace_echo_script(
            "import os\n"
            "print('PYTHONPATH:' + os.environ['PYTHONPATH'])\n"
        )
        inherited = os.pathsep.join(
            [
                str(self.project_root / "src"),
                str(sentinel),
                str(self.project_root / "scripts" / "utilities"),
            ]
        )

        with patch.dict(os.environ, {"PYTHONPATH": inherited}, clear=False):
            return_code, stdout, stderr = self.invoke("run", "echo-args")

        self.assertEqual(return_code, 0)
        child_pythonpath = stdout.removeprefix("PYTHONPATH:").strip().split(os.pathsep)
        expected = [
            self.project_root / "src",
            self.project_root / "scripts" / "utilities",
            self.project_root / "scripts" / "rootzone",
            self.project_root / "scripts" / "setup",
            sentinel,
        ]
        self.assertEqual(
            [self._path_key(entry) for entry in child_pythonpath],
            [self._path_key(entry) for entry in expected],
        )
        self.assertEqual(stderr, "")

    def test_script_catalog_source_overrides_document_record_type(self):
        self._update_catalog_record(
            self.project_root / "scripts" / "script_catalog.csv",
            "echo-args",
            record_type="document",
        )

        list_code, stdout, stderr = self.invoke("list", "--type", "scripts")
        run_code, run_stdout, run_stderr = self.invoke("run", "echo-args")

        self.assertEqual(list_code, 0)
        self.assertIn("script\techo-args", stdout)
        self.assertEqual(stderr, "")
        self.assertEqual(run_code, 7)
        self.assertIn("ARGS:", run_stdout)
        self.assertEqual(run_stderr, "")

    def test_document_catalog_source_overrides_runnable_script_values(self):
        self._update_catalog_record(
            self.project_root / "docs" / "document_catalog.csv",
            "rootzone-report",
            record_type="script",
            runnable="true",
        )

        list_code, stdout, stderr = self.invoke("list", "--type", "docs")
        run_code, _, run_stderr = self.invoke("run", "rootzone-report")

        self.assertEqual(list_code, 0)
        self.assertIn("document\trootzone-report", stdout)
        self.assertEqual(stderr, "")
        self.assertNotEqual(run_code, 0)
        self.assertIn("not runnable", run_stderr)

    def test_run_rejects_document(self):
        return_code, _, stderr = self.invoke("run", "rootzone-report")

        self.assertNotEqual(return_code, 0)
        self.assertIn("not runnable", stderr)

    def test_run_rejects_non_runnable_script(self):
        self._update_catalog_record(
            self.project_root / "scripts" / "script_catalog.csv",
            "prepare-only",
            status="archived",
        )

        return_code, _, stderr = self.invoke("run", "prepare-only")

        self.assertNotEqual(return_code, 0)
        self.assertIn("not runnable", stderr)
        self.assertIn("archived", stderr)

    def test_missing_catalog_is_a_concise_error(self):
        (self.project_root / "docs" / "document_catalog.csv").unlink()

        return_code, _, stderr = self.invoke("list")

        self.assertNotEqual(return_code, 0)
        self.assertIn("catalog", stderr.casefold())
        self.assertIn("document_catalog.csv", stderr)
        self.assertNotIn("Traceback", stderr)


if __name__ == "__main__":
    unittest.main()
