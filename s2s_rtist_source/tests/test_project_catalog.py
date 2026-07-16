import csv
import importlib.util
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

from s2s_rtist.catalog import Catalog, CatalogRecord, sha256_file


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MIGRATION_SCRIPT = (
    PROJECT_ROOT / "scripts" / "archive" / "one_off" / "migrate_root_files_20260715.py"
)


def load_migration_module():
    spec = importlib.util.spec_from_file_location("migrate_root_files_20260715", MIGRATION_SCRIPT)
    if spec is None or spec.loader is None:
        raise AssertionError(f"cannot load migration planner: {MIGRATION_SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


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


class MigrationPlannerTests(unittest.TestCase):
    maxDiff = None

    def setUp(self):
        self.planner = load_migration_module()

    @staticmethod
    def _write(root, relative_path, contents=None):
        path = root / relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(contents or f"# {relative_path}\n", encoding="utf-8")
        return path

    def _record(self, root, source, target, **overrides):
        source_path = self._write(root, source)
        values = {
            "id": Path(source).stem.replace("_", "-"),
            "record_type": "script",
            "original_path": source,
            "current_path": target,
            "status": "active",
            "purpose": Path(source).stem,
            "category": "diagnostics",
            "source_sha256": sha256_file(source_path),
            "current_sha256": sha256_file(source_path),
        }
        values.update(overrides)
        return self.planner.MoveRecord(**values)

    def test_locked_core_and_runner_mappings_are_exact(self):
        names = set(self.planner.CORE_TARGETS) | set(self.planner.RUNNER_TARGETS)
        classifications = {
            name: self.planner.classify_script(name, names) for name in sorted(names)
        }

        self.assertEqual(
            {name: result.current_path for name, result in classifications.items()},
            {**self.planner.CORE_TARGETS, **self.planner.RUNNER_TARGETS},
        )
        for name in self.planner.CORE_TARGETS:
            self.assertEqual(classifications[name].status, "formal")
            self.assertEqual(classifications[name].runnable, "false")
        expected_runner_ids = {
            "run_confirmed_5site_restart_generation_smoke_v1.py": "confirmed-5site-smoke",
            "run_continuous_ir_12site_restart_generation_v1.py": "continuous-12site-generation",
            "run_gefs_gridmet_bias_validation_v1.py": "gefs-gridmet-bias",
            "run_rootzone_flux_frequency_validation_v1.py": "rootzone-frequency",
            "restart_raw_audit_v1.py": "restart-raw-audit",
        }
        for name, record_id in expected_runner_ids.items():
            self.assertEqual(classifications[name].id, record_id)
            self.assertEqual(classifications[name].status, "formal")
            self.assertEqual(classifications[name].runnable, "true")

    def test_locked_document_mappings_are_exact(self):
        expected = {
            "fixed_0_100cm_5site_smoke_server_run_20260715.md": "docs/operations/server/fixed_0_100cm_5site_smoke_server_run_20260715.md",
            "formal_npd24_5site_smoke_server_run_20260714.md": "docs/operations/server/formal_npd24_5site_smoke_server_run_20260714.md",
            "gefs_gridmet_bias_validation_server_run_20260715.md": "docs/operations/server/gefs_gridmet_bias_validation_server_run_20260715.md",
            "first_step_reproduction_notes_2026-05-29.md": "docs/research/reproduction/first_step_reproduction_notes_2026-05-29.md",
            "server_restart_smoke_notes_2026-05-30.md": "docs/archive/historical_notes/server_restart_smoke_notes_2026-05-30.md",
            "Instructions.txt": "docs/archive/historical_notes/Instructions.txt",
            "paper_keyword_snippets.txt": "docs/archive/paper_extracts/paper_keyword_snippets.txt",
            "paper_text_2026ems.txt": "docs/archive/paper_extracts/paper_text_2026ems.txt",
            "requirements_gefs_gridmet_bias_validation_v1.txt": "requirements/requirements_gefs_gridmet_bias_validation_v1.txt",
        }

        self.assertEqual(
            {name: self.planner.classify_document(name).current_path for name in expected},
            expected,
        )

    def test_special_script_rules_precede_prefix_rules(self):
        names = {
            "Main_win.py",
            "Main_win_ensemble_mean.py",
            "compare_expanded_policy_results_v1.py",
            "compare_expanded_policy_results_v2.py",
            "compare_expanded_policy_results_v3.py",
        }

        main = self.planner.classify_script("Main_win.py", names)
        ensemble = self.planner.classify_script("Main_win_ensemble_mean.py", names)
        v1 = self.planner.classify_script("compare_expanded_policy_results_v1.py", names)
        v2 = self.planner.classify_script("compare_expanded_policy_results_v2.py", names)
        v3 = self.planner.classify_script("compare_expanded_policy_results_v3.py", names)

        self.assertEqual(main.current_path, "scripts/archive/original_application/Main_win.py")
        self.assertEqual(ensemble.status, "historical")
        self.assertEqual(v1.current_path, "scripts/archive/superseded/compare_expanded_policy_results_v1.py")
        self.assertEqual(v2.status, "superseded")
        self.assertEqual(v1.replaced_by, "compare_expanded_policy_results_v3.py")
        self.assertEqual(v2.replaced_by, "compare_expanded_policy_results_v3.py")
        self.assertEqual(v3.current_path, "scripts/diagnostics/compare_expanded_policy_results_v3.py")
        self.assertEqual(v3.status, "active")

    def test_prefix_categories_use_locked_order_and_unmatched_fallback(self):
        cases = {
            "plot_result.py": ("visualization", "active"),
            "visualize_result.py": ("visualization", "active"),
            "optimize_policy.py": ("training", "active"),
            "apply_safe_policy.py": ("evaluation", "active"),
            "apply_lstm_policy.py": ("evaluation", "active"),
            "apply_inputs.py": ("data_preparation", "active"),
            "compare_runs.py": ("diagnostics", "active"),
            "generate_data.py": ("simulation", "active"),
            "mystery.py": ("archive", "legacy_unreviewed"),
        }
        names = set(cases)

        for name, (category, status) in cases.items():
            classification = self.planner.classify_script(name, names)
            self.assertEqual(classification.category, category, name)
            self.assertEqual(classification.status, status, name)
        self.assertEqual(
            self.planner.classify_script("mystery.py", names).current_path,
            "scripts/archive/one_off/mystery.py",
        )

    def test_build_plan_excludes_project_cli_and_does_not_move_sources(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = self._write(root, "analyze_example.py")
            cli = self._write(root, "project_cli.py")

            records = self.planner.build_plan(root, "all")
            self.planner.validate_plan(root, records)

            self.assertEqual([record.original_path for record in records], ["analyze_example.py"])
            self.assertTrue(source.is_file())
            self.assertTrue(cli.is_file())
            self.assertFalse((root / records[0].current_path).exists())

    def test_duplicate_targets_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            first = self._record(root, "first.py", "scripts/diagnostics/same.py")
            second = self._record(root, "second.py", "scripts/diagnostics/same.py")

            with self.assertRaisesRegex(ValueError, "duplicate target"):
                self.planner.validate_plan(root, [first, second])

    def test_preexisting_unexpected_target_fails_before_any_move(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            first = self._record(root, "first.py", "scripts/diagnostics/first.py")
            second = self._record(root, "second.py", "scripts/diagnostics/second.py")
            target = self._write(root, second.current_path, "unexpected\n")

            with self.assertRaisesRegex(ValueError, "collision"):
                self.planner.apply_plan(root, [first, second])

            self.assertTrue((root / first.original_path).is_file())
            self.assertFalse((root / first.current_path).exists())
            self.assertEqual(target.read_text(encoding="utf-8"), "unexpected\n")

    def test_validate_recognizes_already_moved_state_and_apply_is_idempotent(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            record = self._record(root, "source.py", "scripts/diagnostics/source.py")
            target = root / record.current_path
            target.parent.mkdir(parents=True)
            (root / record.original_path).rename(target)

            states = self.planner.validate_plan(root, [record])
            self.planner.apply_plan(root, [record])

            self.assertEqual(states[record.original_path], "already_moved")
            self.assertFalse((root / record.original_path).exists())
            self.assertTrue(target.is_file())

    def test_validate_rejects_wrong_hash_and_missing_states(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            wrong_hash = self._record(root, "wrong.py", "scripts/diagnostics/wrong.py")
            target = root / wrong_hash.current_path
            target.parent.mkdir(parents=True)
            (root / wrong_hash.original_path).rename(target)
            target.write_text("changed\n", encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "hash"):
                self.planner.validate_plan(root, [wrong_hash])

            missing = replace(
                wrong_hash,
                original_path="missing.py",
                current_path="scripts/diagnostics/missing.py",
            )
            with self.assertRaisesRegex(ValueError, "missing"):
                self.planner.validate_plan(root, [missing])

    def test_apply_rolls_back_completed_moves_after_rename_failure(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            records = [
                self._record(root, "first.py", "scripts/diagnostics/first.py"),
                self._record(root, "second.py", "scripts/diagnostics/second.py"),
            ]
            real_rename = Path.rename
            calls = []

            def fail_second_rename(path, target):
                calls.append((Path(path), Path(target)))
                if len(calls) == 2:
                    raise OSError("injected rename failure")
                return real_rename(path, target)

            with patch.object(Path, "rename", autospec=True, side_effect=fail_second_rename):
                with self.assertRaisesRegex(RuntimeError, "injected rename failure"):
                    self.planner.apply_plan(root, records)

            for record in records:
                self.assertTrue((root / record.original_path).is_file())
                self.assertFalse((root / record.current_path).exists())

    def test_inventory_round_trip_reuses_immutable_rows(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self._write(root, "analyze_one.py")
            self._write(root, "train_two.py")
            self._write(root, "Instructions.txt")
            records = self.planner.build_plan(root, "all")

            self.planner.write_inventories(root, records)
            script_inventory = root / self.planner.PYTHON_INVENTORY_PATH
            document_inventory = root / self.planner.DOCUMENT_INVENTORY_PATH
            original_script_text = script_inventory.read_text(encoding="utf-8")
            original_document_text = document_inventory.read_text(encoding="utf-8")
            with script_inventory.open(encoding="utf-8", newline="") as handle:
                self.assertEqual(len(list(csv.DictReader(handle))), 2)
            with document_inventory.open(encoding="utf-8", newline="") as handle:
                self.assertEqual(len(list(csv.DictReader(handle))), 1)

            first = next(record for record in records if record.original_path == "analyze_one.py")
            target = root / first.current_path
            target.parent.mkdir(parents=True, exist_ok=True)
            (root / first.original_path).rename(target)
            self._write(root, "new_after_inventory.py")

            reloaded = self.planner.build_plan(root, "all")
            self.planner.write_inventories(root, reloaded)

            self.assertEqual(len(reloaded), 3)
            self.assertNotIn("new_after_inventory.py", {record.original_path for record in reloaded})
            self.assertEqual(script_inventory.read_text(encoding="utf-8"), original_script_text)
            self.assertEqual(document_inventory.read_text(encoding="utf-8"), original_document_text)

    def test_partial_migration_without_inventories_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            target = self.planner.CORE_TARGETS["gefs_gridmet_bias_validation_v1.py"]
            self._write(root, target)

            with self.assertRaisesRegex(ValueError, "partial"):
                self.planner.build_plan(root, "all")


if __name__ == "__main__":
    unittest.main()
