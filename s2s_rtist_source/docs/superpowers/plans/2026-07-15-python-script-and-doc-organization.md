# Python Script and Documentation Organization Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Move all 140 root-level Python scripts and all 9 root-level Markdown/TXT files into a searchable package, categorized script tree, document tree, and archive while preserving every original file and leaving `project_cli.py` as the only root-level Python entry point.

**Architecture:** Reusable formal logic moves into `src/s2s_rtist/`; runnable research scripts move into categorized `scripts/` directories; superseded and original application scripts move into `scripts/archive/`. Two CSV catalogs preserve original paths, hashes, status, purpose, and relationships. `project_cli.py` bootstraps `src/` and delegates list/find/show/run behavior to `s2s_rtist.cli`, using subprocess execution for script isolation.

**Tech Stack:** Python 3.10+, standard library (`argparse`, `csv`, `dataclasses`, `hashlib`, `pathlib`, `subprocess`), pandas/numpy/h5py/eccodes for existing workflows, `unittest`, Git.

---

## Locked File Structure

**Create:**

- `project_cli.py` - the only root-level Python entry point.
- `pyproject.toml` - editable-install metadata for the `src/` package.
- `src/s2s_rtist/__init__.py` - package marker.
- `src/s2s_rtist/catalog.py` - catalog records, CSV loading, search, and validation.
- `src/s2s_rtist/cli.py` - list/find/show/run command implementation.
- `src/s2s_rtist/weather/gefs_gridmet_bias.py` - current GEFS/gridMET reusable logic.
- `src/s2s_rtist/physics/rootzone_flux_frequency.py` - current control-volume and flux logic.
- `src/s2s_rtist/labels/swap_three_output_labels.py` - current SWAP label extraction.
- `src/s2s_rtist/validation/three_output_smoke.py` - current smoke validator.
- `src/s2s_rtist/pipelines/restart_decision_dataset.py` - current restart dataset pipeline.
- `scripts/script_catalog.csv` - all 140 original root script mappings plus new maintenance entry points.
- `docs/document_catalog.csv` - all 9 original root Markdown/TXT mappings.
- `scripts/archive/one_off/migrate_root_files_20260715.py` - deterministic, collision-safe one-time migration utility.
- `tests/test_project_catalog.py` - catalog and migration invariants.
- `tests/test_project_cli.py` - CLI behavior.
- `tests/test_project_layout.py` - final layout, count, hash, link, and import checks.
- `docs/migration/root_python_inventory_20260715.csv` - immutable pre-migration script inventory.
- `docs/migration/root_document_inventory_20260715.csv` - immutable pre-migration document inventory.
- `docs/README.md` and category/archive README files - human navigation.

**Move formal runners:**

- `run_confirmed_5site_restart_generation_smoke_v1.py` -> `scripts/simulation/run_confirmed_5site_restart_generation_smoke_v1.py`
- `run_continuous_ir_12site_restart_generation_v1.py` -> `scripts/simulation/run_continuous_ir_12site_restart_generation_v1.py`
- `run_gefs_gridmet_bias_validation_v1.py` -> `scripts/diagnostics/run_gefs_gridmet_bias_validation_v1.py`
- `run_rootzone_flux_frequency_validation_v1.py` -> `scripts/diagnostics/run_rootzone_flux_frequency_validation_v1.py`
- `restart_raw_audit_v1.py` -> `scripts/diagnostics/restart_raw_audit_v1.py`

**Move root documents:**

- `fixed_0_100cm_5site_smoke_server_run_20260715.md` -> `docs/operations/server/fixed_0_100cm_5site_smoke_server_run_20260715.md`
- `formal_npd24_5site_smoke_server_run_20260714.md` -> `docs/operations/server/formal_npd24_5site_smoke_server_run_20260714.md`
- `gefs_gridmet_bias_validation_server_run_20260715.md` -> `docs/operations/server/gefs_gridmet_bias_validation_server_run_20260715.md`
- `first_step_reproduction_notes_2026-05-29.md` -> `docs/research/reproduction/first_step_reproduction_notes_2026-05-29.md`
- `server_restart_smoke_notes_2026-05-30.md` -> `docs/archive/historical_notes/server_restart_smoke_notes_2026-05-30.md`
- `Instructions.txt` -> `docs/archive/historical_notes/Instructions.txt`
- `paper_keyword_snippets.txt` -> `docs/archive/paper_extracts/paper_keyword_snippets.txt`
- `paper_text_2026ems.txt` -> `docs/archive/paper_extracts/paper_text_2026ems.txt`
- `requirements_gefs_gridmet_bias_validation_v1.txt` -> `requirements/requirements_gefs_gridmet_bias_validation_v1.txt`

The two paper extracts remain local-only and ignored by Git. Their document catalog entries retain paths and SHA256 values with `status=local_reference` and `formal_reference=false`.

---

### Task 1: Add Catalog Model and Immutable Inventory Support

**Files:**

- Create: `src/s2s_rtist/__init__.py`
- Create: `src/s2s_rtist/catalog.py`
- Create: `tests/test_project_catalog.py`
- Create: `pyproject.toml`

- [ ] **Step 1: Write failing catalog tests**

Create `tests/test_project_catalog.py` with tests for CSV loading, duplicate rejection, case-insensitive search, SHA256, and path validation:

```python
from __future__ import annotations

import csv
import tempfile
import unittest
from pathlib import Path

from s2s_rtist.catalog import Catalog, CatalogRecord, sha256_file


class CatalogTests(unittest.TestCase):
    def test_rejects_duplicate_ids(self) -> None:
        records = [
            CatalogRecord("same", "script", "old_a.py", "scripts/a.py", "formal", "A"),
            CatalogRecord("same", "script", "old_b.py", "scripts/b.py", "historical", "B"),
        ]
        with self.assertRaisesRegex(ValueError, "duplicate catalog id"):
            Catalog(records)

    def test_find_searches_original_path_and_purpose_case_insensitively(self) -> None:
        catalog = Catalog([
            CatalogRecord(
                "rootzone-frequency",
                "script",
                "run_rootzone_flux_frequency_validation_v1.py",
                "scripts/diagnostics/run_rootzone_flux_frequency_validation_v1.py",
                "formal",
                "Root-zone flux frequency diagnostic",
            )
        ])
        self.assertEqual([item.id for item in catalog.find("ROOTZONE")], ["rootzone-frequency"])
        self.assertEqual([item.id for item in catalog.find("flux frequency")], ["rootzone-frequency"])

    def test_sha256_file_matches_known_payload(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "sample.txt"
            path.write_bytes(b"abc")
            self.assertEqual(
                sha256_file(path),
                "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad",
            )


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run tests and verify RED**

Run:

```powershell
$env:PYTHONPATH='D:\study\s2s_rtist_source\src'
python -m unittest tests.test_project_catalog -v
```

Expected: `ModuleNotFoundError: No module named 's2s_rtist'`.

- [ ] **Step 3: Implement the catalog model**

Create `src/s2s_rtist/catalog.py` with this public API:

```python
from __future__ import annotations

import csv
import hashlib
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Iterable


@dataclass(frozen=True)
class CatalogRecord:
    id: str
    record_type: str
    original_path: str
    current_path: str
    status: str
    purpose: str
    category: str = ""
    document_type: str = ""
    replaced_by: str = ""
    related_script_ids: str = ""
    formal_reference: str = "false"
    runnable: str = "false"
    source_sha256: str = ""
    current_sha256: str = ""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class Catalog:
    def __init__(self, records: Iterable[CatalogRecord]):
        self.records = list(records)
        ids = [record.id for record in self.records]
        if len(ids) != len(set(ids)):
            raise ValueError("duplicate catalog id")
        paths = [record.current_path for record in self.records]
        if len(paths) != len(set(paths)):
            raise ValueError("duplicate catalog current_path")

    @classmethod
    def from_csv(cls, path: Path) -> "Catalog":
        names = {field.name for field in fields(CatalogRecord)}
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            rows = []
            for raw in csv.DictReader(handle):
                values = {name: raw.get(name, "") for name in names}
                rows.append(CatalogRecord(**values))
        return cls(rows)

    def find(self, query: str) -> list[CatalogRecord]:
        needle = query.casefold()
        return [
            record
            for record in self.records
            if needle in " ".join(str(value) for value in record.__dict__.values()).casefold()
        ]

    def get(self, record_id: str) -> CatalogRecord:
        matches = [record for record in self.records if record.id == record_id]
        if not matches:
            raise KeyError(record_id)
        return matches[0]

    def validate_paths(self, root: Path) -> None:
        missing = [record.current_path for record in self.records if not (root / record.current_path).is_file()]
        if missing:
            raise ValueError(f"missing catalog paths: {missing[:5]}")
```

Create package markers under `src/s2s_rtist/` and configure `pyproject.toml`:

```toml
[build-system]
requires = ["setuptools>=61"]
build-backend = "setuptools.build_meta"

[project]
name = "s2s-rtist"
version = "0.1.0"
requires-python = ">=3.10"

[tool.setuptools.packages.find]
where = ["src"]
```

- [ ] **Step 4: Run tests and verify GREEN**

Run the Task 1 test command again.

Expected: all catalog tests pass.

- [ ] **Step 5: Commit**

```powershell
git add pyproject.toml src/s2s_rtist tests/test_project_catalog.py
git commit -m "feat: add project catalog model"
```

---

### Task 2: Build the Unified CLI with Script and Document Search

**Files:**

- Create: `project_cli.py`
- Create: `src/s2s_rtist/cli.py`
- Create: `tests/test_project_cli.py`

- [ ] **Step 1: Write failing CLI tests**

Use a temporary project containing one script record and one document record. Test `list --type docs`, `find`, `show`, unknown-ID suggestions, argument forwarding, and child exit-code propagation. The test helper calls `main(project_root=..., argv=...)` directly and captures its streams:

```python
from contextlib import redirect_stderr, redirect_stdout
from io import StringIO
from types import SimpleNamespace

from s2s_rtist.cli import main


class ProjectCliTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        (self.root / "scripts").mkdir()
        (self.root / "docs").mkdir()
        echo = self.root / "scripts" / "echo_args.py"
        echo.write_text(
            "import sys\nprint(' '.join(sys.argv[1:]))\nraise SystemExit(7)\n",
            encoding="utf-8",
        )
        (self.root / "scripts" / "rootzone_script.py").write_text(
            "print('rootzone')\n", encoding="utf-8"
        )
        (self.root / "docs" / "rootzone.md").write_text("rootzone", encoding="utf-8")
        self.write_catalogs()

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def write_catalogs(self) -> None:
        fieldnames = [
            "id", "record_type", "original_path", "current_path", "status",
            "purpose", "category", "document_type", "replaced_by",
            "related_script_ids", "formal_reference", "runnable",
            "source_sha256", "current_sha256",
        ]
        script_rows = [
            {
                "id": "rootzone-frequency", "record_type": "script",
                "original_path": "run_rootzone.py",
                "current_path": "scripts/rootzone_script.py", "status": "formal",
                "purpose": "Rootzone flux frequency", "category": "diagnostics",
                "runnable": "true",
            },
            {
                "id": "echo-args", "record_type": "script",
                "original_path": "echo_args.py",
                "current_path": "scripts/echo_args.py", "status": "active",
                "purpose": "Echo forwarded args", "category": "diagnostics",
                "runnable": "true",
            },
        ]
        document_rows = [{
            "id": "rootzone-report", "record_type": "document",
            "original_path": "rootzone.md", "current_path": "docs/rootzone.md",
            "status": "formal", "purpose": "Rootzone validation report",
            "document_type": "report", "formal_reference": "true",
        }]
        for path, rows in (
            (self.root / "scripts" / "script_catalog.csv", script_rows),
            (self.root / "docs" / "document_catalog.csv", document_rows),
        ):
            with path.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(rows)

    def run_cli(self, *args: str) -> SimpleNamespace:
        stdout, stderr = StringIO(), StringIO()
        with redirect_stdout(stdout), redirect_stderr(stderr):
            code = main(project_root=self.root, argv=list(args))
        return SimpleNamespace(returncode=code, stdout=stdout.getvalue(), stderr=stderr.getvalue())

    def test_find_returns_script_and_document_matches(self) -> None:
        completed = self.run_cli("find", "rootzone")
        self.assertEqual(completed.returncode, 0)
        self.assertIn("script\trootzone-frequency", completed.stdout)
        self.assertIn("document\trootzone-report", completed.stdout)

    def test_run_forwards_arguments_and_exit_code(self) -> None:
        completed = self.run_cli("run", "echo-args", "--", "--value", "7")
        self.assertEqual(completed.returncode, 7)
        self.assertIn("--value 7", completed.stdout)

    def test_run_rejects_document_record(self) -> None:
        completed = self.run_cli("run", "rootzone-report")
        self.assertNotEqual(completed.returncode, 0)
        self.assertIn("not runnable", completed.stderr)
```

- [ ] **Step 2: Run tests and verify RED**

Run:

```powershell
$env:PYTHONPATH='D:\study\s2s_rtist_source\src'
python -m unittest tests.test_project_cli -v
```

Expected: import or missing CLI failures.

- [ ] **Step 3: Implement CLI behavior**

`project_cli.py` only bootstraps `src/`:

```python
#!/usr/bin/env python3
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from s2s_rtist.cli import main

if __name__ == "__main__":
    raise SystemExit(main(project_root=ROOT))
```

`src/s2s_rtist/cli.py` must expose `main(*, project_root: Path, argv: list[str] | None = None) -> int` and:

- load `scripts/script_catalog.csv` and `docs/document_catalog.csv`;
- tag each record with `record_type`;
- implement `list --type scripts|docs|all`;
- implement case-insensitive `find`;
- implement exact-ID `show`;
- implement `run ID -- args` only for `runnable=true` script records;
- remove a leading literal `--` from forwarded arguments before launching the child;
- launch `[sys.executable, script_path, *forwarded_args]` with `cwd=project_root`;
- prepend `src`, the target script directory, and script category directories to `PYTHONPATH`;
- use `difflib.get_close_matches` for unknown IDs;
- return the child exit code unchanged.

Use this parser contract:

```python
parser = argparse.ArgumentParser(description="Search and run project scripts")
subparsers = parser.add_subparsers(dest="command", required=True)
subparsers.add_parser("list").add_argument("--type", choices=("scripts", "docs", "all"), default="scripts")
subparsers.add_parser("find").add_argument("query")
subparsers.add_parser("show").add_argument("id")
run_parser = subparsers.add_parser("run")
run_parser.add_argument("id")
run_parser.add_argument("args", nargs=argparse.REMAINDER)
```

- [ ] **Step 4: Run CLI tests and verify GREEN**

Run the Task 2 test command.

Expected: all CLI tests pass.

- [ ] **Step 5: Commit**

```powershell
git add project_cli.py src/s2s_rtist/cli.py tests/test_project_cli.py
git commit -m "feat: add searchable project CLI"
```

---

### Task 3: Create a Collision-Safe Migration Planner

**Files:**

- Create: `scripts/archive/one_off/migrate_root_files_20260715.py`
- Modify: `tests/test_project_catalog.py`
- Create at runtime: `docs/migration/root_python_inventory_20260715.csv`
- Create at runtime: `docs/migration/root_document_inventory_20260715.csv`

- [ ] **Step 1: Add failing classification and collision tests**

Test these locked mappings:

```python
EXPECTED_DOCUMENT_TARGETS = {
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
```

Also test:

- core module mappings are exact;
- `Main_win.py` and `Main_win_ensemble_mean.py` map to `archive/original_application`;
- `compare_expanded_policy_results_v1.py` and `v2.py` map to `archive/superseded` because `v3.py` exists;
- prefix categories map deterministically;
- any existing target causes the planner to fail before moving files;
- dry-run does not alter source paths.

- [ ] **Step 2: Run tests and verify RED**

Run:

```powershell
$env:PYTHONPATH='D:\study\s2s_rtist_source\src'
python -m unittest tests.test_project_catalog -v
```

Expected: missing migration planner functions.

- [ ] **Step 3: Implement deterministic planning**

The migration utility must expose:

```python
CORE_TARGETS = {
    "gefs_gridmet_bias_validation_v1.py": "src/s2s_rtist/weather/gefs_gridmet_bias.py",
    "generate_restart_decision_dataset.py": "src/s2s_rtist/pipelines/restart_decision_dataset.py",
    "rootzone_flux_frequency_diagnostic_v1.py": "src/s2s_rtist/physics/rootzone_flux_frequency.py",
    "swap_three_output_labels_v1.py": "src/s2s_rtist/labels/swap_three_output_labels.py",
    "validate_three_output_smoke_v1.py": "src/s2s_rtist/validation/three_output_smoke.py",
}

RUNNER_TARGETS = {
    "run_confirmed_5site_restart_generation_smoke_v1.py": "scripts/simulation/run_confirmed_5site_restart_generation_smoke_v1.py",
    "run_continuous_ir_12site_restart_generation_v1.py": "scripts/simulation/run_continuous_ir_12site_restart_generation_v1.py",
    "run_gefs_gridmet_bias_validation_v1.py": "scripts/diagnostics/run_gefs_gridmet_bias_validation_v1.py",
    "run_rootzone_flux_frequency_validation_v1.py": "scripts/diagnostics/run_rootzone_flux_frequency_validation_v1.py",
    "restart_raw_audit_v1.py": "scripts/diagnostics/restart_raw_audit_v1.py",
}
```

Use explicit special cases before prefix classification. Prefix classification must use this order:

```python
PREFIX_CATEGORIES = (
    (("plot_", "visualize_"), "visualization"),
    (("train_", "calibrate_", "sweep_", "resweep_", "optimize_", "smooth_"), "training"),
    (("evaluate_", "summarize_", "collect_", "finalize_", "apply_safe_", "apply_lstm_"), "evaluation"),
    (("analyze_", "audit_", "compare_", "diagnose_", "map_"), "diagnostics"),
    (("apply_", "build_", "extract_", "merge_", "prepare_", "plan_"), "data_preparation"),
    (("decision_", "generate_", "run_"), "simulation"),
)
```

Files not matched by an explicit rule or prefix go to `scripts/archive/one_off/` with `status=legacy_unreviewed`.

The original script inventory explicitly excludes the newly created `project_cli.py`; it must contain the 140 Python files that existed before this reorganization.

The utility must support:

```text
--project-root PATH
--phase formal|remaining|documents|all
--write-inventory
--dry-run
--apply
```

Before any move, build the complete selected plan, reject duplicate targets, reject existing targets, and verify every source exists. Use `Path.rename()` only after the entire plan validates. Track completed moves; if an operating-system error occurs mid-apply, reverse completed moves in reverse order and fail with both the original and rollback errors. Never overwrite.

- [ ] **Step 4: Run tests and verify GREEN**

Run the Task 3 test command.

Expected: all planner and catalog tests pass.

- [ ] **Step 5: Generate and inspect immutable inventories**

Run:

```powershell
$env:PYTHONPATH='D:\study\s2s_rtist_source\src'
python scripts/archive/one_off/migrate_root_files_20260715.py --project-root . --phase all --write-inventory --dry-run
```

Expected:

- `root_python_inventory_20260715.csv` has exactly 140 original root script rows;
- `root_document_inventory_20260715.csv` has exactly 9 rows;
- no target collisions;
- no files moved.

- [ ] **Step 6: Commit planner and inventories**

```powershell
git add scripts/archive/one_off/migrate_root_files_20260715.py tests/test_project_catalog.py docs/migration
git commit -m "chore: inventory project scripts and documents"
```

---

### Task 4: Migrate Formal Modules and Runners Without Behavioral Changes

**Files:**

- Move the five `CORE_TARGETS` files.
- Move the five `RUNNER_TARGETS` files.
- Create package `__init__.py` files under `weather`, `physics`, `labels`, `validation`, and `pipelines`.
- Modify: `tests/test_gefs_gridmet_bias_validation_v1.py`
- Modify: `tests/test_rootzone_flux_frequency_diagnostic_v1.py`
- Modify: `tests/test_swap_three_output_labels_v1.py`
- Modify: `tests/test_three_output_smoke_validation_v1.py`
- Modify imports inside moved formal modules and runners.
- Create/modify: `scripts/script_catalog.csv`

- [ ] **Step 1: Update tests first to require package imports**

Replace root-module imports with:

```python
from s2s_rtist.weather.gefs_gridmet_bias import ...
from s2s_rtist.physics.rootzone_flux_frequency import ...
from s2s_rtist.labels.swap_three_output_labels import ...
from s2s_rtist.validation.three_output_smoke import ...
```

For runner tests, load moved scripts using `importlib.util.spec_from_file_location` from their new paths, because `scripts/` is not a package API.

- [ ] **Step 2: Run the 68 tests and verify RED**

Run:

```powershell
$env:PYTHONPATH='D:\study\s2s_rtist_source\src;D:\study\s2s_rtist_source\.gefs_validation_deps'
python -m unittest tests.test_gefs_gridmet_bias_validation_v1 tests.test_rootzone_flux_frequency_diagnostic_v1 tests.test_swap_three_output_labels_v1 tests.test_three_output_smoke_validation_v1
```

Expected: package modules are missing before formal migration.

- [ ] **Step 3: Apply only the formal migration phase**

Run:

```powershell
$env:PYTHONPATH='D:\study\s2s_rtist_source\src'
python scripts/archive/one_off/migrate_root_files_20260715.py --project-root . --phase formal --apply
```

Expected: exactly 10 original scripts move; no other root scripts move.

- [ ] **Step 4: Update imports in moved code**

Apply these exact import changes:

```python
# physics/rootzone_flux_frequency.py
from s2s_rtist.labels.swap_three_output_labels import _read_crop_table, _read_profile_table, _read_swap_csv

# pipelines/restart_decision_dataset.py
from s2s_rtist.labels.swap_three_output_labels import ...

# scripts/diagnostics/run_gefs_gridmet_bias_validation_v1.py
from s2s_rtist.weather.gefs_gridmet_bias import ...

# scripts/diagnostics/run_rootzone_flux_frequency_validation_v1.py
from s2s_rtist.physics.rootzone_flux_frequency import ...

# scripts/simulation/run_confirmed_5site_restart_generation_smoke_v1.py
from s2s_rtist.validation.three_output_smoke import ...
from s2s_rtist.pipelines import restart_decision_dataset as base

# scripts/simulation/run_continuous_ir_12site_restart_generation_v1.py
from s2s_rtist.pipelines import restart_decision_dataset as base
```

Do not rename public functions or change calculations.

- [ ] **Step 5: Add formal catalog rows**

Register stable runnable IDs:

```text
confirmed-5site-smoke
continuous-12site-generation
gefs-gridmet-bias
rootzone-frequency
restart-raw-audit
```

Register the five package modules with `runnable=false`, `status=formal`, and their original paths/hashes.

- [ ] **Step 6: Run tests and verify GREEN**

Run the Step 2 command.

Expected: `Ran 68 tests ... OK`. Existing deprecation warnings may remain, but no new import errors are allowed.

- [ ] **Step 7: Verify formal CLI lookup**

Run:

```powershell
python project_cli.py find rootzone
python project_cli.py show rootzone-frequency
python project_cli.py find GEFS
```

Expected: each command returns the new path and formal status.

- [ ] **Step 8: Commit**

```powershell
git add src scripts tests project_cli.py pyproject.toml
git commit -m "refactor: package formal weather and physics workflows"
```

---

### Task 5: Move and Catalog All Remaining Root Scripts

**Files:**

- Move: every remaining original root `*.py` except `project_cli.py`.
- Update: `scripts/script_catalog.csv`
- Create: category and archive directories as needed.
- Modify: `tests/test_project_layout.py`

- [ ] **Step 1: Write failing final-layout tests**

Create `tests/test_project_layout.py` with these assertions:

```python
class ProjectLayoutTests(unittest.TestCase):
    def test_root_contains_only_project_cli_python_file(self) -> None:
        root_scripts = sorted(path.name for path in PROJECT_ROOT.glob("*.py"))
        self.assertEqual(root_scripts, ["project_cli.py"])

    def test_script_catalog_covers_all_140_original_scripts(self) -> None:
        inventory = list(csv.DictReader(INVENTORY.open(encoding="utf-8-sig")))
        catalog = Catalog.from_csv(SCRIPT_CATALOG)
        original_paths = {row["original_path"] for row in inventory}
        catalog_originals = {record.original_path for record in catalog.records if record.original_path.endswith(".py")}
        self.assertEqual(len(original_paths), 140)
        self.assertEqual(catalog_originals, original_paths)

    def test_unchanged_files_keep_source_hash(self) -> None:
        catalog = Catalog.from_csv(SCRIPT_CATALOG)
        for record in catalog.records:
            current = sha256_file(PROJECT_ROOT / record.current_path)
            self.assertEqual(current, record.current_sha256)
            if record.status in {"historical", "superseded", "legacy_unreviewed"}:
                self.assertEqual(record.source_sha256, record.current_sha256)
```

- [ ] **Step 2: Run layout tests and verify RED**

Run:

```powershell
$env:PYTHONPATH='D:\study\s2s_rtist_source\src'
python -m unittest tests.test_project_layout -v
```

Expected: root still contains remaining scripts and catalog coverage is incomplete.

- [ ] **Step 3: Preview remaining moves**

Run:

```powershell
python scripts/archive/one_off/migrate_root_files_20260715.py --project-root . --phase remaining --dry-run
```

Expected: every remaining original root script appears once, targets are unique, and no existing target is overwritten.

- [ ] **Step 4: Apply remaining moves and regenerate catalog hashes**

Run:

```powershell
python scripts/archive/one_off/migrate_root_files_20260715.py --project-root . --phase remaining --apply
python scripts/archive/one_off/migrate_root_files_20260715.py --project-root . --phase all --dry-run
```

The second command must report that all original paths are already represented at their target paths, not treat moved sources as missing.

- [ ] **Step 5: Compile every migrated script**

Run:

```powershell
$files=(Import-Csv scripts\script_catalog.csv | Where-Object current_path -like '*.py').current_path
python -m py_compile $files
```

Expected: exit code 0 for all cataloged Python files. A syntax failure blocks completion. Repair only the incompatible syntax while preserving behavior, record the content change and old/new hashes in the archive README and catalog, then rerun compilation until every file passes.

- [ ] **Step 6: Run layout tests and verify GREEN**

Run the Step 2 command.

Expected: root only contains `project_cli.py`; all 140 originals are cataloged; hash checks pass.

- [ ] **Step 7: Commit**

```powershell
git add scripts tests/test_project_layout.py
git commit -m "chore: categorize historical research scripts"
```

---

### Task 6: Move, Catalog, and Search the Nine Root Documents

**Files:**

- Move: all 9 locked root Markdown/TXT files.
- Create: `docs/document_catalog.csv`
- Modify: `src/s2s_rtist/cli.py`
- Modify: `tests/test_project_cli.py`
- Modify: `tests/test_project_layout.py`
- Modify: `.gitignore`

- [ ] **Step 1: Add failing document coverage tests**

Add tests asserting:

```python
def test_no_original_markdown_or_text_files_remain_at_root(self) -> None:
    remaining = sorted(
        path.name for path in PROJECT_ROOT.iterdir()
        if path.is_file() and path.suffix.lower() in {".md", ".txt"}
    )
    self.assertEqual(remaining, [])

def test_document_catalog_covers_all_nine_original_files(self) -> None:
    inventory = list(csv.DictReader(DOCUMENT_INVENTORY.open(encoding="utf-8-sig")))
    catalog = Catalog.from_csv(DOCUMENT_CATALOG)
    self.assertEqual(len(inventory), 9)
    self.assertEqual(
        {row["original_path"] for row in inventory},
        {record.original_path for record in catalog.records},
    )
```

Add a CLI test that `find smoke` returns both server documents and smoke scripts.

- [ ] **Step 2: Run tests and verify RED**

Run:

```powershell
$env:PYTHONPATH='D:\study\s2s_rtist_source\src'
python -m unittest tests.test_project_cli tests.test_project_layout -v
```

Expected: root documents remain and the document catalog is absent.

- [ ] **Step 3: Update Git ignore policy for local paper extracts**

Add:

```gitignore
/docs/archive/paper_extracts/
```

Do not ignore `docs/document_catalog.csv`; it must track both local paper extract paths and their hashes.

- [ ] **Step 4: Apply document migration**

Run:

```powershell
python scripts/archive/one_off/migrate_root_files_20260715.py --project-root . --phase documents --dry-run
python scripts/archive/one_off/migrate_root_files_20260715.py --project-root . --phase documents --apply
```

Expected: exactly 9 files move, the two paper extracts remain present locally but ignored, and no root Markdown/TXT files remain.

- [ ] **Step 5: Populate document catalog metadata**

Use these statuses:

- three 2026-07-14/15 server guides: `formal`, `formal_reference=true`;
- first-step reproduction notes: `historical`, `formal_reference=false`;
- server restart notes and `Instructions.txt`: `historical`, `formal_reference=false`;
- paper extracts: `local_reference`, `formal_reference=false`;
- requirements file: `active`, `formal_reference=false`, `document_type=requirements`.

Relate each formal server guide to its CLI ID through `related_script_ids`.

- [ ] **Step 6: Run tests and verify GREEN**

Run the Step 2 command.

Expected: document coverage and mixed script/document search pass.

- [ ] **Step 7: Commit tracked document moves and catalog**

```powershell
git add .gitignore docs requirements src/s2s_rtist/cli.py tests/test_project_cli.py tests/test_project_layout.py
git commit -m "docs: organize project guides and research notes"
```

Verify the ignored paper extracts are not staged:

```powershell
git diff --cached --name-only | Select-String 'paper_extracts'
```

Expected: no output.

---

### Task 7: Add Human Navigation and Version Documentation

**Files:**

- Create: `scripts/README.md`
- Create: one `README.md` in every script category and archive directory.
- Create: `scripts/archive/VERSIONS.md`
- Create: `docs/README.md`
- Create: `docs/operations/server/README.md`
- Create: `docs/research/reproduction/README.md`
- Create: `docs/archive/README.md`
- Modify: `tests/test_project_layout.py`

- [ ] **Step 1: Add failing README coverage tests**

Test exact required paths:

```python
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

def test_required_navigation_files_exist(self) -> None:
    self.assertEqual(
        [path for path in REQUIRED_READMES if not (PROJECT_ROOT / path).is_file()],
        [],
    )
```

- [ ] **Step 2: Run and verify RED**

Run `python -m unittest tests.test_project_layout -v` with `PYTHONPATH=src`.

Expected: required README files are missing.

- [ ] **Step 3: Write navigation files from catalog facts**

Every script category README must contain:

```markdown
# <Category Name>

## Purpose
<One paragraph defining the directory boundary.>

## Current Scripts
| ID | Status | Script | Purpose |
|---|---|---|---|
<Rows copied from script_catalog.csv for this category.>

## Usage
`python project_cli.py show <id>`
`python project_cli.py run <id> -- <args>`

## Version Notes
<Explicit references to scripts/archive/VERSIONS.md when applicable.>
```

`scripts/archive/VERSIONS.md` must explicitly record:

- `compare_expanded_policy_results_v1.py` -> `v3.py`;
- `compare_expanded_policy_results_v2.py` -> `v3.py`;
- `extract_weather_sequences_v1.py` -> `v2.py` only if imports/results confirm replacement; otherwise mark both historical without `replaced_by`;
- `evaluate_learned_trigger_curve_policy_v1.py` -> `v2.py` only if results/docs confirm replacement;
- the reason `Main_win.py` and `Main_win_ensemble_mean.py` are preserved as original application entry points.

Do not infer a replacement solely from a higher version number.

- [ ] **Step 4: Run layout tests and verify GREEN**

Expected: all required README/version files exist.

- [ ] **Step 5: Commit**

```powershell
git add scripts docs tests/test_project_layout.py
git commit -m "docs: add script and document navigation"
```

---

### Task 8: Update Formal Imports, Server Commands, Bundles, and Cross-References

**Files:**

- Modify: `docs/operations/server/*.md`
- Modify: formal tracked reports under `site_general_surrogate_eval/` that reference moved root scripts.
- Modify: `docs/superpowers/plans/2026-07-13-gefs-three-output-physics-tta.md`
- Modify: any bundle builder or runner that copies formal files by old root filename.
- Modify: `tests/test_project_layout.py`

- [ ] **Step 1: Add a failing stale-reference test**

Define the moved formal names and reject root-level command/path references:

```python
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

def test_formal_docs_do_not_use_old_root_commands(self) -> None:
    failures = []
    for path in FORMAL_DOCUMENTS:
        text = path.read_text(encoding="utf-8")
        for name in MOVED_FORMAL_NAMES:
            if f"python3 {name}" in text or f"python {name}" in text:
                failures.append(f"{path}:{name}")
    self.assertEqual(failures, [])
```

Also add a Markdown relative-link validator for tracked formal docs.

- [ ] **Step 2: Run and verify RED**

Run `python -m unittest tests.test_project_layout -v`.

Expected: existing server guides still contain old root commands.

- [ ] **Step 3: Replace formal commands with stable CLI IDs**

Use:

```text
python3 project_cli.py run confirmed-5site-smoke -- <args>
python3 project_cli.py run continuous-12site-generation -- <args>
python3 project_cli.py run gefs-gridmet-bias -- <args>
python3 project_cli.py run rootzone-frequency -- <args>
python3 project_cli.py run restart-raw-audit -- <args>
```

When a document discusses implementation rather than execution, reference the new `src/s2s_rtist/...` module path.

- [ ] **Step 4: Update server bundle contents**

Any bundle that previously copied individual root modules must include:

```text
project_cli.py
src/s2s_rtist/
scripts/script_catalog.csv
the required scripts/<category>/ runner
requirements/ when applicable
```

Do not package all historical scripts into formal server bundles.

- [ ] **Step 5: Search for stale formal references**

Run:

```powershell
rg -n 'python3? (run_|generate_restart_decision_dataset|gefs_gridmet_bias_validation|rootzone_flux_frequency_diagnostic|swap_three_output_labels|validate_three_output_smoke)' docs site_general_surrogate_eval -g '*.md'
```

Expected: no executable old-root commands. Historical prose may mention `original_path` only inside catalogs or migration docs.

- [ ] **Step 6: Run tests and verify GREEN**

Run the layout tests and the existing 68-test suite.

Expected: stale-reference and relative-link checks pass; 68 existing tests remain green.

- [ ] **Step 7: Commit**

```powershell
git add docs site_general_surrogate_eval scripts src tests
git commit -m "docs: update commands for organized project layout"
```

---

### Task 9: Final Verification and Repository Scope Audit

**Files:**

- Modify only if verification exposes an in-scope defect.
- Update: `scripts/script_catalog.csv` and `docs/document_catalog.csv` final hashes.

- [ ] **Step 1: Verify root cleanliness and counts**

Run:

```powershell
(Get-ChildItem -LiteralPath . -File -Filter '*.py').Name
Get-ChildItem -LiteralPath . -File | Where-Object { $_.Extension -in @('.md','.txt') }
(Import-Csv docs\migration\root_python_inventory_20260715.csv).Count
(Import-Csv docs\migration\root_document_inventory_20260715.csv).Count
```

Expected:

- Python output is only `project_cli.py`;
- Markdown/TXT output is empty;
- inventory counts are `140` and `9`.

- [ ] **Step 2: Verify catalog paths and hashes**

Run:

```powershell
$env:PYTHONPATH='D:\study\s2s_rtist_source\src'
python -m unittest tests.test_project_catalog tests.test_project_layout -v
```

Expected: all paths, counts, IDs, and hashes pass.

- [ ] **Step 3: Verify CLI search and lookup**

Run:

```powershell
python project_cli.py list
python project_cli.py list --type docs
python project_cli.py find rootzone
python project_cli.py find smoke
python project_cli.py show rootzone-frequency
```

Expected: formal entries appear first; script and document matches show record types and current paths.

- [ ] **Step 4: Compile cataloged Python files**

Run:

```powershell
$files=(Import-Csv scripts\script_catalog.csv | Where-Object { $_.current_path -like '*.py' }).current_path
python -m py_compile $files
```

Expected: exit code 0 for every cataloged Python file. There are no compile exclusions in the completion criteria.

- [ ] **Step 5: Run full relevant test suite**

Run:

```powershell
$env:PYTHONPATH='D:\study\s2s_rtist_source\src;D:\study\s2s_rtist_source\.gefs_validation_deps'
python -m unittest tests.test_project_catalog tests.test_project_cli tests.test_project_layout tests.test_gefs_gridmet_bias_validation_v1 tests.test_rootzone_flux_frequency_diagnostic_v1 tests.test_swap_three_output_labels_v1 tests.test_three_output_smoke_validation_v1 -v
```

Expected: all new organization tests and all 68 existing tests pass.

- [ ] **Step 6: Audit Git scope**

Run from `D:\study`:

```powershell
git status --short --branch
git diff --cached --name-only
```

Expected:

- no files outside `s2s_rtist_source/` are staged;
- the six pre-existing user-modified root files remain modified and unstaged;
- large results, weather data, model files, archives, and paper extracts remain untracked/ignored as designed.

- [ ] **Step 7: Commit final catalog hashes or fixes**

If final hashes changed during documentation/import updates:

```powershell
git add scripts/script_catalog.csv docs/document_catalog.csv
git commit -m "chore: finalize project organization catalogs"
```

If no files changed, do not create an empty commit.

- [ ] **Step 8: Record final result**

Report:

- final commit range;
- root script/document counts;
- catalog counts;
- test totals;
- confirmation that all cataloged Python files compile under Python 3.10;
- exact CLI examples;
- confirmation that no script or document was deleted.
