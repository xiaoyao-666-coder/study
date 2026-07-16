#!/usr/bin/env python3
"""Inventory and collision-safe migration planner for root scripts and documents."""

import argparse
import csv
import hashlib
from dataclasses import asdict, dataclass, fields, replace
from pathlib import Path
from typing import Iterable


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

RUNNER_IDS = {
    "run_confirmed_5site_restart_generation_smoke_v1.py": "confirmed-5site-smoke",
    "run_continuous_ir_12site_restart_generation_v1.py": "continuous-12site-generation",
    "run_gefs_gridmet_bias_validation_v1.py": "gefs-gridmet-bias",
    "run_rootzone_flux_frequency_validation_v1.py": "rootzone-frequency",
    "restart_raw_audit_v1.py": "restart-raw-audit",
}

DOCUMENT_TARGETS = {
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

PREFIX_CATEGORIES = (
    (("plot_", "visualize_"), "visualization"),
    (("train_", "calibrate_", "sweep_", "resweep_", "optimize_", "smooth_"), "training"),
    (("evaluate_", "summarize_", "collect_", "finalize_", "apply_safe_", "apply_lstm_"), "evaluation"),
    (("analyze_", "audit_", "compare_", "diagnose_", "map_"), "diagnostics"),
    (("apply_", "build_", "extract_", "merge_", "prepare_", "plan_"), "data_preparation"),
    (("decision_", "generate_", "run_"), "simulation"),
)

PYTHON_INVENTORY_PATH = Path("docs/migration/root_python_inventory_20260715.csv")
DOCUMENT_INVENTORY_PATH = Path("docs/migration/root_document_inventory_20260715.csv")
MIGRATION_SCRIPT_NAME = "migrate_root_files_20260715.py"


@dataclass(frozen=True)
class MoveRecord:
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


@dataclass(frozen=True)
class Classification:
    id: str
    current_path: str
    status: str
    purpose: str
    category: str = ""
    document_type: str = ""
    replaced_by: str = ""
    related_script_ids: str = ""
    formal_reference: str = "false"
    runnable: str = "false"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _slug(name: str) -> str:
    return Path(name).stem.lower().replace("_", "-")


def _purpose(name: str) -> str:
    return Path(name).stem.replace("_", " ")


def _script_ids(names: Iterable[str]) -> dict[str, str]:
    ordered = sorted(set(names), key=lambda value: (value.casefold(), value))
    assigned = {name: RUNNER_IDS[name] for name in ordered if name in RUNNER_IDS}
    used = set(assigned.values())
    for name in ordered:
        if name in assigned:
            continue
        base = _slug(name)
        candidate = base
        suffix = 2
        while candidate in used:
            candidate = f"{base}-{suffix}"
            suffix += 1
        assigned[name] = candidate
        used.add(candidate)
    return assigned


def classify_script(name: str, all_root_script_names: Iterable[str]) -> Classification:
    names = set(all_root_script_names) | {name}
    record_id = _script_ids(names)[name]
    purpose = _purpose(name)

    if name in CORE_TARGETS:
        category = Path(CORE_TARGETS[name]).parent.name
        return Classification(
            record_id,
            CORE_TARGETS[name],
            "formal",
            purpose,
            category=category,
            formal_reference="true",
        )
    if name in RUNNER_TARGETS:
        category = Path(RUNNER_TARGETS[name]).parent.name
        return Classification(
            record_id,
            RUNNER_TARGETS[name],
            "formal",
            purpose,
            category=category,
            formal_reference="true",
            runnable="true",
        )
    if name in {"Main_win.py", "Main_win_ensemble_mean.py"}:
        return Classification(
            record_id,
            f"scripts/archive/original_application/{name}",
            "historical",
            purpose,
            category="archive",
        )
    if name in {
        "compare_expanded_policy_results_v1.py",
        "compare_expanded_policy_results_v2.py",
    } and "compare_expanded_policy_results_v3.py" in names:
        return Classification(
            record_id,
            f"scripts/archive/superseded/{name}",
            "superseded",
            purpose,
            category="archive",
            replaced_by="compare_expanded_policy_results_v3.py",
        )
    for prefixes, category in PREFIX_CATEGORIES:
        if name.startswith(prefixes):
            return Classification(
                record_id,
                f"scripts/{category}/{name}",
                "active",
                purpose,
                category=category,
            )
    return Classification(
        record_id,
        f"scripts/archive/one_off/{name}",
        "legacy_unreviewed",
        purpose,
        category="archive",
    )


def classify_document(name: str) -> Classification:
    if name not in DOCUMENT_TARGETS:
        raise ValueError(f"unrecognized root document: {name}")

    target = DOCUMENT_TARGETS[name]
    if name in {
        "fixed_0_100cm_5site_smoke_server_run_20260715.md",
        "formal_npd24_5site_smoke_server_run_20260714.md",
        "gefs_gridmet_bias_validation_server_run_20260715.md",
    }:
        related = {
            "fixed_0_100cm_5site_smoke_server_run_20260715.md": "confirmed-5site-smoke",
            "formal_npd24_5site_smoke_server_run_20260714.md": "confirmed-5site-smoke,rootzone-frequency",
            "gefs_gridmet_bias_validation_server_run_20260715.md": "gefs-gridmet-bias",
        }[name]
        return Classification(
            _slug(name),
            target,
            "formal",
            _purpose(name),
            document_type="server_guide",
            related_script_ids=related,
            formal_reference="true",
        )
    if name == "first_step_reproduction_notes_2026-05-29.md":
        return Classification(
            _slug(name), target, "historical", _purpose(name), document_type="reproduction_notes"
        )
    if name in {"server_restart_smoke_notes_2026-05-30.md", "Instructions.txt"}:
        return Classification(
            _slug(name), target, "historical", _purpose(name), document_type="historical_notes"
        )
    if name in {"paper_keyword_snippets.txt", "paper_text_2026ems.txt"}:
        return Classification(
            _slug(name), target, "local_reference", _purpose(name), document_type="paper_extract"
        )
    return Classification(
        _slug(name),
        target,
        "active",
        _purpose(name),
        document_type="requirements",
        related_script_ids="gefs-gridmet-bias",
    )


def _record_from_classification(
    root: Path, name: str, record_type: str, classification: Classification
) -> MoveRecord:
    digest = sha256_file(root / name)
    return MoveRecord(
        id=classification.id,
        record_type=record_type,
        original_path=name,
        current_path=classification.current_path,
        status=classification.status,
        purpose=classification.purpose,
        category=classification.category,
        document_type=classification.document_type,
        replaced_by=classification.replaced_by,
        related_script_ids=classification.related_script_ids,
        formal_reference=classification.formal_reference,
        runnable=classification.runnable,
        source_sha256=digest,
        current_sha256=digest,
    )


def _validate_unique_records(records: Iterable[MoveRecord]) -> None:
    seen_sources: set[str] = set()
    seen_targets: set[str] = set()
    seen_ids: set[tuple[str, str]] = set()
    for record in records:
        if record.original_path in seen_sources:
            raise ValueError(f"duplicate source: {record.original_path}")
        seen_sources.add(record.original_path)
        if record.current_path in seen_targets:
            raise ValueError(f"duplicate target: {record.current_path}")
        seen_targets.add(record.current_path)
        id_key = (record.record_type, record.id)
        if id_key in seen_ids:
            raise ValueError(f"duplicate {record.record_type} id: {record.id}")
        seen_ids.add(id_key)


def _inventory_paths(root: Path) -> tuple[Path, Path]:
    return root / PYTHON_INVENTORY_PATH, root / DOCUMENT_INVENTORY_PATH


def _load_inventory(path: Path, record_type: str) -> list[MoveRecord]:
    field_names = [field.name for field in fields(MoveRecord)]
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != field_names:
            raise ValueError(f"invalid immutable inventory headers: {path}")
        records = [MoveRecord(**row) for row in reader]
    if any(record.record_type != record_type for record in records):
        raise ValueError(f"invalid record_type in immutable inventory: {path}")
    return records


def _load_existing_inventories(root: Path) -> list[MoveRecord] | None:
    script_path, document_path = _inventory_paths(root)
    if script_path.exists() != document_path.exists():
        raise ValueError("partial immutable inventory set; both CSV files are required")
    if not script_path.exists():
        return None
    records = _load_inventory(script_path, "script") + _load_inventory(document_path, "document")
    _validate_unique_records(records)
    return records


def _detect_partial_tree_without_inventory(root: Path) -> None:
    known_pairs = {**CORE_TARGETS, **RUNNER_TARGETS, **DOCUMENT_TARGETS}
    known_pairs.update(
        {
            "Main_win.py": "scripts/archive/original_application/Main_win.py",
            "Main_win_ensemble_mean.py": "scripts/archive/original_application/Main_win_ensemble_mean.py",
            "compare_expanded_policy_results_v1.py": "scripts/archive/superseded/compare_expanded_policy_results_v1.py",
            "compare_expanded_policy_results_v2.py": "scripts/archive/superseded/compare_expanded_policy_results_v2.py",
        }
    )
    for source, target in known_pairs.items():
        if not (root / source).exists() and (root / target).exists():
            raise ValueError(f"partial migration detected without immutable inventories: {target}")

    migration_roots = [
        root / "scripts" / category
        for category in (
            "data_preparation",
            "simulation",
            "diagnostics",
            "training",
            "evaluation",
            "visualization",
        )
    ] + [
        root / "scripts" / "archive" / "superseded",
        root / "scripts" / "archive" / "original_application",
        root / "scripts" / "archive" / "one_off",
    ]
    for directory in migration_roots:
        if not directory.is_dir():
            continue
        for path in directory.rglob("*.py"):
            if path.name != MIGRATION_SCRIPT_NAME:
                raise ValueError(
                    f"partial migration detected without immutable inventories: {path.relative_to(root)}"
                )


def _discover_records(root: Path) -> list[MoveRecord]:
    _detect_partial_tree_without_inventory(root)
    script_names = sorted(
        (path.name for path in root.glob("*.py") if path.name != "project_cli.py"),
        key=lambda value: (value.casefold(), value),
    )
    document_names = [name for name in DOCUMENT_TARGETS if (root / name).is_file()]
    records = [
        _record_from_classification(root, name, "script", classify_script(name, script_names))
        for name in script_names
    ]
    records.extend(
        _record_from_classification(root, name, "document", classify_document(name))
        for name in sorted(document_names, key=lambda value: (value.casefold(), value))
    )
    _validate_unique_records(records)
    return records


def _filter_phase(records: Iterable[MoveRecord], phase: str) -> list[MoveRecord]:
    records = list(records)
    if phase == "all":
        return records
    if phase == "documents":
        return [record for record in records if record.record_type == "document"]
    formal_names = set(CORE_TARGETS) | set(RUNNER_TARGETS)
    if phase == "formal":
        return [
            record
            for record in records
            if record.record_type == "script" and record.original_path in formal_names
        ]
    if phase == "remaining":
        return [
            record
            for record in records
            if record.record_type == "script" and record.original_path not in formal_names
        ]
    raise ValueError(f"unknown migration phase: {phase}")


def build_plan(project_root: Path, phase: str) -> list[MoveRecord]:
    root = Path(project_root).resolve()
    records = _load_existing_inventories(root)
    if records is None:
        records = _discover_records(root)
    return _filter_phase(records, phase)


def validate_plan(project_root: Path, records: Iterable[MoveRecord]) -> dict[str, str]:
    root = Path(project_root).resolve()
    records = list(records)
    _validate_unique_records(records)
    states: dict[str, str] = {}

    for record in records:
        source = root / record.original_path
        target = root / record.current_path
        source_exists = source.exists()
        target_exists = target.exists()
        if source_exists and target_exists:
            raise ValueError(f"collision: source and target both exist for {record.original_path}")
        if not source_exists and not target_exists:
            raise ValueError(f"missing source and target for {record.original_path}")
        if source_exists:
            if not source.is_file():
                raise ValueError(f"source is not a file: {record.original_path}")
            if sha256_file(source) != record.source_sha256:
                raise ValueError(f"source hash mismatch: {record.original_path}")
            states[record.original_path] = "pending"
            continue
        if not target.is_file():
            raise ValueError(f"target is not a file: {record.current_path}")
        if sha256_file(target) != record.source_sha256:
            raise ValueError(f"target hash mismatch: {record.current_path}")
        states[record.original_path] = "already_moved"

    return states


def apply_plan(project_root: Path, records: Iterable[MoveRecord]) -> list[MoveRecord]:
    root = Path(project_root).resolve()
    records = list(records)
    states = validate_plan(root, records)
    pending = [record for record in records if states[record.original_path] == "pending"]
    for record in pending:
        (root / record.current_path).parent.mkdir(parents=True, exist_ok=True)

    completed: list[MoveRecord] = []
    try:
        for record in pending:
            source = root / record.original_path
            target = root / record.current_path
            if target.exists():
                raise OSError(f"target appeared after validation: {record.current_path}")
            source.rename(target)
            completed.append(record)
    except OSError as failure:
        rollback_failures: list[str] = []
        for record in reversed(completed):
            source = root / record.original_path
            target = root / record.current_path
            try:
                if source.exists():
                    raise OSError(f"rollback source exists: {record.original_path}")
                target.rename(source)
            except OSError as rollback_failure:
                rollback_failures.append(f"{record.current_path}: {rollback_failure}")
        details = "; ".join(rollback_failures) if rollback_failures else "none"
        raise RuntimeError(f"migration failed: {failure}; rollback failures: {details}") from failure
    return completed


def _write_inventory(path: Path, records: Iterable[MoveRecord]) -> None:
    field_names = [field.name for field in fields(MoveRecord)]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=field_names, lineterminator="\n")
        writer.writeheader()
        writer.writerows(asdict(record) for record in records)


def write_inventories(project_root: Path, full_records: Iterable[MoveRecord]) -> None:
    root = Path(project_root).resolve()
    records = list(full_records)
    _validate_unique_records(records)
    scripts = [record for record in records if record.record_type == "script"]
    documents = [record for record in records if record.record_type == "document"]
    script_path, document_path = _inventory_paths(root)
    existing = _load_existing_inventories(root)
    if existing is not None:
        if existing != scripts + documents:
            raise ValueError("immutable inventories differ from the proposed records")
        return
    script_path.parent.mkdir(parents=True, exist_ok=True)
    _write_inventory(script_path, scripts)
    _write_inventory(document_path, documents)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--phase", choices=("formal", "remaining", "documents", "all"), required=True)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--dry-run", action="store_true")
    mode.add_argument("--apply", action="store_true")
    parser.add_argument("--write-inventory", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    root = args.project_root.resolve()
    if args.write_inventory:
        full_records = build_plan(root, "all")
        write_inventories(root, full_records)
        records = _filter_phase(full_records, args.phase)
    else:
        records = build_plan(root, args.phase)

    states = validate_plan(root, records)
    for record in records:
        print(f"{record.original_path} -> {record.current_path} [{states[record.original_path]}]")
    if args.apply:
        moved = apply_plan(root, records)
        print(f"moved {len(moved)} file(s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
