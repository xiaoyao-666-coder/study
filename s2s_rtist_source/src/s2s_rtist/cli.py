"""Command-line search and execution for project catalog records."""

from __future__ import annotations

import argparse
import difflib
import os
import subprocess
import sys
from dataclasses import fields, replace
from pathlib import Path

from .catalog import Catalog, CatalogRecord


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Search and run project scripts")
    subparsers = parser.add_subparsers(dest="command", required=True)

    list_parser = subparsers.add_parser("list")
    list_parser.add_argument(
        "--type", choices=("scripts", "docs", "all"), default="scripts"
    )

    find_parser = subparsers.add_parser("find")
    find_parser.add_argument("query", metavar="QUERY")

    show_parser = subparsers.add_parser("show")
    show_parser.add_argument("record_id", metavar="ID")

    run_parser = subparsers.add_parser("run")
    run_parser.add_argument("record_id", metavar="ID")
    run_parser.add_argument("args", nargs=argparse.REMAINDER, metavar="ARGS")
    return parser


def _load_catalogs(project_root: Path) -> tuple[Catalog, Catalog, Catalog]:
    script_catalog = Catalog.from_csv(project_root / "scripts" / "script_catalog.csv")
    script_catalog = Catalog(
        replace(record, record_type="script") for record in script_catalog.records
    )
    document_catalog = Catalog.from_csv(project_root / "docs" / "document_catalog.csv")
    document_catalog = Catalog(
        replace(record, record_type="document")
        for record in document_catalog.records
    )
    combined = Catalog([*script_catalog.records, *document_catalog.records])
    return script_catalog, document_catalog, combined


def _print_record_summary(record: CatalogRecord) -> None:
    print(f"{record.record_type}\t{record.id}\t{record.current_path}\t{record.purpose}")


def _show_record(record: CatalogRecord) -> None:
    for field in fields(record):
        print(f"{field.name}: {getattr(record, field.name)}")


def _unknown_id(record_id: str, catalog: Catalog) -> int:
    suggestions = difflib.get_close_matches(
        record_id, [record.id for record in catalog.records], n=3
    )
    message = f"error: unknown ID '{record_id}'"
    if suggestions:
        message += f"; did you mean: {', '.join(suggestions)}?"
    print(message, file=sys.stderr)
    return 2


def _get_record(record_id: str, catalog: Catalog) -> CatalogRecord | None:
    try:
        return catalog.get(record_id)
    except KeyError:
        _unknown_id(record_id, catalog)
        return None


def _path_key(path: str) -> str:
    return os.path.normcase(os.path.normpath(path))


def _child_pythonpath(project_root: Path, script_path: Path) -> str:
    entries = [str(project_root / "src"), str(script_path.parent)]
    scripts_root = project_root / "scripts"
    if scripts_root.is_dir():
        entries.extend(
            str(path)
            for path in sorted(scripts_root.iterdir(), key=lambda path: path.name.casefold())
            if path.is_dir()
        )
    entries.extend(filter(None, os.environ.get("PYTHONPATH", "").split(os.pathsep)))

    unique_entries: list[str] = []
    seen: set[str] = set()
    for entry in entries:
        key = _path_key(entry)
        if key not in seen:
            seen.add(key)
            unique_entries.append(entry)
    return os.pathsep.join(unique_entries)


def _supports_child_inheritance(stream: object) -> bool:
    try:
        stream.fileno()  # type: ignore[attr-defined]
    except (AttributeError, OSError, ValueError):
        return False
    return True


def _run_record(
    project_root: Path, record: CatalogRecord, forwarded_args: list[str]
) -> int:
    if (
        record.record_type.casefold() != "script"
        or record.runnable.casefold() != "true"
    ):
        print(f"error: record '{record.id}' is not runnable", file=sys.stderr)
        return 2

    if forwarded_args[:1] == ["--"]:
        forwarded_args = forwarded_args[1:]

    script_path = project_root / record.current_path
    environment = os.environ.copy()
    environment["PYTHONPATH"] = _child_pythonpath(project_root, script_path)
    command = [sys.executable, str(script_path), *forwarded_args]

    capture_output = not (
        _supports_child_inheritance(sys.stdout) and _supports_child_inheritance(sys.stderr)
    )
    try:
        if capture_output:
            completed = subprocess.run(
                command,
                cwd=project_root,
                env=environment,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                errors="replace",
            )
            sys.stdout.write(completed.stdout)
            sys.stderr.write(completed.stderr)
        else:
            completed = subprocess.run(command, cwd=project_root, env=environment)
    except OSError as error:
        print(f"error: could not run '{record.id}': {error}", file=sys.stderr)
        return 2
    return completed.returncode


def main(*, project_root: Path, argv: list[str] | None = None) -> int:
    parser = _build_parser()
    arguments = parser.parse_args(argv)
    root = Path(project_root)
    try:
        scripts, documents, combined = _load_catalogs(root)
    except (OSError, TypeError, ValueError, KeyError) as error:
        print(f"error: could not load catalog: {error}", file=sys.stderr)
        return 2

    if arguments.command == "list":
        records = {
            "scripts": scripts.records,
            "docs": documents.records,
            "all": combined.records,
        }[arguments.type]
        for record in records:
            _print_record_summary(record)
        return 0

    if arguments.command == "find":
        for record in combined.find(arguments.query):
            _print_record_summary(record)
        return 0

    record = _get_record(arguments.record_id, combined)
    if record is None:
        return 2
    if arguments.command == "show":
        _show_record(record)
        return 0
    return _run_record(root, record, arguments.args)
