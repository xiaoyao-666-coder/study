"""Catalog model for the collision-safe project inventory."""

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
    """Return the SHA-256 digest of *path*, reading at most 1 MiB at a time."""

    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


class Catalog:
    """Validated collection of immutable catalog records."""

    def __init__(self, records: Iterable[CatalogRecord]):
        self.records = list(records)
        seen_ids: set[str] = set()
        seen_paths: set[str] = set()
        for record in self.records:
            if record.id in seen_ids:
                raise ValueError(f"duplicate catalog id: {record.id}")
            seen_ids.add(record.id)
            if record.current_path in seen_paths:
                raise ValueError(f"duplicate current_path: {record.current_path}")
            seen_paths.add(record.current_path)

    @classmethod
    def from_csv(cls, path: Path) -> "Catalog":
        """Load catalog records from a UTF-8 (optionally BOM-prefixed) CSV."""

        field_names = {field.name for field in fields(CatalogRecord)}
        records: list[CatalogRecord] = []
        with Path(path).open("r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                values = {
                    name: row[name]
                    for name in field_names
                    if name in row and row[name] is not None
                }
                records.append(CatalogRecord(**values))
        return cls(records)

    def find(self, query: str) -> list[CatalogRecord]:
        """Find records whose serialized field values contain *query*."""

        needle = query.casefold()
        return [
            record
            for record in self.records
            if any(needle in str(getattr(record, field.name)).casefold() for field in fields(record))
        ]

    def get(self, record_id: str) -> CatalogRecord:
        for record in self.records:
            if record.id == record_id:
                return record
        raise KeyError(record_id)

    def validate_paths(self, root: Path) -> None:
        missing = [
            record.current_path
            for record in self.records
            if not (Path(root) / record.current_path).is_file()
        ]
        if missing:
            preview = ", ".join(missing[:5])
            raise ValueError(f"missing catalog paths: {preview}")
