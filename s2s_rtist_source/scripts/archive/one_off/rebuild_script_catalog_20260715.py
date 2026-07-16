#!/usr/bin/env python3
"""Rebuild scripts/script_catalog.csv from the immutable inventory + current hashes."""

from __future__ import annotations

import csv
from pathlib import Path

from s2s_rtist.catalog import sha256_file


def main() -> int:
    root = Path(__file__).resolve().parents[3]
    inventory = root / "docs" / "migration" / "root_python_inventory_20260715.csv"
    output = root / "scripts" / "script_catalog.csv"

    with inventory.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        fieldnames = reader.fieldnames
        if not fieldnames:
            raise SystemExit("inventory has no headers")
        rows = []
        for row in reader:
            target = root / row["current_path"]
            if not target.is_file():
                raise SystemExit(f"missing {row['current_path']}")
            row["current_sha256"] = sha256_file(target)
            # Keep source_sha256 from the pre-migration inventory snapshot.
            # Formal modules rewired after the move intentionally differ.
            if row["status"] in {"historical", "superseded", "legacy_unreviewed"}:
                if row["current_sha256"] != row["source_sha256"]:
                    raise SystemExit(
                        f"archived script content changed: {row['original_path']}"
                    )
            rows.append(row)

    with output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)

    print(f"wrote {len(rows)} script catalog rows to {output.relative_to(root)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
