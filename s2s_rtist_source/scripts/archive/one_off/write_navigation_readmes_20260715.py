#!/usr/bin/env python3
"""Write navigation README files from current catalog facts."""

from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]

SCRIPT_CATEGORY_META = {
    "data_preparation": (
        "Data Preparation",
        "Scripts that prepare workspaces, features, weather/soil inputs, and surrogate tables before training or evaluation.",
    ),
    "simulation": (
        "Simulation",
        "Scripts that launch SWAP restart generation, decision smokes, and related candidate simulation workflows.",
    ),
    "diagnostics": (
        "Diagnostics",
        "Scripts that audit inputs, compare policies, diagnose failures, and run formal validation diagnostics.",
    ),
    "training": (
        "Training",
        "Scripts that train, calibrate, sweep, optimize, or smooth surrogate and ranking models.",
    ),
    "evaluation": (
        "Evaluation",
        "Scripts that evaluate policies, summarize model results, finalize tables, and apply learned decision rules.",
    ),
    "visualization": (
        "Visualization",
        "Scripts that plot or visualize schedules, labels, and bilingual result figures.",
    ),
    "archive": (
        "Archive",
        "Preserved original application entry points, superseded policy comparators, and one-off migration utilities.",
    ),
}


def _load_script_rows() -> list[dict[str, str]]:
    path = ROOT / "scripts" / "script_catalog.csv"
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def _load_document_rows() -> list[dict[str, str]]:
    path = ROOT / "docs" / "document_catalog.csv"
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def _table(rows: list[dict[str, str]]) -> str:
    lines = [
        "| ID | Status | Script | Purpose |",
        "|---|---|---|---|",
    ]
    for row in sorted(rows, key=lambda item: (item["id"], item["current_path"])):
        script = Path(row["current_path"]).name
        purpose = row["purpose"].replace("|", "\\|")
        lines.append(f"| `{row['id']}` | {row['status']} | `{script}` | {purpose} |")
    return "\n".join(lines)


def _write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content.rstrip() + "\n", encoding="utf-8")


def write_script_category_readmes(rows: list[dict[str, str]]) -> None:
    by_category: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        category = row["category"] or "archive"
        if category in {"weather", "physics", "labels", "validation", "pipelines"}:
            continue
        by_category[category].append(row)

    for category, (title, purpose) in SCRIPT_CATEGORY_META.items():
        items = by_category.get(category, [])
        version_notes = (
            "See `scripts/archive/VERSIONS.md` for superseded comparators and original application notes."
            if category == "archive"
            else "Formal package modules live under `src/s2s_rtist/`. See `scripts/archive/VERSIONS.md` when a higher-version script replaces an older one."
        )
        content = f"""# {title}

## Purpose
{purpose}

## Current Scripts
{_table(items)}

## Usage
`python project_cli.py show <id>`
`python project_cli.py run <id> -- <args>`

## Version Notes
{version_notes}
"""
        _write(ROOT / "scripts" / category / "README.md", content)


def write_scripts_readme(rows: list[dict[str, str]]) -> None:
    formal = [row for row in rows if row["status"] == "formal" and row["runnable"] == "true"]
    content = f"""# Scripts

## Purpose
Categorized research and formal runners for the S2S RTIST project. The only root-level Python entry point is `project_cli.py`.

## Categories
| Category | Path | Count |
|---|---|---|
| Data preparation | `scripts/data_preparation/` | {sum(1 for r in rows if r['category']=='data_preparation')} |
| Simulation | `scripts/simulation/` | {sum(1 for r in rows if r['category']=='simulation')} |
| Diagnostics | `scripts/diagnostics/` | {sum(1 for r in rows if r['category']=='diagnostics')} |
| Training | `scripts/training/` | {sum(1 for r in rows if r['category']=='training')} |
| Evaluation | `scripts/evaluation/` | {sum(1 for r in rows if r['category']=='evaluation')} |
| Visualization | `scripts/visualization/` | {sum(1 for r in rows if r['category']=='visualization')} |
| Archive | `scripts/archive/` | {sum(1 for r in rows if r['category']=='archive')} |

Reusable formal libraries live under `src/s2s_rtist/` and are also listed in `scripts/script_catalog.csv`.

## Formal Runnable IDs
{_table(formal)}

## Usage
`python project_cli.py list`
`python project_cli.py find <query>`
`python project_cli.py show <id>`
`python project_cli.py run <id> -- <args>`

## Catalog
All 140 original root scripts are tracked in `scripts/script_catalog.csv` with original path, current path, status, purpose, and SHA256 hashes.
"""
    _write(ROOT / "scripts" / "README.md", content)


def write_versions() -> None:
    content = """# Script Version Notes

This file records explicit replacement relationships. Do not infer a replacement solely from a higher version number.

## Confirmed Replacements

| Older script | Replaced by | Evidence |
|---|---|---|
| `compare_expanded_policy_results_v1.py` | `compare_expanded_policy_results_v3.py` | Locked migration rule: v3 is the active comparator and v1/v2 are archived as superseded. |
| `compare_expanded_policy_results_v2.py` | `compare_expanded_policy_results_v3.py` | Locked migration rule: v3 is the active comparator and v1/v2 are archived as superseded. |

## Historical Without Confirmed Replacement

| Script | Notes |
|---|---|
| `extract_weather_sequences_v1.py` | Coexists with `extract_weather_sequences_v2.py`. Keep both historical/active as cataloged; do not set `replaced_by` without import or result evidence. |
| `evaluate_learned_trigger_curve_policy_v1.py` | Coexists with `evaluate_learned_trigger_curve_policy_v2.py` and the expanded evaluator. Keep both until results/docs confirm a single replacement. |

## Original Application Entry Points

| Script | Status | Reason preserved |
|---|---|---|
| `Main_win.py` | historical | Original SWAP/RTIST application entry point retained for reference and reproducibility. |
| `Main_win_ensemble_mean.py` | historical | Original ensemble-mean application entry point retained for reference and reproducibility. |

Both live under `scripts/archive/original_application/` and are not formal package APIs.
"""
    _write(ROOT / "scripts" / "archive" / "VERSIONS.md", content)


def write_docs_readmes(doc_rows: list[dict[str, str]]) -> None:
    server = [row for row in doc_rows if "operations/server" in row["current_path"]]
    research = [row for row in doc_rows if "research/reproduction" in row["current_path"]]
    archive = [
        row
        for row in doc_rows
        if "docs/archive/" in row["current_path"] or row["status"] == "local_reference"
    ]

    def doc_table(rows: list[dict[str, str]]) -> str:
        lines = [
            "| ID | Status | Path | Purpose |",
            "|---|---|---|---|",
        ]
        for row in sorted(rows, key=lambda item: item["id"]):
            purpose = row["purpose"].replace("|", "\\|")
            lines.append(
                f"| `{row['id']}` | {row['status']} | `{row['current_path']}` | {purpose} |"
            )
        return "\n".join(lines)

    _write(
        ROOT / "docs" / "README.md",
        f"""# Documentation

## Purpose
Project guides, server run notes, research reproduction notes, and archived local extracts.

## Document Catalog
All nine original root Markdown/TXT files are tracked in `docs/document_catalog.csv`.

## Sections
| Section | Path |
|---|---|
| Server operations | `docs/operations/server/` |
| Research reproduction | `docs/research/reproduction/` |
| Archive | `docs/archive/` |
| Requirements | `requirements/` |

## Current Documents
{doc_table(doc_rows)}

## Usage
`python project_cli.py list --type docs`
`python project_cli.py find <query>`
`python project_cli.py show <id>`
""",
    )

    _write(
        ROOT / "docs" / "operations" / "server" / "README.md",
        f"""# Server Operations Guides

## Purpose
Formal server-run instructions for multi-site smoke and GEFS bias validation workflows.

## Current Documents
{doc_table(server)}

## Usage
`python project_cli.py show <id>`
`python project_cli.py run <related-script-id> -- <args>`
""",
    )

    _write(
        ROOT / "docs" / "research" / "reproduction" / "README.md",
        f"""# Research Reproduction Notes

## Purpose
Historical notes that record early reproduction steps and intermediate research decisions.

## Current Documents
{doc_table(research)}
""",
    )

    _write(
        ROOT / "docs" / "archive" / "README.md",
        f"""# Documentation Archive

## Purpose
Historical notes and local paper extracts that remain cataloged for search but are not formal operational guides.

## Current Documents
{doc_table(archive)}

## Local Paper Extracts
`docs/archive/paper_extracts/` is gitignored. Catalog rows retain paths and hashes with `status=local_reference`.
""",
    )


def main() -> int:
    script_rows = _load_script_rows()
    document_rows = _load_document_rows()
    write_scripts_readme(script_rows)
    write_script_category_readmes(script_rows)
    write_versions()
    write_docs_readmes(document_rows)
    print("wrote navigation README files")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
