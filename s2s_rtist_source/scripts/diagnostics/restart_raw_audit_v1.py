"""Preserve a bounded raw SWAP audit subset during restart generation."""

from __future__ import annotations

import json
import math
import shutil
from datetime import datetime
from pathlib import Path
from typing import Sequence


FORMAL_NPRINTDAY = 24
RAW_AUDIT_FILES = (
    "result_restart.inc",
    "result_restart.vap",
    "result_restart.crp",
    "result_restart.wba",
    "result_restart.dwb",
    "result_restart.end",
    "restart_initial.end",
    "swap.swp",
)


def _safe_label(text: str) -> str:
    return text.replace("-", "").lower()


def _safe_ir_label(irrigation_mm: float) -> str:
    return f"{float(irrigation_mm):g}".replace("-", "m").replace(".", "p")


def should_preserve_raw_candidate(
    irrigation_mm: float,
    irrigation_options_mm: Sequence[float],
) -> bool:
    options = [float(value) for value in irrigation_options_mm]
    if not options:
        raise ValueError("irrigation_options_mm cannot be empty")
    value = float(irrigation_mm)
    return math.isclose(value, 0.0, abs_tol=1.0e-9) or math.isclose(
        value,
        max(options),
        abs_tol=1.0e-9,
    )


def preserve_candidate_raw_outputs(
    *,
    date_t: str,
    decision_doy: int,
    irrigation_mm: float,
    irrigation_options_mm: Sequence[float],
    selection_irrigation_mm: float | None = None,
    source_dir: Path = Path("."),
    audit_root: Path = Path("candidate_raw_audit"),
    nprintday: int = FORMAL_NPRINTDAY,
) -> Path | None:
    selection_value = (
        float(irrigation_mm)
        if selection_irrigation_mm is None
        else float(selection_irrigation_mm)
    )
    if not should_preserve_raw_candidate(selection_value, irrigation_options_mm):
        return None

    target = (
        audit_root
        / str(datetime.strptime(date_t, "%d-%b-%Y").year)
        / _safe_label(date_t)
        / f"ir_{_safe_ir_label(selection_value)}mm"
    )
    target.mkdir(parents=True, exist_ok=True)

    copied_files = []
    for name in RAW_AUDIT_FILES:
        source = source_dir / name
        if source.exists():
            shutil.copy2(source, target / name)
            copied_files.append(name)

    required = {"result_restart.inc", "result_restart.vap", "result_restart.crp"}
    missing = sorted(required.difference(copied_files))
    if missing:
        raise FileNotFoundError(
            "raw audit is missing required SWAP outputs: " + ", ".join(missing)
        )

    manifest = {
        "date_t": date_t,
        "decision_doy": int(decision_doy),
        "irrigation_mm": float(irrigation_mm),
        "requested_ir_mm": selection_value,
        "simulated_ir_mm": float(irrigation_mm),
        "numerical_irrigation_fallback": not math.isclose(
            selection_value, float(irrigation_mm), abs_tol=1.0e-9
        ),
        "numerical_endpoint_fallback": not math.isclose(
            selection_value, float(irrigation_mm), abs_tol=1.0e-9
        ),
        "irrigation_options_mm": [float(value) for value in irrigation_options_mm],
        "selection_rule": "zero_and_maximum_irrigation_endpoints",
        "nprintday": int(nprintday),
        "files": copied_files,
    }
    (target / "raw_audit_manifest.json").write_text(
        json.dumps(manifest, indent=2),
        encoding="utf-8",
    )
    return target
