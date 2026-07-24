#!/usr/bin/env python3
"""Fit and freeze the selected weekly GEFS precipitation correction artifact."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import pandas as pd

from scripts.diagnostics.run_gefs_qdm_volume_preserving_2019_validation_v1 import (
    load_2019_target,
)
from scripts.diagnostics.run_gefs_qm_qdm_expanding_cv_2000_2018_v1 import (
    DEFAULT_MEMBER_2015_2019,
    DEFAULT_PAIRED_2000_2002,
    DEFAULT_PAIRED_2003_2014,
    DEFAULT_REFERENCE_2000_2019,
    load_inputs,
)
from scripts.diagnostics.run_gefs_weekly_two_stage_linear_scaling_cv_v1 import (
    fit_two_stage_factors,
    weekly_cycle_table,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONTRACT_PATH = (
    PROJECT_ROOT
    / "site_general_surrogate_eval"
    / "gefs_weekly_linear_final_fit_contract_v1.json"
)
DEFAULT_SELECTION_GATE = (
    PROJECT_ROOT
    / "site_general_surrogate_eval"
    / "gefs_qdm_station_reference_v1"
    / "gefs_weekly_linear_factor_shrinkage_selection_server_v1"
    / "weekly_factor_shrinkage_selection_gate_v1.json"
)
DEFAULT_OUTPUT_DIR = (
    PROJECT_ROOT
    / "site_general_surrogate_eval"
    / "gefs_qdm_station_reference_v1"
    / "gefs_weekly_linear_final_artifact_v1"
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def artifact_hash(payload: dict[str, Any]) -> str:
    body = {key: value for key, value in payload.items() if key != "artifact_sha256"}
    canonical = json.dumps(
        body, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


def load_contract(path: Path) -> dict[str, Any]:
    contract = json.loads(path.read_text(encoding="utf-8"))
    if contract.get("contract_id") != "gefs-weekly-linear-final-fit-v1":
        raise ValueError("final weekly linear fit contract id mismatch")
    if float(contract.get("factor_shrinkage_alpha")) != 0.75:
        raise ValueError("final factor shrinkage alpha must be 0.75")
    if contract.get("group_keys") != ["site_id"]:
        raise ValueError("final correction must be grouped by site only")
    if contract["scope"]["use_2024_for_fit"] or contract["scope"]["use_2024_for_selection"]:
        raise ValueError("2024 must be prohibited from fitting and selection")
    return contract


def validate_selection_gate(gate: dict[str, Any], contract: dict[str, Any]) -> None:
    required = contract["required_selection_gate"]
    for key, expected in required.items():
        if gate.get(key) != expected:
            raise ValueError(f"selection gate mismatch for {key}: {gate.get(key)!r}")


def load_history_2000_2019(args: argparse.Namespace) -> pd.DataFrame:
    history = load_inputs(args)
    target_2019 = load_2019_target(args)
    target_2019["reference_valid_unflagged"] = target_2019[
        "precipitation_mm_reference"
    ].notna()
    keep = [
        "site_id",
        "decision_date",
        "valid_date_utc",
        "gefs_member",
        "precipitation_mm_raw",
        "precipitation_mm_reference",
        "reference_valid_unflagged",
    ]
    combined = pd.concat([history[keep], target_2019[keep]], ignore_index=True)
    combined["decision_date"] = pd.to_datetime(combined["decision_date"])
    combined["valid_date_utc"] = pd.to_datetime(combined["valid_date_utc"])
    years = set(combined["decision_date"].dt.year.astype(int))
    if years != set(range(2000, 2020)):
        raise ValueError(f"final fit years mismatch: {sorted(years)}")
    key = ["site_id", "decision_date", "valid_date_utc", "gefs_member"]
    if combined.duplicated(key).any():
        raise ValueError("duplicate final-fit member key")
    expected_rows = 20 * 6 * 5 * 5 * 7
    if len(combined) != expected_rows:
        raise ValueError(f"final fit rows={len(combined)}, expected={expected_rows}")
    return combined.sort_values(key).reset_index(drop=True)


def build_artifact(
    *,
    factors: pd.DataFrame,
    contract: dict[str, Any],
    selection_gate_sha256: str,
    input_hashes: dict[str, str],
) -> dict[str, Any]:
    alpha = float(contract["factor_shrinkage_alpha"])
    groups = []
    for row in factors.sort_values("site_id").to_dict(orient="records"):
        base_overall = float(row["overall_factor"])
        base_extreme = float(row["final_extreme_factor"])
        groups.append(
            {
                "site_id": str(row["site_id"]),
                "fit_complete_cycle_count": int(row["fit_complete_cycle_count"]),
                "fit_extreme_cycle_count": int(row["fit_extreme_cycle_count"]),
                "extreme_quantile": float(row["extreme_quantile"]),
                "raw_ensemble_mean_7d_q90_mm": float(
                    row["raw_ensemble_mean_7d_q90_mm"]
                ),
                "base_extreme_factor": float(row["extreme_factor"]),
                "base_overall_factor": base_overall,
                "base_final_extreme_factor": base_extreme,
                "effective_overall_factor": 1.0 + alpha * (base_overall - 1.0),
                "effective_extreme_factor": 1.0 + alpha * (base_extreme - 1.0),
            }
        )
    artifact: dict[str, Any] = {
        "artifact_contract_id": contract["contract_id"],
        "artifact_contract_version": contract["contract_version"],
        "candidate_id": contract["candidate_id"],
        "base_candidate_id": contract["base_candidate_id"],
        "group_keys": contract["group_keys"],
        "factor_shrinkage_alpha": alpha,
        "fit_years": contract["fit_years"],
        "selection_gate_sha256": selection_gate_sha256,
        "input_file_sha256": input_hashes,
        "2024_used_for_fit_or_selection": False,
        "application_rule": (
            "if ensemble_mean_raw_7d_mm > site_q90 use effective_extreme_factor "
            "else use effective_overall_factor"
        ),
        "groups": groups,
    }
    artifact["artifact_sha256"] = artifact_hash(artifact)
    return artifact


def run(args: argparse.Namespace) -> dict[str, Path]:
    contract = load_contract(args.contract)
    selection_gate = json.loads(args.selection_gate.read_text(encoding="utf-8"))
    validate_selection_gate(selection_gate, contract)
    history = load_history_2000_2019(args)
    if history["decision_date"].dt.year.eq(2024).any():
        raise ValueError("2024 entered final fitting data")
    cycles = weekly_cycle_table(history, require_reference=True)
    factors = fit_two_stage_factors(
        cycles,
        group_keys=("site_id",),
        extreme_quantile=float(contract["extreme_quantile"]),
    )
    if set(factors["site_id"].astype(str)) != set(contract["expected_sites"]):
        raise ValueError("final fit site set mismatch")
    input_paths = {
        "paired_2000_2002": args.paired_2000_2002,
        "paired_2003_2014": args.paired_2003_2014,
        "member_2015_2019": args.member_2015_2019,
        "reference_2000_2019": args.reference_2000_2019,
    }
    artifact = build_artifact(
        factors=factors,
        contract=contract,
        selection_gate_sha256=sha256_file(args.selection_gate),
        input_hashes={key: sha256_file(path) for key, path in input_paths.items()},
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    artifact_path = args.output_dir / "gefs_weekly_linear_final_artifact_2000_2019_v1.json"
    summary_path = args.output_dir / "gefs_weekly_linear_final_fit_summary_v1.csv"
    manifest_path = args.output_dir / "gefs_weekly_linear_final_fit_manifest_v1.json"
    artifact_path.write_text(
        json.dumps(artifact, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    pd.DataFrame(artifact["groups"]).to_csv(
        summary_path, index=False, encoding="utf-8-sig"
    )
    manifest = {
        "contract_id": contract["contract_id"],
        "candidate_id": contract["candidate_id"],
        "factor_shrinkage_alpha": contract["factor_shrinkage_alpha"],
        "fit_years": contract["fit_years"],
        "fit_member_rows": int(len(history)),
        "fit_complete_site_cycle_count": int(len(cycles)),
        "artifact_sha256": artifact["artifact_sha256"],
        "artifact_file_sha256": sha256_file(artifact_path),
        "selection_gate_sha256": artifact["selection_gate_sha256"],
        "2024_used_for_fit_or_selection": False,
        "status": "final_artifact_frozen_ready_for_2024_application",
    }
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    paths = {"artifact": artifact_path, "summary": summary_path, "manifest": manifest_path}
    print(json.dumps({key: str(value) for key, value in paths.items()}, indent=2))
    return paths


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--contract", type=Path, default=CONTRACT_PATH)
    parser.add_argument("--selection-gate", type=Path, default=DEFAULT_SELECTION_GATE)
    parser.add_argument("--paired-2000-2002", type=Path, default=DEFAULT_PAIRED_2000_2002)
    parser.add_argument("--paired-2003-2014", type=Path, default=DEFAULT_PAIRED_2003_2014)
    parser.add_argument("--member-2015-2019", type=Path, default=DEFAULT_MEMBER_2015_2019)
    parser.add_argument("--reference-2000-2019", type=Path, default=DEFAULT_REFERENCE_2000_2019)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def main() -> None:
    run(parse_args())


if __name__ == "__main__":
    main()
