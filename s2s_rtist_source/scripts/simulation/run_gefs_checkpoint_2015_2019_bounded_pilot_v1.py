#!/usr/bin/env python3
"""Run the bounded five-year checkpoint GEFS-SWAP label pilot."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from scripts.simulation.run_gefs_checkpoint_five_site_eight_ir_smoke_v1 import (
    FORMAL_WORKSPACE_COPIES,
    SITE_ORDER,
    SITE_TO_SOURCE_WORKSPACE,
    run_site,
    validate_source_workspace,
)
from scripts.simulation.run_gefs_checkpoint_one_date_eight_ir_smoke_v1 import (
    IRRIGATION_OPTIONS_MM,
    build_ensemble_mean_weather,
    sha256_file,
)


EXPECTED_YEARS = (2015, 2016, 2017, 2018, 2019)
MINIMUM_RESPONSIVE_SITE_CYCLES = 5
MINIMUM_YEARS_WITH_PROFITABLE_NONZERO_IRRIGATION = 2


def split_for_year(year: int) -> str:
    return "training_oof" if int(year) <= 2018 else "validation"


def selected_cycles(
    weather: pd.DataFrame,
    *,
    expected_years: tuple[int, ...] = EXPECTED_YEARS,
    expected_sites: tuple[str, ...] = SITE_ORDER,
) -> pd.DataFrame:
    required = {"decision_date", "site_id"}
    missing = required.difference(weather.columns)
    if missing:
        raise ValueError(f"all-variable weather is missing fields: {sorted(missing)}")
    data = weather.copy()
    data["decision_date"] = pd.to_datetime(data["decision_date"]).dt.strftime(
        "%Y-%m-%d"
    )
    data["target_year"] = pd.to_datetime(data["decision_date"]).dt.year.astype(int)
    years = tuple(sorted(data["target_year"].unique().tolist()))
    if years != expected_years:
        raise ValueError(
            f"bounded pilot requires years {expected_years}, got {years}"
        )
    sites = set(data["site_id"].astype(str))
    if sites != set(expected_sites):
        raise ValueError(
            f"bounded pilot requires sites {expected_sites}, got {sorted(sites)}"
        )

    rows: list[dict[str, Any]] = []
    for year in expected_years:
        dates = sorted(
            data.loc[data["target_year"].eq(year), "decision_date"].unique().tolist()
        )
        if len(dates) != 1:
            raise ValueError(f"year {year} must contain exactly one frozen cycle")
        decision_date = str(dates[0])
        for site in expected_sites:
            build_ensemble_mean_weather(
                data,
                site_id=site,
                decision_date=decision_date,
            )
        rows.append(
            {
                "target_year": int(year),
                "decision_date": decision_date,
                "split": split_for_year(year),
            }
        )
    return pd.DataFrame(rows)


def boolean_mask(values: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(values.dtype):
        return values.fillna(False)
    return values.astype(str).str.strip().str.lower().eq("true")


def normalize_candidates(candidates: pd.DataFrame) -> pd.DataFrame:
    result = candidates.copy()
    if result.empty:
        return result
    if "decision_date" not in result.columns:
        result["decision_date"] = pd.to_datetime(
            result["date_t"], errors="raise"
        ).dt.strftime("%Y-%m-%d")
    else:
        result["decision_date"] = pd.to_datetime(
            result["decision_date"], errors="raise"
        ).dt.strftime("%Y-%m-%d")
    result["target_year"] = pd.to_numeric(
        result["target_year"], errors="raise"
    ).astype(int)
    return result


def build_response_summary(candidates: pd.DataFrame) -> pd.DataFrame:
    if candidates.empty:
        return pd.DataFrame()
    data = normalize_candidates(candidates)
    best = data.loc[boolean_mask(data["is_best_ir"])].copy()
    ranges = (
        data.groupby(["target_year", "decision_date", "site"], as_index=False)[
            "cwdm_value"
        ]
        .agg(cwdm_min="min", cwdm_max="max")
    )
    ranges["cwdm_range"] = ranges["cwdm_max"] - ranges["cwdm_min"]
    best = best[
        ["target_year", "decision_date", "site", "ir", "net_gain_7d"]
    ].rename(columns={"ir": "best_ir_mm", "net_gain_7d": "best_net_gain_7d"})
    result = ranges.merge(
        best,
        on=["target_year", "decision_date", "site"],
        how="left",
        validate="one_to_one",
    )
    result["has_positive_cwdm_range"] = result["cwdm_range"].gt(1e-9)
    result["has_profitable_nonzero_best_irrigation"] = (
        result["best_ir_mm"].gt(0.0) & result["best_net_gain_7d"].gt(0.0)
    )
    return result.sort_values(["target_year", "site"]).reset_index(drop=True)


def build_bounded_audit(
    *,
    candidates: pd.DataFrame,
    site_audits: list[dict[str, Any]],
    cycles: pd.DataFrame,
    expected_sites: tuple[str, ...] = SITE_ORDER,
) -> dict[str, Any]:
    required_columns = {
        "site",
        "target_year",
        "date_t",
        "ir",
        "is_best_ir",
        "net_gain_7d",
        "cwdm_value",
    }
    schema_ok = required_columns.issubset(candidates.columns)
    data = normalize_candidates(candidates) if schema_ok else pd.DataFrame()
    expected_keys = {
        (int(row.target_year), str(row.decision_date), site)
        for row in cycles.itertuples(index=False)
        for site in expected_sites
    }
    actual_keys = (
        set(
            data[["target_year", "decision_date", "site"]]
            .drop_duplicates()
            .itertuples(index=False, name=None)
        )
        if not data.empty
        else set()
    )
    counts = (
        data.groupby(["target_year", "decision_date", "site"]).size().to_dict()
        if not data.empty
        else {}
    )
    duplicate_count = (
        int(
            data[["target_year", "decision_date", "site", "ir"]]
            .duplicated()
            .sum()
        )
        if not data.empty
        else 0
    )
    irrigation_ok = bool(schema_ok) and all(
        sorted(
            data.loc[
                data["target_year"].eq(year)
                & data["decision_date"].eq(decision)
                & data["site"].astype(str).eq(site),
                "ir",
            ]
            .astype(float)
            .tolist()
        )
        == IRRIGATION_OPTIONS_MM
        for year, decision, site in expected_keys
    )

    audit_keys = {
        (int(row.get("target_year", -1)), str(row.get("decision_date", "")), str(row.get("site_id", "")))
        for row in site_audits
        if bool(row.get("mandatory_gate_passed", False))
    }
    maximum_crop_error = max(
        [float(row.get("maximum_absolute_checkpoint_crop_state_error", np.inf)) for row in site_audits],
        default=float("inf"),
    )
    maximum_profile_error = max(
        [float(row.get("maximum_absolute_checkpoint_profile_state_error", np.inf)) for row in site_audits],
        default=float("inf"),
    )
    maximum_rain_error = max(
        [float(row.get("maximum_absolute_swap_rain_error_mm", np.inf)) for row in site_audits],
        default=float("inf"),
    )
    maximum_residual = max(
        [float(row.get("maximum_absolute_water_balance_residual_mm", np.inf)) for row in site_audits],
        default=float("inf"),
    )
    missing_primary = sum(
        int(row.get("primary_output_missing_value_count", 0)) for row in site_audits
    )
    prestate_reruns = sum(
        int(row.get("prestate_swap_rerun_count", 0)) for row in site_audits
    )

    best = data.loc[boolean_mask(data["is_best_ir"])].copy() if not data.empty else pd.DataFrame()
    best_counts = (
        best.groupby(["target_year", "decision_date", "site"]).size().to_dict()
        if not best.empty
        else {}
    )
    response = build_response_summary(data) if not data.empty else pd.DataFrame()
    responsive_count = (
        int(response["has_positive_cwdm_range"].sum()) if not response.empty else 0
    )
    profitable_count = (
        int(response["has_profitable_nonzero_best_irrigation"].sum())
        if not response.empty
        else 0
    )
    profitable_years = (
        sorted(
            response.loc[
                response["has_profitable_nonzero_best_irrigation"], "target_year"
            ]
            .astype(int)
            .unique()
            .tolist()
        )
        if not response.empty
        else []
    )

    expected_rows = len(expected_keys) * len(IRRIGATION_OPTIONS_MM)
    mandatory_passed = all(
        [
            schema_ok,
            actual_keys == expected_keys,
            len(data) == expected_rows,
            all(int(counts.get(key, 0)) == 8 for key in expected_keys),
            duplicate_count == 0,
            irrigation_ok,
            audit_keys == expected_keys,
            len(best) == len(expected_keys),
            all(int(best_counts.get(key, 0)) == 1 for key in expected_keys),
            maximum_crop_error <= 1e-6,
            maximum_profile_error <= 1e-6,
            maximum_rain_error <= 0.01,
            maximum_residual <= 0.5,
            missing_primary == 0,
            prestate_reruns == 0,
        ]
    )
    response_passed = all(
        [
            responsive_count >= MINIMUM_RESPONSIVE_SITE_CYCLES,
            len(profitable_years)
            >= MINIMUM_YEARS_WITH_PROFITABLE_NONZERO_IRRIGATION,
        ]
    )
    passed = mandatory_passed and response_passed
    if passed:
        status = "verified_checkpoint_2015_2019_bounded_pilot_passed"
        next_gate = "design_seasonal_date_density_without_training"
    elif mandatory_passed:
        status = "verified_checkpoint_2015_2019_bounded_pilot_response_coverage_failed"
        next_gate = "increase_bounded_cycle_coverage_without_training"
    else:
        status = "verified_checkpoint_2015_2019_bounded_pilot_mandatory_gate_failed"
        next_gate = "repair_checkpoint_bounded_pilot"
    return {
        "status": status,
        "mandatory_gate_passed": mandatory_passed,
        "response_coverage_gate_passed": response_passed,
        "bounded_pilot_gate_passed": passed,
        "expected_years": list(EXPECTED_YEARS),
        "selected_cycles": cycles.to_dict(orient="records"),
        "cycle_count": int(len(cycles)),
        "site_count": int(len(expected_sites)),
        "site_cycle_count": int(len(actual_keys)),
        "passed_site_cycle_count": int(len(audit_keys)),
        "candidate_rows": int(len(data)),
        "expected_candidate_rows": int(expected_rows),
        "candidate_rows_by_year": {
            str(year): int((data["target_year"] == year).sum()) if not data.empty else 0
            for year in EXPECTED_YEARS
        },
        "candidate_rows_by_site": {
            site: int(data["site"].astype(str).eq(site).sum()) if not data.empty else 0
            for site in expected_sites
        },
        "duplicate_candidate_key_count": duplicate_count,
        "best_row_count": int(len(best)),
        "responsive_site_cycle_count": responsive_count,
        "profitable_nonzero_site_cycle_count": profitable_count,
        "years_with_profitable_nonzero_irrigation": profitable_years,
        "minimum_responsive_site_cycles": MINIMUM_RESPONSIVE_SITE_CYCLES,
        "minimum_years_with_profitable_nonzero_irrigation": MINIMUM_YEARS_WITH_PROFITABLE_NONZERO_IRRIGATION,
        "maximum_absolute_checkpoint_crop_state_error": maximum_crop_error,
        "maximum_absolute_checkpoint_profile_state_error": maximum_profile_error,
        "maximum_absolute_swap_rain_error_mm": maximum_rain_error,
        "maximum_absolute_water_balance_residual_mm": maximum_residual,
        "primary_output_missing_value_count": missing_primary,
        "prestate_swap_rerun_count": prestate_reruns,
        "weather_driver_source": "frozen_corrected_GEFS_5member_ensemble_mean",
        "weather_label_scenario_consistent": True,
        "bounded_label_generation_performed": True,
        "full_dataset_generation_performed": False,
        "training_eligible": False,
        "surrogate_training_performed": False,
        "tta_performed": False,
        "next_gate": next_gate,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-workspace-root", type=Path, required=True)
    parser.add_argument("--all-variable-weather", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--sites", nargs="+", default=list(SITE_ORDER))
    parser.add_argument("--sowing-month-day", default="04-26")
    parser.add_argument("--harvest-month-day", default="10-10")
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def run(args: argparse.Namespace) -> dict[str, Path]:
    sites = tuple(args.sites)
    if sites != SITE_ORDER:
        raise ValueError(f"formal bounded pilot requires sites in order: {SITE_ORDER}")
    weather = pd.read_csv(args.all_variable_weather)
    cycles = selected_cycles(weather, expected_sites=sites)
    for year in EXPECTED_YEARS:
        for site in sites:
            validate_source_workspace(
                args.source_workspace_root / SITE_TO_SOURCE_WORKSPACE[site],
                year=year,
            )
    for source, _ in FORMAL_WORKSPACE_COPIES:
        if not source.is_file():
            raise FileNotFoundError(f"missing formal dependency: {source}")
    if args.output_dir.exists() and not args.resume:
        raise FileExistsError(f"refusing to overwrite output directory: {args.output_dir}")
    args.output_dir.mkdir(parents=True, exist_ok=args.resume)

    candidates_frames: list[pd.DataFrame] = []
    site_audits: list[dict[str, Any]] = []
    summaries: list[dict[str, Any]] = []
    for cycle in cycles.itertuples(index=False):
        year_root = args.output_dir / str(cycle.target_year)
        year_root.mkdir(exist_ok=args.resume)
        for site in sites:
            print(
                f"[{cycle.target_year}/{site}] starting trunk, checkpoint, and eight branches",
                flush=True,
            )
            try:
                candidates, audit, summary = run_site(
                    site_id=site,
                    source_workspace_root=args.source_workspace_root,
                    all_variable_weather=args.all_variable_weather,
                    output_dir=year_root,
                    year=int(cycle.target_year),
                    decision_date=str(cycle.decision_date),
                    sowing_month_day=args.sowing_month_day,
                    harvest_month_day=args.harvest_month_day,
                    resume=args.resume,
                )
                candidates_frames.append(candidates)
                site_audits.append(audit)
                summaries.append(
                    {
                        "target_year": int(cycle.target_year),
                        "decision_date": str(cycle.decision_date),
                        **summary,
                    }
                )
            except Exception as exc:
                summaries.append(
                    {
                        "target_year": int(cycle.target_year),
                        "decision_date": str(cycle.decision_date),
                        "site_id": site,
                        "status": "failed",
                        "candidate_rows": 0,
                        "site_root": str(year_root / site),
                        "error": f"{type(exc).__name__}: {exc}",
                    }
                )
                print(
                    f"[{cycle.target_year}/{site}] failed: {type(exc).__name__}: {exc}",
                    flush=True,
                )

    candidates = (
        pd.concat(candidates_frames, ignore_index=True)
        if candidates_frames
        else pd.DataFrame()
    )
    if not candidates.empty:
        candidates = normalize_candidates(candidates).sort_values(
            ["target_year", "site", "decision_date", "ir"]
        ).reset_index(drop=True)
    summary = pd.DataFrame(summaries)
    response = build_response_summary(candidates)
    audit = build_bounded_audit(
        candidates=candidates,
        site_audits=site_audits,
        cycles=cycles,
        expected_sites=sites,
    )
    best = (
        candidates.loc[boolean_mask(candidates["is_best_ir"])].copy()
        if not candidates.empty
        else pd.DataFrame()
    )
    outputs = {
        "candidates": args.output_dir / "gefs_checkpoint_2015_2019_bounded_candidates_v1.csv",
        "best": args.output_dir / "gefs_checkpoint_2015_2019_bounded_best_v1.csv",
        "response": args.output_dir / "gefs_checkpoint_2015_2019_response_summary_v1.csv",
        "summary": args.output_dir / "gefs_checkpoint_2015_2019_run_summary_v1.csv",
        "cycles": args.output_dir / "gefs_checkpoint_2015_2019_selected_cycles_v1.csv",
        "audit": args.output_dir / "gefs_checkpoint_2015_2019_bounded_audit_v1.json",
        "manifest": args.output_dir / "gefs_checkpoint_2015_2019_bounded_manifest_v1.json",
    }
    candidates.to_csv(outputs["candidates"], index=False)
    best.to_csv(outputs["best"], index=False)
    response.to_csv(outputs["response"], index=False)
    summary.to_csv(outputs["summary"], index=False)
    cycles.to_csv(outputs["cycles"], index=False)
    outputs["audit"].write_text(
        json.dumps(audit, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    manifest = {
        "status": audit["status"],
        "inputs": {
            "all_variable_weather_sha256": sha256_file(args.all_variable_weather),
            "source_workspace_root": str(args.source_workspace_root),
        },
        "outputs": {
            key: {"path": path.name, "sha256": sha256_file(path)}
            for key, path in outputs.items()
            if key != "manifest"
        },
        "network_download_performed": False,
        "full_dataset_generation_performed": False,
        "surrogate_training_performed": False,
        "tta_performed": False,
    }
    outputs["manifest"].write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    if not audit["bounded_pilot_gate_passed"]:
        raise RuntimeError(f"bounded checkpoint pilot failed; see {outputs['audit']}")
    return outputs


if __name__ == "__main__":
    generated = run(parse_args())
    print(json.dumps({key: str(value) for key, value in generated.items()}, indent=2))
