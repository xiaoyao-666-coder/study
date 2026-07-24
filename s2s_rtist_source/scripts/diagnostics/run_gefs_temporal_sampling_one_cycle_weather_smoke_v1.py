#!/usr/bin/env python3
"""Compare full and conservative GEFS daily weather from one minimal training cycle."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from scripts.data_preparation.audit_gefs_exact_schedule_temporal_sampling_budget_v1 import (
    CONSERVATIVE_CANDIDATE,
    THINNABLE_STATE_PRODUCTS,
)
from scripts.data_preparation.extract_gefs_2015_2019_full_weather_pilot_v1 import (
    run_extraction,
)
from scripts.data_preparation.preflight_gefs_2015_2019_exact_schedule_full_weather_v1 import (
    sha256_file,
    write_json,
)
from s2s_rtist.weather.gefs_quantile_mapping import GEFS_REFORECAST_MEMBERS
from s2s_rtist.weather.gefs_reforecast_full_weather import (
    CANONICAL_WEATHER_COLUMNS,
    REQUIRED_PRODUCT_SPECS,
    _product_cache_paths,
    aggregate_member_weather,
)


EXPECTED_CYCLE_COUNT = 239
MAXIMUM_END_HOUR = 174
EXACT_MATCH_VARIABLES = (
    "precipitation_mm_raw",
    "temperature_min_c",
    "temperature_max_c",
    "solar_kj_m2_day",
)
COMPARISON_KEYS = (
    "decision_date",
    "site_id",
    "gefs_member",
    "local_date",
    "lead_day",
)


def select_smoke_cycle(cycle_plan: pd.DataFrame) -> pd.Series:
    required = {
        "target_year",
        "decision_date",
        "required_site_count",
        "required_sites",
        "expected_output_rows",
        "selected_range_bytes",
    }
    missing = required.difference(cycle_plan.columns)
    if missing:
        raise ValueError(f"batched cycle plan missing fields: {sorted(missing)}")
    plan = cycle_plan.copy()
    plan["target_year"] = pd.to_numeric(plan["target_year"], errors="raise").astype(int)
    plan["required_site_count"] = pd.to_numeric(
        plan["required_site_count"], errors="raise"
    ).astype(int)
    plan["expected_output_rows"] = pd.to_numeric(
        plan["expected_output_rows"], errors="raise"
    ).astype(int)
    plan["selected_range_bytes"] = pd.to_numeric(
        plan["selected_range_bytes"], errors="raise"
    ).astype(int)
    plan["decision_date"] = pd.to_datetime(
        plan["decision_date"], errors="raise"
    ).dt.strftime("%Y-%m-%d")
    if len(plan) != EXPECTED_CYCLE_COUNT or plan["decision_date"].nunique() != EXPECTED_CYCLE_COUNT:
        raise ValueError("batched cycle plan must contain 239 unique cycles")
    training = plan.loc[plan["target_year"].between(2015, 2018)].copy()
    if training.empty:
        raise ValueError("batched cycle plan contains no 2015-2018 training cycles")
    max_site_count = int(training["required_site_count"].max())
    candidates = training.loc[training["required_site_count"].eq(max_site_count)]
    chosen = candidates.sort_values(
        ["selected_range_bytes", "decision_date"], ascending=[True, True]
    ).iloc[0]
    sites = [item for item in str(chosen["required_sites"]).split(",") if item]
    if len(sites) != int(chosen["required_site_count"]):
        raise ValueError("chosen cycle required-site list does not match its count")
    if int(chosen["expected_output_rows"]) != len(sites) * len(GEFS_REFORECAST_MEMBERS) * 7:
        raise ValueError("chosen cycle expected weather row count mismatch")
    return chosen


def seed_preflight_cache(
    *,
    source_cache: Path,
    target_cache: Path,
    cycle_date: str,
    members: tuple[str, ...],
) -> int:
    copied = 0
    cycle = pd.Timestamp(cycle_date).strftime("%Y%m%d")
    for member in members:
        source_inventory = source_cache / "inventories" / f"{cycle}_{member}.xml"
        target_inventory = target_cache / "inventories" / source_inventory.name
        if not source_inventory.is_file():
            raise FileNotFoundError(f"cached inventory is missing: {source_inventory}")
        target_inventory.parent.mkdir(parents=True, exist_ok=True)
        if not target_inventory.exists():
            shutil.copy2(source_inventory, target_inventory)
            copied += 1
        for spec in REQUIRED_PRODUCT_SPECS:
            source_index = _product_cache_paths(
                cycle_date=cycle_date,
                member=member,
                product_id=spec.product_id,
                cache_dir=source_cache,
                maximum_end_hour=MAXIMUM_END_HOUR,
            )["index"]
            target_index = _product_cache_paths(
                cycle_date=cycle_date,
                member=member,
                product_id=spec.product_id,
                cache_dir=target_cache,
                maximum_end_hour=MAXIMUM_END_HOUR,
            )["index"]
            if not source_index.is_file():
                raise FileNotFoundError(f"cached product index is missing: {source_index}")
            target_index.parent.mkdir(parents=True, exist_ok=True)
            if not target_index.exists():
                shutil.copy2(source_index, target_index)
                copied += 1
    return copied


def build_conservative_weather(
    *,
    cache_dir: Path,
    cycle_date: str,
    members: tuple[str, ...],
) -> pd.DataFrame:
    weather_parts: list[pd.DataFrame] = []
    for member in members:
        point_parts: list[pd.DataFrame] = []
        metadata_rows: list[dict[str, Any]] = []
        for spec in REQUIRED_PRODUCT_SPECS:
            paths = _product_cache_paths(
                cycle_date=cycle_date,
                member=member,
                product_id=spec.product_id,
                cache_dir=cache_dir,
                maximum_end_hour=MAXIMUM_END_HOUR,
            )
            if not paths["points"].is_file() or not paths["metadata"].is_file():
                raise FileNotFoundError(
                    f"decoded point cache is incomplete for {cycle_date}/{member}/{spec.product_id}"
                )
            points = pd.read_csv(paths["points"], parse_dates=["cycle_init_utc"])
            if spec.product_id in THINNABLE_STATE_PRODUCTS:
                points = points.loc[
                    pd.to_numeric(points["end_hour"], errors="raise").astype(int).mod(6).eq(0)
                ].copy()
            point_parts.append(points)
            metadata_rows.append(json.loads(paths["metadata"].read_text(encoding="utf-8")))
        weather_parts.append(
            aggregate_member_weather(
                pd.concat(point_parts, ignore_index=True),
                member=member,
                product_manifest=pd.DataFrame(metadata_rows),
            )
        )
    return pd.concat(weather_parts, ignore_index=True).sort_values(
        list(COMPARISON_KEYS)
    ).reset_index(drop=True)


def compare_weather(full: pd.DataFrame, lean: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    left = full.copy()
    right = lean.copy()
    for frame in (left, right):
        frame["decision_date"] = pd.to_datetime(frame["decision_date"]).dt.strftime("%Y-%m-%d")
        frame["local_date"] = pd.to_datetime(frame["local_date"]).dt.strftime("%Y-%m-%d")
    merged = left[list(COMPARISON_KEYS) + list(CANONICAL_WEATHER_COLUMNS)].merge(
        right[list(COMPARISON_KEYS) + list(CANONICAL_WEATHER_COLUMNS)],
        on=list(COMPARISON_KEYS),
        how="outer",
        validate="one_to_one",
        indicator=True,
        suffixes=("_full", "_lean"),
    )
    if not merged["_merge"].eq("both").all():
        raise ValueError("full and lean weather keys differ")
    metric_rows: list[dict[str, Any]] = []
    for variable in CANONICAL_WEATHER_COLUMNS:
        full_values = pd.to_numeric(merged[f"{variable}_full"], errors="raise").to_numpy(float)
        lean_values = pd.to_numeric(merged[f"{variable}_lean"], errors="raise").to_numpy(float)
        difference = lean_values - full_values
        if not np.isfinite(full_values).all() or not np.isfinite(lean_values).all():
            raise ValueError(f"nonfinite values found for {variable}")
        merged[f"{variable}_difference"] = difference
        metric_rows.append(
            {
                "variable": variable,
                "row_count": len(difference),
                "mean_signed_error": float(np.mean(difference)),
                "mean_absolute_error": float(np.mean(np.abs(difference))),
                "root_mean_squared_error": float(np.sqrt(np.mean(difference**2))),
                "maximum_absolute_error": float(np.max(np.abs(difference))),
                "exact_match": bool(np.array_equal(full_values, lean_values)),
            }
        )
    return merged.drop(columns="_merge"), pd.DataFrame(metric_rows)


def build_audit(
    *,
    chosen: pd.Series,
    full_weather: pd.DataFrame,
    lean_weather: pd.DataFrame,
    metrics: pd.DataFrame,
    extraction_audit: dict[str, Any],
) -> dict[str, Any]:
    metric_map = metrics.set_index("variable")
    exact_variables_passed = all(
        bool(metric_map.loc[variable, "exact_match"]) for variable in EXACT_MATCH_VARIABLES
    )
    expected_rows = int(chosen["expected_output_rows"])
    structural_passed = all(
        [
            len(full_weather) == expected_rows,
            len(lean_weather) == expected_rows,
            exact_variables_passed,
            extraction_audit.get("status") == "full_weather_local_extraction_passed",
            extraction_audit.get("retained_grib_file_count") == 0,
        ]
    )
    return {
        "status": (
            "full_vs_conservative_temporal_sampling_weather_smoke_completed"
            if structural_passed
            else "full_vs_conservative_temporal_sampling_weather_smoke_failed"
        ),
        "mandatory_structural_gate_passed": structural_passed,
        "candidate_id": CONSERVATIVE_CANDIDATE,
        "selection_scope": "2015-2018_training_cycles_only",
        "cycle_selection_rule": "maximum_required_site_count_then_minimum_selected_range_bytes",
        "target_year": int(chosen["target_year"]),
        "decision_date": str(chosen["decision_date"]),
        "required_sites": str(chosen["required_sites"]).split(","),
        "member_count": len(GEFS_REFORECAST_MEMBERS),
        "lead_day_count": 7,
        "weather_row_count": len(full_weather),
        "estimated_full_payload_bytes": int(chosen["selected_range_bytes"]),
        "actual_extraction_network_bytes_this_run": int(
            extraction_audit.get("network_bytes_this_run", 0)
        ),
        "same_full_point_records_used_for_lean_reaggregation": True,
        "second_lean_payload_download_performed": False,
        "all_required_weather_variables_retained": True,
        "precipitation_temperature_and_solar_exact_match": exact_variables_passed,
        "weather_difference_metrics_computed": True,
        "weather_equivalence_approved": False,
        "teacher_review_required": True,
        "correction_applied": False,
        "swap_simulation_performed": False,
        "label_generation_performed": False,
        "surrogate_training_performed": False,
        "training_eligible": False,
        "tta_performed": False,
        "next_gate": (
            "teacher_review_full_vs_lean_raw_daily_weather_before_policy_change"
            if structural_passed
            else "repair_one_cycle_weather_smoke"
        ),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cycle-plan", type=Path, required=True)
    parser.add_argument("--temporal-audit", type=Path, required=True)
    parser.add_argument("--preflight-cache-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--timeout", type=int, default=120)
    parser.add_argument("--retries", type=int, default=4)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def run(args: argparse.Namespace) -> dict[str, Path]:
    for path in (args.cycle_plan, args.temporal_audit):
        if not path.is_file():
            raise FileNotFoundError(f"required input is missing: {path}")
    if not args.preflight_cache_dir.is_dir():
        raise FileNotFoundError(f"preflight cache is missing: {args.preflight_cache_dir}")
    if args.output_dir.exists() and not args.resume:
        raise FileExistsError(f"refusing to overwrite output directory: {args.output_dir}")
    temporal_audit = json.loads(args.temporal_audit.read_text(encoding="utf-8"))
    if temporal_audit.get("status") != "exact_schedule_temporal_sampling_metadata_audit_passed":
        raise ValueError("temporal-sampling metadata audit has not passed")
    if temporal_audit.get("product_payload_download_started") is not False:
        raise ValueError("temporal-sampling audit payload state is not clean")
    chosen = select_smoke_cycle(pd.read_csv(args.cycle_plan))
    cycle_date = str(chosen["decision_date"])
    sites = tuple(str(chosen["required_sites"]).split(","))
    members = tuple(GEFS_REFORECAST_MEMBERS)
    args.output_dir.mkdir(parents=True, exist_ok=args.resume)
    extraction_dir = args.output_dir / "full_extraction"
    copied_metadata_files = seed_preflight_cache(
        source_cache=args.preflight_cache_dir,
        target_cache=extraction_dir / "cache",
        cycle_date=cycle_date,
        members=members,
    )
    extraction_outputs = run_extraction(
        cycles=(cycle_date,),
        site_ids=sites,
        members=members,
        output_dir=extraction_dir,
        timeout=args.timeout,
        retries=args.retries,
        workers=1,
    )
    full_weather = pd.read_csv(extraction_outputs["weather"])
    lean_weather = build_conservative_weather(
        cache_dir=extraction_dir / "cache", cycle_date=cycle_date, members=members
    )
    comparison, metrics = compare_weather(full_weather, lean_weather)
    extraction_audit = json.loads(extraction_outputs["audit"].read_text(encoding="utf-8"))
    audit = build_audit(
        chosen=chosen,
        full_weather=full_weather,
        lean_weather=lean_weather,
        metrics=metrics,
        extraction_audit=extraction_audit,
    )
    audit["preflight_metadata_files_copied"] = copied_metadata_files
    outputs = {
        "chosen_cycle": args.output_dir / "gefs_temporal_sampling_smoke_chosen_cycle_v1.json",
        "full_weather": args.output_dir / "gefs_temporal_sampling_smoke_full_weather_v1.csv",
        "lean_weather": args.output_dir / "gefs_temporal_sampling_smoke_lean_weather_v1.csv",
        "comparison": args.output_dir / "gefs_temporal_sampling_smoke_row_comparison_v1.csv",
        "metrics": args.output_dir / "gefs_temporal_sampling_smoke_variable_metrics_v1.csv",
        "audit": args.output_dir / "gefs_temporal_sampling_smoke_audit_v1.json",
        "manifest": args.output_dir / "gefs_temporal_sampling_smoke_manifest_v1.json",
    }
    write_json(
        outputs["chosen_cycle"],
        {
            "target_year": int(chosen["target_year"]),
            "decision_date": str(chosen["decision_date"]),
            "required_site_count": int(chosen["required_site_count"]),
            "required_sites": str(chosen["required_sites"]).split(","),
            "expected_output_rows": int(chosen["expected_output_rows"]),
            "selected_range_bytes": int(chosen["selected_range_bytes"]),
        },
    )
    full_weather.to_csv(outputs["full_weather"], index=False)
    lean_weather.to_csv(outputs["lean_weather"], index=False)
    comparison.to_csv(outputs["comparison"], index=False)
    metrics.to_csv(outputs["metrics"], index=False)
    write_json(outputs["audit"], audit)
    manifest = {
        "status": audit["status"],
        "inputs": {
            "cycle_plan": sha256_file(args.cycle_plan),
            "temporal_audit": sha256_file(args.temporal_audit),
        },
        "outputs": {
            key: {"path": path.name, "sha256": sha256_file(path)}
            for key, path in outputs.items()
            if key != "manifest"
        },
        "full_extraction_manifest": str(extraction_outputs["manifest"]),
        "second_lean_payload_download_performed": False,
    }
    write_json(outputs["manifest"], manifest)
    if not audit["mandatory_structural_gate_passed"]:
        raise RuntimeError(f"one-cycle weather smoke failed; see {outputs['audit']}")
    return outputs


if __name__ == "__main__":
    generated = run(parse_args())
    print(json.dumps({key: str(value) for key, value in generated.items()}, indent=2))
