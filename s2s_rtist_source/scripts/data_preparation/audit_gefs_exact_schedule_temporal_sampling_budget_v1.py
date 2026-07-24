#!/usr/bin/env python3
"""Estimate lean GEFS temporal-sampling bytes from cached indices only."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Sequence

import pandas as pd

from scripts.data_preparation.preflight_gefs_2015_2019_exact_schedule_full_weather_v1 import (
    EXPECTED_UNIQUE_CYCLES,
    sha256_file,
    write_json,
)
from s2s_rtist.weather.gefs_gridmet_bias import (
    GribIndexRecord,
    merge_contiguous_ranges,
)
from s2s_rtist.weather.gefs_reforecast_full_weather import (
    REQUIRED_PRODUCT_SPECS,
    _product_cache_paths,
    select_product_records,
)


EXPECTED_PRODUCT_ROWS = 8365
EXPECTED_TASKS = 1195
MAXIMUM_END_HOUR = 174
FULL_REFERENCE = "full_3hourly_reference"
CONSERVATIVE_CANDIDATE = "state_6hourly_temperature_3hourly_fluxes"
BALANCED_DIAGNOSTIC = "all_instant_6hourly_fluxes_3hourly"
VARIANT_ORDER = (FULL_REFERENCE, CONSERVATIVE_CANDIDATE, BALANCED_DIAGNOSTIC)
THINNABLE_STATE_PRODUCTS = {"spfh_2m", "pres_sfc", "ugrd_hgt", "vgrd_hgt"}
TEMPERATURE_PRODUCT = "tmp_2m"
FLUX_PRODUCTS = {"apcp_sfc", "dswrf_sfc"}


def product_cadence_hours(*, product_id: str, variant_id: str) -> int:
    if variant_id == FULL_REFERENCE:
        return 3
    if variant_id == CONSERVATIVE_CANDIDATE:
        return 6 if product_id in THINNABLE_STATE_PRODUCTS else 3
    if variant_id == BALANCED_DIAGNOSTIC:
        return 6 if product_id in THINNABLE_STATE_PRODUCTS | {TEMPERATURE_PRODUCT} else 3
    raise ValueError(f"unknown temporal-sampling variant: {variant_id}")


def select_records_for_variant(
    records: Sequence[GribIndexRecord], *, product_id: str, variant_id: str
) -> list[GribIndexRecord]:
    cadence = product_cadence_hours(product_id=product_id, variant_id=variant_id)
    selected = [
        record for record in records if int(record.step.end_hour) % cadence == 0
    ]
    if not selected:
        raise ValueError(f"variant {variant_id} selected no records for {product_id}")
    if selected[-1].step.end_hour != MAXIMUM_END_HOUR:
        raise ValueError(
            f"variant {variant_id} does not reach {MAXIMUM_END_HOUR} h for {product_id}"
        )
    if product_id in FLUX_PRODUCTS and len(selected) != len(records):
        raise ValueError(f"flux product {product_id} must retain full three-hour coverage")
    return selected


def selected_range_bytes(records: Sequence[GribIndexRecord]) -> int:
    return int(
        sum(item.end - item.start + 1 for item in merge_contiguous_ranges(records))
    )


def validate_inputs(
    product_preflight: pd.DataFrame, preflight_audit: dict[str, Any]
) -> pd.DataFrame:
    required_columns = {
        "cycle_date",
        "gefs_member",
        "product_id",
        "short_name",
        "selected_message_count",
        "selected_range_bytes",
    }
    missing = required_columns.difference(product_preflight.columns)
    if missing:
        raise ValueError(f"product preflight missing fields: {sorted(missing)}")
    data = product_preflight.copy()
    data["cycle_date"] = pd.to_datetime(data["cycle_date"], errors="raise").dt.strftime(
        "%Y-%m-%d"
    )
    data["target_year"] = pd.to_datetime(data["cycle_date"]).dt.year.astype(int)
    data["selected_message_count"] = pd.to_numeric(
        data["selected_message_count"], errors="raise"
    ).astype(int)
    data["selected_range_bytes"] = pd.to_numeric(
        data["selected_range_bytes"], errors="raise"
    ).astype(int)
    if len(data) != EXPECTED_PRODUCT_ROWS:
        raise ValueError("product preflight row count mismatch")
    if data[["cycle_date", "gefs_member", "product_id"]].duplicated().any():
        raise ValueError("product preflight contains duplicate task-product keys")
    if data["cycle_date"].nunique() != EXPECTED_UNIQUE_CYCLES:
        raise ValueError("product preflight cycle count mismatch")
    if data[["cycle_date", "gefs_member"]].drop_duplicates().shape[0] != EXPECTED_TASKS:
        raise ValueError("product preflight cycle-member task count mismatch")
    expected_products = {item.product_id for item in REQUIRED_PRODUCT_SPECS}
    if set(data["product_id"]) != expected_products:
        raise ValueError("product preflight required product set mismatch")
    required_audit = {
        "status": "exact_schedule_full_weather_preflight_passed",
        "mandatory_structural_gate_passed": True,
        "product_payload_download_started": False,
        "unique_cycle_count": EXPECTED_UNIQUE_CYCLES,
        "cycle_member_task_count": EXPECTED_TASKS,
        "cycle_member_product_row_count": EXPECTED_PRODUCT_ROWS,
    }
    for key, expected in required_audit.items():
        if preflight_audit.get(key) != expected:
            raise ValueError(f"preflight audit contract mismatch for {key}")
    if int(data["selected_range_bytes"].sum()) != int(
        preflight_audit.get("selected_range_bytes", -1)
    ):
        raise ValueError("product preflight bytes do not reconcile to audit")
    return data.sort_values(["cycle_date", "gefs_member", "product_id"]).reset_index(
        drop=True
    )


def estimate_variants(
    product_preflight: pd.DataFrame, *, cache_dir: Path
) -> pd.DataFrame:
    spec_by_product = {item.product_id: item for item in REQUIRED_PRODUCT_SPECS}
    rows: list[dict[str, Any]] = []
    missing_indices: list[str] = []
    reference_mismatches: list[str] = []
    for item in product_preflight.itertuples(index=False):
        cache_path = _product_cache_paths(
            cycle_date=item.cycle_date,
            member=item.gefs_member,
            product_id=item.product_id,
            cache_dir=cache_dir,
            maximum_end_hour=MAXIMUM_END_HOUR,
        )["index"]
        if not cache_path.is_file():
            missing_indices.append(str(cache_path))
            continue
        records = select_product_records(
            cache_path.read_text(encoding="utf-8"),
            spec=spec_by_product[item.product_id],
            maximum_end_hour=MAXIMUM_END_HOUR,
        )
        for variant_id in VARIANT_ORDER:
            selected = select_records_for_variant(
                records, product_id=item.product_id, variant_id=variant_id
            )
            byte_count = selected_range_bytes(selected)
            if (
                variant_id == FULL_REFERENCE
                and byte_count != int(item.selected_range_bytes)
            ):
                reference_mismatches.append(
                    f"{item.cycle_date}/{item.gefs_member}/{item.product_id}: "
                    f"{byte_count} != {item.selected_range_bytes}"
                )
            rows.append(
                {
                    "variant_id": variant_id,
                    "target_year": int(item.target_year),
                    "cycle_date": item.cycle_date,
                    "gefs_member": item.gefs_member,
                    "product_id": item.product_id,
                    "short_name": item.short_name,
                    "cadence_hours": product_cadence_hours(
                        product_id=item.product_id, variant_id=variant_id
                    ),
                    "selected_message_count": len(selected),
                    "selected_range_bytes": byte_count,
                }
            )
    if missing_indices:
        raise FileNotFoundError(
            f"{len(missing_indices)} cached indices are missing; first={missing_indices[0]}"
        )
    if reference_mismatches:
        raise ValueError(
            f"full-reference bytes differ from preflight; first={reference_mismatches[0]}"
        )
    return pd.DataFrame(rows).sort_values(
        ["variant_id", "cycle_date", "gefs_member", "product_id"]
    ).reset_index(drop=True)


def build_budget_tables(
    detail: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    reference_bytes = int(
        detail.loc[detail["variant_id"].eq(FULL_REFERENCE), "selected_range_bytes"].sum()
    )
    variant = (
        detail.groupby("variant_id", as_index=False)
        .agg(
            cycle_count=("cycle_date", "nunique"),
            cycle_member_product_rows=("product_id", "size"),
            selected_message_count=("selected_message_count", "sum"),
            selected_range_bytes=("selected_range_bytes", "sum"),
        )
        .reset_index(drop=True)
    )
    variant["selected_range_gib"] = variant["selected_range_bytes"] / 1024**3
    variant["reduction_bytes_vs_full"] = reference_bytes - variant["selected_range_bytes"]
    variant["reduction_gib_vs_full"] = variant["reduction_bytes_vs_full"] / 1024**3
    variant["reduction_pct_vs_full"] = (
        variant["reduction_bytes_vs_full"] / reference_bytes * 100.0
    )
    science_status = {
        FULL_REFERENCE: "reference",
        CONSERVATIVE_CANDIDATE: "requires_one_cycle_weather_equivalence_smoke",
        BALANCED_DIAGNOSTIC: "diagnostic_only_requires_temperature_extrema_validation",
    }
    variant["science_status"] = variant["variant_id"].map(science_status)
    order = {name: index for index, name in enumerate(VARIANT_ORDER)}
    variant["_order"] = variant["variant_id"].map(order)
    variant = variant.sort_values("_order").drop(columns="_order").reset_index(drop=True)

    product = (
        detail.groupby(["variant_id", "product_id", "short_name", "cadence_hours"], as_index=False)
        .agg(
            selected_message_count=("selected_message_count", "sum"),
            selected_range_bytes=("selected_range_bytes", "sum"),
        )
        .reset_index(drop=True)
    )
    product["selected_range_gib"] = product["selected_range_bytes"] / 1024**3
    product["share_of_variant_pct"] = product.groupby("variant_id")[
        "selected_range_bytes"
    ].transform(lambda values: values / values.sum() * 100.0)

    year = (
        detail.groupby(["variant_id", "target_year"], as_index=False)
        .agg(
            cycle_count=("cycle_date", "nunique"),
            selected_message_count=("selected_message_count", "sum"),
            selected_range_bytes=("selected_range_bytes", "sum"),
        )
        .reset_index(drop=True)
    )
    year["selected_range_gib"] = year["selected_range_bytes"] / 1024**3
    return variant, product, year


def build_audit(
    *,
    detail: pd.DataFrame,
    variant_budget: pd.DataFrame,
    preflight_reference_bytes: int,
    expected_product_rows_per_variant: int = EXPECTED_PRODUCT_ROWS,
) -> dict[str, Any]:
    reference = variant_budget.loc[variant_budget["variant_id"].eq(FULL_REFERENCE)].iloc[0]
    conservative = variant_budget.loc[
        variant_budget["variant_id"].eq(CONSERVATIVE_CANDIDATE)
    ].iloc[0]
    structural_passed = all(
        [
            len(detail) == expected_product_rows_per_variant * len(VARIANT_ORDER),
            int(reference["selected_range_bytes"]) == int(preflight_reference_bytes),
            int(conservative["selected_range_bytes"]) < int(reference["selected_range_bytes"]),
            detail.loc[
                detail["product_id"].isin(FLUX_PRODUCTS), "cadence_hours"
            ].eq(3).all(),
        ]
    )
    return {
        "status": (
            "exact_schedule_temporal_sampling_metadata_audit_passed"
            if structural_passed
            else "exact_schedule_temporal_sampling_metadata_audit_failed"
        ),
        "mandatory_structural_gate_passed": structural_passed,
        "variant_count": len(VARIANT_ORDER),
        "cycle_count": EXPECTED_UNIQUE_CYCLES,
        "cycle_member_task_count": EXPECTED_TASKS,
        "cycle_member_product_rows_per_variant": expected_product_rows_per_variant,
        "required_weather_variable_count": 6,
        "all_required_weather_variables_retained": True,
        "precipitation_three_hourly_retained": True,
        "solar_radiation_three_hourly_retained": True,
        "full_reference_reconciled": int(reference["selected_range_bytes"])
        == int(preflight_reference_bytes),
        "full_reference_gib": float(reference["selected_range_gib"]),
        "conservative_candidate_id": CONSERVATIVE_CANDIDATE,
        "conservative_candidate_gib": float(conservative["selected_range_gib"]),
        "conservative_reduction_gib": float(conservative["reduction_gib_vs_full"]),
        "conservative_reduction_pct": float(conservative["reduction_pct_vs_full"]),
        "metadata_network_download_performed": False,
        "product_payload_download_started": False,
        "weather_equivalence_smoke_performed": False,
        "swap_simulation_performed": False,
        "label_generation_performed": False,
        "surrogate_training_performed": False,
        "training_eligible": False,
        "tta_performed": False,
        "next_gate": (
            "run_one_cycle_full_vs_conservative_temporal_sampling_weather_equivalence_smoke"
            if structural_passed
            else "repair_temporal_sampling_metadata_audit"
        ),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--product-preflight", type=Path, required=True)
    parser.add_argument("--preflight-audit", type=Path, required=True)
    parser.add_argument("--preflight-cache-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def run(args: argparse.Namespace) -> dict[str, Path]:
    for path in (args.product_preflight, args.preflight_audit):
        if not path.is_file():
            raise FileNotFoundError(f"required input is missing: {path}")
    if not args.preflight_cache_dir.is_dir():
        raise FileNotFoundError(f"preflight cache directory is missing: {args.preflight_cache_dir}")
    if args.output_dir.exists():
        raise FileExistsError(f"refusing to overwrite output directory: {args.output_dir}")
    preflight_audit = json.loads(args.preflight_audit.read_text(encoding="utf-8"))
    product_preflight = validate_inputs(
        pd.read_csv(args.product_preflight), preflight_audit
    )
    detail = estimate_variants(product_preflight, cache_dir=args.preflight_cache_dir)
    variant, product, year = build_budget_tables(detail)
    audit = build_audit(
        detail=detail,
        variant_budget=variant,
        preflight_reference_bytes=int(preflight_audit["selected_range_bytes"]),
    )
    args.output_dir.mkdir(parents=True)
    outputs = {
        "variant_budget": args.output_dir / "gefs_temporal_sampling_variant_budget_v1.csv",
        "product_budget": args.output_dir / "gefs_temporal_sampling_product_budget_v1.csv",
        "year_budget": args.output_dir / "gefs_temporal_sampling_year_budget_v1.csv",
        "audit": args.output_dir / "gefs_temporal_sampling_metadata_audit_v1.json",
        "manifest": args.output_dir / "gefs_temporal_sampling_metadata_manifest_v1.json",
    }
    variant.to_csv(outputs["variant_budget"], index=False)
    product.to_csv(outputs["product_budget"], index=False)
    year.to_csv(outputs["year_budget"], index=False)
    write_json(outputs["audit"], audit)
    manifest = {
        "status": audit["status"],
        "inputs": {
            "product_preflight": sha256_file(args.product_preflight),
            "preflight_audit": sha256_file(args.preflight_audit),
        },
        "outputs": {
            key: {"path": path.name, "sha256": sha256_file(path)}
            for key, path in outputs.items()
            if key != "manifest"
        },
        "cache_index_files_read": EXPECTED_PRODUCT_ROWS,
        "metadata_network_download_performed": False,
        "product_payload_download_started": False,
    }
    write_json(outputs["manifest"], manifest)
    if not audit["mandatory_structural_gate_passed"]:
        raise RuntimeError(f"temporal-sampling audit failed; see {outputs['audit']}")
    return outputs


if __name__ == "__main__":
    generated = run(parse_args())
    print(json.dumps({key: str(value) for key, value in generated.items()}, indent=2))
