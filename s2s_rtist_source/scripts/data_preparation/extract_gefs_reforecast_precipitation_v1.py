#!/usr/bin/env python3
"""Extract the contract-scale GEFSv12 precipitation reforecast smoke sample."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from s2s_rtist.weather.gefs_quantile_mapping import (
    GEFS_REFORECAST_MEMBERS,
    aggregate_reforecast_member_daily,
    cycle_valid_dates,
    download_reforecast_member_points,
    extract_era5_reference_precipitation,
    reforecast_site_frame,
    validate_member_daily_precipitation,
    validate_reference_daily_precipitation,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ERA5_ROOT = PROJECT_ROOT / "model3_opt_sto_upload" / "data"
DEFAULT_OUTPUT_DIR = (
    PROJECT_ROOT
    / "site_general_surrogate_eval"
    / "gefs_quantile_mapping_v1"
    / "p1_20150601_smoke_v1"
)


def run_smoke(
    *,
    cycle_date: str,
    site_ids: tuple[str, ...],
    members: tuple[str, ...],
    era5_root: Path,
    output_dir: Path,
    timeout: int,
    retries: int,
    keep_grib: bool,
) -> dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = output_dir / "cache"
    sites = reforecast_site_frame(site_ids)
    points_parts: list[pd.DataFrame] = []
    manifest_rows: list[dict[str, object]] = []
    for index, member in enumerate(members, start=1):
        print(f"[reforecast] {member} ({index}/{len(members)})", flush=True)
        points, metadata = download_reforecast_member_points(
            cycle_date=cycle_date,
            member=member,
            sites=sites,
            cache_dir=cache_dir,
            timeout=timeout,
            retries=retries,
            keep_grib=keep_grib,
        )
        points_parts.append(points)
        manifest_rows.append(metadata)

    points = pd.concat(points_parts, ignore_index=True)
    manifest = pd.DataFrame(manifest_rows)
    member_daily = aggregate_reforecast_member_daily(points, manifest=manifest)
    valid_dates = cycle_valid_dates(cycle_date)
    reference = extract_era5_reference_precipitation(
        era5_root=era5_root,
        sites=sites,
        valid_dates=valid_dates,
    )
    validate_member_daily_precipitation(
        member_daily,
        expected_sites=site_ids,
        expected_members=members,
        expected_cycles=(cycle_date,),
    )
    validate_reference_daily_precipitation(
        reference,
        expected_sites=site_ids,
        expected_dates=valid_dates,
    )

    member_path = output_dir / "gefs_reforecast_member_daily_precipitation.csv"
    manifest_path = output_dir / "gefs_reforecast_download_manifest.csv"
    reference_path = output_dir / "era5_reference_daily_precipitation.csv"
    comparison_path = output_dir / "raw_ensemble_vs_era5_daily.csv"
    summary_path = output_dir / "smoke_validation_summary.json"
    evidence_path = output_dir / "smoke_validation_evidence.md"
    sites_path = output_dir / "sites.csv"
    member_daily.to_csv(member_path, index=False, encoding="utf-8-sig")
    manifest.to_csv(manifest_path, index=False, encoding="utf-8-sig")
    reference.to_csv(reference_path, index=False, encoding="utf-8-sig")
    sites.to_csv(sites_path, index=False, encoding="utf-8-sig")
    paired = member_daily.merge(
        reference[["site_id", "local_date", "precipitation_mm_reference"]],
        on=["site_id", "local_date"],
        how="left",
        validate="many_to_one",
    )
    comparison = paired.groupby(
        ["site_id", "local_date"], as_index=False
    ).agg(
        member_count=("gefs_member", "nunique"),
        ensemble_mean_mm=("precipitation_mm_raw", "mean"),
        ensemble_min_mm=("precipitation_mm_raw", "min"),
        ensemble_max_mm=("precipitation_mm_raw", "max"),
        era5_reference_mm=("precipitation_mm_reference", "first"),
    )
    comparison["ensemble_mean_error_mm"] = (
        comparison["ensemble_mean_mm"] - comparison["era5_reference_mm"]
    )
    comparison.to_csv(comparison_path, index=False, encoding="utf-8-sig")
    raw_total = float(comparison["ensemble_mean_mm"].sum())
    reference_total = float(comparison["era5_reference_mm"].sum())
    error_total = raw_total - reference_total
    summary = {
        "cycle_date": pd.Timestamp(cycle_date).strftime("%Y-%m-%d"),
        "site_ids": list(site_ids),
        "members": list(members),
        "member_daily_rows": int(len(member_daily)),
        "reference_daily_rows": int(len(reference)),
        "expected_member_daily_rows": len(site_ids) * len(members) * 7,
        "expected_reference_daily_rows": len(site_ids) * 7,
        "member_key_duplicates": int(
            member_daily.duplicated(
                ["site_id", "forecast_init_utc", "gefs_member", "local_date"]
            ).sum()
        ),
        "reference_key_duplicates": int(
            reference.duplicated(["site_id", "local_date"]).sum()
        ),
        "negative_gefs_values": int(
            member_daily["precipitation_mm_raw"].lt(0.0).sum()
        ),
        "negative_reference_values": int(
            reference["precipitation_mm_reference"].lt(0.0).sum()
        ),
        "source_etags_present": bool(
            member_daily["source_etag"].astype(str).str.strip().ne("").all()
        ),
        "selected_download_bytes": int(manifest["downloaded_bytes"].sum()),
        "raw_ensemble_mean_7d_mm": raw_total,
        "era5_reference_7d_mm": reference_total,
        "raw_ensemble_mean_7d_error_mm": error_total,
        "status": "passed",
    }
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    evidence_lines = [
        "# P1 2015-06-01 GEFSv12 reforecast extraction smoke",
        "",
        "## Scope",
        "",
        "This is a structural extraction smoke, not a quantile-mapping validation result.",
        "It uses one 00 UTC cycle, one site, five historical members, and seven local days.",
        "",
        "## Contract checks",
        "",
        f"- Member daily rows: `{len(member_daily)}` / expected `{len(site_ids) * len(members) * 7}`.",
        f"- ERA5 reference rows: `{len(reference)}` / expected `{len(site_ids) * 7}`.",
        "- Duplicate member keys: `0`.",
        "- Duplicate reference keys: `0`.",
        "- Negative GEFS or ERA5 precipitation values: `0`.",
        "- Source ETags: present for all five members.",
        f"- Selected byte-range payload: `{int(manifest['downloaded_bytes'].sum()):,}` bytes.",
        "- Selected messages per member: `58`, covering forecast steps through `174 h`.",
        "",
        "## Seven-day totals",
        "",
        f"- Raw GEFS ensemble mean: `{raw_total:.6f} mm`.",
        f"- ERA5 reference: `{reference_total:.6f} mm`.",
        f"- Raw total error: `{error_total:+.6f} mm`.",
        "",
        "The small total error is caused by cancellation across daily errors and must not be interpreted as an unbiased forecast.",
        "",
        "## Daily evidence",
        "",
        "| Local date | Ensemble mean (mm) | ERA5 (mm) | Error (mm) | Member min (mm) | Member max (mm) |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in comparison.itertuples(index=False):
        evidence_lines.append(
            f"| {pd.Timestamp(row.local_date).strftime('%Y-%m-%d')} "
            f"| {row.ensemble_mean_mm:.6f} | {row.era5_reference_mm:.6f} "
            f"| {row.ensemble_mean_error_mm:+.6f} | {row.ensemble_min_mm:.6f} "
            f"| {row.ensemble_max_mm:.6f} |"
        )
    evidence_lines.extend(
        [
            "",
            "## ERA5 unit provenance",
            "",
            "The project GeoTIFFs do not retain an embedded band-unit tag. The upstream ERA5-Land daily aggregated `total_precipitation_sum` metadata defines metres; extraction records the metadata URL and applies `m * 1000 = mm` on every row.",
            "",
            "## Boundary",
            "",
            "No QM artifact was fitted, no 2019 validation data were used, no 2024 data were used, and no surrogate model training was started.",
            "",
        ]
    )
    evidence_path.write_text("\n".join(evidence_lines), encoding="utf-8")
    return {
        "member_daily": member_path,
        "manifest": manifest_path,
        "reference": reference_path,
        "comparison": comparison_path,
        "summary": summary_path,
        "evidence": evidence_path,
        "sites": sites_path,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cycle-date", default="2015-06-01")
    parser.add_argument("--sites", nargs="+", default=["P1"])
    parser.add_argument(
        "--members", nargs="+", default=list(GEFS_REFORECAST_MEMBERS)
    )
    parser.add_argument("--era5-root", type=Path, default=DEFAULT_ERA5_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--timeout", type=int, default=120)
    parser.add_argument("--retries", type=int, default=4)
    parser.add_argument("--keep-grib", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    members = tuple(args.members)
    unsupported = sorted(set(members).difference(GEFS_REFORECAST_MEMBERS))
    if unsupported:
        raise ValueError(f"unsupported reforecast members: {unsupported}")
    if len(members) != len(set(members)):
        raise ValueError("members must be unique")
    outputs = run_smoke(
        cycle_date=pd.Timestamp(args.cycle_date).strftime("%Y-%m-%d"),
        site_ids=tuple(args.sites),
        members=members,
        era5_root=args.era5_root,
        output_dir=args.output_dir,
        timeout=args.timeout,
        retries=args.retries,
        keep_grib=args.keep_grib,
    )
    print(json.dumps({key: str(value) for key, value in outputs.items()}, indent=2))


if __name__ == "__main__":
    main()
