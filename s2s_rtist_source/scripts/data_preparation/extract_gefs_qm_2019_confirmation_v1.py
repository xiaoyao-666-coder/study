#!/usr/bin/env python3
"""Extract the locked 2019 independent confirmation GEFS sample locally."""

from __future__ import annotations

import argparse
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Sequence

import pandas as pd

from s2s_rtist.weather.gefs_quantile_mapping import (
    CONTRACT_ID_V2,
    CONTRACT_VERSION_V2,
    GEFS_REFORECAST_MEMBERS,
    UTC_DAY_BOUNDARY,
    aggregate_reforecast_member_daily_utc,
    cycle_valid_dates,
    download_reforecast_member_points,
    extract_era5_reference_precipitation_utc,
    read_quantile_mapping_artifact,
    reforecast_site_frame,
    validate_member_daily_precipitation,
    validate_reference_daily_precipitation,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONTRACT_PATH = (
    PROJECT_ROOT
    / "site_general_surrogate_eval"
    / "gefs_quantile_mapping_2019_confirmation_contract_v1.json"
)
DEFAULT_OUTPUT_DIR = (
    PROJECT_ROOT
    / "site_general_surrogate_eval"
    / "gefs_quantile_mapping_v2"
    / "gefs_qm_2019_confirmation_v1"
)
DEFAULT_ERA5_ROOT = PROJECT_ROOT / "model3_opt_sto_upload" / "data"
SITES = ("P1", "P2", "P3", "P4", "P15")
MAXIMUM_END_HOUR = 168


def _write_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False, encoding="utf-8-sig")


def load_contract() -> dict[str, object]:
    contract = json.loads(CONTRACT_PATH.read_text(encoding="utf-8"))
    if contract["contract_id"] != "gefs-precipitation-quantile-mapping-v2-2019-confirmation-v1":
        raise ValueError("confirmation contract id mismatch")
    if contract["parent_contract"]["contract_id"] != CONTRACT_ID_V2:
        raise ValueError("confirmation parent contract mismatch")
    return contract


def confirmation_dates(contract: dict[str, object]) -> tuple[str, ...]:
    selection = set(contract["strategy_selection_dates_2019"])
    dates = tuple(contract["independent_confirmation_dates_2019"])
    if len(dates) != len(set(dates)):
        raise ValueError("confirmation dates are duplicated")
    overlap = selection.intersection(dates)
    if overlap:
        raise ValueError(f"confirmation dates overlap strategy dates: {sorted(overlap)}")
    if tuple(sorted(dates)) != dates:
        raise ValueError("confirmation dates must be chronological")
    return dates


def verify_frozen_artifact(contract: dict[str, object]) -> dict[str, object]:
    frozen = contract["frozen_mapping"]
    artifact_path = PROJECT_ROOT / frozen["artifact_relative_path"]
    artifact = read_quantile_mapping_artifact(
        artifact_path, expected_contract_id=CONTRACT_ID_V2
    )
    if artifact["contract_version"] != CONTRACT_VERSION_V2:
        raise ValueError("frozen artifact contract version mismatch")
    if artifact["artifact_sha256"] != frozen["artifact_sha256"]:
        raise ValueError("frozen artifact SHA-256 does not match confirmation contract")
    if artifact["aggregation_day_boundary"] != UTC_DAY_BOUNDARY:
        raise ValueError("frozen artifact does not use UTC day aggregation")
    return artifact


def write_locked_plan(
    *,
    output_dir: Path,
    contract: dict[str, object],
    dates: Sequence[str],
) -> dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    tasks = [
        {
            "cycle_date": cycle_date,
            "member": member,
            "site_count": len(SITES),
            "maximum_end_hour": MAXIMUM_END_HOUR,
            "expected_selected_message_count": 56,
            "role": "independent_confirmation_2019",
        }
        for cycle_date in dates
        for member in GEFS_REFORECAST_MEMBERS
    ]
    dates_path = output_dir / "confirmation_dates_locked.json"
    dates_path.write_text(
        json.dumps(
            {
                "contract_id": contract["contract_id"],
                "contract_version": contract["contract_version"],
                "strategy_selection_dates_2019": contract[
                    "strategy_selection_dates_2019"
                ],
                "independent_confirmation_dates_2019": list(dates),
                "dates_are_disjoint": True,
                "network_download_started": False,
                "frozen_artifact_sha256": contract["frozen_mapping"][
                    "artifact_sha256"
                ],
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    plan_path = output_dir / "extraction_plan.csv"
    _write_csv(pd.DataFrame(tasks), plan_path)
    return {"dates": dates_path, "plan": plan_path}


def run_extraction(
    *,
    output_dir: Path,
    era5_root: Path,
    workers: int,
    timeout: int,
    retries: int,
    plan_only: bool,
) -> dict[str, Path]:
    contract = load_contract()
    artifact = verify_frozen_artifact(contract)
    dates = confirmation_dates(contract)
    plan_outputs = write_locked_plan(
        output_dir=output_dir,
        contract=contract,
        dates=dates,
    )
    if plan_only:
        print("[confirmation] plan-only: no network download started", flush=True)
        return plan_outputs

    locked = json.loads(plan_outputs["dates"].read_text(encoding="utf-8"))
    locked["network_download_started"] = True
    plan_outputs["dates"].write_text(
        json.dumps(locked, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    sites = reforecast_site_frame(SITES)
    cache_dir = output_dir / "cache"
    partial_manifest_path = (
        output_dir / "gefs_reforecast_download_manifest_2019_confirmation.partial.csv"
    )
    tasks = [(cycle_date, member) for cycle_date in dates for member in GEFS_REFORECAST_MEMBERS]
    point_parts: list[pd.DataFrame] = []
    manifest_rows: list[dict[str, object]] = []

    def run_task(task: tuple[str, str]):
        cycle_date, member = task

        def report_range(completed: int, total: int) -> None:
            if completed == 1 or completed == total or completed % 8 == 0:
                print(
                    f"[range] {cycle_date} {member} {completed}/{total}",
                    flush=True,
                )

        points, metadata = download_reforecast_member_points(
            cycle_date=cycle_date,
            member=member,
            sites=sites,
            cache_dir=cache_dir,
            timeout=timeout,
            retries=retries,
            keep_grib=False,
            maximum_end_hour=MAXIMUM_END_HOUR,
            range_progress=report_range,
        )
        metadata = dict(metadata)
        metadata.update(
            {
                "confirmation_role": "independent_confirmation_2019",
                "maximum_end_hour": MAXIMUM_END_HOUR,
                "expected_selected_message_count": 56,
                "network_fallback_used": False,
                "member_fallback_used": False,
            }
        )
        return task, points, metadata

    errors: list[tuple[tuple[str, str], Exception]] = []
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(run_task, task): task for task in tasks}
        for completed, future in enumerate(as_completed(futures), start=1):
            task = futures[future]
            try:
                (cycle_date, member), points, metadata = future.result()
            except Exception as exc:
                errors.append((task, exc))
                print(f"[confirmation] {task[0]} {task[1]} failed: {exc}", flush=True)
                continue
            point_parts.append(points)
            manifest_rows.append(metadata)
            print(
                f"[confirmation] {cycle_date} {member} ready ({completed}/{len(tasks)})",
                flush=True,
            )
            if completed % 5 == 0 or completed == len(tasks):
                partial = pd.DataFrame(manifest_rows).sort_values(
                    ["cycle_date", "gefs_member"]
                )
                _write_csv(partial, partial_manifest_path)
    if errors:
        examples = "; ".join(
            f"{cycle} {member}: {error}" for (cycle, member), error in errors[:5]
        )
        raise RuntimeError(
            f"{len(errors)} confirmation tasks failed after retries: {examples}"
        )

    points = pd.concat(point_parts, ignore_index=True)
    manifest = pd.DataFrame(manifest_rows).sort_values(
        ["cycle_date", "gefs_member"]
    ).reset_index(drop=True)
    if set(manifest["selected_message_count"].astype(int)) != {56}:
        raise ValueError("confirmation manifest does not contain exactly 56 messages per task")
    if set(manifest["selected_end_step"].astype(int)) != {MAXIMUM_END_HOUR}:
        raise ValueError("confirmation manifest does not end at 168 hours")
    if manifest["source_etag"].astype(str).str.strip().eq("").any():
        raise ValueError("confirmation manifest contains an empty source ETag")
    if manifest["downloaded_bytes"].astype(int).le(0).any():
        raise ValueError("confirmation manifest contains a nonpositive byte count")
    manifest_path = output_dir / "gefs_reforecast_download_manifest_2019_confirmation.csv"
    _write_csv(manifest, manifest_path)
    partial_manifest_path.unlink(missing_ok=True)

    member_daily = aggregate_reforecast_member_daily_utc(points, manifest=manifest)
    all_valid_dates = sorted(
        {valid_date for cycle in dates for valid_date in cycle_valid_dates(cycle)}
    )
    reference = extract_era5_reference_precipitation_utc(
        era5_root=era5_root,
        sites=sites,
        valid_dates=all_valid_dates,
    )
    validate_member_daily_precipitation(
        member_daily,
        expected_sites=SITES,
        expected_members=GEFS_REFORECAST_MEMBERS,
        expected_cycles=dates,
        date_column="valid_date_utc",
    )
    validate_reference_daily_precipitation(
        reference,
        expected_sites=SITES,
        expected_dates=all_valid_dates,
        date_column="valid_date_utc",
    )
    expected = contract["expected_counts"]
    if len(member_daily) != expected["confirmation_member_rows"]:
        raise ValueError("confirmation member row count does not match contract")
    if len(reference) != expected["confirmation_unique_reference_observations"]:
        raise ValueError("confirmation reference row count does not match contract")

    member_path = (
        output_dir / "gefs_reforecast_member_daily_precipitation_utc_2019_confirmation.csv"
    )
    reference_path = (
        output_dir / "era5_reference_daily_precipitation_utc_2019_confirmation.csv"
    )
    _write_csv(member_daily, member_path)
    _write_csv(reference, reference_path)
    retained_grib_count = len(list((cache_dir / "minigrib").glob("*.grib2")))
    if retained_grib_count:
        raise ValueError(f"confirmation extraction retained {retained_grib_count} GRIB files")
    frozen_manifest_path = output_dir / "frozen_qm_artifact_load_manifest.json"
    frozen_manifest_path.write_text(
        json.dumps(
            {
                "contract_id": contract["contract_id"],
                "parent_contract_id": CONTRACT_ID_V2,
                "artifact_sha256": artifact["artifact_sha256"],
                "artifact_path": str(
                    PROJECT_ROOT / contract["frozen_mapping"]["artifact_relative_path"]
                ),
                "fit_years": artifact["fit_years"],
                "refit_performed": False,
                "strategy_change_performed": False,
                "network_download_started": True,
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "member_daily": str(member_path),
                "reference": str(reference_path),
                "manifest": str(manifest_path),
                "frozen_manifest": str(frozen_manifest_path),
                "plan": str(plan_outputs["plan"]),
            },
            indent=2,
        )
    )
    return {
        "member_daily": member_path,
        "reference": reference_path,
        "manifest": manifest_path,
        "frozen_manifest": frozen_manifest_path,
        "plan": plan_outputs["plan"],
        "dates": plan_outputs["dates"],
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--era5-root", type=Path, default=DEFAULT_ERA5_ROOT)
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--timeout", type=int, default=120)
    parser.add_argument("--retries", type=int, default=5)
    parser.add_argument("--plan-only", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.workers < 1:
        raise ValueError("workers must be positive")
    run_extraction(
        output_dir=args.output_dir,
        era5_root=args.era5_root,
        workers=args.workers,
        timeout=args.timeout,
        retries=args.retries,
        plan_only=args.plan_only,
    )


if __name__ == "__main__":
    main()
