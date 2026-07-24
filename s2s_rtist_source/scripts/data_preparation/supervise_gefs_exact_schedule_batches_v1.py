#!/usr/bin/env python3
"""Keep two formal local GEFS batch lanes filled between deep audit checks."""

from __future__ import annotations

import argparse
import ctypes
import json
import msvcrt
import os
from pathlib import Path
import subprocess
import sys
import time
from typing import Any

import pandas as pd


BATCH_AUDIT_NAME = "gefs_exact_schedule_batch_full_weather_audit_v1.json"
CYCLE_AUDIT_NAME = "gefs_2015_2019_full_weather_audit_v1.json"
STILL_ACTIVE = 259
PROCESS_QUERY_LIMITED_INFORMATION = 0x1000


class MemoryStatusEx(ctypes.Structure):
    _fields_ = [
        ("dwLength", ctypes.c_ulong),
        ("dwMemoryLoad", ctypes.c_ulong),
        ("ullTotalPhys", ctypes.c_ulonglong),
        ("ullAvailPhys", ctypes.c_ulonglong),
        ("ullTotalPageFile", ctypes.c_ulonglong),
        ("ullAvailPageFile", ctypes.c_ulonglong),
        ("ullTotalVirtual", ctypes.c_ulonglong),
        ("ullAvailVirtual", ctypes.c_ulonglong),
        ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
    ]


class FileTime(ctypes.Structure):
    _fields_ = [
        ("dwLowDateTime", ctypes.c_ulong),
        ("dwHighDateTime", ctypes.c_ulong),
    ]


def process_is_running(pid: int) -> bool:
    handle = ctypes.windll.kernel32.OpenProcess(
        PROCESS_QUERY_LIMITED_INFORMATION, False, int(pid)
    )
    if not handle:
        return False
    try:
        exit_code = ctypes.c_ulong()
        if not ctypes.windll.kernel32.GetExitCodeProcess(
            handle, ctypes.byref(exit_code)
        ):
            return False
        return int(exit_code.value) == STILL_ACTIVE
    finally:
        ctypes.windll.kernel32.CloseHandle(handle)


def process_creation_time(pid: int) -> float | None:
    handle = ctypes.windll.kernel32.OpenProcess(
        PROCESS_QUERY_LIMITED_INFORMATION, False, int(pid)
    )
    if not handle:
        return None
    try:
        creation = FileTime()
        exit_time = FileTime()
        kernel = FileTime()
        user = FileTime()
        if not ctypes.windll.kernel32.GetProcessTimes(
            handle,
            ctypes.byref(creation),
            ctypes.byref(exit_time),
            ctypes.byref(kernel),
            ctypes.byref(user),
        ):
            return None
        ticks = (int(creation.dwHighDateTime) << 32) | int(
            creation.dwLowDateTime
        )
        return (ticks - 116444736000000000) / 10_000_000.0
    finally:
        ctypes.windll.kernel32.CloseHandle(handle)


def available_physical_gib() -> float:
    status = MemoryStatusEx()
    status.dwLength = ctypes.sizeof(MemoryStatusEx)
    if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
        raise OSError("GlobalMemoryStatusEx failed")
    return float(status.ullAvailPhys) / (1024**3)


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def batch_is_strictly_complete(
    output_root: Path, batch_row: Any, cycle_rows: pd.DataFrame, run_root: Path
) -> bool:
    batch_id = str(batch_row.batch_id)
    batch_dir = output_root / batch_id
    audit_path = batch_dir / BATCH_AUDIT_NAME
    if not audit_path.is_file():
        return False
    try:
        audit = _read_json(audit_path)
    except (OSError, ValueError, json.JSONDecodeError):
        return False
    if not all(
        [
            audit.get("status") == "exact_schedule_full_weather_local_batch_passed",
            audit.get("mandatory_structural_gate_passed") is True,
            audit.get("full_three_hourly_records") is True,
            audit.get("all_required_weather_variables_retained") is True,
            audit.get("temporary_grib_retained") is False,
            int(audit.get("member_count", -1)) == 5,
            int(audit.get("cycle_count", -1)) == int(batch_row.cycle_count),
            int(audit.get("completed_cycle_count", -1)) == int(batch_row.cycle_count),
            int(audit.get("expected_rows", -1)) == int(batch_row.expected_output_rows),
        ]
    ):
        return False
    if list(batch_dir.rglob("*.grib2")):
        return False
    stderr_path = run_root / f"{batch_id}.workers8_ranges4.stderr.log"
    if stderr_path.is_file() and stderr_path.stat().st_size:
        return False
    if len(cycle_rows) != int(batch_row.cycle_count):
        return False
    for cycle in cycle_rows.itertuples(index=False):
        cycle_dir = batch_dir / str(cycle.decision_date).replace("-", "")
        cycle_audit_path = cycle_dir / CYCLE_AUDIT_NAME
        if not cycle_audit_path.is_file():
            return False
        try:
            cycle_audit = _read_json(cycle_audit_path)
        except (OSError, ValueError, json.JSONDecodeError):
            return False
        if not all(
            [
                cycle_audit.get("status") == "full_weather_local_extraction_passed",
                int(cycle_audit.get("row_count", -1))
                == int(cycle.expected_output_rows),
                int(cycle_audit.get("expected_row_count", -1))
                == int(cycle.expected_output_rows),
                int(cycle_audit.get("member_count", -1)) == 5,
                int(cycle_audit.get("canonical_missing_value_count", -1)) == 0,
                int(cycle_audit.get("canonical_nonfinite_value_count", -1)) == 0,
                int(cycle_audit.get("duplicate_sample_key_count", -1)) == 0,
                int(cycle_audit.get("retained_grib_file_count", -1)) == 0,
            ]
        ):
            return False
    return True


def active_batch_ids(run_root: Path, batch_ids: list[str]) -> set[str]:
    active: set[str] = set()
    for batch_id in batch_ids:
        pid_path = run_root / f"{batch_id}.pid"
        if not pid_path.is_file():
            continue
        try:
            pid = int(pid_path.read_text(encoding="ascii").strip())
        except (OSError, ValueError):
            continue
        creation_time = process_creation_time(pid)
        pid_file_time = pid_path.stat().st_mtime
        if (
            process_is_running(pid)
            and creation_time is not None
            and creation_time <= pid_file_time + 5.0
        ):
            active.add(batch_id)
    return active


def launch_batch(launcher: Path, repo: Path, batch_id: str) -> None:
    flags = subprocess.CREATE_NO_WINDOW | subprocess.CREATE_NEW_PROCESS_GROUP
    subprocess.Popen(
        [sys.executable, str(launcher), batch_id],
        cwd=repo,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        close_fds=True,
        creationflags=flags,
    )


def supervise_once(args: argparse.Namespace) -> list[str]:
    batches = pd.read_csv(args.batch_budget).sort_values(
        ["target_year", "batch_number_in_year"]
    )
    cycles = pd.read_csv(args.cycle_plan)
    batch_ids = batches["batch_id"].astype(str).tolist()
    complete: set[str] = set()
    for batch in batches.itertuples(index=False):
        selected_cycles = cycles.loc[cycles["batch_id"].eq(batch.batch_id)]
        if batch_is_strictly_complete(
            args.output_root, batch, selected_cycles, args.run_root
        ):
            complete.add(str(batch.batch_id))
    active = active_batch_ids(args.run_root, batch_ids)
    lane_limit = 1 if available_physical_gib() < args.minimum_memory_gib else 2
    slots = max(0, lane_limit - len(active))
    launched: list[str] = []
    for batch_id in batch_ids:
        if slots <= 0:
            break
        if batch_id in complete or batch_id in active:
            continue
        pid_path = args.run_root / f"{batch_id}.pid"
        if pid_path.exists():
            # An incomplete dead batch requires audited log rotation/recovery.
            break
        launch_batch(args.launcher, args.repo, batch_id)
        launched.append(batch_id)
        active.add(batch_id)
        slots -= 1
    return launched


def log_line(path: Path, message: str) -> None:
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    with path.open("a", encoding="utf-8") as stream:
        stream.write(f"[{timestamp}] {message}\n")


def run(args: argparse.Namespace) -> None:
    args.run_root.mkdir(parents=True, exist_ok=True)
    lock_path = args.run_root / "gefs_batch_supervisor_v1.lock"
    lock_stream = lock_path.open("a+b")
    if lock_stream.tell() == 0:
        lock_stream.write(b"0")
        lock_stream.flush()
    lock_stream.seek(0)
    try:
        msvcrt.locking(lock_stream.fileno(), msvcrt.LK_NBLCK, 1)
    except OSError as error:
        raise SystemExit("GEFS batch supervisor is already running") from error
    (args.run_root / "gefs_batch_supervisor_v1.pid").write_text(
        f"{os.getpid()}\n", encoding="ascii"
    )
    log_path = args.run_root / "gefs_batch_supervisor_v1.log"
    log_line(log_path, f"supervisor started pid={os.getpid()}")
    while True:
        try:
            launched = supervise_once(args)
            for batch_id in launched:
                log_line(log_path, f"launched {batch_id}")
            batches = pd.read_csv(args.batch_budget)
            audits = [
                args.output_root / str(batch_id) / BATCH_AUDIT_NAME
                for batch_id in batches["batch_id"]
            ]
            if audits and all(path.is_file() for path in audits):
                # The deep automation performs final strict all-batch/year closeout.
                log_line(log_path, "all batch audit files exist; supervisor stopped")
                return
        except Exception as error:  # keep the scheduler alive; deep audit reports faults
            log_line(log_path, f"check failed: {type(error).__name__}: {error}")
        time.sleep(args.interval_seconds)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--batch-budget", type=Path, required=True)
    parser.add_argument("--cycle-plan", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--launcher", type=Path, required=True)
    parser.add_argument("--interval-seconds", type=int, default=20)
    parser.add_argument("--minimum-memory-gib", type=float, default=1.5)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
