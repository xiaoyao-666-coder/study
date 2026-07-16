"""Parallel smoke test for one decision date and eight irrigation candidates.

Run this from the server parent directory that contains Maize/, for example:

    cd /media/data_hot/lzx_projs/soil_moisture_otw/s2s_rtist_source
    nohup env CUDA_VISIBLE_DEVICES="" LD_LIBRARY_PATH="/media/data_hot/lzx_projs/soil_moisture_otw/s2s_rtist_source/local_libs/gcc_runtime/usr/lib/x86_64-linux-gnu:$LD_LIBRARY_PATH" python3 decision_smoke_8ir_parallel.py --jobs 8 > decision_smoke_8ir_parallel.log 2>&1 &

Each irrigation candidate runs in an independent copied work directory, so SWAP
outputs do not overwrite each other.
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import pandas as pd


IRRIGATION_OPTIONS_MM = [0, 10, 15, 20, 25, 30, 40, 60]
YIELD_PRICE_PER_KG = 0.20
WATER_COST_PER_HA_PER_MM = 2.0
WEIGHT_INDEX = 0.7


def copy_template(template_dir: Path, dst_dir: Path) -> None:
    ignore = shutil.ignore_patterns(
        "__pycache__",
        "*.log",
        "candidate_result.csv",
        "decision_smoke_8ir*.csv",
        "decision_smoke_8ir*.log",
        "decision_smoke_8ir*.pid",
    )
    shutil.copytree(template_dir, dst_dir, ignore=ignore)


def launch_candidate(worker_script: Path, work_dir: Path, irrigation_mm: int) -> subprocess.Popen:
    log_path = work_dir / "candidate.log"
    log = open(log_path, "w", encoding="utf-8", errors="ignore")
    return subprocess.Popen(
        [sys.executable, str(worker_script), "--ir", str(irrigation_mm)],
        cwd=str(work_dir),
        stdout=log,
        stderr=subprocess.STDOUT,
    )


def poll_finished(running: list[tuple[int, Path, subprocess.Popen]]) -> int:
    finished = 0
    for item in list(running):
        ir, work_dir, proc = item
        ret = proc.poll()
        if ret is None:
            continue
        running.remove(item)
        finished += 1
        print(f"candidate {ir} mm finished with exit code {ret}: {work_dir}", flush=True)
        if ret != 0:
            raise RuntimeError(f"candidate {ir} mm failed; see {work_dir / 'candidate.log'}")
    return finished


def wait_for_slot(running: list[tuple[int, Path, subprocess.Popen]], jobs: int) -> None:
    while len(running) >= jobs:
        finished = poll_finished(running)
        if finished == 0 and len(running) >= jobs:
            time.sleep(5)


def wait_for_all(running: list[tuple[int, Path, subprocess.Popen]]) -> None:
    while running:
        finished = poll_finished(running)
        if finished == 0:
            active = ", ".join(str(ir) for ir, _, _ in running)
            print(f"waiting for candidates: {active}", flush=True)
            time.sleep(5)


def collect_results(work_root: Path) -> pd.DataFrame:
    rows = []
    for ir in IRRIGATION_OPTIONS_MM:
        result_path = work_root / f"ir_{ir}" / "candidate_result.csv"
        if not result_path.exists():
            raise FileNotFoundError(f"missing {result_path}")
        rows.append(pd.read_csv(result_path))

    out = pd.concat(rows, ignore_index=True).sort_values("ir")
    cwdm_ir0 = float(out.loc[out["ir"] == 0, "cwdm_value"].iloc[0])
    out["target_value"] = (
        (out["cwdm_value"] - cwdm_ir0) * YIELD_PRICE_PER_KG
        - out["ir"] * WATER_COST_PER_HA_PER_MM * WEIGHT_INDEX
    )
    out.loc[out["ir"] == 0, "target_value"] = 0.0
    return out[
        [
            "date_t",
            "ir",
            "end_daynr",
            "dvs",
            "cwdm_value",
            "cwso_value",
            "target_value",
        ]
    ]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--template", default="Maize", help="Template Maize directory to copy.")
    parser.add_argument("--jobs", type=int, default=4, help="Number of candidates to run in parallel.")
    parser.add_argument("--work-root", default=None, help="Output work root. Default uses a timestamp.")
    args = parser.parse_args()

    base_dir = Path.cwd()
    template_dir = (base_dir / args.template).resolve()
    if not template_dir.exists():
        raise FileNotFoundError(f"template directory not found: {template_dir}")
    if not (template_dir / "swap_test").exists():
        raise FileNotFoundError(f"swap_test not found in template directory: {template_dir}")

    worker_script = (base_dir / "decision_candidate_worker.py").resolve()
    if not worker_script.exists():
        raise FileNotFoundError(f"worker script not found: {worker_script}")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    work_root = Path(args.work_root or f"decision_smoke_parallel_{timestamp}").resolve()
    if work_root.exists():
        raise FileExistsError(f"{work_root} already exists; choose another --work-root")
    work_root.mkdir(parents=True)

    jobs = max(1, min(args.jobs, len(IRRIGATION_OPTIONS_MM)))
    print(f"template: {template_dir}", flush=True)
    print(f"work root: {work_root}", flush=True)
    print(f"parallel jobs: {jobs}", flush=True)

    running: list[tuple[int, Path, subprocess.Popen]] = []
    for ir in IRRIGATION_OPTIONS_MM:
        wait_for_slot(running, jobs)
        work_dir = work_root / f"ir_{ir}"
        copy_template(template_dir, work_dir)
        proc = launch_candidate(worker_script, work_dir, ir)
        running.append((ir, work_dir, proc))
        print(f"candidate {ir} mm started: {work_dir}", flush=True)

    wait_for_all(running)

    out = collect_results(work_root)
    out_path = work_root / "decision_smoke_8ir_parallel.csv"
    out.to_csv(out_path, index=False)
    out.to_csv(base_dir / "decision_smoke_8ir_parallel_latest.csv", index=False)

    best = out.loc[out["target_value"].idxmax()]
    print("\nresults:", flush=True)
    print(out.to_string(index=False), flush=True)
    print("\nbest candidate:", flush=True)
    print(best.to_string(), flush=True)
    print(f"\nwrote {out_path}", flush=True)


if __name__ == "__main__":
    main()
