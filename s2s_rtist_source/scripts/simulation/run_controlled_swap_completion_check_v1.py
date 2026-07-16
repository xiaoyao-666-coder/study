"""Run a bounded short-period SWAP completion check in an isolated workspace.

Designed for the server environment. It copies one Maize workspace, patches
`swap.swp` to a short TSTART/TEND window, runs SWAP, and writes a compact
completion report. It does not train any surrogate model.

Example on server:

    cd /media/data_hot/lzx_projs/soil_moisture_otw/s2s_rtist_source
    python run_controlled_swap_completion_check_v1.py --site P1 --days 5 --timeout 180
"""

from __future__ import annotations

import argparse
from datetime import datetime, timedelta
import os
from pathlib import Path
import platform
import re
import shutil
import subprocess
import time

import pandas as pd


OUT_DIR = Path("site_general_surrogate_eval")
WORKSPACE_ROOT = OUT_DIR / "controlled_swap_completion_check_v1"
CONFIRMED_WORKSPACES = OUT_DIR / "confirmed_5site_workspaces"
DEFAULT_SOURCE_MAIZE = Path("model3_opt_sto_upload") / "Maize"

SITE_TO_WORKSPACE = {
    "P1": "P1_N1_Maize",
    "P2": "P2_N2_Maize",
    "P3": "P3_N3_Maize",
    "P4": "P4_N4_Maize",
    "P15": "P15_coord_12_Maize",
}


def dd_mmm_yyyy(dt: datetime) -> str:
    return dt.strftime("%d-%b-%Y").lower()


def pick_source_workspace(site: str, explicit_source: str | None) -> Path:
    if explicit_source:
        source = Path(explicit_source)
    else:
        source = CONFIRMED_WORKSPACES / SITE_TO_WORKSPACE.get(site, "")
        if not source.exists():
            source = DEFAULT_SOURCE_MAIZE
    if not source.exists():
        raise FileNotFoundError(f"Source workspace not found: {source}")
    return source


def patch_swap_period(swp_path: Path, tstart: datetime, tend: datetime) -> None:
    lines = swp_path.read_text(encoding="utf-8", errors="ignore").splitlines(keepends=True)
    out = []
    saw_start = False
    saw_end = False
    for line in lines:
        if re.match(r"\s*TSTART\s*=", line):
            out.append(
                f"  TSTART  = {dd_mmm_yyyy(tstart)} ! Start date of simulation run, give day-month-year, [dd-mmm-yyyy]\n"
            )
            saw_start = True
        elif re.match(r"\s*TEND\s*=", line):
            out.append(
                f"  TEND    = {dd_mmm_yyyy(tend)} ! End   date of simulation run, give day-month-year, [dd-mmm-yyyy]\n"
            )
            saw_end = True
        else:
            out.append(line)
    if not saw_start or not saw_end:
        raise RuntimeError(f"Could not find TSTART/TEND in {swp_path}")
    swp_path.write_text("".join(out), encoding="utf-8")


def choose_executable(workspace: Path, explicit_exe: str | None) -> Path:
    if explicit_exe:
        exe = workspace / explicit_exe
    elif platform.system().lower().startswith("win"):
        exe = workspace / "Swap.exe"
    else:
        exe = workspace / "swap_test"
        if not exe.exists():
            exe = workspace / "swap"
    if not exe.exists():
        raise FileNotFoundError(f"SWAP executable not found: {exe}")
    return exe


def modified_outputs(workspace: Path, start_time: float) -> list[str]:
    names = []
    for path in workspace.iterdir():
        if path.is_file() and path.stat().st_mtime >= start_time:
            if path.name.lower().startswith(("result", "swap")) or path.suffix.lower() in {".log", ".ok"}:
                names.append(path.name)
    return sorted(names)


def markdown_table(df: pd.DataFrame) -> str:
    cols = list(df.columns)
    rows = ["| " + " | ".join(cols) + " |", "| " + " | ".join(["---"] * len(cols)) + " |"]
    for row in df.itertuples(index=False):
        rows.append("| " + " | ".join(str(v) for v in row) + " |")
    return "\n".join(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--site", default="P1", choices=sorted(SITE_TO_WORKSPACE))
    parser.add_argument("--source-workspace", default=None)
    parser.add_argument("--executable", default=None, help="Executable name inside the copied workspace.")
    parser.add_argument("--start", default="2024-03-01", help="YYYY-MM-DD")
    parser.add_argument("--days", type=int, default=5)
    parser.add_argument("--timeout", type=int, default=180)
    args = parser.parse_args()

    tstart = datetime.strptime(args.start, "%Y-%m-%d")
    tend = tstart + timedelta(days=args.days - 1)
    source = pick_source_workspace(args.site, args.source_workspace)

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    work = WORKSPACE_ROOT / f"{args.site}_short_{args.days}d_{stamp}"
    WORKSPACE_ROOT.mkdir(parents=True, exist_ok=True)
    shutil.copytree(source, work)

    swp_path = work / "swap.swp"
    patch_swap_period(swp_path, tstart, tend)

    exe = choose_executable(work, args.executable)
    if not platform.system().lower().startswith("win"):
        current_mode = exe.stat().st_mode
        os.chmod(exe, current_mode | 0o111)

    run_start = time.time()
    started_at = datetime.now().isoformat(timespec="seconds")
    try:
        result = subprocess.run(
            [str(exe.resolve())],
            cwd=work,
            capture_output=True,
            text=True,
            timeout=args.timeout,
        )
        elapsed = time.time() - run_start
        status = "completed" if result.returncode == 0 else "returned_nonzero"
        returncode = result.returncode
        stdout_tail = result.stdout[-2000:]
        stderr_tail = result.stderr[-2000:]
    except subprocess.TimeoutExpired as exc:
        elapsed = time.time() - run_start
        status = "timeout"
        returncode = ""
        stdout_tail = (exc.stdout or "")[-2000:] if isinstance(exc.stdout, str) else ""
        stderr_tail = (exc.stderr or "")[-2000:] if isinstance(exc.stderr, str) else ""

    outputs = modified_outputs(work, run_start)
    row = {
        "site": args.site,
        "source_workspace": str(source),
        "run_workspace": str(work),
        "executable": str(exe),
        "tstart": dd_mmm_yyyy(tstart),
        "tend": dd_mmm_yyyy(tend),
        "days": args.days,
        "timeout_s": args.timeout,
        "status": status,
        "returncode": returncode,
        "elapsed_s": round(elapsed, 3),
        "started_at": started_at,
        "modified_output_count": len(outputs),
        "modified_outputs": ";".join(outputs),
        "stdout_tail": stdout_tail,
        "stderr_tail": stderr_tail,
    }
    df = pd.DataFrame([row])

    csv_out = WORKSPACE_ROOT / f"{args.site}_short_{args.days}d_completion_check_{stamp}.csv"
    md_out = WORKSPACE_ROOT / f"{args.site}_short_{args.days}d_completion_check_{stamp}.md"
    df.to_csv(csv_out, index=False)

    report = [
        "# Controlled SWAP Completion Check V1",
        "",
        "## Summary",
        markdown_table(
            df[
                [
                    "site",
                    "run_workspace",
                    "executable",
                    "tstart",
                    "tend",
                    "status",
                    "returncode",
                    "elapsed_s",
                    "modified_output_count",
                ]
            ]
        ),
        "",
        "## Modified Outputs",
        "",
        "\n".join(f"- `{name}`" for name in outputs) if outputs else "_No modified outputs detected._",
        "",
        "## Stdout Tail",
        "",
        "```text",
        stdout_tail.strip(),
        "```",
        "",
        "## Stderr Tail",
        "",
        "```text",
        stderr_tail.strip(),
        "```",
    ]
    md_out.write_text("\n".join(report) + "\n", encoding="utf-8")

    print("Controlled SWAP completion check v1")
    print(f"site: {args.site}")
    print(f"workspace: {work}")
    print(f"period: {dd_mmm_yyyy(tstart)} to {dd_mmm_yyyy(tend)}")
    print(f"status: {status}")
    print(f"returncode: {returncode}")
    print(f"elapsed_s: {elapsed:.3f}")
    print(f"modified_output_count: {len(outputs)}")
    print(f"csv: {csv_out}")
    print(f"md: {md_out}")


if __name__ == "__main__":
    main()
