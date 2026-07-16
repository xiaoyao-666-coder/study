#!/usr/bin/env python3
"""Run the approved 3-sample x 3-frequency SWAP diagnostic."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys

import pandas as pd

from s2s_rtist.physics.rootzone_flux_frequency import (
    analyze_case_outputs,
    patch_nprintday_text,
)


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_WORKSPACE_ROOT = (
    PROJECT_ROOT
    / "site_general_surrogate_eval"
    / "continuous_ir_12site_workspaces_v1"
)
DEFAULT_OUTPUT_ROOT = (
    PROJECT_ROOT
    / "site_general_surrogate_eval"
    / "rootzone_flux_frequency_validation_v1"
)
SUPPORT_FILES = [
    "generate_restart_decision_dataset.py",
    "swap_three_output_labels_v1.py",
    "rootzone_flux_frequency_diagnostic_v1.py",
    "run_rootzone_flux_frequency_validation_v1.py",
]
SWP_FILES = ["SwapOriginal.swp", "Swap1.swp", "swap.swp"]


@dataclass(frozen=True)
class ValidationCase:
    site: str
    decision_date: str
    decision_doy: int
    irrigation_mm: float
    nprintday: int


def validation_cases() -> list[ValidationCase]:
    cases: list[ValidationCase] = []
    for nprintday in [1, 4, 24]:
        cases.append(
            ValidationCase("code_C2", "16-Jul-2024", 198, 30.0, nprintday)
        )
        cases.append(
            ValidationCase("code_C2", "16-Jul-2024", 198, 60.0, nprintday)
        )
        cases.append(
            ValidationCase("code_N2", "15-May-2024", 136, 30.0, nprintday)
        )
    return cases


def case_directory_name(case: ValidationCase) -> str:
    date_label = datetime.strptime(case.decision_date, "%d-%b-%Y").strftime(
        "%Y%m%d"
    )
    irrigation_label = f"{case.irrigation_mm:g}".replace(".", "p")
    return (
        f"{case.site}_{date_label}_ir{irrigation_label}_npd{case.nprintday}"
    )


def patch_workspace_nprintday(workspace: Path, nprintday: int) -> None:
    for name in SWP_FILES:
        path = Path(workspace) / name
        if not path.exists():
            raise FileNotFoundError(f"missing SWAP config: {path}")
        text = path.read_text(encoding="utf-8", errors="ignore")
        path.write_text(
            patch_nprintday_text(text, nprintday),
            encoding="utf-8",
        )


def copy_support_files(workspace: Path) -> None:
    for name in SUPPORT_FILES:
        source = PROJECT_ROOT / name
        if not source.exists():
            raise FileNotFoundError(f"missing support file: {source}")
        shutil.copy2(source, workspace / name)


def choose_swap_executable() -> Path:
    if platform.system().lower().startswith("win"):
        names = ["Swap.exe", "swap_test", "swap"]
    else:
        names = ["swap_test", "swap", "Swap.exe"]
    for name in names:
        candidate = Path.cwd() / name
        if candidate.exists():
            if not platform.system().lower().startswith("win"):
                os.chmod(candidate, candidate.stat().st_mode | 0o111)
            return candidate
    raise FileNotFoundError("no SWAP executable found")


def portable_run_swap(log_name: str) -> None:
    executable = choose_swap_executable()
    with Path(log_name).open("w", encoding="utf-8", errors="ignore") as log:
        returncode = subprocess.call(
            [str(executable.resolve())],
            cwd=str(Path.cwd()),
            stdout=log,
            stderr=subprocess.STDOUT,
        )
    log_text = Path(log_name).read_text(encoding="utf-8", errors="ignore")
    normal_completion = "normal completion" in log_text.lower()
    if returncode != 0 and normal_completion:
        return
    if returncode != 0:
        raise RuntimeError(
            f"SWAP returned {returncode}; see {log_name}\n"
            + "\n".join(log_text.splitlines()[-40:])
        )
    if not normal_completion:
        raise RuntimeError(
            f"SWAP did not report normal completion; see {log_name}\n"
            + "\n".join(log_text.splitlines()[-40:])
        )


def import_local_generator():
    sys.path.insert(0, str(Path.cwd()))
    from s2s_rtist.pipelines import restart_decision_dataset as base

    base.run_swap = portable_run_swap
    return base


def assert_outputs(prefix: str, extensions: list[str]) -> None:
    missing = [extension for extension in extensions if not Path(f"{prefix}.{extension}").exists()]
    if missing:
        raise FileNotFoundError(f"missing {prefix} outputs: {missing}")


def run_pre_worker(decision_date: str, decision_doy: int) -> None:
    base = import_local_generator()
    base.YEAR = datetime.strptime(decision_date, "%d-%b-%Y").year
    base.configure_irrigation(decision_date, None)
    base.write_forecaststep_swp(decision_doy, decision_doy - 1)
    patch_workspace_nprintday(Path.cwd(), 1)
    base.run_swap("pre_state.log")
    assert_outputs("result_forec", ["end", "vap", "crp", "inc"])
    shutil.copy2("result_forec.end", "restart_initial.end")
    Path("pre_state_complete.json").write_text(
        json.dumps(
            {
                "decision_date": decision_date,
                "decision_doy": decision_doy,
                "completed_at": datetime.now().isoformat(timespec="seconds"),
            },
            indent=2,
        ),
        encoding="utf-8",
    )


def run_restart_worker(case: ValidationCase) -> None:
    base = import_local_generator()
    base.YEAR = datetime.strptime(case.decision_date, "%d-%b-%Y").year
    shutil.copy2("result_forec.end", "restart_initial.end")
    base.configure_irrigation(case.decision_date, case.irrigation_mm)
    patch_workspace_nprintday(Path.cwd(), case.nprintday)
    end_doy = base.inclusive_horizon_end_doy(case.decision_doy, base.HORIZON_DAYS)
    base.set_swp_for_restart(case.decision_doy, end_doy, outfil="result_restart")
    patch_workspace_nprintday(Path.cwd(), case.nprintday)
    base.run_swap("restart.log")
    assert_outputs("result_restart", ["vap", "crp", "inc", "bal", "wba"])

    result = analyze_case_outputs(
        pre_crop_path=Path("result_forec.crp"),
        pre_profile_path=Path("result_forec.vap"),
        restart_crop_path=Path("result_restart.crp"),
        restart_profile_path=Path("result_restart.vap"),
        restart_increment_path=Path("result_restart.inc"),
        decision_date=case.decision_date,
        nprintday=case.nprintday,
        horizon_days=base.HORIZON_DAYS,
    )
    summary = {
        "site": case.site,
        "irrigation_mm": case.irrigation_mm,
        **result.summary,
    }
    pd.DataFrame([summary]).to_csv("diagnostic_summary.csv", index=False)
    result.samples.to_csv("diagnostic_subdaily_samples.csv", index=False)
    result.daily.to_csv("diagnostic_daily_moving_boundary.csv", index=False)
    Path("diagnostic_metadata.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )


def run_subprocess(command: list[str], workspace: Path, log_name: str, timeout: int) -> None:
    log_path = workspace / log_name
    with log_path.open("w", encoding="utf-8", errors="ignore") as log:
        result = subprocess.run(
            command,
            cwd=workspace,
            stdout=log,
            stderr=subprocess.STDOUT,
            timeout=timeout,
        )
    if result.returncode != 0:
        text = log_path.read_text(encoding="utf-8", errors="ignore")
        raise RuntimeError(
            f"command failed in {workspace}; see {log_path}\n"
            + "\n".join(text.splitlines()[-60:])
        )


def prepare_workspace(source: Path, target: Path) -> None:
    if target.exists():
        raise FileExistsError(
            f"target already exists: {target}; use a new --run-id for a clean run"
        )
    shutil.copytree(source, target)
    copy_support_files(target)


def pre_key(case: ValidationCase) -> tuple[str, str, int]:
    return case.site, case.decision_date, case.decision_doy


def run_orchestrator(args: argparse.Namespace) -> Path:
    workspace_root = Path(args.workspace_root).resolve()
    output_root = Path(args.output_root).resolve()
    run_dir = output_root / args.run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    pre_root = run_dir / "pre_states"
    cases_root = run_dir / "cases"
    pre_root.mkdir()
    cases_root.mkdir()

    pre_directories: dict[tuple[str, str, int], Path] = {}
    for case in validation_cases():
        key = pre_key(case)
        if key in pre_directories:
            continue
        source = workspace_root / f"{case.site}_Maize"
        if not source.exists():
            raise FileNotFoundError(f"missing source workspace: {source}")
        date_label = datetime.strptime(case.decision_date, "%d-%b-%Y").strftime(
            "%Y%m%d"
        )
        pre_dir = pre_root / f"{case.site}_{date_label}"
        prepare_workspace(source, pre_dir)
        print(
            f"[pre] starting {case.site} {case.decision_date} in {pre_dir}",
            flush=True,
        )
        command = [
            args.python,
            "run_rootzone_flux_frequency_validation_v1.py",
            "--worker-mode",
            "pre",
            "--decision-date",
            case.decision_date,
            "--decision-doy",
            str(case.decision_doy),
        ]
        run_subprocess(command, pre_dir, "pre_worker_stdout_stderr.log", args.timeout)
        print(f"[pre] completed {case.site} {case.decision_date}", flush=True)
        pre_directories[key] = pre_dir

    summary_rows: list[pd.DataFrame] = []
    for case in validation_cases():
        source = workspace_root / f"{case.site}_Maize"
        case_dir = cases_root / case_directory_name(case)
        prepare_workspace(source, case_dir)
        print(f"[case] starting {case_directory_name(case)}", flush=True)
        pre_dir = pre_directories[pre_key(case)]
        for name in ["result_forec.end", "result_forec.vap", "result_forec.crp", "result_forec.inc"]:
            shutil.copy2(pre_dir / name, case_dir / name)
        command = [
            args.python,
            "run_rootzone_flux_frequency_validation_v1.py",
            "--worker-mode",
            "restart",
            "--site",
            case.site,
            "--decision-date",
            case.decision_date,
            "--decision-doy",
            str(case.decision_doy),
            "--irrigation-mm",
            str(case.irrigation_mm),
            "--nprintday",
            str(case.nprintday),
        ]
        run_subprocess(command, case_dir, "restart_worker_stdout_stderr.log", args.timeout)
        case_summary = pd.read_csv(case_dir / "diagnostic_summary.csv")
        summary_rows.append(case_summary)
        pd.concat(summary_rows, ignore_index=True).to_csv(
            run_dir / "rootzone_flux_frequency_validation_summary_v1.partial.csv",
            index=False,
        )
        residual = float(
            case_summary.iloc[0]["water_balance_residual_corrected_7d_mm"]
        )
        print(
            f"[case] completed {case_directory_name(case)} "
            f"corrected_residual_mm={residual:.6f}",
            flush=True,
        )

    summary = pd.concat(summary_rows, ignore_index=True)
    summary.to_csv(run_dir / "rootzone_flux_frequency_validation_summary_v1.csv", index=False)
    Path(run_dir / "run_manifest.json").write_text(
        json.dumps(
            {
                "run_id": args.run_id,
                "completed_at": datetime.now().isoformat(timespec="seconds"),
                "workspace_root": str(workspace_root),
                "case_count": len(summary),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return run_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace-root", default=str(DEFAULT_WORKSPACE_ROOT))
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument(
        "--run-id",
        default=datetime.now().strftime("validation_%Y%m%d_%H%M%S"),
    )
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--timeout", type=int, default=3600)
    parser.add_argument("--worker-mode", choices=["pre", "restart"])
    parser.add_argument("--site")
    parser.add_argument("--decision-date")
    parser.add_argument("--decision-doy", type=int)
    parser.add_argument("--irrigation-mm", type=float)
    parser.add_argument("--nprintday", type=int)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.worker_mode == "pre":
        run_pre_worker(args.decision_date, args.decision_doy)
        return
    if args.worker_mode == "restart":
        run_restart_worker(
            ValidationCase(
                site=args.site,
                decision_date=args.decision_date,
                decision_doy=args.decision_doy,
                irrigation_mm=args.irrigation_mm,
                nprintday=args.nprintday,
            )
        )
        return

    run_dir = run_orchestrator(args)
    print(f"completed root-zone flux-frequency validation: {run_dir}")


if __name__ == "__main__":
    main()
