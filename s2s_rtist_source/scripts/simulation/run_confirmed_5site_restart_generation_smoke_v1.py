#!/usr/bin/env python3
"""Run a bounded restart/decision generation smoke check for confirmed sites.

This driver does not train a surrogate model. It copies each confirmed Maize
workspace into a timestamped run directory, runs a small restart-based decision
generation job, and summarizes whether each site can produce the 8-irrigation
candidate table needed by the later surrogate workflow.

Default scope:

- Sites: P1, P2, P3, P4, P15
- Decision dates: 16-Jul-2024 only
- Irrigation candidates: inherited from generate_restart_decision_dataset.py

Server example:

    cd /media/data_hot/lzx_projs/soil_moisture_otw/s2s_rtist_source
    export LD_LIBRARY_PATH=/media/data_hot/lzx_projs/soil_moisture_otw/s2s_rtist_source/local_libs/gcc_runtime/usr/lib/x86_64-linux-gnu:$LD_LIBRARY_PATH
    python3 project_cli.py run confirmed-5site-smoke --
"""

from __future__ import annotations

import argparse
from datetime import datetime
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys

import pandas as pd

from s2s_rtist.validation.three_output_smoke import (
    SmokeValidationError,
    validate_smoke_dataset,
    write_validation_outputs,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = Path("site_general_surrogate_eval")
CONFIRMED_WORKSPACES = OUT_DIR / "confirmed_5site_workspaces"
RUN_ROOT = OUT_DIR / "confirmed_5site_restart_generation_smoke_v1"
# Workspace copies keep the historical root filenames so isolated SWAP
# workspaces can import helpers without installing the package.
FORMAL_WORKSPACE_COPIES = (
    (PROJECT_ROOT / "src" / "s2s_rtist" / "pipelines" / "restart_decision_dataset.py", "generate_restart_decision_dataset.py"),
    (PROJECT_ROOT / "src" / "s2s_rtist" / "labels" / "swap_three_output_labels.py", "swap_three_output_labels_v1.py"),
    (PROJECT_ROOT / "src" / "s2s_rtist" / "physics" / "rootzone_flux_frequency.py", "rootzone_flux_frequency_diagnostic_v1.py"),
    (PROJECT_ROOT / "scripts" / "diagnostics" / "restart_raw_audit_v1.py", "restart_raw_audit_v1.py"),
)
DEFAULT_SOURCE_MAIZE = Path("model3_opt_sto_upload") / "Maize"

SITE_TO_WORKSPACE = {
    "P1": "P1_N1_Maize",
    "P2": "P2_N2_Maize",
    "P3": "P3_N3_Maize",
    "P4": "P4_N4_Maize",
    "P15": "P15_coord_12_Maize",
}

SITE_COORDINATES = {
    "P1": {"code_site_id": "N1", "longitude": -98.224144, "latitude": 42.015928},
    "P2": {"code_site_id": "N2", "longitude": -88.415, "latitude": 40.595},
    "P3": {"code_site_id": "N3", "longitude": -96.877, "latitude": 46.321},
    "P4": {"code_site_id": "N4", "longitude": -94.6686, "latitude": 42.6816},
    "P15": {"code_site_id": "coord_12", "longitude": -112.265, "latitude": 41.735},
}

DEFAULT_DECISION_DATES = ["16-Jul-2024:198"]
SITE_PLAN_FILE = "site_sampling_plan.csv"

RUNNER_SOURCE = r'''
from __future__ import annotations

import argparse
import os
import platform
import shutil
import subprocess
from pathlib import Path

import pandas as pd

from s2s_rtist.pipelines import restart_decision_dataset as base
import real_ir_update


def portable_format_date(date_obj) -> str:
    return f"{date_obj.day}-{date_obj.strftime('%b-%Y')}"


real_ir_update.format_date = portable_format_date


def parse_decision_date(text: str) -> tuple[str, int]:
    if ":" not in text:
        raise ValueError(f"Decision date must be DATE:DOY, got {text!r}")
    date_t, doy = text.split(":", 1)
    return date_t, int(doy)


def choose_swap_executable() -> Path:
    if platform.system().lower().startswith("win"):
        candidates = [Path.cwd() / "Swap.exe", Path.cwd() / "swap_test", Path.cwd() / "swap"]
    else:
        candidates = [Path.cwd() / "swap_test", Path.cwd() / "swap", Path.cwd() / "Swap.exe"]
    for exe in candidates:
        if exe.exists():
            if not platform.system().lower().startswith("win"):
                os.chmod(exe, exe.stat().st_mode | 0o111)
            return exe
    raise FileNotFoundError("No SWAP executable found: expected swap_test, swap, or Swap.exe")


def run_swap(log_name: str) -> None:
    if not Path(base.SWP_FILE).exists():
        raise FileNotFoundError(f"{base.SWP_FILE} was not created before running SWAP")

    exe = choose_swap_executable()
    with open(log_name, "w", encoding="utf-8", errors="ignore") as log:
        ret = subprocess.call(
            [str(exe.resolve())],
            cwd=str(Path.cwd()),
            stdout=log,
            stderr=subprocess.STDOUT,
        )

    text = Path(log_name).read_text(encoding="utf-8", errors="ignore")
    normal_completion = "normal completion" in text.lower()
    if ret != 0 and normal_completion:
        print(f"{log_name}: SWAP returned {ret}, but log says normal completion; continuing", flush=True)
    elif ret != 0:
        raise RuntimeError(f"SWAP failed with exit code {ret}; see {log_name}\n" + "\n".join(text.splitlines()[-40:]))
    elif not normal_completion:
        raise RuntimeError(f"SWAP did not report normal completion; see {log_name}\n" + "\n".join(text.splitlines()[-40:]))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--site", required=True)
    parser.add_argument("--decision-date", action="append", help="DATE:DOY, e.g. 16-Jul-2024:198")
    parser.add_argument("--sampling-plan", help="Per-site CSV with date_t, decision_doy, irrigation_mm columns.")
    args = parser.parse_args()

    base.run_swap = run_swap

    all_rows = []
    if args.sampling_plan:
        plan = pd.read_csv(args.sampling_plan)
        required = {"date_t", "decision_doy", "irrigation_mm"}
        missing = required.difference(plan.columns)
        if missing:
            raise ValueError(f"Sampling plan is missing columns: {sorted(missing)}")
        date_groups = []
        for (date_t, decision_doy), group in plan.groupby(["date_t", "decision_doy"], sort=False):
            irrigation_values = sorted(pd.to_numeric(group["irrigation_mm"], errors="coerce").dropna().unique())
            date_groups.append((str(date_t), int(decision_doy), irrigation_values))
    else:
        if not args.decision_date:
            raise ValueError("Pass --decision-date or --sampling-plan")
        date_groups = []
        for raw in args.decision_date:
            date_t, decision_doy = parse_decision_date(raw)
            date_groups.append((date_t, decision_doy, None))

    for date_t, decision_doy, irrigation_values in date_groups:
        print(f"[{args.site}] processing {date_t} DOY={decision_doy}", flush=True)
        df = base.run_one_date(date_t, decision_doy, irrigation_options_mm=irrigation_values)
        df.insert(0, "site", args.site)
        label = base.safe_label(date_t)
        if Path("result_forec.end").exists():
            shutil.copyfile("result_forec.end", f"result_pre_{label}.end")
        if Path("result_forec.crp").exists():
            shutil.copyfile("result_forec.crp", f"result_pre_{label}.crp")
        all_rows.append(df)

    dataset = pd.concat(all_rows, ignore_index=True)
    best = dataset[dataset["is_best_ir"]][
        ["site", "date_t", "decision_doy", "best_ir_for_date", "best_target_for_date"]
    ].drop_duplicates()

    dataset.to_csv("site_restart_generation_smoke.csv", index=False)
    best.to_csv("site_restart_generation_smoke_best_by_date.csv", index=False)

    print(f"[{args.site}] wrote site_restart_generation_smoke.csv rows={len(dataset)}", flush=True)
    print(best.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
'''


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sites", nargs="+", default=sorted(SITE_TO_WORKSPACE))
    parser.add_argument("--decision-dates", nargs="+", default=DEFAULT_DECISION_DATES)
    parser.add_argument("--sampling-plan", help="CSV with site_id, date_t, decision_doy, irrigation_mm columns.")
    parser.add_argument("--run-id", default=datetime.now().strftime("%Y%m%d_%H%M%S"))
    parser.add_argument("--timeout-per-site", type=int, default=1800)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument(
        "--allow-template-fallback",
        action="store_true",
        help="Debug only: use the generic Maize template when a confirmed site workspace is missing.",
    )
    return parser.parse_args()


def validate_decision_dates(raw_dates: list[str]) -> None:
    for raw in raw_dates:
        if ":" not in raw:
            raise ValueError(f"Decision date must be DATE:DOY, got {raw!r}")
        date_t, doy = raw.split(":", 1)
        if not date_t or not doy.isdigit():
            raise ValueError(f"Decision date must be DATE:DOY, got {raw!r}")
        actual_doy = datetime.strptime(date_t, "%d-%b-%Y").timetuple().tm_yday
        if int(doy) != int(actual_doy):
            raise ValueError(
                f"{raw} has mismatched DOY: {date_t} is DOY {actual_doy}, "
                f"but got {doy}. Use {date_t}:{actual_doy}."
            )


def site_aliases(site: str) -> set[str]:
    meta = SITE_COORDINATES[site]
    code_site_id = str(meta["code_site_id"])
    return {site, code_site_id, f"code_{code_site_id}"}


def prepare_site_sampling_plan(full_plan: pd.DataFrame, site: str, site_work: Path) -> tuple[Path, int, int]:
    required = {"site_id", "date_t", "decision_doy", "irrigation_mm"}
    missing = required.difference(full_plan.columns)
    if missing:
        raise ValueError(f"Sampling plan is missing columns: {sorted(missing)}")

    aliases = site_aliases(site)
    site_plan = full_plan[full_plan["site_id"].astype(str).isin(aliases)].copy()
    if site_plan.empty:
        raise ValueError(f"Sampling plan has no rows for {site}; accepted aliases: {sorted(aliases)}")

    site_plan_path = site_work / SITE_PLAN_FILE
    site_plan.to_csv(site_plan_path, index=False)
    n_dates = site_plan[["date_t", "decision_doy"]].drop_duplicates().shape[0]
    return site_plan_path, int(len(site_plan)), int(n_dates)


def copy_generator_files(workspace: Path) -> None:
    for source, target_name in FORMAL_WORKSPACE_COPIES:
        if not source.exists():
            raise FileNotFoundError(f"Missing formal dependency: {source}")
        shutil.copy2(source, workspace / target_name)
    (workspace / "run_restart_smoke_one_site.py").write_text(RUNNER_SOURCE, encoding="utf-8")


def source_workspace(site: str, *, allow_template_fallback: bool = False) -> Path:
    if site not in SITE_TO_WORKSPACE:
        raise ValueError(f"Unknown site {site!r}; expected one of {sorted(SITE_TO_WORKSPACE)}")
    source = CONFIRMED_WORKSPACES / SITE_TO_WORKSPACE[site]
    if source.exists():
        return source
    if allow_template_fallback and DEFAULT_SOURCE_MAIZE.exists():
        print(
            f"Confirmed workspace missing for {site}; falling back to {DEFAULT_SOURCE_MAIZE}",
            flush=True,
        )
        return DEFAULT_SOURCE_MAIZE
    raise FileNotFoundError(
        f"Missing confirmed workspace for {site}: {source}. "
        "Formal smoke runs do not use the generic Maize template."
    )
    return source


def write_site_config(site: str, workspace: Path, source: Path) -> None:
    meta = SITE_COORDINATES[site]
    config = {
        "paper_site_id": site,
        "code_site_id": meta["code_site_id"],
        "longitude": meta["longitude"],
        "latitude": meta["latitude"],
        "source_workspace": str(source),
        "generation_smoke_note": (
            "This run workspace was prepared by run_confirmed_5site_restart_generation_smoke_v1.py. "
            "If source_workspace is model3_opt_sto_upload/Maize, the run is a workflow smoke fallback "
            "rather than a fully regenerated site-specific SWAP input directory."
        ),
    }
    (workspace / "site_config.json").write_text(
        json.dumps(config, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    pd.DataFrame([config]).to_csv(workspace / "site_config.csv", index=False)


def prepend_server_runtime_library(env: dict[str, str]) -> dict[str, str]:
    src = str(PROJECT_ROOT / "src")
    existing_pythonpath = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = src + (os.pathsep + existing_pythonpath if existing_pythonpath else "")
    if platform.system().lower().startswith("win"):
        return env
    local_lib = Path.cwd() / "local_libs" / "gcc_runtime" / "usr" / "lib" / "x86_64-linux-gnu"
    if local_lib.exists():
        existing = env.get("LD_LIBRARY_PATH", "")
        env["LD_LIBRARY_PATH"] = str(local_lib) + (":" + existing if existing else "")
    return env


def run_site(
    site: str,
    site_work: Path,
    decision_dates: list[str],
    timeout: int,
    python_exe: str,
    use_sampling_plan: bool = False,
) -> dict:
    cmd = [python_exe, "run_restart_smoke_one_site.py", "--site", site]
    if use_sampling_plan:
        cmd.extend(["--sampling-plan", SITE_PLAN_FILE])
    else:
        for raw in decision_dates:
            cmd.extend(["--decision-date", raw])

    env = prepend_server_runtime_library(dict(os.environ))
    started_at = datetime.now().isoformat(timespec="seconds")
    try:
        result = subprocess.run(
            cmd,
            cwd=site_work,
            capture_output=True,
            text=True,
            timeout=timeout,
            env=env,
        )
        status = "completed" if result.returncode == 0 else "returned_nonzero"
        returncode: str | int = result.returncode
        stdout_tail = result.stdout[-4000:]
        stderr_tail = result.stderr[-4000:]
    except subprocess.TimeoutExpired as exc:
        status = "timeout"
        returncode = ""
        stdout_tail = (exc.stdout or "")[-4000:] if isinstance(exc.stdout, str) else ""
        stderr_tail = (exc.stderr or "")[-4000:] if isinstance(exc.stderr, str) else ""

    dataset = site_work / "site_restart_generation_smoke.csv"
    best = site_work / "site_restart_generation_smoke_best_by_date.csv"
    n_rows = ""
    n_best = ""
    if dataset.exists():
        n_rows = len(pd.read_csv(dataset))
    if best.exists():
        n_best = len(pd.read_csv(best))

    return {
        "site": site,
        "run_workspace": str(site_work),
        "status": status,
        "returncode": returncode,
        "started_at": started_at,
        "decision_dates": ";".join(decision_dates),
        "candidate_rows": n_rows,
        "best_rows": n_best,
        "dataset_csv": str(dataset) if dataset.exists() else "",
        "best_csv": str(best) if best.exists() else "",
        "stdout_tail": stdout_tail,
        "stderr_tail": stderr_tail,
    }


def markdown_table(df: pd.DataFrame) -> str:
    if df.empty:
        return "_No rows._"
    cols = list(df.columns)
    rows = ["| " + " | ".join(cols) + " |", "| " + " | ".join(["---"] * len(cols)) + " |"]
    for row in df.itertuples(index=False):
        rows.append("| " + " | ".join(str(v) for v in row) + " |")
    return "\n".join(rows)


def write_report(run_dir: Path, summary: pd.DataFrame, best_rows: pd.DataFrame, decision_dates: list[str]) -> Path:
    report = run_dir / "confirmed_5site_restart_generation_smoke_v1.md"
    summary_view_cols = [
        "site",
        "status",
        "returncode",
        "candidate_rows",
        "best_rows",
        "run_workspace",
    ]
    lines = [
        "# Confirmed 5-Site Restart Generation Smoke V1",
        "",
        "## Scope",
        "",
        "- Sites: P1, P2, P3, P4, P15 unless overridden.",
        f"- Decision dates: `{'; '.join(decision_dates)}`.",
        "- Each decision date should produce 8 irrigation candidate rows.",
        "- This is a bounded generation smoke check, not surrogate model training.",
        "",
        "## Site Run Summary",
        "",
        markdown_table(summary[summary_view_cols] if not summary.empty else summary),
        "",
        "## Best Irrigation By Site/Date",
        "",
        markdown_table(best_rows),
        "",
        "## Notes",
        "",
        "- A successful full default run should produce 40 candidate rows: 5 sites x 1 date x 8 irrigation options.",
        "- Formal label QA is written to `three_output_smoke_validation_v1.md` and its summary CSV.",
        "- On Linux, the script prepends `local_libs/gcc_runtime/usr/lib/x86_64-linux-gnu` to `LD_LIBRARY_PATH` when present.",
        "- Expand date count only after this smoke check completes for all five sites.",
    ]
    report.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report


def main() -> None:
    args = parse_args()
    sampling_plan = pd.read_csv(args.sampling_plan) if args.sampling_plan else None
    if args.sampling_plan:
        decision_dates = [
            f"{row.date_t}:{int(row.decision_doy)}"
            for row in sampling_plan[["date_t", "decision_doy"]].drop_duplicates().itertuples(index=False)
        ]
        validate_decision_dates(decision_dates)
    else:
        decision_dates = args.decision_dates
        validate_decision_dates(decision_dates)

    run_dir = RUN_ROOT / args.run_id
    run_dir.mkdir(parents=True, exist_ok=False)

    prep_rows = []
    for site in args.sites:
        source = source_workspace(
            site,
            allow_template_fallback=args.allow_template_fallback,
        )
        site_work = run_dir / site
        shutil.copytree(source, site_work)
        write_site_config(site, site_work, source)
        copy_generator_files(site_work)
        plan_rows = ""
        plan_dates = ""
        plan_path = ""
        if sampling_plan is not None:
            site_plan_path, plan_rows, plan_dates = prepare_site_sampling_plan(sampling_plan, site, site_work)
            plan_path = str(site_plan_path)
        prep_rows.append(
            {
                "site": site,
                "source_workspace": str(source),
                "run_workspace": str(site_work),
                "prepared": True,
                "sampling_plan_rows": plan_rows,
                "sampling_plan_dates": plan_dates,
                "sampling_plan_csv": plan_path,
            }
        )

    prep_df = pd.DataFrame(prep_rows)
    prep_df.to_csv(run_dir / "prepared_site_workspaces.csv", index=False)

    if args.prepare_only:
        summary_df = pd.DataFrame(
            [
                {
                    "site": row["site"],
                    "run_workspace": row["run_workspace"],
                    "status": "prepared_only",
                    "returncode": "",
                    "started_at": "",
                    "decision_dates": ";".join(decision_dates),
                    "candidate_rows": "",
                    "best_rows": "",
                    "dataset_csv": "",
                    "best_csv": "",
                    "stdout_tail": "",
                    "stderr_tail": "",
                }
                for row in prep_rows
            ]
        )
    else:
        summary_rows = []
        for row in prep_rows:
            print(f"Running restart generation smoke for {row['site']} in {row['run_workspace']}", flush=True)
            summary_rows.append(
                run_site(
                    site=row["site"],
                    site_work=Path(row["run_workspace"]),
                    decision_dates=decision_dates,
                    timeout=args.timeout_per_site,
                    python_exe=args.python,
                    use_sampling_plan=sampling_plan is not None,
                )
            )
        summary_df = pd.DataFrame(summary_rows)

    summary_df.to_csv(run_dir / "confirmed_5site_restart_generation_smoke_summary_v1.csv", index=False)

    best_frames = []
    for best_csv in summary_df["best_csv"].dropna():
        if best_csv and Path(best_csv).exists():
            best_frames.append(pd.read_csv(best_csv))
    best_df = pd.concat(best_frames, ignore_index=True) if best_frames else pd.DataFrame()
    best_df.to_csv(run_dir / "confirmed_5site_restart_generation_smoke_best_by_date_v1.csv", index=False)

    validation_paths: tuple[Path, Path] | None = None
    validation_error = ""
    if not args.prepare_only:
        try:
            incomplete = summary_df[summary_df["status"] != "completed"]
            if not incomplete.empty:
                failed_sites = ", ".join(incomplete["site"].astype(str))
                raise SmokeValidationError(
                    f"site runs did not complete: {failed_sites}"
                )
            dataset_frames = [
                pd.read_csv(path)
                for path in summary_df["dataset_csv"]
                if path and Path(path).exists()
            ]
            if len(dataset_frames) != len(args.sites):
                raise SmokeValidationError(
                    f"expected {len(args.sites)} site datasets, got {len(dataset_frames)}"
                )
            validation_result = validate_smoke_dataset(
                pd.concat(dataset_frames, ignore_index=True)
            )
            validation_paths = write_validation_outputs(validation_result, run_dir)
        except (OSError, ValueError, SmokeValidationError) as exc:
            validation_error = str(exc)

    report = write_report(run_dir, summary_df, best_df, decision_dates)

    latest_report = OUT_DIR / "confirmed_5site_restart_generation_smoke_v1_latest.md"
    latest_report.write_text(report.read_text(encoding="utf-8"), encoding="utf-8")

    print("Confirmed 5-site restart generation smoke v1")
    print(f"run_dir: {run_dir}")
    print(f"summary: {run_dir / 'confirmed_5site_restart_generation_smoke_summary_v1.csv'}")
    print(f"best: {run_dir / 'confirmed_5site_restart_generation_smoke_best_by_date_v1.csv'}")
    print(f"report: {report}")
    if validation_paths is not None:
        print(f"validation_summary: {validation_paths[0]}")
        print(f"validation_report: {validation_paths[1]}")
    if validation_error:
        raise RuntimeError(f"formal three-output smoke validation failed: {validation_error}")


if __name__ == "__main__":
    main()
