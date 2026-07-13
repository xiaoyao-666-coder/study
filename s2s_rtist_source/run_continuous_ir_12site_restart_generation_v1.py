#!/usr/bin/env python3
"""Run continuous-irrigation restart generation for prepared 12-site workspaces.

The runner consumes a sampling plan with site_id/date_t/decision_doy/
irrigation_mm columns. It expects workspaces prepared by
prepare_continuous_ir_12site_workspaces_v1.py and deliberately does not fall
back to the base Maize template, because fallback runs would pollute the
site-general surrogate training set.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime
import os
from pathlib import Path
import shutil
import subprocess
import sys

import pandas as pd


OUT_DIR = Path("site_general_surrogate_eval")
DEFAULT_WORKSPACE_ROOT = OUT_DIR / "continuous_ir_12site_workspaces_v1"
DEFAULT_RUN_ROOT = OUT_DIR / "continuous_ir_12site_restart_generation_v1"
DEFAULT_SITE_FEATURE_CSV = OUT_DIR / "site_feature_screening_12_code_sites.csv"
GENERATOR_SCRIPT = Path("generate_restart_decision_dataset.py")
THREE_OUTPUT_LABEL_SCRIPT = Path("swap_three_output_labels_v1.py")
SITE_PLAN_FILE = "site_sampling_plan.csv"
REAL_IR_HELPER_CANDIDATES = [
    Path("real_ir_update.py"),
    Path("model3_opt_sto_upload") / "Maize" / "real_ir_update.py",
    Path("rtist_minimal_work") / "Maize" / "real_ir_update.py",
]


RUNNER_SOURCE = r'''
from __future__ import annotations

import argparse
import os
import platform
import shutil
import subprocess
import traceback
from pathlib import Path

import pandas as pd

import generate_restart_decision_dataset as base
import real_ir_update


def portable_format_date(date_obj) -> str:
    return f"{date_obj.day}-{date_obj.strftime('%b-%Y')}"


real_ir_update.format_date = portable_format_date


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
    parser.add_argument("--sampling-plan", required=True)
    parser.add_argument("--continue-on-date-error", action="store_true")
    args = parser.parse_args()

    base.run_swap = run_swap
    plan = pd.read_csv(args.sampling_plan)
    required = {"date_t", "decision_doy", "irrigation_mm"}
    missing = required.difference(plan.columns)
    if missing:
        raise ValueError(f"Sampling plan is missing columns: {sorted(missing)}")

    all_rows = []
    error_rows = []
    for (date_t, decision_doy), group in plan.groupby(["date_t", "decision_doy"], sort=False):
        irrigation_values = sorted(pd.to_numeric(group["irrigation_mm"], errors="coerce").dropna().unique())
        print(f"[{args.site}] processing {date_t} DOY={int(decision_doy)} ir={irrigation_values}", flush=True)
        try:
            df = base.run_one_date(str(date_t), int(decision_doy), irrigation_options_mm=irrigation_values)
            df.insert(0, "site", args.site)
            label = base.safe_label(str(date_t))
            if Path("result_forec.end").exists():
                shutil.copyfile("result_forec.end", f"result_pre_{label}.end")
            if Path("result_forec.crp").exists():
                shutil.copyfile("result_forec.crp", f"result_pre_{label}.crp")
            all_rows.append(df)
            pd.concat(all_rows, ignore_index=True).to_csv("site_restart_generation_smoke.partial.csv", index=False)
        except Exception as exc:
            error_rows.append(
                {
                    "site": args.site,
                    "date_t": str(date_t),
                    "decision_doy": int(decision_doy),
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                    "traceback_tail": traceback.format_exc()[-4000:],
                }
            )
            pd.DataFrame(error_rows).to_csv("site_restart_generation_errors.csv", index=False)
            print(f"[{args.site}] ERROR on {date_t}: {type(exc).__name__}: {exc}", flush=True)
            if not args.continue_on_date_error:
                raise
            print(f"[{args.site}] continuing after failed date {date_t}", flush=True)

    if not all_rows:
        raise RuntimeError("No successful site-date rows; see site_restart_generation_errors.csv")

    dataset = pd.concat(all_rows, ignore_index=True)
    best = dataset[dataset["is_best_ir"]][
        ["site", "date_t", "decision_doy", "best_ir_for_date", "best_target_for_date"]
    ].drop_duplicates()

    dataset.to_csv("site_restart_generation_smoke.csv", index=False)
    best.to_csv("site_restart_generation_smoke_best_by_date.csv", index=False)

    print(f"[{args.site}] wrote site_restart_generation_smoke.csv rows={len(dataset)}", flush=True)
    if error_rows:
        print(f"[{args.site}] skipped failed site-dates={len(error_rows)}; see site_restart_generation_errors.csv", flush=True)
    print(best.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
'''


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sampling-plan", required=True)
    parser.add_argument("--site-feature-csv", default=str(DEFAULT_SITE_FEATURE_CSV))
    parser.add_argument("--workspace-root", default=str(DEFAULT_WORKSPACE_ROOT))
    parser.add_argument("--run-root", default=str(DEFAULT_RUN_ROOT))
    parser.add_argument("--run-id", default=datetime.now().strftime("%Y%m%d_%H%M%S"))
    parser.add_argument("--sites", nargs="+", help="Optional subset of site ids.")
    parser.add_argument("--timeout-per-site", type=int, default=7200)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--continue-on-date-error", action="store_true")
    return parser.parse_args()


def markdown_table(df: pd.DataFrame) -> str:
    if df.empty:
        return "_No rows._"
    cols = list(df.columns)
    rows = ["| " + " | ".join(cols) + " |", "| " + " | ".join(["---"] * len(cols)) + " |"]
    for row in df.itertuples(index=False):
        rows.append("| " + " | ".join(str(v) for v in row) + " |")
    return "\n".join(rows)


def paper_alignment_metadata(prep_df: pd.DataFrame, sampling_plan: Path) -> dict[str, object]:
    irrigation_values: list[float] = []
    if not prep_df.empty and "site_sampling_plan" in prep_df.columns:
        first_plan = Path(prep_df.iloc[0]["site_sampling_plan"])
        if first_plan.exists():
            site_plan = pd.read_csv(first_plan)
            if "irrigation_mm" in site_plan.columns:
                irrigation_values = sorted(pd.to_numeric(site_plan["irrigation_mm"], errors="coerce").dropna().unique().tolist())
    paper_list = [0.0, 10.0, 15.0, 20.0, 25.0, 30.0, 40.0, 60.0]
    return {
        "sampling_plan": str(sampling_plan),
        "weather_scenario_count": 1,
        "weather_scenario_label": "era5_single_scenario",
        "uses_paper_fixed_irrigation_list": irrigation_values == paper_list,
        "irrigation_option_count": len(irrigation_values),
        "irrigation_option_values_mm": irrigation_values,
        "paper_irrigation_option_values_mm": paper_list,
        "paper_alignment_note": (
            "Irrigation options match the paper fixed list when uses_paper_fixed_irrigation_list=true. "
            "Weather remains a single ERA5-driven scenario, not the paper 9-member S2S ensemble."
        ),
    }


def load_sites(site_feature_csv: Path, requested: list[str] | None) -> list[str]:
    if not site_feature_csv.exists():
        raise FileNotFoundError(f"Missing site feature CSV: {site_feature_csv}")
    df = pd.read_csv(site_feature_csv)
    if "site" not in df.columns:
        raise ValueError(f"{site_feature_csv} is missing required column: site")
    sites = [str(v) for v in df["site"].dropna().tolist()]
    if requested:
        missing = sorted(set(requested).difference(sites))
        if missing:
            raise ValueError(f"Requested sites not present in site feature CSV: {missing}")
        sites = requested
    return sites


def workspace_name(site: str) -> str:
    return f"{site}_Maize"


def site_aliases(site: str) -> set[str]:
    aliases = {site}
    if site.startswith("code_"):
        aliases.add(site.replace("code_", "", 1))
    else:
        aliases.add(f"code_{site}")
    return aliases


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


def copy_generator_files(site_work: Path) -> None:
    if not GENERATOR_SCRIPT.exists():
        raise FileNotFoundError(f"Missing generator script: {GENERATOR_SCRIPT}")
    if not THREE_OUTPUT_LABEL_SCRIPT.exists():
        raise FileNotFoundError(
            f"Missing three-output label helper: {THREE_OUTPUT_LABEL_SCRIPT}"
        )
    shutil.copyfile(GENERATOR_SCRIPT, site_work / GENERATOR_SCRIPT.name)
    shutil.copyfile(
        THREE_OUTPUT_LABEL_SCRIPT,
        site_work / THREE_OUTPUT_LABEL_SCRIPT.name,
    )

    helper_target = site_work / "real_ir_update.py"
    if not helper_target.exists():
        for candidate in REAL_IR_HELPER_CANDIDATES:
            if candidate.exists():
                shutil.copyfile(candidate, helper_target)
                break
        else:
            raise FileNotFoundError(
                "Missing helper script real_ir_update.py. Expected it in the "
                "prepared workspace, project root, model3_opt_sto_upload/Maize, "
                "or rtist_minimal_work/Maize."
            )
    (site_work / "run_restart_continuous_ir_one_site.py").write_text(RUNNER_SOURCE, encoding="utf-8")


def source_workspace(site: str, workspace_root: Path) -> Path:
    source = workspace_root / workspace_name(site)
    if not source.exists():
        raise FileNotFoundError(
            f"Missing prepared workspace for {site}: {source}. "
            "Run prepare_continuous_ir_12site_workspaces_v1.py first."
        )
    return source


def prepend_server_runtime_library(env: dict[str, str]) -> dict[str, str]:
    lib = (
        Path.cwd()
        / "local_libs"
        / "gcc_runtime"
        / "usr"
        / "lib"
        / "x86_64-linux-gnu"
    )
    if lib.exists():
        current = env.get("LD_LIBRARY_PATH", "")
        env["LD_LIBRARY_PATH"] = str(lib) + (":" + current if current else "")
    return env


def run_site(
    *,
    site: str,
    site_work: Path,
    sampling_plan: Path,
    python_exe: str,
    timeout: int,
    continue_on_date_error: bool,
) -> dict[str, object]:
    cmd = [
        python_exe,
        "run_restart_continuous_ir_one_site.py",
        "--site",
        site,
        "--sampling-plan",
        str(sampling_plan.name),
    ]
    if continue_on_date_error:
        cmd.append("--continue-on-date-error")
    env = prepend_server_runtime_library(os.environ.copy())
    site_log = site_work / "site_runner_stdout_stderr.log"
    try:
        with site_log.open("w", encoding="utf-8", errors="ignore") as log:
            result = subprocess.run(
                cmd,
                cwd=site_work,
                stdout=log,
                stderr=subprocess.STDOUT,
                text=True,
                timeout=timeout,
                env=env,
            )
        status = "completed" if result.returncode == 0 else "returned_nonzero"
    except subprocess.TimeoutExpired as exc:
        text = site_log.read_text(encoding="utf-8", errors="ignore") if site_log.exists() else ""
        return {
            "site": site,
            "run_workspace": str(site_work),
            "status": "timeout",
            "returncode": "",
            "site_log": str(site_log),
            "stdout_tail": text[-4000:],
            "stderr_tail": "",
            "rows": 0,
            "error_rows": 0,
        }

    dataset = site_work / "site_restart_generation_smoke.csv"
    rows = int(pd.read_csv(dataset).shape[0]) if dataset.exists() else 0
    errors = site_work / "site_restart_generation_errors.csv"
    error_rows = int(pd.read_csv(errors).shape[0]) if errors.exists() else 0
    text = site_log.read_text(encoding="utf-8", errors="ignore") if site_log.exists() else ""
    return {
        "site": site,
        "run_workspace": str(site_work),
        "status": status,
        "returncode": result.returncode,
        "site_log": str(site_log),
        "stdout_tail": text[-4000:],
        "stderr_tail": "",
        "rows": rows,
        "error_rows": error_rows,
    }


def write_report(run_dir: Path, prep_df: pd.DataFrame, summary_df: pd.DataFrame) -> Path:
    report = run_dir / "continuous_ir_12site_restart_generation_v1.md"
    metadata_path = run_dir / "continuous_ir_12site_restart_generation_metadata_v1.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8")) if metadata_path.exists() else {}
    lines = [
        "# Continuous Irrigation 12-Site Restart Generation V1",
        "",
        f"- run dir: `{run_dir}`",
        f"- prepared sites: `{len(prep_df)}`",
        f"- completed sites: `{int((summary_df['status'] == 'completed').sum()) if not summary_df.empty else 0}`",
        f"- weather scenario label: `{metadata.get('weather_scenario_label', 'unknown')}`",
        f"- weather scenario count: `{metadata.get('weather_scenario_count', 'unknown')}`",
        f"- paper fixed irrigation list matched: `{metadata.get('uses_paper_fixed_irrigation_list', 'unknown')}`",
        f"- irrigation option values mm: `{metadata.get('irrigation_option_values_mm', [])}`",
        "",
        "## Prepared Workspaces",
        markdown_table(prep_df),
        "",
        "## Run Summary",
        markdown_table(
            summary_df[
                [
                    "site",
                    "status",
                    "returncode",
                    "rows",
                    "error_rows",
                    "plan_rows",
                    "plan_dates",
                    "site_log",
                    "run_workspace",
                ]
            ]
        )
        if not summary_df.empty
        else "_No run rows._",
    ]
    report.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report


def main() -> None:
    args = parse_args()
    sampling_plan = Path(args.sampling_plan)
    workspace_root = Path(args.workspace_root)
    run_root = Path(args.run_root)
    run_dir = run_root / args.run_id

    if not sampling_plan.exists():
        raise FileNotFoundError(f"Missing sampling plan: {sampling_plan}")
    full_plan = pd.read_csv(sampling_plan)
    sites = load_sites(Path(args.site_feature_csv), args.sites)

    run_dir.mkdir(parents=True, exist_ok=True)
    prep_rows = []
    for site in sites:
        source = source_workspace(site, workspace_root)
        site_work = run_dir / site
        if site_work.exists():
            shutil.rmtree(site_work)
        shutil.copytree(source, site_work)
        copy_generator_files(site_work)
        site_plan_path, plan_rows, plan_dates = prepare_site_sampling_plan(full_plan, site, site_work)
        prep_rows.append(
            {
                "site": site,
                "source_workspace": str(source),
                "run_workspace": str(site_work),
                "site_sampling_plan": str(site_plan_path),
                "plan_rows": plan_rows,
                "plan_dates": plan_dates,
            }
        )

    prep_df = pd.DataFrame(prep_rows)
    prep_df.to_csv(run_dir / "prepared_site_workspaces.csv", index=False)
    metadata_path = run_dir / "continuous_ir_12site_restart_generation_metadata_v1.json"
    metadata = paper_alignment_metadata(prep_df, sampling_plan)
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")

    if args.prepare_only:
        empty = pd.DataFrame(columns=["site", "status", "returncode", "rows", "plan_rows", "plan_dates", "run_workspace"])
        report = write_report(run_dir, prep_df, empty)
        print("Continuous irrigation 12-site restart generation v1")
        print(f"run_dir: {run_dir}")
        print(f"prepared: {len(prep_df)}")
        print(f"prepare_only: True")
        print(f"report: {report}")
        return

    summary_rows = []
    for row in prep_df.itertuples(index=False):
        print(f"Running continuous-ir restart generation for {row.site}", flush=True)
        result = run_site(
            site=row.site,
            site_work=Path(row.run_workspace),
            sampling_plan=Path(row.site_sampling_plan),
            python_exe=args.python,
            timeout=args.timeout_per_site,
            continue_on_date_error=args.continue_on_date_error,
        )
        result["plan_rows"] = row.plan_rows
        result["plan_dates"] = row.plan_dates
        summary_rows.append(result)

    summary_df = pd.DataFrame(summary_rows)
    summary_path = run_dir / "continuous_ir_12site_restart_generation_summary_v1.csv"
    summary_df.to_csv(summary_path, index=False)

    dataset_frames = []
    best_frames = []
    candidate_error_frames = []
    for row in prep_df.itertuples(index=False):
        site_work = Path(row.run_workspace)
        dataset = site_work / "site_restart_generation_smoke.csv"
        best = site_work / "site_restart_generation_smoke_best_by_date.csv"
        candidate_errors = list(site_work.glob("restart_decision_candidate_errors_*.csv"))
        if dataset.exists():
            dataset_frames.append(pd.read_csv(dataset))
        if best.exists():
            best_frames.append(pd.read_csv(best))
        for candidate_error in candidate_errors:
            tmp = pd.read_csv(candidate_error)
            tmp.insert(0, "site", row.site)
            candidate_error_frames.append(tmp)
    if dataset_frames:
        pd.concat(dataset_frames, ignore_index=True).to_csv(
            run_dir / "continuous_ir_12site_restart_generation_merged_v1.csv",
            index=False,
        )
    if best_frames:
        pd.concat(best_frames, ignore_index=True).to_csv(
            run_dir / "continuous_ir_12site_restart_generation_best_by_date_v1.csv",
            index=False,
        )
    if candidate_error_frames:
        pd.concat(candidate_error_frames, ignore_index=True).to_csv(
            run_dir / "continuous_ir_12site_restart_generation_candidate_errors_v1.csv",
            index=False,
        )

    report = write_report(run_dir, prep_df, summary_df)
    print("Continuous irrigation 12-site restart generation v1")
    print(f"run_dir: {run_dir}")
    print(f"summary: {summary_path}")
    print(f"report: {report}")
    print(summary_df[["site", "status", "rows", "error_rows", "plan_rows", "plan_dates"]].to_string(index=False))


if __name__ == "__main__":
    main()
