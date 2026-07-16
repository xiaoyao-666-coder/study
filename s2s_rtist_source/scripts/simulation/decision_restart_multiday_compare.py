"""Compare full-season vs .end-restart decisions for multiple decision dates.

This is a small stability test before using restart-based SWAP runs for larger
surrogate-dataset generation.
"""

from __future__ import annotations

import shutil
import subprocess
from datetime import datetime
from pathlib import Path

import pandas as pd

import ForecastStep
import real_ir_update


YEAR = 2024
START_DOY = 61
DECISION_DATES = [
    ("16-Jul-2024", 198),
    ("20-Jul-2024", 202),
    ("24-Jul-2024", 206),
]
HORIZON_DAYS = 7
IRRIGATION_OPTIONS_MM = [0, 10, 15, 20, 25, 30, 40, 60]

YIELD_PRICE_PER_KG = 0.20
WATER_COST_PER_HA_PER_MM = 2.0
WEIGHT_INDEX = 0.7

SWP_TEMPLATE = "Swap1.swp"
SWP_FILE = "swap.swp"
SWP_FILES_TO_UPDATE = ["SwapOriginal.swp", "Swap1.swp", "swap.swp"]

COLUMNS = [
    "Date",
    "Daynr",
    "Daycrp",
    "DVS",
    "TSUM",
    "LAIpot",
    "LAI",
    "Height",
    "CrpFac",
    "RootdPot",
    "Rootd",
    "PWLV",
    "WLV",
    "PWST",
    "WST",
    "PWRT",
    "WRT",
    "CPWDM",
    "CWDM",
    "CPWSO",
    "CWSO",
    "PGRASSDM",
    "GRASSDM",
    "PMOWDM",
    "MOWDM",
    "PGRAZDM",
    "GRAZDM",
    "DWLVCROP",
    "DWLVSOIL",
    "DWST",
    "DWRT",
    "DWSO",
    "HarLosOrm",
]


def safe_label(date_t: str) -> str:
    return date_t.replace("-", "").lower()


def doy_to_swap_date(year: int, doy: int) -> str:
    return datetime.strptime(f"{year}-{doy}", "%Y-%j").strftime("%d-%b-%Y").lower()


def run_swap(log_name: str) -> None:
    if not Path(SWP_FILE).exists():
        raise FileNotFoundError(f"{SWP_FILE} was not created before running swap_test")
    with open(log_name, "w", encoding="utf-8", errors="ignore") as log:
        ret = subprocess.call(
            [str(Path.cwd() / "swap_test")],
            cwd=str(Path.cwd()),
            stdout=log,
            stderr=subprocess.STDOUT,
        )
    text = Path(log_name).read_text(encoding="utf-8", errors="ignore")
    normal_completion = "normal completion" in text.lower()
    if ret != 0 and normal_completion:
        print(f"{log_name}: swap_test returned exit code {ret}, but log says normal completion; continuing", flush=True)
    elif ret != 0:
        raise RuntimeError(f"swap_test failed with exit code {ret}; see {log_name}\n" + "\n".join(text.splitlines()[-40:]))
    elif not normal_completion:
        raise RuntimeError(f"swap_test did not report normal completion; see {log_name}\n" + "\n".join(text.splitlines()[-40:]))


def skip_swap_system(command: str) -> int:
    return 0


def write_forecaststep_swp(decision_doy: int, end_doy: int) -> None:
    ForecastStep.os.system = skip_swap_system
    ForecastStep.run_sub1(
        START_DOY,
        YEAR,
        end_doy,
        YEAR,
        decision_doy,
        YEAR,
        "gmaized.crp",
        SWP_FILE,
        divide=0,
    )
    if not Path(SWP_FILE).exists():
        raise FileNotFoundError(f"ForecastStep did not write {SWP_FILE}")


def run_forecaststep(log_name: str, decision_doy: int, end_doy: int) -> None:
    write_forecaststep_swp(decision_doy, end_doy)
    run_swap(log_name)
    if not Path("result_forec.crp").exists() or not Path("result_forec.end").exists():
        raise FileNotFoundError("SWAP completed but result_forec.crp/result_forec.end was not found")


def set_swp_for_restart(decision_doy: int, end_doy: int, outfil: str) -> None:
    lines = Path(SWP_TEMPLATE).read_text(encoding="utf-8", errors="ignore").splitlines(keepends=True)
    tstart = doy_to_swap_date(YEAR, decision_doy)
    tend = doy_to_swap_date(YEAR, end_doy)
    crop_start = doy_to_swap_date(YEAR, START_DOY)
    crop_end = doy_to_swap_date(YEAR, end_doy)

    for i, line in enumerate(lines):
        if "TSTART  =" in line and "Start date of simulation run" in line:
            lines[i] = f"  TSTART  = {tstart} ! Start date of simulation run, give day-month-year, [dd-mmm-yyyy]\n"
        elif "TEND    =" in line and "End   date of simulation run" in line:
            lines[i] = f"  TEND    = {tend} ! End   date of simulation run, give day-month-year, [dd-mmm-yyyy]\n"
        elif "OUTFIL   =" in line and "Generic file name of output files" in line:
            lines[i] = f"  OUTFIL   = '{outfil}' ! Generic file name of output files, [A16]\n"
        elif "SWINCO =" in line and "type of initial soil moisture condition" in line:
            lines[i] = " SWINCO = 3 ! Switch, type of initial soil moisture condition:\n"
        elif "INIFIL =" in line and "name of final with extension" in line:
            lines[i] = "  INIFIL = 'restart_initial.end'   ! name of final with extension [a200]\n"

    for i, line in enumerate(lines):
        if line.strip().startswith("INITCRP") and "CROPSTART" in line:
            lines[i + 1] = f"     2       {crop_start}    {crop_end}   'mais'    'gmaized'      2\n"
            break

    Path(SWP_FILE).write_text("".join(lines), encoding="utf-8")


def configure_irrigation(date_t: str, irrigation_mm: float | None) -> None:
    for swp_file in SWP_FILES_TO_UPDATE:
        if irrigation_mm is None:
            real_ir_update.modify_irrigation_swp(swp_file, 0)
        else:
            real_ir_update.modify_irrigation_swp(swp_file, 1)
            real_ir_update.update_swp_file(swp_file, [date_t], [irrigation_mm])


def read_last(path: str) -> dict:
    rows = []
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            if not line.startswith("2024-"):
                continue
            values = [v.strip() for v in line.rstrip("\n").split(",")]
            if len(values) == len(COLUMNS):
                rows.append(values)
    if not rows:
        raise RuntimeError(f"No crop rows found in {path}")
    df = pd.DataFrame(rows, columns=COLUMNS)
    for col in ["Daynr", "DVS", "CWDM", "CWSO"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    row = df.dropna(subset=["Daynr", "DVS", "CWDM", "CWSO"]).iloc[-1]
    return {
        "end_daynr": int(row["Daynr"]),
        "dvs": float(row["DVS"]),
        "cwdm_value": float(row["CWDM"]),
        "cwso_value": float(row["CWSO"]),
    }


def score_targets(df: pd.DataFrame, mode: str) -> pd.DataFrame:
    scored = []
    for date_t, group in df.groupby("date_t", sort=False):
        out = group.copy()
        cwdm0 = float(out.loc[out["ir"] == 0, "cwdm_value"].iloc[0])
        out["target_value"] = (
            (out["cwdm_value"] - cwdm0) * YIELD_PRICE_PER_KG
            - out["ir"] * WATER_COST_PER_HA_PER_MM * WEIGHT_INDEX
        )
        out.loc[out["ir"] == 0, "target_value"] = 0.0
        out.insert(0, "mode", mode)
        scored.append(out)
    return pd.concat(scored, ignore_index=True)


def run_one_date(date_t: str, decision_doy: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    label = safe_label(date_t)
    pre_end_doy = decision_doy - 1
    end_doy = decision_doy + HORIZON_DAYS

    print(f"\n=== {date_t}: shared pre-decision state ===", flush=True)
    configure_irrigation(date_t, None)
    run_forecaststep(f"multi_{label}_pre.log", decision_doy, pre_end_doy)
    shutil.copyfile("result_forec.end", "restart_initial.end")
    shutil.copyfile("result_forec.end", f"restart_initial_{label}.end")

    full_rows = []
    restart_rows = []
    for ir in IRRIGATION_OPTIONS_MM:
        print(f"{date_t}: running full candidate {ir} mm", flush=True)
        configure_irrigation(date_t, ir)
        run_forecaststep(f"multi_{label}_full_ir_{ir}.log", decision_doy, end_doy)
        full_rows.append({"date_t": date_t, "ir": ir, **read_last("result_forec.crp")})

        print(f"{date_t}: running restart candidate {ir} mm", flush=True)
        configure_irrigation(date_t, ir)
        set_swp_for_restart(decision_doy, end_doy, outfil="result_restart")
        run_swap(f"multi_{label}_restart_ir_{ir}.log")
        restart_rows.append({"date_t": date_t, "ir": ir, **read_last("result_restart.crp")})

    return pd.DataFrame(full_rows), pd.DataFrame(restart_rows)


def main() -> None:
    if not Path("swap_test").exists():
        raise FileNotFoundError("Run inside a copied Maize directory containing swap_test.")

    all_full = []
    all_restart = []
    for date_t, decision_doy in DECISION_DATES:
        full, restart = run_one_date(date_t, decision_doy)
        all_full.append(full)
        all_restart.append(restart)

    full = score_targets(pd.concat(all_full, ignore_index=True), "full")
    restart = score_targets(pd.concat(all_restart, ignore_index=True), "restart")
    merged = full.merge(restart, on=["date_t", "ir"], suffixes=("_full", "_restart"))
    for col in ["dvs", "cwdm_value", "cwso_value", "target_value"]:
        merged[f"{col}_diff"] = merged[f"{col}_restart"] - merged[f"{col}_full"]

    best_rows = []
    for date_t in full["date_t"].drop_duplicates():
        full_date = full[full["date_t"] == date_t]
        restart_date = restart[restart["date_t"] == date_t]
        best_full = full_date.loc[full_date["target_value"].idxmax()]
        best_restart = restart_date.loc[restart_date["target_value"].idxmax()]
        best_rows.append(
            {
                "date_t": date_t,
                "best_ir_full": best_full["ir"],
                "target_full": best_full["target_value"],
                "best_ir_restart": best_restart["ir"],
                "target_restart": best_restart["target_value"],
                "same_best_ir": best_full["ir"] == best_restart["ir"],
            }
        )
    best = pd.DataFrame(best_rows)

    full.to_csv("decision_restart_multiday_full.csv", index=False)
    restart.to_csv("decision_restart_multiday_restart.csv", index=False)
    merged.to_csv("decision_restart_multiday_compare.csv", index=False)
    best.to_csv("decision_restart_multiday_best.csv", index=False)

    print("\nbest-ir comparison:", flush=True)
    print(best.to_string(index=False), flush=True)
    print("\ndiff summary by date:", flush=True)
    summary = merged.groupby("date_t").agg(
        max_abs_cwdm_diff=("cwdm_value_diff", lambda s: float(s.abs().max())),
        max_abs_cwso_diff=("cwso_value_diff", lambda s: float(s.abs().max())),
        max_abs_target_diff=("target_value_diff", lambda s: float(s.abs().max())),
    )
    print(summary.to_string(), flush=True)
    print("\nwrote decision_restart_multiday_compare.csv and decision_restart_multiday_best.csv", flush=True)


if __name__ == "__main__":
    main()
