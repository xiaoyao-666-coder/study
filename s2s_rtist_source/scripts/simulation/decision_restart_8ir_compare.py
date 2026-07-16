"""Compare full-season vs .end-restart decisions for eight irrigation options.

Run inside a clean copied Maize directory on the Linux server. It checks whether
the faster restart workflow preserves the irrigation decision ranking.
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
DECISION_DOY = 198
PRE_END_DOY = DECISION_DOY - 1
END_DOY = DECISION_DOY + 7
DECISION_DATE = "16-Jul-2024"
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


def write_forecaststep_swp(end_doy: int) -> None:
    ForecastStep.os.system = skip_swap_system
    ForecastStep.run_sub1(
        START_DOY,
        YEAR,
        end_doy,
        YEAR,
        DECISION_DOY,
        YEAR,
        "gmaized.crp",
        SWP_FILE,
        divide=0,
    )
    if not Path(SWP_FILE).exists():
        raise FileNotFoundError(f"ForecastStep did not write {SWP_FILE}")


def run_forecaststep(log_name: str, end_doy: int) -> None:
    write_forecaststep_swp(end_doy)
    run_swap(log_name)
    if not Path("result_forec.crp").exists() or not Path("result_forec.end").exists():
        raise FileNotFoundError("SWAP completed but result_forec.crp/result_forec.end was not found")


def set_swp_for_restart(outfil: str) -> None:
    lines = Path(SWP_TEMPLATE).read_text(encoding="utf-8", errors="ignore").splitlines(keepends=True)
    tstart = doy_to_swap_date(YEAR, DECISION_DOY)
    tend = doy_to_swap_date(YEAR, END_DOY)
    crop_start = doy_to_swap_date(YEAR, START_DOY)
    crop_end = doy_to_swap_date(YEAR, END_DOY)

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


def configure_irrigation(irrigation_mm: float | None) -> None:
    for swp_file in SWP_FILES_TO_UPDATE:
        if irrigation_mm is None:
            real_ir_update.modify_irrigation_swp(swp_file, 0)
        else:
            real_ir_update.modify_irrigation_swp(swp_file, 1)
            real_ir_update.update_swp_file(swp_file, [DECISION_DATE], [irrigation_mm])


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
    out = df.copy()
    cwdm0 = float(out.loc[out["ir"] == 0, "cwdm_value"].iloc[0])
    out["target_value"] = (
        (out["cwdm_value"] - cwdm0) * YIELD_PRICE_PER_KG
        - out["ir"] * WATER_COST_PER_HA_PER_MM * WEIGHT_INDEX
    )
    out.loc[out["ir"] == 0, "target_value"] = 0.0
    out.insert(0, "mode", mode)
    return out


def main() -> None:
    if not Path("swap_test").exists():
        raise FileNotFoundError("Run inside a copied Maize directory containing swap_test.")

    print("running shared pre-decision state", flush=True)
    configure_irrigation(None)
    run_forecaststep("compare_pre.log", PRE_END_DOY)
    shutil.copyfile("result_forec.end", "restart_initial.end")

    full_rows = []
    restart_rows = []
    for ir in IRRIGATION_OPTIONS_MM:
        print(f"running full candidate {ir} mm", flush=True)
        configure_irrigation(ir)
        run_forecaststep(f"compare_full_ir_{ir}.log", END_DOY)
        full_row = {"ir": ir, **read_last("result_forec.crp")}
        full_rows.append(full_row)
        shutil.copyfile("result_forec.crp", f"result_full_ir_{ir}.crp")

        print(f"running restart candidate {ir} mm", flush=True)
        configure_irrigation(ir)
        set_swp_for_restart(outfil="result_restart")
        run_swap(f"compare_restart_ir_{ir}.log")
        restart_row = {"ir": ir, **read_last("result_restart.crp")}
        restart_rows.append(restart_row)
        shutil.copyfile("result_restart.crp", f"result_restart_ir_{ir}.crp")

    full = score_targets(pd.DataFrame(full_rows), "full")
    restart = score_targets(pd.DataFrame(restart_rows), "restart")
    merged = full.merge(
        restart,
        on="ir",
        suffixes=("_full", "_restart"),
    )
    for col in ["dvs", "cwdm_value", "cwso_value", "target_value"]:
        merged[f"{col}_diff"] = merged[f"{col}_restart"] - merged[f"{col}_full"]

    full.to_csv("decision_restart_8ir_full.csv", index=False)
    restart.to_csv("decision_restart_8ir_restart.csv", index=False)
    merged.to_csv("decision_restart_8ir_compare.csv", index=False)

    print("\nfull results:", flush=True)
    print(full.to_string(index=False), flush=True)
    print("\nrestart results:", flush=True)
    print(restart.to_string(index=False), flush=True)
    print("\ndiff summary:", flush=True)
    print(merged[["ir", "cwdm_value_diff", "cwso_value_diff", "dvs_diff", "target_value_diff"]].to_string(index=False), flush=True)
    print("\nbest full:", flush=True)
    print(full.loc[full["target_value"].idxmax()].to_string(), flush=True)
    print("\nbest restart:", flush=True)
    print(restart.loc[restart["target_value"].idxmax()].to_string(), flush=True)
    print("\nwrote decision_restart_8ir_compare.csv", flush=True)


if __name__ == "__main__":
    main()
