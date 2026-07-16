"""Probe whether SWAP can restart from a saved .end state for 7-day decisions.

Run inside a copied Maize directory on the Linux server. The script compares:

1. full-season candidate run: 2024-03-01 -> 2024-07-23
2. restart candidate run:
   - base state: 2024-03-01 -> 2024-07-15
   - candidate: 2024-07-16 -> 2024-07-23 using the saved .end state

If the restart result matches the full-season result closely enough, later
dataset generation can avoid re-running from March for every irrigation option.
"""

from __future__ import annotations

import shutil
import subprocess
import sys
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
IRRIGATION_MM = 20
DECISION_DATE = "16-Jul-2024"

SWP_TEMPLATE = "Swap1.swp"
SWP_FILE = "swap.swp"
SWP_FILES_TO_UPDATE = ["SwapOriginal.swp", "Swap1.swp", "swap.swp"]
CURRENT_SWAP_LOG = "swap_run.log"

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
    if ret != 0 and not normal_completion:
        tail = text.splitlines()[-40:]
        raise RuntimeError(
            f"swap_test failed with exit code {ret}; see {log_name}\n"
            + "\n".join(tail)
        )
    if ret != 0 and normal_completion:
        print(f"swap_test returned exit code {ret}, but log says normal completion; continuing")


def quiet_swap_system(command: str) -> int:
    with open(CURRENT_SWAP_LOG, "w", encoding="utf-8", errors="ignore") as log:
        return subprocess.call(
            [command],
            cwd=str(Path.cwd()),
            stdout=log,
            stderr=subprocess.STDOUT,
        )


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
    if "normal completion" not in Path(log_name).read_text(encoding="utf-8", errors="ignore").lower():
        tail = Path(log_name).read_text(encoding="utf-8", errors="ignore").splitlines()[-40:]
        raise RuntimeError(f"ForecastStep/SWAP may have failed; see {log_name}\n" + "\n".join(tail))
    if not Path("result_forec.crp").exists() or not Path("result_forec.end").exists():
        raise FileNotFoundError("SWAP completed but result_forec.crp/result_forec.end was not found")


def set_swp(
    *,
    tstart_doy: int,
    tend_doy: int,
    crop_start_doy: int,
    crop_end_doy: int,
    swinco: int,
    inifil: str | None,
    outfil: str,
) -> None:
    lines = Path(SWP_TEMPLATE).read_text(encoding="utf-8", errors="ignore").splitlines(keepends=True)
    tstart = doy_to_swap_date(YEAR, tstart_doy)
    tend = doy_to_swap_date(YEAR, tend_doy)
    crop_start = doy_to_swap_date(YEAR, crop_start_doy)
    crop_end = doy_to_swap_date(YEAR, crop_end_doy)

    for i, line in enumerate(lines):
        if "TSTART  =" in line and "Start date of simulation run" in line:
            lines[i] = f"  TSTART  = {tstart} ! Start date of simulation run, give day-month-year, [dd-mmm-yyyy]\n"
        elif "TEND    =" in line and "End   date of simulation run" in line:
            lines[i] = f"  TEND    = {tend} ! End   date of simulation run, give day-month-year, [dd-mmm-yyyy]\n"
        elif "OUTFIL   =" in line and "Generic file name of output files" in line:
            lines[i] = f"  OUTFIL   = '{outfil}' ! Generic file name of output files, [A16]\n"
        elif "SWINCO =" in line and "type of initial soil moisture condition" in line:
            lines[i] = f" SWINCO = {swinco} ! Switch, type of initial soil moisture condition:\n"
        elif inifil and "INIFIL =" in line and "name of final with extension" in line:
            lines[i] = f"  INIFIL = '{inifil}'   ! name of final with extension [a200]\n"

    for i, line in enumerate(lines):
        if line.strip().startswith("INITCRP") and "CROPSTART" in line:
            lines[i + 1] = f"     2       {crop_start}    {crop_end}   'mais'    'gmaized'      2\n"
            break

    Path(SWP_FILE).write_text("".join(lines), encoding="utf-8")


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


def configure_irrigation(irrigation_mm: float | None) -> None:
    for swp_file in SWP_FILES_TO_UPDATE:
        if irrigation_mm is None:
            # SWAP still parses the table shape; an empty fixed-irrigation table
            # causes "Previous table is incomplete" even when SWIRFIX is 0.
            real_ir_update.modify_irrigation_swp(swp_file, 0)
        else:
            real_ir_update.modify_irrigation_swp(swp_file, 1)
            real_ir_update.update_swp_file(swp_file, [DECISION_DATE], [irrigation_mm])


def main() -> None:
    if not Path("swap_test").exists():
        raise FileNotFoundError("Run inside a copied Maize directory containing swap_test.")

    print("running full-season candidate")
    configure_irrigation(IRRIGATION_MM)
    run_forecaststep("restart_probe_full.log", END_DOY)
    shutil.copyfile("result_forec.crp", "result_full.crp")
    shutil.copyfile("result_forec.end", "result_full.end")
    full = read_last("result_full.crp")

    print("running base state to day before decision")
    configure_irrigation(None)
    run_forecaststep("restart_probe_pre.log", PRE_END_DOY)
    shutil.copyfile("result_forec.crp", "result_pre.crp")
    shutil.copyfile("result_forec.end", "result_pre.end")
    shutil.copyfile("result_forec.end", "restart_initial.end")

    print("running 7-day restart candidate")
    configure_irrigation(IRRIGATION_MM)
    set_swp(
        tstart_doy=DECISION_DOY,
        tend_doy=END_DOY,
        crop_start_doy=START_DOY,
        crop_end_doy=END_DOY,
        swinco=3,
        inifil="restart_initial.end",
        outfil="result_restart",
    )
    run_swap("restart_probe_restart.log")
    restart = read_last("result_restart.crp")

    rows = [
        {"mode": "full", **full},
        {"mode": "restart", **restart},
        {
            "mode": "restart_minus_full",
            "end_daynr": restart["end_daynr"] - full["end_daynr"],
            "dvs": restart["dvs"] - full["dvs"],
            "cwdm_value": restart["cwdm_value"] - full["cwdm_value"],
            "cwso_value": restart["cwso_value"] - full["cwso_value"],
        },
    ]
    out = pd.DataFrame(rows)
    out.to_csv("decision_restart_probe.csv", index=False)
    print(out.to_string(index=False))
    print("\nwrote decision_restart_probe.csv")


if __name__ == "__main__":
    main()
