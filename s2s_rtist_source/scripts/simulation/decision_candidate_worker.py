"""Worker for one irrigation candidate in a copied Maize directory.

This script is launched by decision_smoke_8ir_parallel.py with cwd set to an
independent Maize work directory. It keeps the original smoke-test logic intact:
update irrigation -> run ForecastStep.run_sub1 -> parse result_forec.crp.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import pandas as pd


DECISION_DATE = "16-Jul-2024"
START_DOY = 61
DECISION_DOY = 198
END_DOY = DECISION_DOY + 7
YEAR = 2024

CRP_FILE = "gmaized.crp"
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


def quiet_swap_system(command: str) -> int:
    with open("swap_stdout.log", "w", encoding="utf-8", errors="ignore") as log:
        return subprocess.call(command, shell=True, stdout=log, stderr=subprocess.STDOUT)


def read_result_crp(path: str | Path = "result_forec.crp") -> pd.DataFrame:
    rows = []
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            if not line.startswith("2024-"):
                continue
            values = [v.strip() for v in line.rstrip("\n").split(",")]
            if len(values) == len(COLUMNS):
                rows.append(values)

    if not rows:
        raise RuntimeError(f"No crop output rows found in {path}")

    df = pd.DataFrame(rows, columns=COLUMNS)
    for col in ["Daynr", "DVS", "CWDM", "CWSO"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=["Daynr", "DVS", "CWDM", "CWSO"])
    if df.empty:
        raise RuntimeError(f"No numeric CWDM/CWSO/DVS rows found in {path}")
    return df


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ir", type=float, required=True)
    args = parser.parse_args()

    work_dir = Path.cwd()
    sys.path.insert(0, str(work_dir))

    import ForecastStep  # noqa: WPS433
    import real_ir_update  # noqa: WPS433

    if not Path("swap_test").exists():
        raise FileNotFoundError("swap_test not found; run inside a copied Maize directory.")

    ForecastStep.os.system = quiet_swap_system

    for swp_file in SWP_FILES_TO_UPDATE:
        real_ir_update.update_swp_file(swp_file, [DECISION_DATE], [args.ir])

    ForecastStep.run_sub1(
        START_DOY,
        YEAR,
        END_DOY,
        YEAR,
        DECISION_DOY,
        YEAR,
        CRP_FILE,
        SWP_FILE,
        divide=0,
    )

    last = read_result_crp("result_forec.crp").iloc[-1]
    row = {
        "date_t": DECISION_DATE,
        "ir": args.ir,
        "end_daynr": int(last["Daynr"]),
        "dvs": float(last["DVS"]),
        "cwdm_value": float(last["CWDM"]),
        "cwso_value": float(last["CWSO"]),
    }
    pd.DataFrame([row]).to_csv("candidate_result.csv", index=False)
    print(pd.Series(row).to_string())


if __name__ == "__main__":
    main()
