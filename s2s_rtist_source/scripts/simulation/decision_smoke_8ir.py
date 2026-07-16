"""Smoke test for one decision date and eight irrigation candidates.

Run this script inside a copied Maize directory on the Linux server, e.g.

    cd /media/data_hot/lzx_projs/soil_moisture_otw/s2s_rtist_source/Maize_decision_smoke
    export LD_LIBRARY_PATH="/media/data_hot/lzx_projs/soil_moisture_otw/s2s_rtist_source/local_libs/gcc_runtime/usr/lib/x86_64-linux-gnu:$LD_LIBRARY_PATH"
    python decision_smoke_8ir.py

It verifies the core loop needed for surrogate-data generation:
update irrigation candidate -> run SWAP -> read CWDM/CWSO/DVS -> score target_value.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import pandas as pd

import ForecastStep
import real_ir_update


DECISION_DATE = "16-Jul-2024"
START_DOY = 61  # 2024-03-01
DECISION_DOY = 198  # 2024-07-16
END_DOY = DECISION_DOY + 7  # 2024-07-23
YEAR = 2024
IRRIGATION_OPTIONS_MM = [0, 10, 15, 20, 25, 30, 40, 60]

YIELD_PRICE_PER_KG = 0.20
WATER_COST_PER_HA_PER_MM = 2.0
WEIGHT_INDEX = 0.7

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
    """Replacement for ForecastStep.os.system that keeps SWAP output quiet."""
    with open("swap_last_stdout.log", "w", encoding="utf-8", errors="ignore") as log:
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


def run_candidate(irrigation_mm: float) -> dict:
    for swp_file in SWP_FILES_TO_UPDATE:
        real_ir_update.update_swp_file(swp_file, [DECISION_DATE], [irrigation_mm])

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

    df = read_result_crp("result_forec.crp")
    last = df.iloc[-1]
    return {
        "date_t": DECISION_DATE,
        "ir": irrigation_mm,
        "end_daynr": int(last["Daynr"]),
        "dvs": float(last["DVS"]),
        "cwdm_value": float(last["CWDM"]),
        "cwso_value": float(last["CWSO"]),
    }


def main() -> None:
    if not Path("swap_test").exists():
        raise FileNotFoundError("Run this script inside the Maize directory containing swap_test.")

    ForecastStep.os.system = quiet_swap_system

    results = []
    cwdm_ir0 = None
    for ir in IRRIGATION_OPTIONS_MM:
        print(f"running irrigation candidate {ir} mm")
        row = run_candidate(ir)
        if ir == 0:
            cwdm_ir0 = row["cwdm_value"]
            row["target_value"] = 0.0
        else:
            row["target_value"] = (
                (row["cwdm_value"] - cwdm_ir0) * YIELD_PRICE_PER_KG
                - ir * WATER_COST_PER_HA_PER_MM * WEIGHT_INDEX
            )
        results.append(row)

    out = pd.DataFrame(results)
    out = out[
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
    out.to_csv("decision_smoke_8ir.csv", index=False)

    best = out.loc[out["target_value"].idxmax()]
    print("\nresults:")
    print(out.to_string(index=False))
    print("\nbest candidate:")
    print(best.to_string())
    print("\nwrote decision_smoke_8ir.csv")


if __name__ == "__main__":
    main()
