#!/usr/bin/env python3
"""Run SWAP to each decision eve and save true pre-decision state features.

Run inside a copied Maize/SWAP experiment directory that contains swap_test,
ForecastStep.py, real_ir_update.py, Swap1.swp, and crop/weather inputs.

This script does not rerun irrigation candidates. It only runs the base
simulation to decision_doy - 1 for each decision date, saves result_forec.crp
and result_forec.end, and writes current-state tables for the short-term
surrogate v1 dataset.
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
SWP_FILE = "swap.swp"
SWP_FILES_TO_UPDATE = ["SwapOriginal.swp", "Swap1.swp", "swap.swp"]

DECISION_DATES = [
    ("16-Jul-2024", 198),
    ("20-Jul-2024", 202),
    ("24-Jul-2024", 206),
    ("28-Jul-2024", 210),
    ("01-Aug-2024", 214),
    ("05-Aug-2024", 218),
    ("09-Aug-2024", 222),
    ("13-Aug-2024", 226),
    ("17-Aug-2024", 230),
    ("21-Aug-2024", 234),
]

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


def configure_no_irrigation(date_t: str) -> None:
    for swp_file in SWP_FILES_TO_UPDATE:
        real_ir_update.modify_irrigation_swp(swp_file, 0)


def read_last_crp(path: Path) -> dict:
    rows = []
    with path.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            if not line.startswith("2024-"):
                continue
            values = [v.strip() for v in line.rstrip("\n").split(",")]
            if len(values) == len(COLUMNS):
                rows.append(values)
    if not rows:
        raise RuntimeError(f"No crop rows found in {path}")

    df = pd.DataFrame(rows, columns=COLUMNS)
    numeric_cols = ["Daynr", "DVS", "LAI", "Rootd", "CWDM", "CWSO"]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    row = df.dropna(subset=["Daynr", "DVS", "LAI", "Rootd", "CWDM", "CWSO"]).iloc[-1]
    return {
        "state_daynr": int(row["Daynr"]),
        "state_dvs": float(row["DVS"]),
        "state_lai": float(row["LAI"]),
        "state_rootd": float(row["Rootd"]),
        "state_cwdm": float(row["CWDM"]),
        "state_cwso": float(row["CWSO"]),
    }


def markdown_table(df: pd.DataFrame) -> str:
    cols = list(df.columns)
    lines = [
        "| " + " | ".join(cols) + " |",
        "| " + " | ".join(["---"] * len(cols)) + " |",
    ]
    for row in df.itertuples(index=False):
        lines.append("| " + " | ".join(str(v) for v in row) + " |")
    return "\n".join(lines)


def main() -> None:
    if not Path("swap_test").exists():
        raise FileNotFoundError("Run inside a Maize/SWAP directory containing swap_test.")

    output_dir = Path("Maize_shortterm_surrogate_v1")
    output_dir.mkdir(exist_ok=True)

    rows = []
    for date_t, decision_doy in DECISION_DATES:
        label = safe_label(date_t)
        pre_end_doy = decision_doy - 1
        print(f"\n=== {date_t}: true pre-decision state to DOY {pre_end_doy} ===", flush=True)
        configure_no_irrigation(date_t)
        write_forecaststep_swp(decision_doy, pre_end_doy)
        run_swap(f"true_state_{label}_pre.log")

        crp_path = Path(f"result_pre_{label}.crp")
        end_path = Path(f"result_pre_{label}.end")
        shutil.copyfile("result_forec.crp", crp_path)
        shutil.copyfile("result_forec.end", end_path)

        rows.append(
            {
                "date_t": date_t,
                "decision_doy": decision_doy,
                "pre_end_doy": pre_end_doy,
                "state_source": str(crp_path),
                **read_last_crp(crp_path),
                "state_soil_water_layers_status": "pending_extract_from_swap_water_output",
            }
        )

    state = pd.DataFrame(rows)
    state_path = output_dir / "current_state_by_date_true.csv"
    report_path = output_dir / "true_current_state_extract_report.md"
    state.to_csv(state_path, index=False)

    report = [
        "# True Pre-Decision Current State Extract Report",
        "",
        f"- Decision dates: {len(state)}",
        f"- Output: `{state_path}`",
        "- Source: SWAP base run ending at decision_doy - 1.",
        "- Soil water layers are still pending and should be extracted from SWAP water output in the next step.",
        "",
        "## Extracted State",
        "",
        markdown_table(state),
        "",
    ]
    report_path.write_text("\n".join(report), encoding="utf-8")
    print(f"\nWrote {state_path}")
    print(f"Wrote {report_path}")


if __name__ == "__main__":
    main()
