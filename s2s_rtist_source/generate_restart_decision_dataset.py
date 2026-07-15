"""Generate a restart-based decision dataset for one Maize/SWAP setup.

This script uses the validated fast path:

1. For each decision date, run SWAP once to the day before the decision and
   save the .end state.
2. For each irrigation candidate, restart from that .end state and run the
   7-day decision horizon.
3. Score all candidates for the date and mark the best irrigation amount.

Run inside a clean copied Maize directory on the Linux server.
"""

from __future__ import annotations

import shutil
import subprocess
from datetime import datetime
from pathlib import Path

import pandas as pd

import ForecastStep
import real_ir_update
from restart_raw_audit_v1 import preserve_candidate_raw_outputs
from swap_three_output_labels_v1 import (
    extract_candidate_labels,
    flatten_candidate_labels,
    inclusive_horizon_end_doy,
    patch_nprintday_text,
)


YEAR = 2024
START_DOY = 61
HORIZON_DAYS = 7
RESTART_NPRINTDAY = 24

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

IRRIGATION_OPTIONS_MM = [0, 10, 15, 20, 25, 30, 40, 60]
PAPER_IRRIGATION_OPTIONS_MM = [0.0, 10.0, 15.0, 20.0, 25.0, 30.0, 40.0, 60.0]

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


def safe_ir_label(irrigation_mm: float) -> str:
    return str(irrigation_mm).replace("-", "m").replace(".", "p")


def irrigation_signature(irrigation_values: list[float]) -> str:
    return ",".join(f"{float(v):g}" for v in irrigation_values)


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


def run_pre_state(log_name: str, decision_doy: int) -> None:
    pre_end_doy = decision_doy - 1
    write_forecaststep_swp(decision_doy, pre_end_doy)
    run_swap(log_name)
    if not Path("result_forec.end").exists():
        raise FileNotFoundError("SWAP completed but result_forec.end was not found")


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

    swp_text = patch_nprintday_text("".join(lines), RESTART_NPRINTDAY)
    Path(SWP_FILE).write_text(swp_text, encoding="utf-8")


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
            if len(line) < 5 or not line[:4].isdigit() or line[4] != "-":
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
    row = df.dropna(subset=["Daynr", "DVS", "CWDM", "CWSO"]).iloc[-1]
    return {
        "end_daynr": int(row["Daynr"]),
        "dvs": float(row["DVS"]),
        "lai": float(row["LAI"]),
        "rootd": float(row["Rootd"]),
        "cwdm_value": float(row["CWDM"]),
        "cwso_value": float(row["CWSO"]),
    }


def score_one_date(rows: list[dict]) -> pd.DataFrame:
    out = pd.DataFrame(rows).sort_values("ir").reset_index(drop=True)
    cwdm0 = float(out.loc[out["ir"] == 0, "cwdm_value"].iloc[0])
    out["target_value"] = (
        (out["cwdm_value"] - cwdm0) * YIELD_PRICE_PER_KG
        - out["ir"] * WATER_COST_PER_HA_PER_MM * WEIGHT_INDEX
    )
    out.loc[out["ir"] == 0, "target_value"] = 0.0
    out["net_gain_7d"] = out["target_value"]
    best = out.loc[out["target_value"].idxmax()]
    out["best_ir_for_date"] = float(best["ir"])
    out["best_target_for_date"] = float(best["target_value"])
    out["is_best_ir"] = out["ir"] == best["ir"]
    return out


def write_candidate_artifacts(label: str, rows: list[dict], irrigation_options: list[float], candidate_errors: list[dict]) -> None:
    if rows:
        partial = pd.DataFrame(rows).sort_values("ir").reset_index(drop=True)
        partial["weather_scenario_count"] = 1
        partial["weather_scenario_label"] = "single_scenario"
        partial["irrigation_option_count"] = len(irrigation_options)
        partial["irrigation_option_values_mm"] = irrigation_signature(irrigation_options)
        partial["matches_paper_irrigation_option_list"] = int(
            [float(v) for v in irrigation_options] == PAPER_IRRIGATION_OPTIONS_MM
        )
        partial.to_csv(f"restart_decision_dataset_{label}.partial.csv", index=False)
    if candidate_errors:
        pd.DataFrame(candidate_errors).to_csv(f"restart_decision_candidate_errors_{label}.csv", index=False)


def run_one_date(date_t: str, decision_doy: int, irrigation_options_mm: list[float] | None = None) -> pd.DataFrame:
    label = safe_label(date_t)
    end_doy = inclusive_horizon_end_doy(decision_doy, HORIZON_DAYS)
    irrigation_options = irrigation_options_mm if irrigation_options_mm is not None else IRRIGATION_OPTIONS_MM
    irrigation_options = [float(v) for v in irrigation_options]

    print(f"\n=== {date_t}: pre-decision state ===", flush=True)
    configure_irrigation(date_t, None)
    run_pre_state(f"dataset_{label}_pre.log", decision_doy)
    shutil.copyfile("result_forec.end", "restart_initial.end")
    shutil.copyfile("result_forec.end", f"restart_initial_{label}.end")

    rows = []
    candidate_errors = []
    for ir in irrigation_options:
        print(f"{date_t}: running restart candidate {ir} mm", flush=True)
        try:
            configure_irrigation(date_t, ir)
            set_swp_for_restart(decision_doy, end_doy, outfil="result_restart")
            run_swap(f"dataset_{label}_restart_ir_{safe_ir_label(float(ir))}.log")
            physical_labels = extract_candidate_labels(
                pre_crop_path=Path("result_forec.crp"),
                pre_profile_path=Path("result_forec.vap"),
                restart_crop_path=Path("result_restart.crp"),
                restart_profile_path=Path("result_restart.vap"),
                restart_increment_path=Path("result_restart.inc"),
                decision_date=date_t,
                horizon_days=HORIZON_DAYS,
                nprintday=RESTART_NPRINTDAY,
            )
            raw_audit_dir = preserve_candidate_raw_outputs(
                date_t=date_t,
                decision_doy=decision_doy,
                irrigation_mm=ir,
                irrigation_options_mm=irrigation_options,
                nprintday=RESTART_NPRINTDAY,
            )
            rows.append(
                {
                    "date_t": date_t,
                    "decision_doy": decision_doy,
                    "horizon_end_doy": end_doy,
                    "horizon_days": HORIZON_DAYS,
                    "ir": ir,
                    "swap_version": "4.0.1",
                    "water_depth_unit": "mm",
                    "flux_rate_source_unit": "cm/day",
                    "root_depth_unit": "cm",
                    "soil_vwc_0_100cm_unit": "cm3/cm3",
                    "control_volume_type": "fixed_0_100cm",
                    "control_depth_cm": 100.0,
                    "data_processing_spec_version": "three_output_surrogate_data_processing_spec_v1_fixed_0_100cm",
                    "raw_audit_preserved": int(raw_audit_dir is not None),
                    "raw_audit_dir": str(raw_audit_dir) if raw_audit_dir else "",
                    **read_last("result_restart.crp"),
                    **flatten_candidate_labels(physical_labels),
                }
            )
        except Exception as exc:
            candidate_errors.append(
                {
                    "date_t": date_t,
                    "decision_doy": decision_doy,
                    "horizon_end_doy": end_doy,
                    "ir": ir,
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                }
            )
    if candidate_errors:
        write_candidate_artifacts(label, rows, irrigation_options, candidate_errors)
        raise RuntimeError(
            f"{date_t}: {len(candidate_errors)} candidate runs failed; "
            f"partial_rows={len(rows)}; see restart_decision_candidate_errors_{label}.csv"
        )
    scored = score_one_date(rows)
    scored["weather_scenario_count"] = 1
    scored["weather_scenario_label"] = "single_scenario"
    scored["irrigation_option_count"] = len(irrigation_options)
    scored["irrigation_option_values_mm"] = irrigation_signature(irrigation_options)
    scored["matches_paper_irrigation_option_list"] = int(irrigation_options == PAPER_IRRIGATION_OPTIONS_MM)
    scored.to_csv(f"restart_decision_dataset_{label}.csv", index=False)
    return scored


def main() -> None:
    if not Path("swap_test").exists():
        raise FileNotFoundError("Run inside a copied Maize directory containing swap_test.")

    all_rows = []
    for date_t, decision_doy in DECISION_DATES:
        all_rows.append(run_one_date(date_t, decision_doy))

    dataset = pd.concat(all_rows, ignore_index=True)
    best = dataset[dataset["is_best_ir"]][
        ["date_t", "decision_doy", "best_ir_for_date", "best_target_for_date"]
    ].drop_duplicates()

    dataset.to_csv("restart_decision_dataset.csv", index=False)
    best.to_csv("restart_decision_best_by_date.csv", index=False)

    print("\nbest irrigation by date:", flush=True)
    print(best.to_string(index=False), flush=True)
    print("\nwrote restart_decision_dataset.csv", flush=True)
    print("wrote restart_decision_best_by_date.csv", flush=True)


if __name__ == "__main__":
    main()
