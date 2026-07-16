#!/usr/bin/env python3
"""Apply confirmed 5-site POLARIS soil hydraulic inputs to SWAP workspaces.

This is the second true site-specific input layer after static SWP inputs. It
uses the existing POLARIS feature table extracted for the confirmed sites and
rebuilds each workspace's `df_polaris_soil_hydraulic.csv` plus the SWP
`ISOILLAY1` soil hydraulic table.

It intentionally reuses the original project transformations from
`Extract_tif.process_polaris_data`:

- alpha = 10 ** alpha_log
- ksat = 24 * (10 ** ksat_log)
- n = 5 * n_raw
- alphaw = 2 * alpha
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil

import pandas as pd


OUT_DIR = Path("site_general_surrogate_eval")
FEATURES = OUT_DIR / "site_feature_screening_12_code_sites.csv"
CONFIRMED_WORKSPACES = OUT_DIR / "confirmed_5site_workspaces"
DEFAULT_SOURCE_MAIZE = Path("model3_opt_sto_upload") / "Maize"
REPORT_CSV = OUT_DIR / "confirmed_5site_polaris_soil_input_application_v1.csv"
PROFILE_CSV = OUT_DIR / "confirmed_5site_polaris_soil_profiles_v1.csv"
REPORT_MD = OUT_DIR / "confirmed_5site_polaris_soil_input_application_v1.md"

SITE_TO_WORKSPACE = {
    "P1": "P1_N1_Maize",
    "P2": "P2_N2_Maize",
    "P3": "P3_N3_Maize",
    "P4": "P4_N4_Maize",
    "P15": "P15_coord_12_Maize",
}

SITE_TO_FEATURE = {
    "P1": "code_N1",
    "P2": "code_N2",
    "P3": "code_N3",
    "P4": "code_N4",
    "P15": "code_active",
}

DEPTHS = [
    ("0_5", 0),
    ("5_15", 5),
    ("15_30", 15),
    ("30_60", 30),
    ("60_100", 60),
    ("100_200", 100),
]

SWP_FILES = ["SwapOriginal.swp", "Swap1.swp", "swap.swp"]

EMBEDDED_PROFILES = [
    ("P1", "code_N1", 0, 0.0501339733600616, 0.4756147861480713, 0.5274284206721523, 1.745023727416992, 100.7682558027957, 1.0548568413443047),
    ("P1", "code_N1", 5, 0.0511137954890728, 0.4687872529029846, 0.5254285914528232, 1.7507289350032804, 98.62726395778984, 1.0508571829056463),
    ("P1", "code_N1", 15, 0.0513937026262283, 0.4551889896392822, 0.5069017014561265, 1.7498517036437986, 85.33790562961296, 1.013803402912253),
    ("P1", "code_N1", 30, 0.0510112456977367, 0.4558333456516266, 0.4946455370950675, 1.7605376243591306, 70.59279446057928, 0.989291074190135),
    ("P1", "code_N1", 60, 0.0508394353091716, 0.4597046971321106, 0.4485283860142286, 1.730131208896637, 59.773013281309105, 0.8970567720284572),
    ("P1", "code_N1", 100, 0.0544827170670032, 0.4492292702198028, 0.4119969798838674, 1.6886860132217405, 57.03007861836225, 0.8239939597677348),
    ("P15", "code_active", 0, 0.0584954991936683, 0.5160518884658813, 0.415560351294534, 1.6024954617023466, 64.8411671533729, 0.831120702589068),
    ("P15", "code_active", 5, 0.0588359758257865, 0.515586793422699, 0.4135920397361283, 1.59910187125206, 65.19170841203314, 0.8271840794722566),
    ("P15", "code_active", 15, 0.0602730959653854, 0.5138465762138367, 0.4067656804648812, 1.5864324569702146, 60.75159587617409, 0.8135313609297624),
    ("P15", "code_active", 30, 0.0601534582674503, 0.5088334083557129, 0.37757189907637473, 1.570598483085632, 42.81429767545413, 0.7551437981527495),
    ("P15", "code_active", 60, 0.0504913218319416, 0.5071223378181458, 0.3887181915045565, 1.6188247501850126, 41.20550764723859, 0.777436383009113),
    ("P15", "code_active", 100, 0.0482074245810508, 0.5097353458404541, 0.4096385028309583, 1.6473180055618286, 60.064998779703444, 0.8192770056619166),
    ("P2", "code_N2", 0, 0.0793901979923248, 0.5066052675247192, 0.2481096206722904, 1.2986229360103605, 33.35032502585139, 0.4962192413445808),
    ("P2", "code_N2", 5, 0.0809100568294525, 0.504959225654602, 0.2496354872318057, 1.292121112346649, 32.88717927418077, 0.4992709744636114),
    ("P2", "code_N2", 15, 0.0862621217966079, 0.5025236010551453, 0.2516671745630198, 1.2708042562007904, 30.370072507663757, 0.5033343491260396),
    ("P2", "code_N2", 30, 0.101154588162899, 0.4820543229579925, 0.24075491787772707, 1.234391704201698, 19.907555204130016, 0.48150983575545414),
    ("P2", "code_N2", 60, 0.0903471186757087, 0.4343261420726776, 0.2255436763817234, 1.2682990729808805, 12.55878834455163, 0.4510873527634468),
    ("P2", "code_N2", 100, 0.0796822980046272, 0.3856716156005859, 0.23005966683605336, 1.3566061854362486, 9.49763843180105, 0.4601193336721067),
    ("P3", "code_N3", 0, 0.0671637952327728, 0.5343822240829468, 0.41941424110637116, 1.498132050037384, 43.76569731872699, 0.8388284822127423),
    ("P3", "code_N3", 5, 0.0679817274212837, 0.5300089120864868, 0.4206689022894804, 1.5048995614051819, 56.05334106112865, 0.8413378045789608),
    ("P3", "code_N3", 15, 0.0652087479829788, 0.5110798478126526, 0.40799280690438605, 1.5363804996013641, 58.3684350460602, 0.8159856138087721),
    ("P3", "code_N3", 30, 0.060776300728321, 0.4847043752670288, 0.4057363170026126, 1.583770215511322, 64.06782885470739, 0.8114726340052252),
    ("P3", "code_N3", 60, 0.0584152713418006, 0.4687479734420776, 0.4083255801113438, 1.6112156212329865, 70.36659227324691, 0.8166511602226876),
    ("P3", "code_N3", 100, 0.0601931288838386, 0.460406482219696, 0.36337302395725485, 1.5791940689086914, 58.61751893977974, 0.7267460479145097),
    ("P4", "code_N4", 0, 0.0905365198850631, 0.5654973387718201, 0.44464728833053346, 1.3591207563877106, 19.085775813023158, 0.8892945766610669),
    ("P4", "code_N4", 5, 0.09211216121912, 0.5581141710281372, 0.43369663289472926, 1.3490717113018036, 17.09943219600277, 0.8673932657894585),
    ("P4", "code_N4", 15, 0.0990772992372512, 0.5363020896911621, 0.403988792075674, 1.3311827182769775, 14.868023668949597, 0.807977584151348),
    ("P4", "code_N4", 30, 0.0930704921483993, 0.5018664002418518, 0.3619832390227635, 1.341039389371872, 12.829027408041922, 0.723966478045527),
    ("P4", "code_N4", 60, 0.0812915414571762, 0.4691485166549682, 0.38727412106327924, 1.436333507299423, 13.41165742456182, 0.7745482421265585),
    ("P4", "code_N4", 100, 0.0716632679104805, 0.4155083298683166, 0.36790624352910434, 1.5396615862846375, 17.15242997434262, 0.7358124870582087),
]


def file_hash(path: Path) -> str:
    if not path.exists():
        return ""
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()[:16]


def ensure_workspace(site: str, create_missing: bool) -> Path:
    workspace = CONFIRMED_WORKSPACES / SITE_TO_WORKSPACE[site]
    if workspace.exists():
        return workspace
    if not create_missing:
        raise FileNotFoundError(f"Missing workspace: {workspace}")
    if not DEFAULT_SOURCE_MAIZE.exists():
        raise FileNotFoundError(f"Missing base Maize template: {DEFAULT_SOURCE_MAIZE}")
    workspace.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(DEFAULT_SOURCE_MAIZE, workspace)
    return workspace


def build_polaris_profile(feature_row: pd.Series) -> pd.DataFrame:
    rows = []
    for suffix, start_depth_cm in DEPTHS:
        alpha = 10 ** float(feature_row[f"alpha_{suffix}"])
        ksat = 24.0 * (10 ** float(feature_row[f"ksat_{suffix}"]))
        rows.append(
            {
                "depth_start_cm": start_depth_cm,
                "theta_r_": float(feature_row[f"theta_r_{suffix}"]),
                "theta_s_": float(feature_row[f"theta_s_{suffix}"]),
                "alpha_": alpha,
                "n_": 5.0 * float(feature_row[f"n_{suffix}"]),
                "ksat_": ksat,
                "alphaw_": 2.0 * alpha,
            }
        )
    return pd.DataFrame(rows).set_index("depth_start_cm")


def build_embedded_profile(site: str) -> tuple[str, pd.DataFrame]:
    rows = []
    feature_site_key = SITE_TO_FEATURE[site]
    for embedded in EMBEDDED_PROFILES:
        embedded_site, embedded_key, depth, theta_r, theta_s, alpha, npar, ksat, alphaw = embedded
        if embedded_site != site:
            continue
        feature_site_key = embedded_key
        rows.append(
            {
                "depth_start_cm": depth,
                "theta_r_": theta_r,
                "theta_s_": theta_s,
                "alpha_": alpha,
                "n_": npar,
                "ksat_": ksat,
                "alphaw_": alphaw,
            }
        )
    if not rows:
        raise ValueError(f"No embedded POLARIS profile for site: {site}")
    return feature_site_key, pd.DataFrame(rows).set_index("depth_start_cm")


def profile_to_swp_rows(profile: pd.DataFrame) -> list[str]:
    rows = []
    for idx, row in enumerate(profile.itertuples(), start=1):
        rows.append(
            f"{idx} {row.theta_r_:.3f} {row.theta_s_:.3f} {row.alpha_:.4f} "
            f"{row.n_:.3f} {row.ksat_:.3f} 0.500 {row.alphaw_:.4f}  -4.0\n"
        )
    return rows


def update_swp_soil_table(path: Path, profile: pd.DataFrame) -> dict:
    if not path.exists():
        return {"file": path.name, "status": "missing", "sha256_16": ""}

    lines = path.read_text(encoding="utf-8", errors="ignore").splitlines(keepends=True)
    header_idx = None
    end_idx = None
    for i, line in enumerate(lines):
        if line.lstrip().startswith("ISOILLAY1") and "ORES" in line and "KSAT" in line:
            header_idx = i
            break
    if header_idx is None:
        return {"file": path.name, "status": "missing_isoillay1_header", "sha256_16": file_hash(path)}

    for j in range(header_idx + 1, len(lines)):
        if lines[j].startswith("* --- end of table"):
            end_idx = j
            break
    if end_idx is None:
        return {"file": path.name, "status": "missing_isoillay1_end", "sha256_16": file_hash(path)}

    new_lines = lines[: header_idx + 1] + profile_to_swp_rows(profile) + lines[end_idx:]
    path.write_text("".join(new_lines), encoding="utf-8")
    return {
        "file": path.name,
        "status": "updated",
        "soil_rows_written": len(profile),
        "sha256_16": file_hash(path),
    }


def write_workspace_profile(workspace: Path, site: str, profile: pd.DataFrame) -> str:
    csv_profile = profile[["theta_r_", "theta_s_", "alpha_", "n_", "ksat_", "alphaw_"]]
    out = workspace / "df_polaris_soil_hydraulic.csv"
    csv_profile.to_csv(out)
    profile.reset_index().assign(paper_site_id=site).to_csv(
        workspace / "polaris_soil_hydraulic_application_v1.csv",
        index=False,
    )
    return file_hash(out)


def markdown_table(df: pd.DataFrame) -> str:
    if df.empty:
        return "_No rows._"
    cols = list(df.columns)
    rows = ["| " + " | ".join(cols) + " |", "| " + " | ".join(["---"] * len(cols)) + " |"]
    for row in df.itertuples(index=False):
        vals = []
        for value in row:
            if isinstance(value, float):
                vals.append(f"{value:.6g}")
            else:
                vals.append(str(value))
        rows.append("| " + " | ".join(vals) + " |")
    return "\n".join(rows)


def write_report(report: pd.DataFrame, profiles: pd.DataFrame) -> None:
    summary = (
        profiles.groupby("paper_site_id")
        .agg(
            theta_r_mean=("theta_r_", "mean"),
            theta_s_mean=("theta_s_", "mean"),
            alpha_mean=("alpha_", "mean"),
            n_mean=("n_", "mean"),
            ksat_mean=("ksat_", "mean"),
            ksat_min=("ksat_", "min"),
            ksat_max=("ksat_", "max"),
        )
        .reset_index()
    )
    hash_view = report[
        [
            "paper_site_id",
            "feature_site_key",
            "profile_source",
            "file",
            "status",
            "soil_rows_written",
            "df_polaris_sha256_16",
            "sha256_16",
        ]
    ]
    lines = [
        "# Confirmed 5-Site POLARIS Soil Input Application V1",
        "",
        "## Scope",
        "",
        "- Rebuilds `df_polaris_soil_hydraulic.csv` from the existing confirmed-site POLARIS feature table.",
        "- Applies the same transformations used by the original `Extract_tif.process_polaris_data`.",
        "- Rewrites the SWP `ISOILLAY1` soil hydraulic table in `SwapOriginal.swp`, `Swap1.swp`, and `swap.swp`.",
        "- Does not regenerate site-specific weather.",
        "",
        "## Site Soil Summary",
        "",
        markdown_table(summary),
        "",
        "## Updated Files",
        "",
        markdown_table(hash_view),
        "",
        "## Interpretation",
        "",
        "The confirmed workspaces now contain site-specific POLARIS soil hydraulic curves. "
        "Next, rerun the restart-generation smoke and curve audit. If the run remains stable and "
        "candidate curves still differ, the next layer is site-specific weather extraction.",
    ]
    REPORT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sites", nargs="+", default=sorted(SITE_TO_WORKSPACE))
    parser.add_argument("--create-missing", action="store_true")
    parser.add_argument("--use-embedded-profiles", action="store_true", help="Use bundled confirmed-site POLARIS profiles instead of the feature CSV.")
    args = parser.parse_args()

    features = None if args.use_embedded_profiles else pd.read_csv(FEATURES) if FEATURES.exists() else None
    profile_source = "feature_table" if features is not None else "embedded_confirmed_profiles"
    report_rows = []
    profile_rows = []

    for site in args.sites:
        if site not in SITE_TO_FEATURE:
            raise ValueError(f"Unknown site: {site}")
        feature_key = SITE_TO_FEATURE[site]
        if features is not None:
            matches = features[features["site"] == feature_key]
            if matches.empty:
                raise ValueError(f"Missing feature row for {site}: {feature_key}")
            feature_row = matches.iloc[0]
            profile = build_polaris_profile(feature_row)
        else:
            feature_key, profile = build_embedded_profile(site)
        workspace = ensure_workspace(site, create_missing=args.create_missing)
        df_hash = write_workspace_profile(workspace, site, profile)

        for row in profile.reset_index().to_dict(orient="records"):
            profile_rows.append({"paper_site_id": site, "feature_site_key": feature_key, "profile_source": profile_source, **row})

        for swp_name in SWP_FILES:
            result = update_swp_soil_table(workspace / swp_name, profile)
            report_rows.append(
                {
                    "paper_site_id": site,
                    "feature_site_key": feature_key,
                    "profile_source": profile_source,
                    "workspace": str(workspace),
                    "df_polaris_sha256_16": df_hash,
                    **result,
                }
            )

    report = pd.DataFrame(report_rows)
    profiles = pd.DataFrame(profile_rows)
    REPORT_CSV.parent.mkdir(parents=True, exist_ok=True)
    report.to_csv(REPORT_CSV, index=False)
    profiles.to_csv(PROFILE_CSV, index=False)
    write_report(report, profiles)

    print("Confirmed 5-site POLARIS soil input application v1")
    print(f"profile_source: {profile_source}")
    print(f"csv: {REPORT_CSV}")
    print(f"profiles: {PROFILE_CSV}")
    print(f"md: {REPORT_MD}")
    print(report[["paper_site_id", "file", "status", "soil_rows_written", "df_polaris_sha256_16", "sha256_16"]].to_string(index=False))


if __name__ == "__main__":
    main()
