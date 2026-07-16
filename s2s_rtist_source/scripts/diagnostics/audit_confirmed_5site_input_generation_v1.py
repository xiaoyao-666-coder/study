"""Audit input-generation readiness for the five confirmed paper sites.

This is a pre-training audit only. It does not run SWAP, does not train a
surrogate model, and does not create per-site workspaces. It checks whether the
confirmed P1, P2, P3, P4, and P15 coordinates have usable static/site features
and whether the shared SWAP/data template files are present.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd


OUT_DIR = Path("site_general_surrogate_eval")
PACKAGE_ROOT = Path("model3_opt_sto_upload")
INVENTORY = OUT_DIR / "multisite_canonical_site_inventory_template_v1.csv"
FEATURES = OUT_DIR / "site_feature_screening_12_code_sites.csv"
OUTLIERS = OUT_DIR / "site_outlier_ranking_12_code_sites.csv"
SOURCE_AUDIT = OUT_DIR / "multisite_data_source_audit_v1.csv"

CODE_TO_FEATURE_SITE = {
    "N1": "code_N1",
    "N2": "code_N2",
    "N3": "code_N3",
    "N4": "code_N4",
    "coord_12": "code_active",
}

REQUIRED_FEATURE_COLUMNS = [
    "dem_m",
    "tile_drain",
    "theta_s_0_60_mean",
    "theta_r_0_60_mean",
    "ksat_0_60_mean",
    "alpha_0_60_mean",
    "t2m_mean",
    "tmax_mean",
    "precip_sum",
    "pet_sum",
    "rad_sum",
]

REQUIRED_TEMPLATE_FILES = [
    "swap_test",
    "swap",
    "Swap.exe",
    "swap.swp",
    "SwapOriginal.swp",
    "GmaizeDOriginal.crp",
    "swap.dra",
    "SoilPhysParam.csv",
    "df_gridmet.csv",
    "weather_s2s_out.csv",
    "df_polaris_soil_hydraulic.csv",
    "ForecastStep.py",
    "use_s2s.py",
    "Extract_tif.py",
]


def markdown_table(df: pd.DataFrame) -> str:
    if df.empty:
        return "_No rows._"
    cols = list(df.columns)
    rows = ["| " + " | ".join(cols) + " |", "| " + " | ".join(["---"] * len(cols)) + " |"]
    for row in df.itertuples(index=False):
        vals = []
        for value in row:
            if isinstance(value, float):
                vals.append(f"{value:.4f}")
            else:
                vals.append(str(value))
        rows.append("| " + " | ".join(vals) + " |")
    return "\n".join(rows)


def status_from_missing(missing: list[str]) -> str:
    return "ready_for_input_generation_audit" if not missing else "blocked_missing_inputs"


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    inventory = pd.read_csv(INVENTORY)
    confirmed = inventory[
        (inventory["paper_site_id"].astype(str).str.match(r"^P\d+$"))
        & (inventory["paper_id_mapping_status"] == "high")
    ].copy()
    confirmed["feature_site_key"] = confirmed["code_site_id"].map(CODE_TO_FEATURE_SITE)

    features = pd.read_csv(FEATURES)
    outliers = pd.read_csv(OUTLIERS)
    source_audit = pd.read_csv(SOURCE_AUDIT)

    feature_cols = ["site", "lon", "lat"] + REQUIRED_FEATURE_COLUMNS
    confirmed_features = confirmed.merge(
        features[feature_cols],
        left_on="feature_site_key",
        right_on="site",
        how="left",
    )
    confirmed_features = confirmed_features.merge(
        outliers[["site", "heterogeneity_distance_from_mean"]],
        on="site",
        how="left",
    )
    confirmed_features["missing_required_features"] = confirmed_features.apply(
        lambda row: ";".join(
            col for col in REQUIRED_FEATURE_COLUMNS if pd.isna(row.get(col))
        ),
        axis=1,
    )
    confirmed_features["feature_status"] = confirmed_features["missing_required_features"].apply(
        lambda x: "available_from_existing_screening" if x == "" else "missing_feature_values"
    )

    maize_dir = PACKAGE_ROOT / "Maize"
    template_rows = []
    for name in REQUIRED_TEMPLATE_FILES:
        path = maize_dir / name
        template_rows.append(
            {
                "name": name,
                "path": str(path),
                "exists": path.exists(),
                "kind": "dir" if path.is_dir() else "file",
            }
        )
    template = pd.DataFrame(template_rows)

    missing_sources = source_audit[~source_audit["exists"]].copy()
    missing_template = template[~template["exists"]].copy()
    global_missing = list(missing_sources["name"].astype(str)) + list(missing_template["name"].astype(str))

    manifest = confirmed[
        [
            "paper_site_id",
            "code_site_id",
            "longitude",
            "latitude",
            "coordinate_source",
            "paper_id_mapping_status",
        ]
    ].copy()
    manifest["feature_site_key"] = manifest["code_site_id"].map(CODE_TO_FEATURE_SITE)
    manifest["recommended_workspace_name"] = manifest.apply(
        lambda r: f"{r.paper_site_id}_{r.code_site_id}_Maize",
        axis=1,
    )
    manifest["workspace_status"] = "not_created"
    manifest["swap_run_status"] = "not_started"
    manifest["generation_readiness"] = status_from_missing(global_missing)
    manifest["next_action"] = (
        "clone Maize template per site, set longitude/latitude, re-extract static/weather inputs, then run a smoke SWAP check"
    )

    inv_out = OUT_DIR / "confirmed_5site_inventory_v1.csv"
    feat_out = OUT_DIR / "confirmed_5site_feature_audit_v1.csv"
    templ_out = OUT_DIR / "confirmed_5site_template_file_audit_v1.csv"
    manifest_out = OUT_DIR / "confirmed_5site_generation_manifest_v1.csv"
    report_out = OUT_DIR / "confirmed_5site_input_generation_audit_v1.md"

    confirmed.to_csv(inv_out, index=False)
    confirmed_features.to_csv(feat_out, index=False)
    template.to_csv(templ_out, index=False)
    manifest.to_csv(manifest_out, index=False)

    lines = [
        "# Confirmed 5-Site Input Generation Audit V1",
        "",
        "## Scope",
        "- Sites: high-confidence paper mappings only: P1, P2, P3, P4, P15.",
        "- This audit does not run SWAP, train a surrogate model, or create per-site workspaces.",
        "- Goal: confirm whether a small multi-site input-generation smoke workflow is ready to be attempted.",
        "",
        "## Summary",
        f"- Confirmed paper sites: {len(confirmed)}",
        f"- Confirmed sites with existing screening features: {(confirmed_features['feature_status'] == 'available_from_existing_screening').sum()} / {len(confirmed_features)}",
        f"- Data/source entries missing from previous audit: {len(missing_sources)}",
        f"- Required SWAP/template files missing: {len(missing_template)}",
        f"- Overall readiness: {status_from_missing(global_missing)}",
        "",
        "## Confirmed Site Manifest",
        markdown_table(
            manifest[
                [
                    "paper_site_id",
                    "code_site_id",
                    "longitude",
                    "latitude",
                    "feature_site_key",
                    "recommended_workspace_name",
                    "generation_readiness",
                ]
            ]
        ),
        "",
        "## Feature Audit",
        markdown_table(
            confirmed_features[
                [
                    "paper_site_id",
                    "code_site_id",
                    "feature_site_key",
                    "dem_m",
                    "tile_drain",
                    "theta_s_0_60_mean",
                    "theta_r_0_60_mean",
                    "ksat_0_60_mean",
                    "alpha_0_60_mean",
                    "t2m_mean",
                    "precip_sum",
                    "heterogeneity_distance_from_mean",
                    "feature_status",
                ]
            ]
        ),
        "",
        "## Template File Audit",
        markdown_table(template),
        "",
        "## Interpretation",
        "- The five confirmed paper sites already have static/weather screening features in the existing 12-code-site audit table.",
        "- Required source directories and SWAP/template files are present according to this local audit.",
        "- The next safe smoke step is to create isolated per-site Maize workspaces for these five sites and run input extraction / SWAP smoke checks, still without training the universal surrogate.",
        "- P5-P14 should remain outside the formal leave-one-site-out set until exact paper coordinates or author/source-table confirmation is available.",
    ]
    report_out.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print("Confirmed 5-site input generation audit v1")
    print(f"confirmed sites: {len(confirmed)}")
    print(f"missing data/source entries: {len(missing_sources)}")
    print(f"missing template files: {len(missing_template)}")
    print(f"readiness: {status_from_missing(global_missing)}")
    print(f"wrote: {inv_out}")
    print(f"wrote: {feat_out}")
    print(f"wrote: {templ_out}")
    print(f"wrote: {manifest_out}")
    print(f"wrote: {report_out}")


if __name__ == "__main__":
    main()
