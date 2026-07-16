"""Map Main_win.py candidate coordinates to paper Fig. 4 site IDs.

This script uses visually digitized point centers from Fig. 4 of the local
paper PDF. The coordinates are approximate and are intended only for auditing
site identity before building a multi-site surrogate dataset.
"""

from __future__ import annotations

import math
from pathlib import Path

import pandas as pd


OUT_DIR = Path("site_general_surrogate_eval")
INVENTORY = OUT_DIR / "multisite_canonical_site_inventory_template_v1.csv"


# Yellow-dot centers extracted from page07_img2_X1.jpg, then labeled by the
# visible P labels in Fig. 4. The lon/lat transform is calibrated with code
# anchors P1, P3, P4, and P15 that match the figure unambiguously.
FIG4_DIGITIZED = [
    ("P1", 670.4226804123712, 255.75257731958763, -98.224, 42.022),
    ("P2", 869.7894736842105, 294.12631578947367, -88.396, 40.669),
    ("P3", 697.8942307692307, 133.7596153846154, -96.870, 46.323),
    ("P4", 742.4318181818181, 237.4318181818182, -94.674, 42.668),
    ("P5", 802.980198019802, 274.2475247524753, -91.689, 41.370),
    ("P6", 316.1980198019802, 228.2970297029703, -115.687, 42.990),
    ("P7", 820.3295454545455, 469.1818181818182, -90.834, 34.496),
    ("P8", 978.8333333333334, 204.03333333333333, -83.020, 43.846),
    ("P9", 791.0505050505051, 308.8686868686869, -92.278, 40.149),
    ("P10", 868.0729166666666, 218.28125, -88.480, 43.343),
    ("P11", 621.9711538461538, 375.7692307692308, -100.613, 37.790),
    ("P12", 945.3541666666666, 312.59375, -84.670, 40.018),
    ("P13", 559.5876288659794, 447.8453608247423, -103.688, 35.249),
    ("P14", 498.5050505050505, 134.8989898989899, -106.700, 46.283),
    ("P15", 385.60360360360363, 263.73873873873873, -112.266, 41.740),
]


def lon_lat_distance_deg(lon1: float, lat1: float, lon2: float, lat2: float) -> float:
    """Small-distance approximation in degrees, longitude scaled by latitude."""
    lat_mid = math.radians((lat1 + lat2) / 2.0)
    return math.hypot((lon1 - lon2) * math.cos(lat_mid), lat1 - lat2)


def confidence(distance_deg: float, duplicate_nonwinner: bool) -> str:
    if duplicate_nonwinner:
        return "needs_manual_confirmation"
    if distance_deg <= 0.10:
        return "high"
    if distance_deg <= 0.25:
        return "possible"
    return "unmatched"


def markdown_table(df: pd.DataFrame) -> str:
    if df.empty:
        return "_None_"
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


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    inventory = pd.read_csv(INVENTORY)
    fig = pd.DataFrame(
        FIG4_DIGITIZED,
        columns=[
            "paper_site_id",
            "fig4_x_px",
            "fig4_y_px",
            "fig4_approx_longitude",
            "fig4_approx_latitude",
        ],
    )

    rows = []
    for row in inventory.itertuples(index=False):
        candidates = []
        for site in fig.itertuples(index=False):
            dist = lon_lat_distance_deg(
                float(row.longitude),
                float(row.latitude),
                float(site.fig4_approx_longitude),
                float(site.fig4_approx_latitude),
            )
            candidates.append((site.paper_site_id, dist))
        candidates.sort(key=lambda x: x[1])
        rows.append(
            {
                "code_site_id": row.code_site_id,
                "code_longitude": row.longitude,
                "code_latitude": row.latitude,
                "nearest_paper_site_id": candidates[0][0],
                "nearest_distance_deg": candidates[0][1],
                "second_nearest_paper_site_id": candidates[1][0],
                "second_nearest_distance_deg": candidates[1][1],
            }
        )

    match = pd.DataFrame(rows)
    best_rank = match.groupby("nearest_paper_site_id")["nearest_distance_deg"].rank(method="first")
    match["duplicate_nearest_paper_site"] = match["nearest_paper_site_id"].map(
        match["nearest_paper_site_id"].value_counts()
    ).gt(1)
    match["duplicate_nonwinner"] = match["duplicate_nearest_paper_site"] & best_rank.gt(1)
    match["mapping_confidence"] = [
        confidence(float(r.nearest_distance_deg), bool(r.duplicate_nonwinner))
        for r in match.itertuples(index=False)
    ]
    match["assigned_paper_site_id"] = match.apply(
        lambda r: r["nearest_paper_site_id"] if r["mapping_confidence"] == "high" else "",
        axis=1,
    )
    match["mapping_note"] = match.apply(
        lambda r: (
            "High-confidence Fig. 4 match."
            if r["mapping_confidence"] == "high"
            else (
                "Nearest Fig. 4 point is duplicated by another code coordinate; keep pending."
                if r["duplicate_nonwinner"]
                else "No close Fig. 4 paper site; likely not part of the paper P1-P15 set or needs author/manual confirmation."
            )
        ),
        axis=1,
    )

    updated = inventory.merge(
        match[
            [
                "code_site_id",
                "assigned_paper_site_id",
                "nearest_paper_site_id",
                "nearest_distance_deg",
                "mapping_confidence",
                "mapping_note",
            ]
        ],
        on="code_site_id",
        how="left",
    )
    updated["paper_site_id"] = updated["assigned_paper_site_id"].fillna("")
    updated["paper_id_mapping_status"] = updated["mapping_confidence"].fillna("pending")
    updated["notes"] = updated["mapping_note"].fillna(updated["notes"])
    updated = updated[
        [
            "paper_site_id",
            "code_site_id",
            "longitude",
            "latitude",
            "coordinate_source",
            "main_win_line",
            "paper_id_mapping_status",
            "static_attribute_status",
            "swap_input_status",
            "multisite_generation_status",
            "notes",
        ]
    ]

    fig_out = OUT_DIR / "paper_fig4_digitized_site_coordinates_v1.csv"
    match_out = OUT_DIR / "code_to_paper_site_mapping_from_fig4_v1.csv"
    updated_out = OUT_DIR / "multisite_canonical_site_inventory_mapped_v1.csv"
    report_out = OUT_DIR / "code_to_paper_site_mapping_from_fig4_v1.md"

    fig.to_csv(fig_out, index=False)
    match.to_csv(match_out, index=False)
    updated.to_csv(updated_out, index=False)

    high = match[match["mapping_confidence"] == "high"]
    pending = match[match["mapping_confidence"] != "high"]
    unmatched_paper = sorted(set(fig["paper_site_id"]) - set(high["assigned_paper_site_id"]))
    lines = [
        "# Code-to-Paper Site Mapping From Fig. 4 V1",
        "",
        "## Scope",
        "- Source figure: local paper PDF, Fig. 4, extracted image `page07_img2_X1.jpg`.",
        "- Method: digitized yellow site dots, calibrated approximate lon/lat with unambiguous code anchors P1, P3, P4, and P15.",
        "- Purpose: audit station identity before leave-one-site-out data generation; this is not model training.",
        "",
        "## Result",
        f"- High-confidence code-to-paper matches: {len(high)} / {len(match)}",
        f"- Pending / unmatched code coordinates: {len(pending)} / {len(match)}",
        f"- Paper Fig. 4 sites not represented by a high-confidence code coordinate: {', '.join(unmatched_paper)}",
        "",
        "## High-Confidence Matches",
        markdown_table(
            high[
                [
                    "code_site_id",
                    "code_longitude",
                    "code_latitude",
                    "assigned_paper_site_id",
                    "nearest_distance_deg",
                ]
            ]
        ),
        "",
        "## Pending Code Coordinates",
        markdown_table(
            pending[
                [
                    "code_site_id",
                    "code_longitude",
                    "code_latitude",
                    "nearest_paper_site_id",
                    "nearest_distance_deg",
                    "mapping_note",
                ]
            ]
        ),
        "",
        "## Interpretation",
        "- The 12 coordinates embedded in `Main_win.py` are not a complete paper P1-P15 inventory.",
        "- Five coordinates can be safely used as paper-labeled sites for the next audit step: P1, P2, P3, P4, and P15.",
        "- The remaining seven coordinates should stay pending until a paper/site source table or author confirmation is found.",
    ]
    report_out.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"wrote: {fig_out}")
    print(f"wrote: {match_out}")
    print(f"wrote: {updated_out}")
    print(f"wrote: {report_out}")
    print(f"high-confidence matches: {len(high)} / {len(match)}")


if __name__ == "__main__":
    main()
