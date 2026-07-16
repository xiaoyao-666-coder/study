from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt


OUT_DIR = Path(
    r"D:\study\s2s_rtist_source\site_general_surrogate_eval\public_data_failure_environment_diagnostic_v1"
)


def configure_fonts() -> None:
    candidates = [
        "Microsoft YaHei",
        "SimHei",
        "Noto Sans CJK SC",
        "Source Han Sans SC",
        "Arial Unicode MS",
    ]
    available = {font.name for font in fm.fontManager.ttflist}
    for name in candidates:
        if name in available:
            plt.rcParams["font.sans-serif"] = [name, "DejaVu Sans"]
            break
    plt.rcParams["axes.unicode_minus"] = False


def bar_with_points(df: pd.DataFrame, cols: list[str], labels: list[str], colors: list[str], ylabel: str, title: str, path: Path, ylim=None) -> None:
    means = df[cols].mean()
    stds = df[cols].std()
    x = np.arange(len(cols))
    plt.figure(figsize=(10, 5.6))
    plt.bar(x, means, yerr=stds, capsize=4, color=colors, alpha=0.86)
    for i, col in enumerate(cols):
        jitter = np.linspace(-0.08, 0.08, len(df))
        plt.scatter(np.full(len(df), i) + jitter, df[col], s=28, color="black", alpha=0.65, zorder=3)
    plt.xticks(x, labels, rotation=15, ha="right")
    plt.ylabel(ylabel)
    plt.title(title, pad=12)
    if ylim is not None:
        plt.ylim(*ylim)
    plt.grid(axis="y", alpha=0.25)
    plt.tight_layout()
    plt.savefig(path, dpi=220)
    plt.close()


def main() -> None:
    configure_fonts()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    seed_df = pd.DataFrame(
        [
            [11, 78.772414, 5.365517, 6.082759, 77.020690, 0.862069, 0.275862, 0.310345, 0.827586],
            [23, 78.531034, 5.013793, 5.165517, 79.813793, 0.827586, 0.275862, 0.275862, 0.827586],
            [37, 77.537931, 5.662069, 5.200000, 77.034483, 0.793103, 0.310345, 0.275862, 0.793103],
            [52, 77.537931, 5.662069, 6.179310, 76.027586, 0.793103, 0.310345, 0.310345, 0.793103],
            [89, 57.303448, 5.055172, 5.200000, 80.144828, 0.827586, 0.275862, 0.275862, 0.862069],
        ],
        columns=[
            "seed",
            "baseline",
            "pm1",
            "pm1_pm3",
            "pm3",
            "baseline_large",
            "pm1_large",
            "pm1_pm3_large",
            "pm3_large",
        ],
    )

    labels = ["baseline", "+/-1 day", "+/-1,+/-3 days", "+/-3 days"]
    regret_cols = ["baseline", "pm1", "pm1_pm3", "pm3"]
    rate_cols = ["baseline_large", "pm1_large", "pm1_pm3_large", "pm3_large"]
    colors = ["#6b7280", "#2563eb", "#16a34a", "#d97706"]

    bar_with_points(
        seed_df,
        regret_cols,
        labels,
        colors,
        "Mean regret vs SWAP oracle",
        "相邻日期 SWAP 补标对失败日期的影响",
        OUT_DIR / "fig1_seed_sweep_mean_regret_cn.png",
    )

    bar_with_points(
        seed_df,
        rate_cols,
        labels,
        colors,
        "Large regret > 5 rate",
        "补标后大误差比例明显下降",
        OUT_DIR / "fig2_seed_sweep_large_regret_rate_cn.png",
        ylim=(0, 1.0),
    )

    site_rows = [
        [11, "code_B1", 86.7, 2.933333],
        [11, "code_C2", 64.175, 9.0],
        [11, "code_N1", 177.433333, 6.466667],
        [11, "code_N2", 12.72, 3.6],
        [11, "code_N4", 30.65, 2.3],
        [23, "code_B1", 86.7, 1.6],
        [23, "code_C2", 64.175, 8.575],
        [23, "code_N1", 177.433333, 6.466667],
        [23, "code_N2", 11.32, 3.84],
        [23, "code_N4", 30.65, 2.3],
        [37, "code_B1", 86.7, 4.8],
        [37, "code_C2", 64.175, 8.575],
        [37, "code_N1", 177.433333, 6.466667],
        [37, "code_N2", 5.56, 3.8],
        [37, "code_N4", 30.65, 2.25],
        [52, "code_B1", 86.7, 4.366667],
        [52, "code_C2", 64.175, 9.0],
        [52, "code_N1", 177.433333, 6.466667],
        [52, "code_N2", 5.56, 3.6],
        [52, "code_N4", 30.65, 2.3],
        [89, "code_B1", 86.7, 2.033333],
        [89, "code_C2", 64.175, 8.575],
        [89, "code_N1", 74.833333, 6.466667],
        [89, "code_N2", 11.32, 3.6],
        [89, "code_N4", 30.65, 2.25],
    ]
    site_df = pd.DataFrame(site_rows, columns=["seed", "site_id", "baseline", "pm1"])
    site_agg = (
        site_df.groupby("site_id")
        .agg(
            baseline_mean=("baseline", "mean"),
            baseline_std=("baseline", "std"),
            pm1_mean=("pm1", "mean"),
            pm1_std=("pm1", "std"),
        )
        .reset_index()
    )

    x = np.arange(len(site_agg))
    width = 0.36
    plt.figure(figsize=(10, 5.6))
    plt.bar(
        x - width / 2,
        site_agg["baseline_mean"],
        yerr=site_agg["baseline_std"],
        capsize=3,
        width=width,
        color="#6b7280",
        label="baseline",
    )
    plt.bar(
        x + width / 2,
        site_agg["pm1_mean"],
        yerr=site_agg["pm1_std"],
        capsize=3,
        width=width,
        color="#2563eb",
        label="+/-1 day labels",
    )
    plt.xticks(x, site_agg["site_id"])
    plt.ylabel("Mean regret vs SWAP oracle")
    plt.title("各失败站点上的改善情况", pad=12)
    plt.legend()
    plt.grid(axis="y", alpha=0.25)
    plt.tight_layout()
    plt.savefig(OUT_DIR / "fig3_by_site_baseline_vs_pm1_cn.png", dpi=220)
    plt.close()

    coverage = pd.DataFrame(
        [
            ["code_A2", 1, 25.0],
            ["code_A2", 3, 25.0],
            ["code_B1", 1, 16.875],
            ["code_B1", 3, 5.0],
            ["code_C2", 1, 12.5],
            ["code_C2", 3, 1.6666666667],
            ["code_N1", 1, 23.5714285714],
            ["code_N1", 3, 0.0],
            ["code_N2", 1, 17.5],
            ["code_N2", 3, 12.5],
            ["code_N4", 1, 23.0],
            ["code_N4", 3, 0.0],
        ],
        columns=["site_id", "nearest_anchor_gap", "mean_best_ir"],
    )
    pivot = coverage.pivot(index="site_id", columns="nearest_anchor_gap", values="mean_best_ir").fillna(0)
    pivot = pivot.reindex(["code_A2", "code_B1", "code_C2", "code_N1", "code_N2", "code_N4"])
    x = np.arange(len(pivot))
    plt.figure(figsize=(10, 5.6))
    plt.bar(x - width / 2, pivot[1], width=width, color="#2563eb", label="+/-1 day")
    plt.bar(x + width / 2, pivot[3], width=width, color="#d97706", label="+/-3 day")
    for i, value in enumerate(pivot[3]):
        if abs(float(value)) < 1e-12:
            plt.text(i + width / 2, 0.35, "0", ha="center", va="bottom", fontsize=10, color="#7c2d12")
    plt.xticks(x, pivot.index)
    plt.ylabel("Mean SWAP oracle irrigation (mm)")
    plt.title("±1 天比 ±3 天更接近高需水窗口", pad=12)
    plt.legend()
    plt.grid(axis="y", alpha=0.25)
    plt.tight_layout()
    plt.savefig(OUT_DIR / "fig4_gap_coverage_mean_best_ir_cn.png", dpi=220)
    plt.close()

    for path in sorted(OUT_DIR.glob("*_cn.png")):
        print(path)


if __name__ == "__main__":
    main()
