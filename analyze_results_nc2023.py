# -*- coding: utf-8 -*-
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import rcParams

from runtime_paths import CHECKPOINT_DIR, CHECKPOINT_NC2023_DIR, PLOTS_DIR

rcParams["font.sans-serif"] = ["SimHei", "DejaVu Sans"]
rcParams["axes.unicode_minus"] = False


def load_results(checkpoint_dir):
    checkpoint_dir = Path(checkpoint_dir)
    with open(checkpoint_dir / "test_results.json", "r", encoding="utf-8") as f:
        results = json.load(f)

    predictions = np.load(checkpoint_dir / "predictions.npy")
    targets = np.load(checkpoint_dir / "targets.npy")
    return results, predictions, targets


def print_metrics(results, standard_results_path):
    nc_global = results.get("Global", results)

    print("\n" + "=" * 60)
    print("GCCL + 2023 NC Loss 模型测试结果")
    print("=" * 60)
    print(f"MSE:  {nc_global.get('MSE', 0):.6f}")
    print(f"RMSE: {nc_global.get('RMSE', 0):.6f}")
    print(f"MAE:  {nc_global.get('MAE', 0):.6f}")
    print(f"Bias: {nc_global.get('Bias', 0):.6f}")
    print(f"R²:   {nc_global.get('R2', 0):.6f}")

    if standard_results_path.exists():
        with open(standard_results_path, "r", encoding="utf-8") as f:
            standard_results = json.load(f)
        std_global = standard_results.get("Global", standard_results)
        print("-" * 60)
        for key in ["MSE", "RMSE", "MAE", "Bias", "R2"]:
            print(f"{key}: baseline={std_global.get(key, 0):.6f}, nc2023={nc_global.get(key, 0):.6f}")
    print("=" * 60)


def plot_predictions_vs_targets(predictions, targets, save_path):
    pred_flat = predictions.flatten()
    target_flat = targets.flatten()
    sample_size = min(5000, len(pred_flat))
    indices = np.random.choice(len(pred_flat), sample_size, replace=False)

    pred_sample = pred_flat[indices]
    target_sample = target_flat[indices]

    plt.figure(figsize=(8, 8))
    plt.scatter(target_sample, pred_sample, alpha=0.5, s=1, color="blue")
    max_val = max(np.max(target_sample), np.max(pred_sample))
    min_val = min(np.min(target_sample), np.min(pred_sample))
    plt.plot([min_val, max_val], [min_val, max_val], "r--", linewidth=2)
    plt.xlabel("True Soil Moisture")
    plt.ylabel("Predicted Soil Moisture")
    plt.title("NC2023 Predictions vs Targets")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()


def plot_time_series(predictions, targets, sample_idx, pixel_h, pixel_w, save_path):
    pred_series = predictions[sample_idx, :, 0, pixel_h, pixel_w]
    target_series = targets[sample_idx, :, 0, pixel_h, pixel_w]

    plt.figure(figsize=(12, 5))
    plt.plot(target_series, "b-", label="True", linewidth=2)
    plt.plot(pred_series, "r--", label="Predicted (NC2023)", linewidth=2)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()


def plot_spatial_comparison(predictions, targets, time_idx, sample_idx, save_path):
    pred_map = predictions[sample_idx, time_idx, 0]
    target_map = targets[sample_idx, time_idx, 0]
    diff_map = pred_map - target_map

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    vmin = min(np.min(target_map), np.min(pred_map))
    vmax = max(np.max(target_map), np.max(pred_map))

    im0 = axes[0].imshow(target_map, cmap="Blues", vmin=vmin, vmax=vmax)
    axes[0].set_title("True")
    axes[0].axis("off")
    plt.colorbar(im0, ax=axes[0])

    im1 = axes[1].imshow(pred_map, cmap="Blues", vmin=vmin, vmax=vmax)
    axes[1].set_title("Predicted")
    axes[1].axis("off")
    plt.colorbar(im1, ax=axes[1])

    im2 = axes[2].imshow(diff_map, cmap="RdBu_r", vmin=-np.max(np.abs(diff_map)), vmax=np.max(np.abs(diff_map)))
    axes[2].set_title("Difference")
    axes[2].axis("off")
    plt.colorbar(im2, ax=axes[2])

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()


def plot_error_distribution(predictions, targets, save_path):
    errors = (predictions - targets).flatten()

    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.hist(errors, bins=50, edgecolor="black", alpha=0.7, color="steelblue")
    plt.grid(True, alpha=0.3)

    plt.subplot(1, 2, 2)
    plt.hist(np.abs(errors), bins=50, edgecolor="black", alpha=0.7, color="orange")
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()


def analyze_by_lead_time(predictions, targets, save_path):
    num_steps = predictions.shape[1]
    rmse_nc = []
    mae_nc = []

    for t in range(num_steps):
        pred_t = predictions[:, t]
        target_t = targets[:, t]
        rmse_nc.append(np.sqrt(np.mean((pred_t - target_t) ** 2)))
        mae_nc.append(np.mean(np.abs(pred_t - target_t)))

    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.plot(range(1, num_steps + 1), rmse_nc, "b-o", linewidth=2, markersize=6)
    plt.title("RMSE vs Lead Time")
    plt.grid(True, alpha=0.3)

    plt.subplot(1, 2, 2)
    plt.plot(range(1, num_steps + 1), mae_nc, "r-o", linewidth=2, markersize=6)
    plt.title("MAE vs Lead Time")
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()


def compare_with_standard(predictions_nc, targets_nc, standard_pred_path, save_path):
    standard_pred_path = Path(standard_pred_path)
    if not standard_pred_path.exists():
        return

    predictions_std = np.load(standard_pred_path)
    num_steps = predictions_nc.shape[1]
    rmse_nc = []
    rmse_std = []

    for t in range(num_steps):
        rmse_nc.append(np.sqrt(np.mean((predictions_nc[:, t] - targets_nc[:, t]) ** 2)))
        rmse_std.append(np.sqrt(np.mean((predictions_std[:, t] - targets_nc[:, t]) ** 2)))

    plt.figure(figsize=(10, 6))
    plt.plot(range(1, num_steps + 1), rmse_std, "b-o", label="Standard MSE", linewidth=2, markersize=6)
    plt.plot(range(1, num_steps + 1), rmse_nc, "r-s", label="NC2023", linewidth=2, markersize=6)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()


def main():
    data_input_dir = CHECKPOINT_NC2023_DIR
    baseline_results_path = CHECKPOINT_DIR / "test_results.json"
    baseline_pred_path = CHECKPOINT_DIR / "predictions.npy"
    plot_output_dir = PLOTS_DIR
    plot_output_dir.mkdir(exist_ok=True)

    results, predictions, targets = load_results(data_input_dir)
    print_metrics(results, baseline_results_path)

    plot_predictions_vs_targets(predictions, targets, plot_output_dir / "pred_vs_target.png")
    plot_time_series(predictions, targets, sample_idx=0, pixel_h=25, pixel_w=48, save_path=plot_output_dir / "time_series.png")
    plot_spatial_comparison(predictions, targets, time_idx=0, sample_idx=0, save_path=plot_output_dir / "spatial_comparison.png")
    plot_error_distribution(predictions, targets, save_path=plot_output_dir / "error_distribution.png")
    analyze_by_lead_time(predictions, targets, save_path=plot_output_dir / "lead_time_analysis.png")
    compare_with_standard(predictions, targets, baseline_pred_path, save_path=plot_output_dir / "comparison.png")

    print(f"分析完成，图片已保存到: {plot_output_dir}")


if __name__ == "__main__":
    main()
