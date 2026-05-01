import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import rcParams

from runtime_paths import CHECKPOINT_DIR, PLOTS_DIR

rcParams["font.sans-serif"] = ["SimHei", "DejaVu Sans"]
rcParams["axes.unicode_minus"] = False


def load_results(checkpoint_dir=None):
    checkpoint_dir = Path(checkpoint_dir or CHECKPOINT_DIR)
    with open(checkpoint_dir / "test_results.json", "r", encoding="utf-8") as f:
        results = json.load(f)

    predictions = np.load(checkpoint_dir / "predictions.npy")
    targets = np.load(checkpoint_dir / "targets.npy")
    return results, predictions, targets


def print_metrics(results):
    global_res = results.get("Global", results)

    print("\n" + "=" * 60)
    print("GCCL 模型测试结果")
    print("=" * 60)
    print(f"MSE:  {global_res.get('MSE', 0):.6f}")
    print(f"RMSE: {global_res.get('RMSE', 0):.6f}")
    print(f"MAE:  {global_res.get('MAE', 0):.6f}")
    print(f"Bias: {global_res.get('Bias', 0):.6f}")
    print(f"R²:   {global_res.get('R2', 0):.6f}")

    if "Steps" in results:
        print("-" * 60)
        for step, metrics in results["Steps"].items():
            print(f"{step}: RMSE={metrics['RMSE']:.4f}, MAE={metrics['MAE']:.4f}, R²={metrics['R2']:.4f}")
    print("=" * 60)


def plot_predictions_vs_targets(predictions, targets, save_path):
    pred_flat = predictions.flatten()
    target_flat = targets.flatten()
    sample_size = min(5000, len(pred_flat))
    indices = np.random.choice(len(pred_flat), sample_size, replace=False)

    pred_sample = pred_flat[indices]
    target_sample = target_flat[indices]

    plt.figure(figsize=(8, 8))
    plt.scatter(target_sample, pred_sample, alpha=0.5, s=1)
    max_val = max(np.max(target_sample), np.max(pred_sample))
    min_val = min(np.min(target_sample), np.min(pred_sample))
    plt.plot([min_val, max_val], [min_val, max_val], "r--", label="Perfect Prediction")
    plt.xlabel("True Soil Moisture")
    plt.ylabel("Predicted Soil Moisture")
    plt.title("Predictions vs Targets")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()


def plot_time_series(predictions, targets, sample_idx, pixel_h, pixel_w, save_path):
    pred_series = predictions[sample_idx, :, 0, pixel_h, pixel_w]
    target_series = targets[sample_idx, :, 0, pixel_h, pixel_w]

    plt.figure(figsize=(12, 5))
    plt.plot(target_series, "b-", label="True", linewidth=2)
    plt.plot(pred_series, "r--", label="Predicted", linewidth=2)
    plt.xlabel("Time Step")
    plt.ylabel("Soil Moisture")
    plt.title(f"Time Series (Sample {sample_idx}, Pixel [{pixel_h}, {pixel_w}])")
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
    plt.hist(errors, bins=50, edgecolor="black", alpha=0.7)
    plt.title("Error Distribution")
    plt.grid(True, alpha=0.3)

    plt.subplot(1, 2, 2)
    plt.hist(np.abs(errors), bins=50, edgecolor="black", alpha=0.7, color="orange")
    plt.title("Absolute Error Distribution")
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()


def analyze_by_lead_time(predictions, targets, save_path):
    num_steps = predictions.shape[1]
    rmse_by_step = []
    mae_by_step = []

    for t in range(num_steps):
        pred_t = predictions[:, t]
        target_t = targets[:, t]
        rmse_by_step.append(np.sqrt(np.mean((pred_t - target_t) ** 2)))
        mae_by_step.append(np.mean(np.abs(pred_t - target_t)))

    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.plot(range(1, num_steps + 1), rmse_by_step, "b-o", linewidth=2)
    plt.title("RMSE vs Lead Time")
    plt.grid(True, alpha=0.3)

    plt.subplot(1, 2, 2)
    plt.plot(range(1, num_steps + 1), mae_by_step, "r-o", linewidth=2)
    plt.title("MAE vs Lead Time")
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()


def main():
    PLOTS_DIR.mkdir(exist_ok=True)
    results, predictions, targets = load_results()
    print_metrics(results)

    plot_predictions_vs_targets(predictions, targets, PLOTS_DIR / "pred_vs_target.png")
    plot_time_series(predictions, targets, sample_idx=0, pixel_h=25, pixel_w=48, save_path=PLOTS_DIR / "time_series.png")
    plot_spatial_comparison(predictions, targets, time_idx=0, sample_idx=0, save_path=PLOTS_DIR / "spatial_comparison.png")
    plot_error_distribution(predictions, targets, PLOTS_DIR / "error_distribution.png")
    analyze_by_lead_time(predictions, targets, PLOTS_DIR / "lead_time_analysis.png")

    print(f"分析完成，图片已保存到: {PLOTS_DIR}")


if __name__ == "__main__":
    main()
