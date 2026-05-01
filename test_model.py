import json
import os
import sys
from pathlib import Path

import numpy as np
import torch

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = Path(os.environ.get("PROJECT_DIR", SCRIPT_DIR))
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from data_loader import prepare_dataloaders, denormalize
from gccl_model import GCCL_Model
from metrics import compute_masked_metrics
from runtime_paths import DATASET_DIR, CHECKPOINT_DIR


def test_model(checkpoint_path, dataset_path=None, use_era5=True):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")

    batch_size = 64
    in_steps = 3
    out_steps = 7
    hidden_dim = 64
    gc_dim = 32
    kernel_size = (3, 3)
    theta = 7
    k_hop = 2

    checkpoint_path = Path(checkpoint_path)
    if dataset_path is None:
        dataset_path = DATASET_DIR
    dataset_path = Path(dataset_path)

    print(f"\n加载模型: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    norm_params = checkpoint.get("norm_params")

    print("\n准备数据...")
    _, _, test_loader, adj_matrix, current_norm_params = prepare_dataloaders(
        dataset_path=str(dataset_path),
        in_steps=in_steps,
        out_steps=out_steps,
        batch_size=batch_size,
        use_era5=use_era5,
        theta=theta,
        k_hop=k_hop,
    )

    if norm_params is None:
        norm_params = current_norm_params

    sample_input, _ = next(iter(test_loader))
    _, _, height, width, channels = sample_input.shape
    print(f"图像尺寸: {height}x{width}, 通道数: {channels}")

    print("\n初始化模型...")
    model = GCCL_Model(
        input_dim=channels,
        gc_dim=gc_dim,
        hidden_dim_g=hidden_dim,
        hidden_dim_c=hidden_dim,
        kernel_size=kernel_size,
        img_height=height,
        img_width=width,
        theta=theta,
        k_hop=k_hop,
        output_dim=1,
    ).to(device)

    model.load_state_dict(checkpoint["model_state_dict"])
    print(f"模型加载成功 (来自 Epoch {checkpoint.get('epoch', -1) + 1})")

    print("\n" + "=" * 60)
    print("开始测试...")
    print("=" * 60)

    model.eval()
    all_predictions = []
    all_targets = []

    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(test_loader):
            inputs = inputs.to(device)
            predictions = model.predict(inputs, adj_matrix, out_steps)

            all_predictions.append(predictions.cpu().numpy())
            all_targets.append(targets.numpy())

            if (batch_idx + 1) % 10 == 0:
                print(f"  已处理 {batch_idx + 1}/{len(test_loader)} 批次")

    predictions = np.concatenate(all_predictions, axis=0)
    targets = np.concatenate(all_targets, axis=0)

    predictions_cf = np.transpose(predictions, (0, 1, 4, 2, 3))
    targets_cf = np.transpose(targets, (0, 1, 4, 2, 3))

    print(f"\n预测形状: {predictions_cf.shape}")
    print(f"目标形状: {targets_cf.shape}")

    predictions_denorm = denormalize(predictions_cf, norm_params["smap_min"], norm_params["smap_max"])
    targets_denorm = denormalize(targets_cf, norm_params["smap_min"], norm_params["smap_max"])

    print("\n计算评估指标...")
    results = compute_masked_metrics(predictions_denorm, targets_denorm, norm_params["valid_mask"].astype(bool))

    return results, predictions_denorm, targets_denorm


if __name__ == "__main__":
    checkpoint_path = CHECKPOINT_DIR / "best_model.pth"

    results, predictions, targets = test_model(checkpoint_path)

    output_dir = PROJECT_DIR / "test_results"
    output_dir.mkdir(exist_ok=True)

    with open(output_dir / "results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    np.save(output_dir / "predictions.npy", predictions)
    np.save(output_dir / "targets.npy", targets)

    print("\n测试完成！结果已保存到:", output_dir)
