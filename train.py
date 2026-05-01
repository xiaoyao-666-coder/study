# -*- coding: utf-8 -*-
"""
GCCL 模型主训练脚本
核心对齐项：
- 先切分数据，再只用训练集拟合归一化
- 训练、建图、评估全部使用 Hubei mask
- 6 通道输入，仅预测 1 通道 SMAP
"""

import os
import sys
from pathlib import Path

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = Path(os.environ.get("PROJECT_DIR", SCRIPT_DIR))
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

import json
import numpy as np
import torch
import torch.optim as optim
from data_loader import prepare_dataloaders, denormalize
from gccl_model import GCCL_Model
from metrics import masked_mse_torch, compute_masked_metrics
from runtime_paths import DATASET_DIR, CHECKPOINT_DIR


def train_epoch(model, train_loader, optimizer, device, adj_matrix, out_steps, valid_mask, grad_clip=1.0):
    model.train()
    total_loss = 0.0
    num_batches = 0
    accumulation_steps = max(1, 64 // train_loader.batch_size)

    optimizer.zero_grad()

    for batch_idx, (inputs, targets) in enumerate(train_loader):
        inputs = inputs.to(device)
        targets = targets.to(device)

        predictions = model(inputs, adj_matrix, out_steps)
        loss = masked_mse_torch(predictions, targets, valid_mask)
        (loss / accumulation_steps).backward()

        if (batch_idx + 1) % accumulation_steps == 0 or (batch_idx + 1) == len(train_loader):
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)
            optimizer.step()
            optimizer.zero_grad()

        total_loss += loss.item()
        num_batches += 1

        if batch_idx % 10 == 0:
            print(f"    Batch [{batch_idx}/{len(train_loader)}] Loss: {loss.item():.6f}")

    return total_loss / max(num_batches, 1)


def validate(model, val_loader, device, adj_matrix, out_steps, valid_mask):
    model.eval()
    total_loss = 0.0
    num_batches = 0

    with torch.no_grad():
        for inputs, targets in val_loader:
            inputs = inputs.to(device)
            targets = targets.to(device)

            predictions = model(inputs, adj_matrix, out_steps)
            loss = masked_mse_torch(predictions, targets, valid_mask)

            total_loss += loss.item()
            num_batches += 1

    return total_loss / max(num_batches, 1)


def test(model, test_loader, device, adj_matrix, out_steps, norm_params):
    model.eval()
    all_predictions = []
    all_targets = []
    valid_mask = norm_params["valid_mask"].astype(bool)

    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs = inputs.to(device)
            predictions = model.predict(inputs, adj_matrix, out_steps)
            all_predictions.append(predictions.cpu().numpy())
            all_targets.append(targets.numpy())

    predictions = np.concatenate(all_predictions, axis=0)
    targets = np.concatenate(all_targets, axis=0)

    predictions_cf = np.transpose(predictions, (0, 1, 4, 2, 3))
    targets_cf = np.transpose(targets, (0, 1, 4, 2, 3))

    predictions_denorm = denormalize(predictions_cf, norm_params["smap_min"], norm_params["smap_max"])
    targets_denorm = denormalize(targets_cf, norm_params["smap_min"], norm_params["smap_max"])

    results = compute_masked_metrics(predictions_denorm, targets_denorm, valid_mask)
    return results, predictions_denorm, targets_denorm


def main():
    batch_size = 16
    learning_rate = 5e-5
    hidden_dim = 64
    kernel_size = (3, 3)
    in_steps = 3
    out_steps = 7
    epochs = 100
    gc_dim = 32
    theta = 7
    k_hop = 2

    dataset_path = DATASET_DIR
    checkpoint_dir = CHECKPOINT_DIR
    checkpoint_dir.mkdir(exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=" * 60)
    print("GCCL Model Training (Masked + Train-Only Normalization)")
    print(f"Device: {device}")
    print(f"Dataset: {dataset_path}")
    print("=" * 60)

    print("\n[1/4] 准备数据 (含 ERA5 气象特征)...")
    train_loader, val_loader, test_loader, adj_matrix, norm_params = prepare_dataloaders(
        dataset_path=str(dataset_path),
        in_steps=in_steps,
        out_steps=out_steps,
        batch_size=batch_size,
        use_era5=True,
        theta=theta,
        k_hop=k_hop,
    )

    sample_input, _ = next(iter(train_loader))
    _, _, height, width, channels = sample_input.shape
    valid_mask = norm_params["valid_mask"].astype(bool)

    print(f"\n图像尺寸: {height}x{width}, 输入通道数: {channels}, 输出通道数: 1")
    print(f"有效像素占比: {valid_mask.mean():.4f}")

    print("\n[2/4] 初始化模型...")
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

    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=10)

    print("\n[3/4] 开始训练...")
    best_val_loss = float("inf")

    for epoch in range(epochs):
        print(f"\nEpoch [{epoch + 1}/{epochs}]")
        train_loss = train_epoch(model, train_loader, optimizer, device, adj_matrix, out_steps, valid_mask)
        val_loss = validate(model, val_loader, device, adj_matrix, out_steps, valid_mask)
        scheduler.step(val_loss)

        print(f"Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            checkpoint_path = checkpoint_dir / "best_model.pth"
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_loss": val_loss,
                    "norm_params": norm_params,
                },
                checkpoint_path,
            )

    print("\n[4/4] 测试评估...")
    best_checkpoint = torch.load(checkpoint_dir / "best_model.pth", map_location=device, weights_only=False)
    model.load_state_dict(best_checkpoint["model_state_dict"])

    results, predictions, targets = test(model, test_loader, device, adj_matrix, out_steps, norm_params)

    print("\n" + "=" * 40)
    print(" 完整模型 - 分步长预测结果")
    print("-" * 40)
    for step in ["1d", "3d", "5d", "7d"]:
        metrics = results["Steps"][step]
        print(f" 第 {step[0]} 天预测 ({step}):")
        print(f"  -> RMSE: {metrics['RMSE']:.4f}")
        print(f"  -> MAE:  {metrics['MAE']:.4f}")
        print(f"  -> R²:   {metrics['R2']:.4f}")
        print("-" * 40)

    results_path = checkpoint_dir / "test_results.json"
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    np.save(checkpoint_dir / "predictions.npy", predictions)
    np.save(checkpoint_dir / "targets.npy", targets)


if __name__ == "__main__":
    main()
