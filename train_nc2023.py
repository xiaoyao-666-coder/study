# -*- coding: utf-8 -*-
"""
GCCL 模型训练脚本 - 使用 2023 NC 论文损失函数

省时微调版：
- 默认从无 LSS 的 best_model.pth 热启动
- k_samples 默认降到 2
- epochs 默认降到 10
- 早停 patience 默认 3
- GPU 上自动启用 AMP 混合精度
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
from loss_lss import RobustNC2023Loss
from metrics import masked_mse_torch, compute_masked_metrics
from runtime_paths import DATASET_DIR, CHECKPOINT_DIR, CHECKPOINT_NC2023_DIR


def train_epoch_nc2023(
    model,
    train_loader,
    criterion,
    optimizer,
    scaler,
    device,
    adj_matrix,
    out_steps,
    valid_mask,
    use_amp,
    grad_clip=1.0,
):
    model.train()
    total_loss, total_mse, total_lss = 0.0, 0.0, 0.0
    num_batches = 0
    accumulation_steps = max(1, 64 // train_loader.batch_size)

    optimizer.zero_grad()

    for batch_idx, (inputs, targets) in enumerate(train_loader):
        inputs = inputs.to(device)
        targets = targets.to(device)

        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
            predictions = model(inputs, adj_matrix, out_steps)
            total_loss_batch, mse_loss, lss_loss = criterion(
                model=model,
                inputs=inputs,
                targets=targets,
                original_preds=predictions,
                adj_matrix=adj_matrix,
                valid_mask=valid_mask,
            )

        loss_to_backward = total_loss_batch / accumulation_steps

        if use_amp:
            scaler.scale(loss_to_backward).backward()
        else:
            loss_to_backward.backward()

        if (batch_idx + 1) % accumulation_steps == 0 or (batch_idx + 1) == len(train_loader):
            if use_amp:
                scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)

            if use_amp:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()

            optimizer.zero_grad()

        total_loss += total_loss_batch.item()
        total_mse += mse_loss.item()
        total_lss += lss_loss.item()
        num_batches += 1

        if batch_idx % 10 == 0:
            print(
                f"    Batch [{batch_idx}/{len(train_loader)}] "
                f"Total: {total_loss_batch.item():.6f} | "
                f"MSE: {mse_loss.item():.6f} | "
                f"LSS: {lss_loss.item():.6f}"
            )

    denom = max(num_batches, 1)
    return total_loss / denom, total_mse / denom, total_lss / denom


def validate_with_nc2023_loss(model, val_loader, criterion, device, adj_matrix, out_steps, valid_mask, use_amp):
    model.eval()
    total_loss = 0.0
    total_mse = 0.0
    total_lss = 0.0
    num_batches = 0

    with torch.no_grad():
        for inputs, targets in val_loader:
            inputs = inputs.to(device)
            targets = targets.to(device)

            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
                predictions = model(inputs, adj_matrix, out_steps)
                total_loss_batch, mse_loss, lss_loss = criterion(
                    model=model,
                    inputs=inputs,
                    targets=targets,
                    original_preds=predictions,
                    adj_matrix=adj_matrix,
                    valid_mask=valid_mask,
                )

            total_loss += total_loss_batch.item()
            total_mse += mse_loss.item()
            total_lss += lss_loss.item()
            num_batches += 1

    denom = max(num_batches, 1)
    return total_loss / denom, total_mse / denom, total_lss / denom


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


def maybe_load_base_checkpoint(model, optimizer, checkpoint_path, device):
    if not checkpoint_path.exists():
        print(f"未找到热启动权重: {checkpoint_path}")
        return 0, None

    print(f"加载热启动权重: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"], strict=False)

    start_epoch = checkpoint.get("epoch", -1) + 1
    return start_epoch, checkpoint


def main():
    batch_size = 16
    learning_rate = 2e-5
    hidden_dim = 64
    kernel_size = (3, 3)
    in_steps = 3
    out_steps = 7
    epochs = 10
    gc_dim = 32
    theta = 7
    k_hop = 2

    # 省时配置
    lambda_reg = 1.0
    k_samples = 2
    q_radius = 0.01
    early_stop_patience = 3
    warm_start_checkpoint = CHECKPOINT_DIR / "best_model.pth"

    dataset_path = DATASET_DIR
    checkpoint_dir = CHECKPOINT_NC2023_DIR
    checkpoint_dir.mkdir(exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    print("=" * 60)
    print("GCCL Model Training with 2023 NC Loss (Fast Fine-tune)")
    print(f"Device: {device}")
    print(f"Dataset: {dataset_path}")
    print(f"Warm start: {warm_start_checkpoint}")
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

    criterion = RobustNC2023Loss(
        lambda_reg=lambda_reg,
        k_samples=k_samples,
        q_radius=q_radius,
    )
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=2)

    start_epoch, base_checkpoint = maybe_load_base_checkpoint(model, optimizer, warm_start_checkpoint, device)

    print("\n[3/4] 开始训练...")
    best_val_loss = float("inf")
    patience_counter = 0

    for epoch in range(start_epoch, start_epoch + epochs):
        print(f"\nEpoch [{epoch + 1}/{start_epoch + epochs}]")
        train_loss, train_mse, train_lss = train_epoch_nc2023(
            model, train_loader, criterion, optimizer, scaler, device, adj_matrix, out_steps, valid_mask, use_amp
        )
        val_loss, val_mse, val_lss = validate_with_nc2023_loss(
            model, val_loader, criterion, device, adj_matrix, out_steps, valid_mask, use_amp
        )
        scheduler.step(val_loss)

        print(
            f"Train Loss: {train_loss:.6f} (MSE: {train_mse:.6f}, LSS: {train_lss:.6f}) | "
            f"Val Loss: {val_loss:.6f} (MSE: {val_mse:.6f}, LSS: {val_lss:.6f})"
        )

        latest_path = checkpoint_dir / "latest_model.pth"
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_loss": val_loss,
                "norm_params": norm_params,
            },
            latest_path,
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
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
            print("  -> 验证集提升，已保存 best_model.pth")
        else:
            patience_counter += 1
            print(f"  -> 验证集未提升，early-stop counter: {patience_counter}/{early_stop_patience}")
            if patience_counter >= early_stop_patience:
                print("  -> 提前停止")
                break

    print("\n[4/4] 测试评估...")
    best_checkpoint = torch.load(checkpoint_dir / "best_model.pth", map_location=device, weights_only=False)
    model.load_state_dict(best_checkpoint["model_state_dict"])

    results, predictions, targets = test(model, test_loader, device, adj_matrix, out_steps, norm_params)

    print("\nLSS 模型 - 分步长预测结果")
    print("-" * 40)
    for step in ["1d", "3d", "5d", "7d"]:
        metrics = results["Steps"][step]
        print(f"第 {step[0]} 天预测 ({step}):")
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
