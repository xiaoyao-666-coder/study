# -*- coding: utf-8 -*-
"""
GCCL training entry for OTW-loss validation.

This script keeps the model and data pipeline unchanged and supports using OTW
as the training objective. By default it follows the teacher's requested
"replace the original loss with OTW" setting:

    J = lambda * J_OTW

An optional compatibility flag can re-enable the earlier hybrid objective:

    J = J_MSE + lambda * J_OTW

Defaults are tuned for the teacher's requested first-step verification:
- 7-day forecast remains the main target
- 1d/3d/5d/7d metrics are still saved for diagnosis
- warm-start from the baseline checkpoint is enabled by default
- AMP is disabled by default because sparse graph ops may not support half
  precision reliably across training environments
"""

import argparse
import json
import os
import sys
from pathlib import Path

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = Path(os.environ.get("PROJECT_DIR", SCRIPT_DIR))
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

import numpy as np
import torch
import torch.optim as optim

from data_loader import denormalize, prepare_dataloaders
from gccl_model import GCCL_Model
from loss_otw import OTWLoss
from metrics import compute_masked_metrics
from runtime_paths import CHECKPOINT_DIR, CHECKPOINT_OTW_DIR, DATASET_DIR


def parse_args():
    parser = argparse.ArgumentParser(description="Train GCCL with OTW loss")
    parser.add_argument("--dataset-path", type=str, default=str(DATASET_DIR))
    parser.add_argument("--checkpoint-dir", type=str, default=str(CHECKPOINT_OTW_DIR))
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--gc-dim", type=int, default=32)
    parser.add_argument("--in-steps", type=int, default=3)
    parser.add_argument("--out-steps", type=int, default=7)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--theta", type=int, default=7)
    parser.add_argument("--k-hop", type=int, default=2)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--early-stop-patience", type=int, default=3)
    parser.add_argument("--lambda-otw", type=float, default=1.0)
    parser.add_argument("--waste-cost", type=float, default=3.0)
    parser.add_argument("--window-size", type=int, default=3)
    parser.add_argument("--beta", type=float, default=0.01)
    parser.add_argument(
        "--include-mse-term",
        action="store_true",
        help="Use the earlier hybrid objective J_MSE + lambda * J_OTW instead of pure OTW",
    )
    parser.add_argument("--use-amp", action="store_true", help="Enable mixed precision if supported")
    parser.add_argument("--warm-start", type=str, default=str(CHECKPOINT_DIR / "best_model.pth"))
    parser.add_argument("--no-warm-start", action="store_true")
    return parser.parse_args()


def build_model(channels, height, width, hidden_dim, gc_dim, theta, k_hop):
    return GCCL_Model(
        input_dim=channels,
        gc_dim=gc_dim,
        hidden_dim_g=hidden_dim,
        hidden_dim_c=hidden_dim,
        kernel_size=(3, 3),
        img_height=height,
        img_width=width,
        theta=theta,
        k_hop=k_hop,
        output_dim=1,
    )


def maybe_load_checkpoint(model, checkpoint_path, device):
    if checkpoint_path is None:
        print("Warm start: disabled")
        return 0, None

    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.exists():
        print(f"Warm start checkpoint not found: {checkpoint_path}")
        return 0, None

    print(f"Loading warm start checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"], strict=False)

    start_epoch = checkpoint.get("epoch", -1) + 1
    return start_epoch, checkpoint


def train_epoch_otw(
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
    grad_clip,
):
    model.train()
    total_loss, total_mse, total_otw = 0.0, 0.0, 0.0
    num_batches = 0
    accumulation_steps = max(1, 64 // train_loader.batch_size)

    optimizer.zero_grad()

    for batch_idx, (inputs, targets) in enumerate(train_loader):
        inputs = inputs.to(device)
        targets = targets.to(device)

        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
            predictions = model(inputs, adj_matrix, out_steps)
            total_loss_batch, mse_loss, otw_loss = criterion(predictions, targets, valid_mask)

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
        total_otw += otw_loss.item()
        num_batches += 1

        if batch_idx % 10 == 0:
            print(
                f"    Batch [{batch_idx}/{len(train_loader)}] "
                f"Total: {total_loss_batch.item():.6f} | "
                f"MSE: {mse_loss.item():.6f} | "
                f"OTW: {otw_loss.item():.6f}"
            )

    denom = max(num_batches, 1)
    return total_loss / denom, total_mse / denom, total_otw / denom


def validate_with_otw_loss(model, val_loader, criterion, device, adj_matrix, out_steps, valid_mask, use_amp):
    model.eval()
    total_loss, total_mse, total_otw = 0.0, 0.0, 0.0
    num_batches = 0

    with torch.no_grad():
        for inputs, targets in val_loader:
            inputs = inputs.to(device)
            targets = targets.to(device)

            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
                predictions = model(inputs, adj_matrix, out_steps)
                total_loss_batch, mse_loss, otw_loss = criterion(predictions, targets, valid_mask)

            total_loss += total_loss_batch.item()
            total_mse += mse_loss.item()
            total_otw += otw_loss.item()
            num_batches += 1

    denom = max(num_batches, 1)
    return total_loss / denom, total_mse / denom, total_otw / denom


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


def save_checkpoint(checkpoint_path, epoch, model, optimizer, val_loss, norm_params, args):
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "val_loss": val_loss,
            "norm_params": norm_params,
            "config": vars(args),
        },
        checkpoint_path,
    )


def save_run_artifacts(checkpoint_dir, args, results, predictions, targets):
    checkpoint_dir.mkdir(exist_ok=True)

    with open(checkpoint_dir / "run_config.json", "w", encoding="utf-8") as f:
        json.dump(vars(args), f, indent=2, ensure_ascii=False)

    with open(checkpoint_dir / "test_results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    np.save(checkpoint_dir / "predictions.npy", predictions)
    np.save(checkpoint_dir / "targets.npy", targets)


def main():
    args = parse_args()

    checkpoint_dir = Path(args.checkpoint_dir)
    checkpoint_dir.mkdir(exist_ok=True)

    dataset_path = Path(args.dataset_path)
    warm_start_checkpoint = None if args.no_warm_start else args.warm_start

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = bool(args.use_amp and device.type == "cuda")
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    print("=" * 60)
    print("GCCL Model Training with OTW Loss")
    print(f"Device: {device}")
    print(f"Dataset: {dataset_path}")
    print(f"Checkpoint dir: {checkpoint_dir}")
    print(f"Warm start: {warm_start_checkpoint or 'disabled'}")
    print(f"Primary validation horizon: {args.out_steps}d")
    print(f"Objective: {'MSE + lambda * OTW' if args.include_mse_term else 'pure OTW'}")
    print("=" * 60)

    print("\n[1/4] Preparing data...")
    train_loader, val_loader, test_loader, adj_matrix, norm_params = prepare_dataloaders(
        dataset_path=str(dataset_path),
        in_steps=args.in_steps,
        out_steps=args.out_steps,
        batch_size=args.batch_size,
        use_era5=True,
        theta=args.theta,
        k_hop=args.k_hop,
    )

    sample_input, _ = next(iter(train_loader))
    _, _, height, width, channels = sample_input.shape
    valid_mask = norm_params["valid_mask"].astype(bool)

    print(f"\nImage size: {height}x{width}, input channels: {channels}, output channels: 1")
    print(f"Valid-pixel ratio: {valid_mask.mean():.4f}")

    print("\n[2/4] Building model and OTW criterion...")
    model = build_model(channels, height, width, args.hidden_dim, args.gc_dim, args.theta, args.k_hop).to(device)
    criterion = OTWLoss(
        lambda_reg=args.lambda_otw,
        waste_cost=args.waste_cost,
        window_size=args.window_size,
        beta=args.beta,
        include_mse_term=args.include_mse_term,
    )
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=2)

    start_epoch, _ = maybe_load_checkpoint(model, warm_start_checkpoint, device)

    print("\n[3/4] Training...")
    best_val_loss = float("inf")
    patience_counter = 0

    for epoch in range(start_epoch, start_epoch + args.epochs):
        print(f"\nEpoch [{epoch + 1}/{start_epoch + args.epochs}]")
        train_loss, train_mse, train_otw = train_epoch_otw(
            model=model,
            train_loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            scaler=scaler,
            device=device,
            adj_matrix=adj_matrix,
            out_steps=args.out_steps,
            valid_mask=valid_mask,
            use_amp=use_amp,
            grad_clip=args.grad_clip,
        )
        val_loss, val_mse, val_otw = validate_with_otw_loss(
            model=model,
            val_loader=val_loader,
            criterion=criterion,
            device=device,
            adj_matrix=adj_matrix,
            out_steps=args.out_steps,
            valid_mask=valid_mask,
            use_amp=use_amp,
        )
        scheduler.step(val_loss)

        print(
            f"Train Loss: {train_loss:.6f} (MSE: {train_mse:.6f}, OTW: {train_otw:.6f}) | "
            f"Val Loss: {val_loss:.6f} (MSE: {val_mse:.6f}, OTW: {val_otw:.6f})"
        )

        save_checkpoint(checkpoint_dir / "latest_model.pth", epoch, model, optimizer, val_loss, norm_params, args)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            save_checkpoint(checkpoint_dir / "best_model.pth", epoch, model, optimizer, val_loss, norm_params, args)
            print("  -> Validation improved, saved best_model.pth")
        else:
            patience_counter += 1
            print(f"  -> Validation did not improve, early-stop counter: {patience_counter}/{args.early_stop_patience}")
            if patience_counter >= args.early_stop_patience:
                print("  -> Early stopping triggered")
                break

    print("\n[4/4] Testing...")
    best_checkpoint = torch.load(checkpoint_dir / "best_model.pth", map_location=device, weights_only=False)
    model.load_state_dict(best_checkpoint["model_state_dict"])

    results, predictions, targets = test(model, test_loader, device, adj_matrix, args.out_steps, norm_params)
    save_run_artifacts(checkpoint_dir, args, results, predictions, targets)

    print("\nOTW model - horizon metrics")
    print("-" * 40)
    for step, metrics in results["Steps"].items():
        print(f"{step}: RMSE={metrics['RMSE']:.4f}, MAE={metrics['MAE']:.4f}, R2={metrics['R2']:.4f}")
    print("-" * 40)
    primary_step = f"{args.out_steps}d"
    if primary_step in results["Steps"]:
        metrics = results["Steps"][primary_step]
        print(
            f"Primary horizon ({primary_step}) -> "
            f"RMSE: {metrics['RMSE']:.4f}, MAE: {metrics['MAE']:.4f}, R2: {metrics['R2']:.4f}"
        )


if __name__ == "__main__":
    main()
