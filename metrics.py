import numpy as np
import torch


def masked_mse_torch(predictions, targets, valid_mask):
    if predictions.shape != targets.shape:
        raise ValueError(f"predictions shape {predictions.shape} != targets shape {targets.shape}")

    if predictions.ndim != 5:
        raise ValueError("masked_mse_torch expects shape (B, T, H, W, C)")

    mask = torch.as_tensor(valid_mask, dtype=predictions.dtype, device=predictions.device)
    if mask.ndim != 2:
        raise ValueError("valid_mask must be 2D")

    mask = mask.view(1, 1, mask.shape[0], mask.shape[1], 1)
    squared_error = (predictions - targets) ** 2
    weighted_error = squared_error * mask

    valid_count = mask.sum() * predictions.shape[0] * predictions.shape[1] * predictions.shape[-1]
    valid_count = valid_count.clamp_min(1.0)
    return weighted_error.sum() / valid_count


def compute_masked_metrics(predictions, targets, valid_mask):
    if predictions.shape != targets.shape:
        raise ValueError(f"predictions shape {predictions.shape} != targets shape {targets.shape}")

    if predictions.ndim != 5:
        raise ValueError("compute_masked_metrics expects shape (N, T, C, H, W)")

    mask = np.asarray(valid_mask, dtype=bool)
    if mask.ndim != 2:
        raise ValueError("valid_mask must be 2D")

    pred_valid = predictions[..., mask]
    target_valid = targets[..., mask]

    global_mse = np.mean((pred_valid - target_valid) ** 2)
    global_rmse = np.sqrt(global_mse)
    global_mae = np.mean(np.abs(pred_valid - target_valid))
    global_bias = np.mean(pred_valid - target_valid)

    ss_res_global = np.sum((target_valid - pred_valid) ** 2)
    ss_tot_global = np.sum((target_valid - np.mean(target_valid)) ** 2)
    global_r2 = 1 - (ss_res_global / (ss_tot_global + 1e-8))

    step_indices = {"1d": 0, "3d": 2, "5d": 4, "7d": 6}
    step_metrics = {}

    for step_name, t_idx in step_indices.items():
        pred_t = predictions[:, t_idx, :, :, :][..., mask]
        target_t = targets[:, t_idx, :, :, :][..., mask]

        t_mse = np.mean((pred_t - target_t) ** 2)
        t_rmse = np.sqrt(t_mse)
        t_mae = np.mean(np.abs(pred_t - target_t))
        t_bias = np.mean(pred_t - target_t)

        t_ss_res = np.sum((target_t - pred_t) ** 2)
        t_ss_tot = np.sum((target_t - np.mean(target_t)) ** 2)
        t_r2 = 1 - (t_ss_res / (t_ss_tot + 1e-8))

        step_metrics[step_name] = {
            "RMSE": float(t_rmse),
            "MAE": float(t_mae),
            "Bias": float(t_bias),
            "R2": float(t_r2),
        }

    return {
        "Global": {
            "MSE": float(global_mse),
            "RMSE": float(global_rmse),
            "MAE": float(global_mae),
            "Bias": float(global_bias),
            "R2": float(global_r2),
        },
        "Steps": step_metrics,
    }
