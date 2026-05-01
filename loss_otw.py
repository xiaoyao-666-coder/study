"""
Optimal Transport Warping loss for spatiotemporal forecasting.

This file is intentionally separated from the baseline and LSS losses so the
OTW experiment can be managed independently.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from metrics import masked_mse_torch


class OTWLoss(nn.Module):
    def __init__(
        self,
        lambda_reg=1.0,
        waste_cost=3.0,
        window_size=3,
        beta=0.01,
        normalize_by_length=True,
        split_positive_negative=True,
        include_mse_term=False,
    ):
        super(OTWLoss, self).__init__()
        self.lambda_reg = lambda_reg
        self.waste_cost = waste_cost
        self.window_size = window_size
        self.beta = beta
        self.normalize_by_length = normalize_by_length
        self.split_positive_negative = split_positive_negative
        self.include_mse_term = include_mse_term

    def _extract_valid_series(self, tensor, valid_mask):
        if tensor.ndim != 5:
            raise ValueError("OTWLoss expects predictions/targets with shape (B, T, H, W, C)")

        batch_size, time_steps, height, width, channels = tensor.shape
        mask = torch.as_tensor(valid_mask, dtype=torch.bool, device=tensor.device)
        if mask.shape != (height, width):
            raise ValueError(
                f"valid_mask shape {tuple(mask.shape)} does not match spatial shape {(height, width)}"
            )

        tensor = tensor.permute(0, 2, 3, 4, 1).contiguous().view(batch_size, height * width, channels, time_steps)
        tensor = tensor[:, mask.reshape(-1), :, :]
        if tensor.numel() == 0:
            raise ValueError("valid_mask contains no valid pixels")

        return tensor.reshape(-1, time_steps)

    def _windowed_cumsum(self, series, window_size):
        if window_size <= 0:
            raise ValueError("window_size must be positive")

        if series.ndim != 2:
            raise ValueError("series must have shape (N, T)")

        effective_window = min(window_size, series.size(-1))
        cumulative = series.cumsum(dim=-1)
        if effective_window >= series.size(-1):
            return cumulative

        shifted = F.pad(cumulative[:, :-effective_window], (effective_window, 0))
        return cumulative - shifted

    def _smooth_abs(self, values):
        zeros = torch.zeros_like(values)
        return F.smooth_l1_loss(values, zeros, reduction="none", beta=self.beta)

    def _otw_component(self, series_a, series_b):
        if series_a.shape != series_b.shape:
            raise ValueError(f"series_a shape {series_a.shape} != series_b shape {series_b.shape}")

        warped_a = self._windowed_cumsum(series_a, self.window_size)
        warped_b = self._windowed_cumsum(series_b, self.window_size)
        diff = warped_a - warped_b

        position_cost = self._smooth_abs(diff)
        if diff.size(-1) > 1:
            position_cost[:, -1] = position_cost[:, -1] * self.waste_cost
        else:
            position_cost = position_cost * self.waste_cost

        if self.normalize_by_length:
            return position_cost.mean()
        return position_cost.sum(dim=-1).mean()

    def _otw_distance(self, predictions, targets, valid_mask):
        pred_series = self._extract_valid_series(predictions, valid_mask)
        target_series = self._extract_valid_series(targets, valid_mask)

        if not self.split_positive_negative:
            return self._otw_component(pred_series, target_series)

        pred_pos = torch.clamp(pred_series, min=0.0)
        target_pos = torch.clamp(target_series, min=0.0)
        pred_neg = torch.clamp(-pred_series, min=0.0)
        target_neg = torch.clamp(-target_series, min=0.0)

        return self._otw_component(pred_pos, target_pos) + self._otw_component(pred_neg, target_neg)

    def forward(self, predictions, targets, valid_mask):
        mse_loss = masked_mse_torch(predictions, targets, valid_mask)
        otw_loss = self._otw_distance(predictions, targets, valid_mask)
        if self.include_mse_term:
            total_loss = mse_loss + self.lambda_reg * otw_loss
        else:
            total_loss = self.lambda_reg * otw_loss
        return total_loss, mse_loss, otw_loss
