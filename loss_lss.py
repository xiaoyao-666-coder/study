"""
2023 Nature Communications loss implementation for robust forecasting.

This file keeps the LSS-related objective separate from the baseline and OTW
losses so the project can upload or exclude it independently.
"""

import math
import torch
import torch.nn as nn

from metrics import masked_mse_torch


def _first_n_primes(n):
    primes = []
    candidate = 2
    while len(primes) < n:
        is_prime = True
        limit = int(math.sqrt(candidate)) + 1
        for p in primes:
            if p > limit:
                break
            if candidate % p == 0:
                is_prime = False
                break
        if is_prime:
            primes.append(candidate)
        candidate += 1
    return primes


def _van_der_corput(index, base):
    value = 0.0
    denom = 1.0
    while index > 0:
        index, remainder = divmod(index, base)
        denom *= base
        value += remainder / denom
    return value


def halton_sequence(num_points, dim, device, dtype):
    if dim <= 0:
        raise ValueError("dim must be positive")

    primes = _first_n_primes(dim)
    seq = torch.empty((num_points, dim), dtype=dtype, device=device)
    for d, base in enumerate(primes):
        values = [_van_der_corput(i + 1, base) for i in range(num_points)]
        seq[:, d] = torch.tensor(values, dtype=dtype, device=device)
    return seq


class RobustNC2023Loss(nn.Module):
    def __init__(self, lambda_reg=1.0, k_samples=10, q_radius=0.01, clamp_inputs=True):
        super(RobustNC2023Loss, self).__init__()
        self.lambda_reg = lambda_reg
        self.k_samples = k_samples
        self.q_radius = q_radius
        self.clamp_inputs = clamp_inputs

    def _build_halton_perturbations(self, inputs):
        _, time_steps, _, _, channels = inputs.shape
        seq = halton_sequence(
            num_points=self.k_samples,
            dim=time_steps * channels,
            device=inputs.device,
            dtype=inputs.dtype,
        )

        seq = (seq * 2.0 - 1.0) * self.q_radius
        seq = seq.view(self.k_samples, time_steps, channels)
        return seq.unsqueeze(2).unsqueeze(3)

    def forward(self, model, inputs, targets, original_preds, adj_matrix=None, valid_mask=None):
        mse_loss = masked_mse_torch(original_preds, targets, valid_mask)
        lss_loss = 0.0

        perturbations = self._build_halton_perturbations(inputs)

        for sample_idx in range(self.k_samples):
            perturbation = perturbations[sample_idx].unsqueeze(0)
            perturbed_inputs = inputs + perturbation

            if self.clamp_inputs:
                perturbed_inputs = torch.clamp(perturbed_inputs, 0.0, 1.0)

            if adj_matrix is not None:
                perturbed_preds = model(perturbed_inputs, adj_matrix, num_output_frames=targets.size(1))
            else:
                perturbed_preds = model(perturbed_inputs)

            lss_loss = lss_loss + masked_mse_torch(perturbed_preds, original_preds.detach(), valid_mask)

        lss_loss = lss_loss / self.k_samples
        total_loss = mse_loss + self.lambda_reg * lss_loss
        return total_loss, mse_loss, lss_loss
