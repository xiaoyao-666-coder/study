"""Baseline loss functions for the GCCL soil-moisture model."""

import torch.nn as nn

from metrics import masked_mse_torch


class MSELoss(nn.Module):
    def __init__(self):
        super(MSELoss, self).__init__()

    def forward(self, predictions, targets, valid_mask):
        return masked_mse_torch(predictions, targets, valid_mask)
