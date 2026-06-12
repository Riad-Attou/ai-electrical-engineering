"""Temporal Convolutional Network (TCN) with dilated causal convolutions."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class _CausalConv1d(nn.Module):
    """Conv1d with left-only zero-padding so the output at step t uses only steps <= t."""

    def __init__(self, in_ch: int, out_ch: int, kernel_size: int, dilation: int):
        super().__init__()
        self._pad = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(in_ch, out_ch, kernel_size, dilation=dilation)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(F.pad(x, (self._pad, 0)))


class _TCNBlock(nn.Module):
    """Two causal dilated convolutions with a residual connection."""

    def __init__(self, channels: int, kernel_size: int, dilation: int, dropout: float):
        super().__init__()
        self.conv1 = _CausalConv1d(channels, channels, kernel_size, dilation)
        self.conv2 = _CausalConv1d(channels, channels, kernel_size, dilation)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.drop(F.relu(self.conv1(x)))
        out = self.drop(F.relu(self.conv2(out)))
        return F.relu(x + out)


class TCNFilter(nn.Module):
    """
    Temporal Convolutional Network for speed denoising.

    Dilation doubles each level: 1, 2, 4, ..., 2^(n_levels-1).
    With two convolutions per block, the receptive field is:
        RF = 1 + 2 * (kernel_size - 1) * (2^n_levels - 1)  timesteps

    Default (kernel=4, n_levels=4):  RF = 91 ms  —  covers the motor's
    mechanical time constant (~100 ms) unlike the plain CNN (15 ms).

    Input  : (B, W, input_size)
    Output : (B,)
    """

    def __init__(
        self,
        input_size: int = 2,
        channels: int = 32,
        kernel_size: int = 4,
        n_levels: int = 4,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.input_proj = nn.Conv1d(input_size, channels, 1)
        self.blocks = nn.Sequential(
            *[
                _TCNBlock(channels, kernel_size, dilation=2**i, dropout=dropout)
                for i in range(n_levels)
            ]
        )
        self.head = nn.Linear(channels, 1)
        self._rf = 1 + 2 * (kernel_size - 1) * (2**n_levels - 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.permute(0, 2, 1)  # (B, input_size, W)
        x = self.input_proj(x)  # (B, channels, W)
        x = self.blocks(x)  # (B, channels, W)
        return self.head(x[:, :, -1]).squeeze(-1)  # (B,)

    @property
    def receptive_field(self) -> int:
        return self._rf
