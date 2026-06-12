"""1D CNN speed filter — drop-in alternative to GRUFilter."""

import torch
import torch.nn as nn


class CNNFilter(nn.Module):
    """
    Stacked causal 1D convolutions for speed denoising.

    Uses valid convolution (no padding): each layer shrinks the sequence
    by (kernel_size - 1).  The receptive field after `depth` layers is
    1 + depth * (kernel_size - 1) timesteps.

    Requires  window >= depth * (kernel_size - 1) + 1.
    Default   depth=2, kernel=8  →  min window = 15  (well under default W=64).

    Input  : (B, W, input_size)  —  [omega_noisy_norm, voltage_norm]
    Output : (B,)                —  omega_true_norm at the last timestep
    """

    def __init__(
        self,
        input_size: int = 2,
        channels: int = 32,
        kernel_size: int = 8,
        depth: int = 2,
    ):
        super().__init__()
        layers: list[nn.Module] = []
        in_ch = input_size
        for _ in range(depth):
            layers += [nn.Conv1d(in_ch, channels, kernel_size), nn.ReLU()]
            in_ch = channels
        self.conv = nn.Sequential(*layers)
        self.head = nn.Linear(channels, 1)
        self._min_window = depth * (kernel_size - 1) + 1

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, W, C) → (B, C, W) for Conv1d
        x = x.permute(0, 2, 1)
        assert x.shape[-1] >= self._min_window, (
            f"Window {x.shape[-1]} < minimum {self._min_window} for this CNNFilter config"
        )
        out = self.conv(x)  # (B, channels, W')
        return self.head(out[:, :, -1]).squeeze(-1)  # last step → (B,)
