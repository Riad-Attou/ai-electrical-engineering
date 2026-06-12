"""GRU-based speed filter."""

import torch
import torch.nn as nn


class GRUFilter(nn.Module):
    """
    Single-output GRU speed filter.

    The GRU hidden state accumulates context across the window, naturally
    matching the motor's first-order electrical and mechanical dynamics.

    Input  : (B, W, input_size)  —  [omega_noisy_norm, voltage_norm]
    Output : (B,)                —  omega_true_norm at the last timestep
    """

    def __init__(
        self,
        input_size: int = 2,
        hidden_size: int = 32,
        num_layers: int = 1,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.head = nn.Linear(hidden_size, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.gru(x)  # (B, W, hidden_size)
        return self.head(out[:, -1, :]).squeeze(-1)  # (B,)
