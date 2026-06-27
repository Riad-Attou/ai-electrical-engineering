"""
PyTorch dataset and normalization for BDC motor speed filter training.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch.utils.data import Dataset


@dataclass
class NormStats:
    """
    Z-score statistics computed from the training split only.
    Apply to all splits before building datasets to avoid leakage.
    """

    noisy_mean: float
    noisy_std: float
    volt_mean: float
    volt_std: float
    true_mean: float
    true_std: float

    @staticmethod
    def from_split(split) -> "NormStats":
        """Compute stats from the training arrays of a MotorSplit."""
        return NormStats(
            noisy_mean=float(split.train_noisy.mean()),
            noisy_std=float(split.train_noisy.std()),
            volt_mean=float(split.train_voltage.mean()),
            volt_std=float(split.train_voltage.std()),
            true_mean=float(split.train_true.mean()),
            true_std=float(split.train_true.std()),
        )

    def _norm(self, x: np.ndarray, mean: float, std: float) -> np.ndarray:
        return (x - mean) / (std + 1e-8)

    def norm_noisy(self, x: np.ndarray) -> np.ndarray:
        return self._norm(x, self.noisy_mean, self.noisy_std)

    def norm_volt(self, x: np.ndarray) -> np.ndarray:
        return self._norm(x, self.volt_mean, self.volt_std)

    def norm_true(self, x: np.ndarray) -> np.ndarray:
        return self._norm(x, self.true_mean, self.true_std)

    def denorm_true(self, x: np.ndarray) -> np.ndarray:
        return x * (self.true_std + 1e-8) + self.true_mean


class BDCFilterDataset(Dataset):
    """
    Sliding-window dataset for causal speed denoising.

    Each sample:
      x : (W, 2)  float32  —  [omega_noisy_norm, voltage_norm]  for W steps
      y : ()      float32  —  omega_true_norm  at the last step of the window

    Windows are extracted within each trajectory independently,
    so no sample ever bridges two trajectories (no leakage).

    Parameters
    ----------
    omega_noisy : (N_traj, T)  already normalized
    omega_true  : (N_traj, T)  already normalized
    voltage     : (N_traj, T)  already normalized
    window      : context length W in timesteps
    """

    def __init__(
        self,
        omega_noisy: np.ndarray,
        omega_true: np.ndarray,
        voltage: np.ndarray,
        window: int,
        use_voltage: bool = True,
    ):
        super().__init__()
        N, T = omega_noisy.shape
        assert window <= T, f"window ({window}) > trajectory length ({T})"

        self._noisy = omega_noisy.astype(np.float32)
        self._true = omega_true.astype(np.float32)
        self._voltage = voltage.astype(np.float32)
        self._W = window
        self._use_voltage = use_voltage

        # Flat index as numpy arrays for memory efficiency
        n_win = T - window + 1
        self._traj = np.repeat(np.arange(N, dtype=np.int32), n_win)
        # t_end is exclusive: window spans [t_end - W, t_end)
        self._tend = np.tile(np.arange(window, T + 1, dtype=np.int32), N)

    def __len__(self) -> int:
        return len(self._traj)

    def __getitem__(self, idx: int):
        n = int(self._traj[idx])
        t = int(self._tend[idx])
        W = self._W
        noisy_win = self._noisy[n, t - W : t]
        if self._use_voltage:
            x = np.stack([noisy_win, self._voltage[n, t - W : t]], axis=-1)  # (W, 2)
        else:
            x = noisy_win[:, np.newaxis]  # (W, 1)
        y = self._true[n, t - 1]
        return torch.from_numpy(x), torch.tensor(y)
