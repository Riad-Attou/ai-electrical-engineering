"""
PyTorch dataset and normalization for servo backlash compensation (ML2).

Feature engineering to handle train/test distribution shift
------------------------------------------------------------
The chirp (train) causes enc_m to drift from 0 to ~51 rad while the
multisine (test) oscillates around 0.  Absolute enc_m cannot be Z-scored
from train stats and applied to test.

Fix — two derived signals that are stationary on both splits:

  enc_m_rel : enc_m relative to the start of each window.
              Shows how much the motor has moved within the window.
              Always starts at 0; amplitude ~ intra-window displacement.

  backlash_error : theta_l − enc_m / N
              Rigid-coupling residual.  Bounded by ±gap_out = ±0.02 rad;
              zero whenever the gear is engaged; nonzero inside the dead zone.

Both signals have similar distributions in train and test and Z-score
cleanly from training statistics.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from utils.servo import ServoSplit


@dataclass
class NormStats:
    """Z-score statistics for the derived signals, computed from training data."""

    enc_rel_std:  float   # std of intra-window relative encoder displacement
    pwm_mean:     float
    pwm_std:      float
    error_std:    float   # std of (theta_l − enc_m/N) on training split

    @staticmethod
    def from_split(split: ServoSplit) -> "NormStats":
        # enc_m_rel: compute per-window deltas from the flat series
        # Use overall std of the first-difference as a proxy for intra-window std
        enc_diff = np.diff(split.train_enc_m, prepend=split.train_enc_m[0])
        error    = split.train_theta_l - split.train_enc_m / split.N
        return NormStats(
            enc_rel_std=float(enc_diff.std()) + 1e-8,
            pwm_mean=float(split.train_pwm.mean()),
            pwm_std=float(split.train_pwm.std()) + 1e-8,
            error_std=float(error.std()) + 1e-8,
        )

    def norm_pwm(self, x: np.ndarray) -> np.ndarray:
        return (x - self.pwm_mean) / self.pwm_std

    def norm_enc_rel(self, x: np.ndarray) -> np.ndarray:
        return x / self.enc_rel_std

    def norm_error(self, x: np.ndarray) -> np.ndarray:
        return x / self.error_std

    def denorm_error(self, x: np.ndarray) -> np.ndarray:
        return x * self.error_std

    def save(self, path: str | Path) -> None:
        np.savez(path, enc_rel_std=self.enc_rel_std,
                 pwm_mean=self.pwm_mean, pwm_std=self.pwm_std,
                 error_std=self.error_std)

    @staticmethod
    def load(path: str | Path) -> "NormStats":
        d = np.load(path)
        return NormStats(**{k: float(d[k]) for k in d.files})


class ServoDataset(Dataset):
    """
    Sliding-window dataset for causal backlash compensation.

    Each sample
    -----------
    x : (W, 2) float32 — [enc_m_rel_norm, pwm_norm]
        enc_m_rel = enc_m − enc_m[window_start]  (relative displacement)
    y : ()     float32 — normalised (theta_l − enc_m/N) at the last step

    Parameters
    ----------
    enc_m        : (T,) raw motor encoder angle [rad]
    pwm          : (T,) normalised PWM command
    backlash_err : (T,) normalised (theta_l − enc_m/N)
    enc_rel_std  : scalar used to normalise enc_m_rel in __getitem__
    window       : context length W in timesteps
    """

    def __init__(
        self,
        enc_m:        np.ndarray,
        pwm_norm:     np.ndarray,
        backlash_err: np.ndarray,
        enc_rel_std:  float,
        window:       int,
    ):
        super().__init__()
        T = len(enc_m)
        assert len(pwm_norm) == T and len(backlash_err) == T
        assert window <= T, f"window ({window}) > series length ({T})"

        self._enc_m  = enc_m.astype(np.float32)
        self._pwm    = pwm_norm.astype(np.float32)
        self._err    = backlash_err.astype(np.float32)
        self._std    = float(enc_rel_std)
        self._W      = window
        self._n      = T - window + 1

    def __len__(self) -> int:
        return self._n

    def __getitem__(self, idx: int):
        W       = self._W
        enc_win = self._enc_m[idx : idx + W]
        enc_rel = (enc_win - enc_win[0]) / self._std   # window-relative, normalised
        pw      = self._pwm[idx : idx + W]
        x       = np.stack([enc_rel, pw], axis=-1)     # (W, 2)
        y       = self._err[idx + W - 1]
        return torch.from_numpy(x), torch.tensor(y)
