"""
PyTorch dataset and normalization for servo backlash compensation (ML2).

Feature engineering to handle train/test distribution shift
------------------------------------------------------------
The chirp (train) causes enc_m to drift from 0 to ~51 rad while the
multisine (test) oscillates around 0.  Absolute enc_m cannot be Z-scored
from train stats and applied to test.

Normalization strategy — per-window velocity scaling
-----------------------------------------------------
The backlash error observed in this dataset is dominated by damper-lag
(the gear spring never engages: phi << gap_motor always).  The error is
approximately proportional to motor velocity:

    backlash_error ≈ f(omega_m) × tau_load / N

The chirp (train) has 2.4× higher peak motor speed than the multisine
(test).  A fixed error_std computed from training data would therefore
cause a 2.4× distribution shift in the normalised target, causing models
to over-correct on the test set and score WORSE than rigid coupling.

Fix: per-window velocity scaling.  For each window of length W:

    local_vel = std(diff(enc_m_win))  — motor-side velocity estimate

Both the input (enc_m_rel) and the target (backlash_error) are
normalised by local_vel so that a window where the motor barely moves
and one where it spins fast present the same normalised scale to the
network.  The network output is then multiplied by (local_vel / N) at
inference time to recover the prediction in radians.
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
    """Statistics for signals that do NOT use per-window scaling."""

    enc_rel_std:  float   # std of diff(enc_m_train) — reference velocity scale
    pwm_mean:     float
    pwm_std:      float
    error_std:    float   # kept for backwards-compat; not used in __getitem__

    @staticmethod
    def from_split(split: ServoSplit) -> "NormStats":
        enc_diff = np.diff(split.train_enc_m, prepend=split.train_enc_m[0])
        enc_rel_std = float(enc_diff.std()) + 1e-8
        return NormStats(
            enc_rel_std=enc_rel_std,
            pwm_mean=float(split.train_pwm.mean()),
            pwm_std=float(split.train_pwm.std()) + 1e-8,
            error_std=enc_rel_std / split.N,   # reference; actual denorm is per-window
        )

    def norm_pwm(self, x: np.ndarray) -> np.ndarray:
        return (x - self.pwm_mean) / self.pwm_std

    def norm_enc_rel(self, x: np.ndarray) -> np.ndarray:
        return x / self.enc_rel_std

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
        enc_m_rel is normalised by the window's own velocity std (local_vel),
        so the network sees scale-invariant motion patterns regardless of
        how fast the motor is spinning.
    y : ()     float32 — per-window-normalised backlash error
        y = (theta_l − enc_m/N) / (local_vel / N)
        Multiply the network output by (local_vel / N) at inference to
        recover the backlash error in radians.

    Parameters
    ----------
    enc_m        : (T,) raw motor encoder angle [rad]
    pwm_norm     : (T,) normalised PWM command (z-scored from NormStats)
    backlash_err : (T,) RAW (un-normalised) backlash error (theta_l − enc_m/N) [rad]
    N            : gear ratio (used to compute per-window output scale)
    window       : context length W in timesteps
    """

    def __init__(
        self,
        enc_m:        np.ndarray,
        pwm_norm:     np.ndarray,
        backlash_err: np.ndarray,   # raw, in radians
        N:            float,
        window:       int,
    ):
        super().__init__()
        T = len(enc_m)
        assert len(pwm_norm) == T and len(backlash_err) == T
        assert window <= T, f"window ({window}) > series length ({T})"

        self._enc_m  = enc_m.astype(np.float32)
        self._pwm    = pwm_norm.astype(np.float32)
        self._err    = backlash_err.astype(np.float32)   # raw [rad]
        self._N      = float(N)
        self._W      = window
        self._n      = T - window + 1

    def __len__(self) -> int:
        return self._n

    def __getitem__(self, idx: int):
        W       = self._W
        enc_win = self._enc_m[idx : idx + W]
        diff_w  = np.diff(enc_win, prepend=enc_win[0])
        local_vel = float(diff_w.std()) + 1e-6     # motor-side velocity scale
        scale   = local_vel / self._N              # output-side velocity scale [rad]

        enc_rel = (enc_win - enc_win[0]) / local_vel   # per-window normalised
        pw      = self._pwm[idx : idx + W]
        x       = np.stack([enc_rel, pw], axis=-1)     # (W, 2)
        y       = self._err[idx + W - 1] / scale       # per-window normalised target
        return torch.from_numpy(x), torch.tensor(y, dtype=torch.float32)
