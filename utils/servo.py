"""
Servo with gearbox + backlash — data loading and dataset split.

Two-inertia model (motor + load) coupled through a reduction gear with
compliant backlash.  Physics originally implemented in MATLAB/Simulink;
the pre-recorded CSV datasets are the source of truth.

ML2 task (backlash compensation)
---------------------------------
  Input  : (pwm, enc_m) — PWM command + motor-side encoder position
  Target : theta_l      — true output (load) angle
  Baseline: enc_m / N   — rigid coupling assumption (no backlash model)
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Physical parameters (mirror of servo_params.m)
# ---------------------------------------------------------------------------

GEAR_RATIO: float = 50.0        # reduction ratio N [-]
ENC_M_BITS: int   = 12          # motor encoder resolution [bits/rev]
ENC_O_BITS: int   = 14          # output encoder resolution [bits/rev]
GAP_OUT:    float = 0.02        # output-side half-backlash [rad]
GAP_MOTOR:  float = GEAR_RATIO * GAP_OUT  # motor-side half-backlash [rad]


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_and_resample_csv(
    path: str | Path,
    dt_target: float = 1e-3,
) -> dict[str, np.ndarray]:
    """
    Load a servo CSV and resample every column to a uniform dt_target grid.

    The Simulink solver uses a variable step (MaxStep = dt/2 = 0.5 ms), so
    the raw CSV has ~66 k rows over 30 s.  After resampling to 1 ms we get
    exactly 30 001 points.

    Returns a dict col → (T,) float32 array, plus 't' as float64.
    """
    df = pd.read_csv(path)
    t_orig = df["t"].to_numpy(dtype=np.float64)
    t_new  = np.arange(0.0, t_orig[-1] + dt_target * 0.5, dt_target)

    result: dict[str, np.ndarray] = {"t": t_new}
    for col in df.columns:
        if col == "t":
            continue
        result[col] = np.interp(t_new, t_orig,
                                df[col].to_numpy(dtype=np.float64)).astype(np.float32)
    return result


# ---------------------------------------------------------------------------
# ServoSplit
# ---------------------------------------------------------------------------

@dataclass
class ServoSplit:
    """
    Train / val / test arrays for the ML2 backlash-compensation task.

    All signal arrays are 1-D, shape (T,), at a fixed dt.

    Features  : pwm   — PWM command in [-1, 1]
                enc_m — motor-side encoder angle [rad]
    Target    : theta_l — true output (load) angle [rad]
    Auxiliary : enc_o — output encoder (noisy, used only as a baseline)
    Time axis : t     — seconds
    """

    # Training split (chirp, first 70 %)
    train_t:       np.ndarray
    train_pwm:     np.ndarray
    train_enc_m:   np.ndarray
    train_enc_o:   np.ndarray
    train_theta_l: np.ndarray

    # Validation split (chirp, last 30 %)
    val_t:       np.ndarray
    val_pwm:     np.ndarray
    val_enc_m:   np.ndarray
    val_enc_o:   np.ndarray
    val_theta_l: np.ndarray

    # Test split (multisine — unseen excitation)
    test_t:       np.ndarray
    test_pwm:     np.ndarray
    test_enc_m:   np.ndarray
    test_enc_o:   np.ndarray
    test_theta_l: np.ndarray

    dt: float = 1e-3
    N:  float = GEAR_RATIO

    def save(self, path: str | Path) -> None:
        np.savez(path, **{k: v for k, v in self.__dict__.items()})

    @staticmethod
    def load(path: str | Path) -> "ServoSplit":
        d = np.load(path)
        return ServoSplit(**{k: (float(d[k]) if d[k].ndim == 0 else d[k])
                             for k in d.files})


# ---------------------------------------------------------------------------
# Builder
# ---------------------------------------------------------------------------

def build_servo_split(
    train_csv: str | Path,
    test_csv:  str | Path,
    val_frac:  float = 0.30,
    dt:        float = 1e-3,
) -> ServoSplit:
    """
    Load train and test CSVs, resample to fixed dt, split training set
    temporally (no shuffle) into train / val.
    """
    train = load_and_resample_csv(train_csv, dt)
    test  = load_and_resample_csv(test_csv,  dt)

    T       = len(train["t"])
    n_train = T - int(T * val_frac)

    def _slc(data: dict, key: str, sl: slice) -> np.ndarray:
        return data[key][sl]

    sl_tr = slice(0, n_train)
    sl_va = slice(n_train, None)

    cols = ("t", "pwm", "enc_m", "enc_o", "theta_l")
    kw: dict = {}
    for c in cols:
        kw[f"train_{c}"] = _slc(train, c, sl_tr)
        kw[f"val_{c}"]   = _slc(train, c, sl_va)
        kw[f"test_{c}"]  = test[c]

    return ServoSplit(**kw, dt=dt, N=GEAR_RATIO)
