"""
Non-learned baselines for ML2 backlash compensation.

Rigid coupling   : theta_l_est = enc_m / N
                   Assumes no backlash — fails at every gear reversal.

Output encoder   : theta_l_est = enc_o
                   The physical output encoder, with eccentricity error,
                   quantisation noise, and 0.5 ms read latency.
"""

from __future__ import annotations

import numpy as np

from utils.servo import ServoSplit


def rigid_coupling_rmse(split: ServoSplit, start_idx: int = 0) -> float:
    """
    RMSE of enc_m / N against theta_l on the test split.

    start_idx aligns the window offset from model predictions so the
    same time range is used for fair comparison.
    """
    pred = split.test_enc_m[start_idx:] / split.N
    true = split.test_theta_l[start_idx:]
    return float(np.sqrt(np.mean((pred - true) ** 2)))


def enc_o_rmse(split: ServoSplit, start_idx: int = 0) -> float:
    """RMSE of the output encoder (enc_o) against theta_l on the test split."""
    pred = split.test_enc_o[start_idx:]
    true = split.test_theta_l[start_idx:]
    return float(np.sqrt(np.mean((pred - true) ** 2)))


def run_all_baselines(split: ServoSplit, start_idx: int = 0) -> dict[str, float]:
    """
    Compute all non-learned baselines on the test split.
    Returns {method_name: RMSE_in_radians}.
    start_idx is used to align with model predictions (window - 1).
    """
    return {
        "Rigid coupling  (enc_m/N)": rigid_coupling_rmse(split, start_idx),
        "Output encoder  (enc_o)":   enc_o_rmse(split, start_idx),
    }
