"""
Non-learned baselines for BDC motor speed filtering.

EMA:    Exponential Moving Average, alpha optimized on the validation set.
        Less lag than a plain MA; single tunable parameter.

Kalman: Steady-state linear Kalman filter using the nominal BDC motor model.
        Represents what a control engineer would deploy: the optimal linear
        estimator under Gaussian noise when the plant model is known.
        Running on test trajectories with ±15% parameter variation tests
        robustness to model mismatch — the realistic deployment scenario.
"""

from __future__ import annotations

import numpy as np
from scipy.linalg import expm, solve_discrete_are
from scipy.signal import lfilter

from utils.motor import BDCMotorParams
from utils.traj import MotorSplit

# ---------------------------------------------------------------------------
# EMA
# ---------------------------------------------------------------------------


def _ema(noisy: np.ndarray, alpha: float) -> np.ndarray:
    """Causal EMA via IIR filter. noisy: (N, T) or (T,). Returns same shape."""
    return lfilter([1.0 - alpha], [1.0, -alpha], noisy, axis=-1)


def ema_rmse(split: MotorSplit, alpha: float, on: str = "test") -> float:
    noisy = getattr(split, f"{on}_noisy")
    true = getattr(split, f"{on}_true")
    return float(np.sqrt(np.mean((_ema(noisy, alpha) - true) ** 2)))


def optimize_ema(split: MotorSplit, n_grid: int = 60) -> tuple[float, float]:
    """Grid search over alpha on the val set. Returns (best_alpha, val_rmse)."""
    candidates = [
        (a, ema_rmse(split, a, on="val")) for a in np.linspace(0.50, 0.999, n_grid)
    ]
    return min(candidates, key=lambda x: x[1])


# ---------------------------------------------------------------------------
# Kalman filter
# ---------------------------------------------------------------------------


def _discretize_motor(params: BDCMotorParams, dt: float):
    """
    ZOH (zero-order hold) exact discretization of the BDC motor [i, ω] state.

    Continuous model:
        d[i]/dt    = -R/L · i  - Kb/L · ω  + 1/L · V
        d[ω]/dt    =  Kt/J · i - B/J  · ω

    ZOH gives A_d = expm(A_c · dt) and the exact input-to-state matrix B_d,
    which avoids the instability of Euler discretization when dt > 2·τ_electrical
    (τ_e = L/R = 0.5 ms here, dt = 1 ms → Euler would be unstable).
    """
    R, L, Kt, Kb, J, B = (params.R, params.L, params.Kt, params.Kb, params.J, params.B)
    A_c = np.array([[-R / L, -Kb / L], [Kt / J, -B / J]])
    A_d = expm(A_c * dt)
    B_c = np.array([1.0 / L, 0.0])
    B_d = np.linalg.solve(A_c, (A_d - np.eye(2)) @ B_c)
    return A_d, B_d


def kalman_predict_one(
    noisy: np.ndarray,
    volt: np.ndarray,
    params: BDCMotorParams,
    dt: float,
    Q_diag: tuple[float, float] = (10.0, 1.0),
    R_var: float = 300.0,
) -> np.ndarray:
    """Steady-state Kalman predictions for a single trajectory. Returns (T,)."""
    A_d, B_d = _discretize_motor(params, dt)
    C = np.array([[0.0, 1.0]])
    Q = np.diag(Q_diag)
    R_m = np.array([[R_var]])
    P_ss = solve_discrete_are(A_d.T, C.T, Q, R_m)
    K_ss = (P_ss @ C.T @ np.linalg.inv(C @ P_ss @ C.T + R_m)).flatten()

    T = len(noisy)
    x = np.zeros(2)
    preds = np.empty(T, dtype=np.float32)
    for t in range(T):
        x_pred = A_d @ x + B_d * volt[t]
        innov = noisy[t] - x_pred[1]
        x = x_pred + K_ss * innov
        preds[t] = x[1]
    return preds


def kalman_rmse(
    split: MotorSplit,
    params: BDCMotorParams,
    Q_diag: tuple[float, float] = (10.0, 1.0),
    R_var: float = 300.0,
    on: str = "test",
) -> float:
    """
    Steady-state Kalman filter RMSE.

    State  : x = [i, ω]ᵀ
    Measure: z = ω + noise,  noise ~ N(0, R_var)
    Q_diag : process noise variances for [i, ω] per timestep.
             Reflects model mismatch from ±15% parameter variation.
    R_var  : measurement noise variance — use the average of the noise
             std range (5–30 rad/s) → (17.5)² ≈ 306.

    The steady-state gain is computed once via the discrete Riccati equation
    and then applied as a constant-gain IIR filter over every trajectory.
    Vectorised over the N test trajectories for speed.
    """
    A_d, B_d = _discretize_motor(params, split.dt)
    C = np.array([[0.0, 1.0]])  # observe ω
    Q = np.diag(Q_diag)
    R_m = np.array([[R_var]])

    # Solve discrete algebraic Riccati equation for steady-state P
    P_ss = solve_discrete_are(A_d.T, C.T, Q, R_m)
    K_ss = (P_ss @ C.T @ np.linalg.inv(C @ P_ss @ C.T + R_m)).flatten()  # (2,)

    noisy = getattr(split, f"{on}_noisy")  # (N, T)
    true = getattr(split, f"{on}_true")
    volt = getattr(split, f"{on}_voltage")
    N, T = noisy.shape

    x = np.zeros((N, 2))  # initial state
    preds = np.empty((N, T), dtype=np.float32)

    for t in range(T):
        x_pred = x @ A_d.T + np.outer(volt[:, t], B_d)  # (N, 2)
        innov = noisy[:, t] - x_pred[:, 1]  # (N,)
        x = x_pred + np.outer(innov, K_ss)  # (N, 2)
        preds[:, t] = x[:, 1]

    return float(np.sqrt(np.mean((preds - true) ** 2)))


# ---------------------------------------------------------------------------
# Convenience: run all non-learned baselines at once
# ---------------------------------------------------------------------------


def run_all_baselines(
    split: MotorSplit,
    params: BDCMotorParams,
    ma_window: int = 64,
) -> dict[str, float]:
    """
    Compute MA, EMA (optimised), and Kalman RMSE on the test set.
    EMA and Kalman are tuned/validated on the val set only.
    Returns a dict mapping method name → test RMSE [rad/s].
    """
    from numpy.lib.stride_tricks import sliding_window_view

    # MA (already computed in train.py, reproduced here for the dict)
    win = sliding_window_view(split.test_noisy, ma_window, axis=1)
    ma_rmse = float(
        np.sqrt(np.mean((win.mean(axis=-1) - split.test_true[:, ma_window - 1 :]) ** 2))
    )

    # EMA — find best alpha on val
    best_alpha, _ = optimize_ema(split)
    ema_test = ema_rmse(split, best_alpha, on="test")

    # Kalman
    kf_test = kalman_rmse(split, params)

    return {
        f"MA  (window={ma_window})": ma_rmse,
        f"EMA (α={best_alpha:.3f})": ema_test,
        "Kalman (nominal model)": kf_test,
    }
