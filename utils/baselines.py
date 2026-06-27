"""
Non-learned baselines for motor+rod speed filtering.

MA:     Causal moving average over the window. Simplest smoother.

EMA:    Exponential Moving Average, alpha optimized on the validation set.
        Less lag than a plain MA; single tunable parameter.

Kalman: Steady-state linear Kalman filter using the nominal *linear* motor
        model [i, omega]. This is what a control engineer would deploy: the
        optimal linear estimator under Gaussian noise when the plant model is
        known. Crucially, the linear model does NOT include the rod's gravity
        torque (proportional to sin(theta)), which on this plant acts as an
        unmodeled, state-dependent disturbance. The measurement-noise variance
        R and process-noise variance Q are tuned on the validation set so the
        Kalman is a fair, well-configured baseline — not a strawman.
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
# Kalman filter (steady-state, nominal linear motor model)
# ---------------------------------------------------------------------------


def _discretize_motor(params: BDCMotorParams, dt: float):
    """
    ZOH (zero-order hold) exact discretization of the BDC motor [i, omega] state.

    Continuous model:
        d[i]/dt     = -R/L * i  - Kb/L * omega  + 1/L * V
        d[omega]/dt =  Kt/J * i - B/J  * omega

    ZOH gives A_d = expm(A_c * dt) and the exact input-to-state matrix B_d,
    which avoids the instability of Euler discretization when dt > 2*tau_e
    (tau_e = L/R; here L/R is small so Euler would be ill-conditioned).
    """
    R, L, Kt, Kb, J, B = (params.R, params.L, params.Kt, params.Kb, params.J, params.B)
    A_c = np.array([[-R / L, -Kb / L], [Kt / J, -B / J]])
    A_d = expm(A_c * dt)
    B_c = np.array([1.0 / L, 0.0])
    B_d = np.linalg.solve(A_c, (A_d - np.eye(2)) @ B_c)
    return A_d, B_d


def _kalman_gain(params: BDCMotorParams, dt: float, Q_diag, R_var: float):
    A_d, B_d = _discretize_motor(params, dt)
    C = np.array([[0.0, 1.0]])  # observe omega
    Q = np.diag(Q_diag)
    R_m = np.array([[R_var]])
    P_ss = solve_discrete_are(A_d.T, C.T, Q, R_m)
    K_ss = (P_ss @ C.T @ np.linalg.inv(C @ P_ss @ C.T + R_m)).flatten()  # (2,)
    return A_d, B_d, K_ss


def _kalman_run(noisy, volt, A_d, B_d, K_ss) -> np.ndarray:
    """Constant-gain filter over (N, T) arrays. Returns omega estimate (N, T)."""
    N, T = noisy.shape
    x = np.zeros((N, 2))
    preds = np.empty((N, T), dtype=np.float32)
    for t in range(T):
        x_pred = x @ A_d.T + np.outer(volt[:, t], B_d)  # (N, 2)
        innov = noisy[:, t] - x_pred[:, 1]  # (N,)
        x = x_pred + np.outer(innov, K_ss)  # (N, 2)
        preds[:, t] = x[:, 1]
    return preds


def kalman_predict_one(
    noisy: np.ndarray,
    volt: np.ndarray,
    params: BDCMotorParams,
    dt: float,
    Q_diag: tuple[float, float] = (1.0, 0.1),
    R_var: float = 1.0,
) -> np.ndarray:
    """Steady-state Kalman omega estimate for a single (T,) trajectory."""
    A_d, B_d, K_ss = _kalman_gain(params, dt, Q_diag, R_var)
    preds = _kalman_run(noisy[None, :], volt[None, :], A_d, B_d, K_ss)
    return preds[0]


def kalman_rmse(
    split: MotorSplit,
    params: BDCMotorParams,
    Q_diag: tuple[float, float] = (1.0, 0.1),
    R_var: float = 1.0,
    on: str = "test",
) -> float:
    """
    Steady-state Kalman filter RMSE on the chosen split.

    State  : x = [i, omega]^T   (linear motor model, no gravity term)
    Measure: z = omega + noise,  noise ~ N(0, R_var)
    Q_diag : process-noise variances for [i, omega] per timestep. The omega
             entry absorbs the unmodeled gravity disturbance.
    R_var  : measurement-noise variance (matched to the sensor noise std).
    """
    A_d, B_d, K_ss = _kalman_gain(params, split.dt, Q_diag, R_var)
    noisy = getattr(split, f"{on}_noisy")
    true = getattr(split, f"{on}_true")
    volt = getattr(split, f"{on}_voltage")
    preds = _kalman_run(noisy, volt, A_d, B_d, K_ss)
    return float(np.sqrt(np.mean((preds - true) ** 2)))


def optimize_kalman(
    split: MotorSplit,
    params: BDCMotorParams,
    r_grid: np.ndarray | None = None,
    q_grid: np.ndarray | None = None,
) -> tuple[tuple[float, float], float, float]:
    """
    Tune (R_var, Q_omega) on the validation set so the Kalman is well-configured.

    Q_i (process noise on current) is held at a small fixed fraction of Q_omega.
    Returns ((Q_i, Q_omega), best_R_var, val_rmse).
    """
    if r_grid is None:
        r_grid = np.logspace(-2.0, 1.8, 10)  # 0.01 .. ~63 (wide; avoids grid-edge optima)
    if q_grid is None:
        q_grid = np.logspace(-3.0, 0.5, 8)  # 1e-3 .. ~3 (process noise on omega)

    best = None
    for q_w in q_grid:
        q_diag = (q_w * 0.1, float(q_w))
        for r in r_grid:
            rmse = kalman_rmse(split, params, Q_diag=q_diag, R_var=float(r), on="val")
            if best is None or rmse < best[2]:
                best = (q_diag, float(r), rmse)
    return best


# ---------------------------------------------------------------------------
# Convenience: run all non-learned baselines at once
# ---------------------------------------------------------------------------


def run_all_baselines(
    split: MotorSplit,
    params: BDCMotorParams,
    ma_window: int = 64,
) -> dict[str, float]:
    """
    Compute MA, EMA (optimised), and Kalman (tuned) RMSE on the test set.
    EMA and Kalman are tuned/validated on the val set only, then evaluated on
    the test set. Returns a dict mapping method label -> test RMSE [rad/s].
    """
    from numpy.lib.stride_tricks import sliding_window_view

    # Raw — no filtering at all (RMSE of the noisy sensor vs truth)
    raw_rmse = float(np.sqrt(np.mean((split.test_noisy - split.test_true) ** 2)))

    # MA — causal moving average
    win = sliding_window_view(split.test_noisy, ma_window, axis=1)
    ma_rmse = float(
        np.sqrt(np.mean((win.mean(axis=-1) - split.test_true[:, ma_window - 1 :]) ** 2))
    )

    # EMA — tune alpha on val, evaluate on test
    best_alpha, _ = optimize_ema(split)
    ema_test = ema_rmse(split, best_alpha, on="test")

    # Kalman — tune (R_var, Q_omega) on val, evaluate on test
    q_diag, r_var, _ = optimize_kalman(split, params)
    kf_test = kalman_rmse(split, params, Q_diag=q_diag, R_var=r_var, on="test")

    return {
        "Raw (no filter)": raw_rmse,
        f"MA  (window={ma_window})": ma_rmse,
        f"EMA (a={best_alpha:.3f})": ema_test,
        f"Kalman (R={r_var:.2g}, Qw={q_diag[1]:.2g})": kf_test,
    }
