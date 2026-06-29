#!/usr/bin/env python3
"""
Compute RMSE and effective delay for every method, then generate two figures:
  figures/metric_variance.png  — RMSE bar chart (error std, rad/s)
  figures/metric_delay.png     — effective delay bar chart (ms, via cross-correlation)
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from numpy.lib.stride_tricks import sliding_window_view
from scipy.signal import lfilter

from models.cnn_filter import CNNFilter
from models.gru_filter import GRUFilter
from models.tcn_filter import TCNFilter
from utils.baselines import optimize_ema, optimize_kalman, kalman_predict_one
from utils.dataset import NormStats
from utils.motor import BDCMotorParams
from utils.traj import MotorSplit

FIG = Path("figures")
FIG.mkdir(exist_ok=True)

WINDOW = 64
_BASE_PARAMS = BDCMotorParams(R=3.0, L=4e-3, Kt=0.05, Kb=0.05, J=7.04e-5, B=0.005, V_max=12.0)

# ── helpers ──────────────────────────────────────────────────────────────────

def estimate_delay_ms(pred_all: np.ndarray, true_all: np.ndarray, dt: float,
                      max_lag: int = 60) -> float:
    """
    Per-trajectory MSE-minimization lag estimate.
    Find d in [-max_lag, +max_lag] that minimizes MSE(pred[t], true[t-d]).
    Positive d = pred lags behind true.
    Returns median delay across trajectories in ms.
    """
    delays = []
    for pred, true in zip(pred_all, true_all):
        best_mse, best_lag = np.inf, 0
        for lag in range(-max_lag, max_lag + 1):
            if lag == 0:
                mse = np.mean((pred - true) ** 2)
            elif lag > 0:
                mse = np.mean((pred[lag:] - true[:-lag]) ** 2)
            else:
                mse = np.mean((pred[:lag] - true[-lag:]) ** 2)
            if mse < best_mse:
                best_mse, best_lag = mse, lag
        delays.append(best_lag * dt * 1000)
    return float(np.median(delays))


def rmse(pred: np.ndarray, true: np.ndarray) -> float:
    return float(np.sqrt(np.mean((pred - true) ** 2)))



# ── load ─────────────────────────────────────────────────────────────────────

split  = MotorSplit.load("data/rod_split.npz")
stats  = NormStats.from_split(split)
dt     = split.dt
device = "cuda" if torch.cuda.is_available() else "cpu"

print("Tuning baselines …")
best_alpha, _ = optimize_ema(split)
q_diag, r_var, _ = optimize_kalman(split, _BASE_PARAMS)
print(f"  EMA α={best_alpha:.3f}   Kalman R={r_var:.2g} Qw={q_diag[1]:.2g}")

ckpt = Path("checkpoints")
model_defs = {
    "GRU": (ckpt / "best_gru.pt", GRUFilter(input_size=2, hidden_size=32, num_layers=1)),
    "CNN": (ckpt / "best_cnn.pt", CNNFilter(input_size=2, channels=32, kernel_size=8, depth=2)),
    "TCN": (ckpt / "best_tcn.pt", TCNFilter(input_size=2, channels=32, kernel_size=4, n_levels=4)),
}
models = {}
for key, (path, m) in model_defs.items():
    if path.exists():
        m.load_state_dict(torch.load(path, weights_only=True, map_location=device))
        models[key] = m.to(device).eval()

# ── compute metrics ───────────────────────────────────────────────────────────

true_2d   = split.test_true       # (N, T)
noisy_2d  = split.test_noisy      # (N, T)

ema_pred_2d = lfilter([1 - best_alpha], [1, -best_alpha], noisy_2d, axis=-1)

kf_pred_2d = np.stack([
    kalman_predict_one(noisy_2d[n], split.test_voltage[n],
                       _BASE_PARAMS, dt, q_diag, r_var)
    for n in range(noisy_2d.shape[0])
])

# Neural model predictions (shorter: T-WINDOW+1 per traj) → pad to align with true_2d
def model_predict_2d(model, split, stats, device):
    preds, trues = [], []
    for n in range(split.test_noisy.shape[0]):
        noisy = split.test_noisy[n]; volt = split.test_voltage[n]
        win_n = sliding_window_view(stats.norm_noisy(noisy), WINDOW)
        win_v = sliding_window_view(stats.norm_volt(volt),   WINDOW)
        x = np.stack([win_n, win_v], axis=-1).astype(np.float32)
        with torch.no_grad():
            p = model(torch.from_numpy(x).to(device)).cpu().numpy()
        preds.append(stats.denorm_true(p))
        trues.append(split.test_true[n, WINDOW - 1:])
    return np.array(preds), np.array(trues)

# Each entry: (pred_2d, true_2d) — same shape per method
results: dict[str, tuple[np.ndarray, np.ndarray]] = {
    "Raw":    (noisy_2d,    true_2d),
    "EMA":    (ema_pred_2d, true_2d),
    "Kalman": (kf_pred_2d,  true_2d),
}
for key, model in models.items():
    pred_2d, true_2d_m = model_predict_2d(model, split, stats, device)
    results[key] = (pred_2d, true_2d_m)

metrics: dict[str, tuple[float, float]] = {}
for name, (pred, true) in results.items():
    r = float(np.sqrt(np.mean((pred - true) ** 2)))
    lag = estimate_delay_ms(pred, true, dt)
    metrics[name] = (r, lag)
    print(f"  {name:8s}  RMSE={r:.3f} rad/s   delay={lag:.1f} ms")

# ── plot ─────────────────────────────────────────────────────────────────────

ORDER  = [k for k in ("Raw", "EMA", "Kalman", "CNN", "TCN", "GRU") if k in metrics]
COLORS = {
    "Raw":    "#aaaaaa",
    "EMA":    "#e07b00",
    "Kalman": "#9b59b6",
    "CNN":    "#3498db",
    "TCN":    "#1abc9c",
    "GRU":    "#2ecc71",
}
NEURAL = {"CNN", "TCN", "GRU"}

BG = "#FBFAF6"

# ── figure 1: RMSE (variance) ─────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(8, 4.8))
fig.patch.set_facecolor(BG); ax.set_facecolor(BG)
vals = [metrics[k][0] for k in ORDER]
bars = ax.bar(ORDER, vals, color=[COLORS[k] for k in ORDER],
              edgecolor="#333", linewidth=0.7, width=0.55)
for b, v in zip(bars, vals):
    ax.text(b.get_x() + b.get_width() / 2, v + 0.03, f"{v:.2f}",
            ha="center", va="bottom", fontsize=11, fontweight="bold")
ax.set_ylabel("RMSE  [rad/s]", fontsize=13)
ax.set_title("Error variance  (RMSE, lower = better)", fontsize=13, fontweight="bold", pad=10)
ax.grid(True, axis="y", alpha=0.35, color="#ccc")
ax.spines[["top", "right"]].set_visible(False)
ax.set_ylim(0, max(vals) * 1.22)
# shade neural region
neural_idx = [i for i, k in enumerate(ORDER) if k in NEURAL]
if neural_idx:
    ax.axvspan(min(neural_idx) - 0.45, max(neural_idx) + 0.45,
               color="#e7f2f0", alpha=0.55, zorder=0, label="neural filters")
    ax.legend(fontsize=10)
plt.tight_layout()
out1 = FIG / "metric_variance.png"
plt.savefig(out1, dpi=160, bbox_inches="tight")
plt.close(fig)
print(f"saved -> {out1}")

# ── figure 2: delay ───────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(8, 4.8))
fig.patch.set_facecolor(BG); ax.set_facecolor(BG)
delays = [metrics[k][1] for k in ORDER]
bars = ax.bar(ORDER, delays, color=[COLORS[k] for k in ORDER],
              edgecolor="#333", linewidth=0.7, width=0.55)
for b, v in zip(bars, delays):
    ax.text(b.get_x() + b.get_width() / 2, v + 0.15, f"{v:.1f}",
            ha="center", va="bottom", fontsize=11, fontweight="bold")
ax.set_ylabel("Effective delay  [ms]", fontsize=13)
ax.set_title("Phase shift  (effective lag, lower = better)", fontsize=13, fontweight="bold", pad=10)
ax.grid(True, axis="y", alpha=0.35, color="#ccc")
ax.spines[["top", "right"]].set_visible(False)
ax.set_ylim(bottom=min(0, min(delays)) - 0.5)
if neural_idx:
    ax.axvspan(min(neural_idx) - 0.45, max(neural_idx) + 0.45,
               color="#e7f2f0", alpha=0.55, zorder=0, label="neural filters")
    ax.legend(fontsize=10)
plt.tight_layout()
out2 = FIG / "metric_delay.png"
plt.savefig(out2, dpi=160, bbox_inches="tight")
plt.close(fig)
print(f"saved -> {out2}")
