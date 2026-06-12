"""
Multi-method comparison figure — all models on one test trajectory.

Usage
-----
python compare.py               # traj 0, full 10 s
python compare.py --traj 2      # different trajectory
python compare.py --t-end 3.0   # zoom to first 3 s
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from numpy.lib.stride_tricks import sliding_window_view
from scipy.signal import lfilter

from models.cnn_filter import CNNFilter
from models.gru_filter import GRUFilter
from models.tcn_filter import TCNFilter
from utils.baselines import kalman_predict_one, optimize_ema
from utils.dataset import NormStats
from utils.motor import BDCMotorParams
from utils.traj import MotorSplit

_BASE_PARAMS = BDCMotorParams(
    R=1.0, L=0.5e-3, Kt=0.01, Kb=0.01, J=1e-5, B=1e-6, V_max=12.0
)

WINDOW = 64

# (label, color, linestyle, linewidth, alpha)
_STYLE: dict[str, tuple] = {
    "Noisy":       ("Noisy measurement",       "#aaaaaa", "-",  1.0, 0.45),
    "EMA":         ("EMA (optimised)",          "#e07b00", "-",  1.6, 0.85),
    "Kalman":      ("Kalman (nominal model)",   "#9b59b6", "-",  1.8, 1.00),
    "GRU_NOVOLT":  ("GRU — speed only",         "#e74c3c", "--", 1.6, 0.90),
    "CNN":         ("CNN (RF = 15 ms)",         "#3498db", "-",  1.6, 0.85),
    "TCN":         ("TCN (RF = 91 ms)",         "#1abc9c", "-",  2.0, 1.00),
    "GRU":         ("GRU — speed + voltage",    "#2ecc71", "-",  2.2, 1.00),
    "True":        ("True speed",               "#1a1a1a", "--", 1.6, 1.00),
}


def _rpm(v: np.ndarray) -> np.ndarray:
    return v * 60 / (2 * np.pi)


def _model_predict(model: torch.nn.Module, noisy: np.ndarray, volt: np.ndarray,
                   stats: NormStats, use_voltage: bool, device: str) -> np.ndarray:
    """Sliding-window inference on one trajectory. Returns (T - W + 1,) rad/s."""
    win_noisy = sliding_window_view(stats.norm_noisy(noisy), WINDOW)
    if use_voltage:
        win_volt = sliding_window_view(stats.norm_volt(volt), WINDOW)
        x = np.stack([win_noisy, win_volt], axis=-1).astype(np.float32)
    else:
        x = win_noisy[:, :, np.newaxis].astype(np.float32)
    model.eval()
    with torch.no_grad():
        pred_n = model(torch.from_numpy(x).to(device)).cpu().numpy()
    return stats.denorm_true(pred_n)


def load_models(device: str) -> dict[str, tuple[torch.nn.Module, bool]]:
    """Load whichever checkpoints exist. Returns {key: (model, use_voltage)}."""
    ckpt_dir = Path("checkpoints")
    candidates = {
        "GRU":        (ckpt_dir / "best_gru.pt",
                       GRUFilter(input_size=2, hidden_size=32, num_layers=1), True),
        "GRU_NOVOLT": (ckpt_dir / "best_gru_novolt.pt",
                       GRUFilter(input_size=1, hidden_size=32, num_layers=1), False),
        "TCN":        (ckpt_dir / "best_tcn.pt",
                       TCNFilter(input_size=2, channels=32, kernel_size=4, n_levels=4), True),
        "CNN":        (ckpt_dir / "best_cnn.pt",
                       CNNFilter(input_size=2, channels=32, kernel_size=8, depth=2), True),
    }
    loaded = {}
    for key, (path, model, use_v) in candidates.items():
        if path.exists():
            model.load_state_dict(torch.load(path, weights_only=True, map_location=device))
            model.to(device)
            loaded[key] = (model, use_v)
            print(f"  Loaded {key:12s} ← {path}")
        else:
            print(f"  Skipped {key:11s} (checkpoint not found: {path})")
    return loaded


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--split",  default="data/motor_split.npz")
    parser.add_argument("--traj",   type=int, default=0, help="Test trajectory index")
    parser.add_argument("--t-end",  type=float, default=None, help="Zoom: plot up to this time (s)")
    parser.add_argument("--out",    default="figures/comparison_all_methods.png")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    figures = Path("figures")
    figures.mkdir(exist_ok=True)

    split = MotorSplit.load(args.split)
    stats = NormStats.from_split(split)
    dt = split.dt
    n = args.traj

    noisy = split.test_noisy[n]
    true  = split.test_true[n]
    volt  = split.test_voltage[n]
    T     = len(noisy)
    t_ax  = np.arange(T) * dt

    # Time window for plotting
    if args.t_end is not None:
        T_plot = min(T, int(args.t_end / dt))
    else:
        T_plot = T

    print("Computing baselines …")
    best_alpha, _ = optimize_ema(split)
    ema_pred = lfilter([1.0 - best_alpha], [1.0, -best_alpha], noisy)
    kalman_pred = kalman_predict_one(noisy, volt, _BASE_PARAMS, dt)

    print("Loading models …")
    models = load_models(device)
    model_preds: dict[str, np.ndarray] = {}
    for key, (model, use_v) in models.items():
        model_preds[key] = _model_predict(model, noisy, volt, stats, use_v, device)

    # -------------------------------------------------------------------------
    # Plot
    # -------------------------------------------------------------------------
    fig, axes = plt.subplots(2, 1, figsize=(14, 7), sharex=True,
                             gridspec_kw={"height_ratios": [3, 2]})

    def _plot(ax, y, key, t_offset=0):
        label, color, ls, lw, alpha = _STYLE[key]
        t = t_ax[t_offset:T_plot]
        ax.plot(t, _rpm(y[:T_plot - t_offset]), color=color, ls=ls,
                lw=lw, alpha=alpha, label=label)

    # Speed panel
    ax = axes[0]
    _plot(ax, noisy, "Noisy")
    _plot(ax, true,  "True")
    _plot(ax, ema_pred, "EMA")
    _plot(ax, kalman_pred, "Kalman")
    for key in ("GRU_NOVOLT", "CNN", "TCN", "GRU"):
        if key in model_preds:
            _plot(ax, model_preds[key], key, t_offset=WINDOW - 1)
    ax.set_ylabel("Speed (RPM)", fontsize=12)
    ax.grid(True, alpha=0.4)
    ax.set_title(
        f"Test trajectory {n} — speed filter comparison  "
        f"(all methods, dt = {dt*1e3:.0f} ms)",
        fontsize=13,
    )

    # Error panel
    ax = axes[1]
    true_full = true
    for key, pred in [("EMA", ema_pred), ("Kalman", kalman_pred)]:
        label, color, ls, lw, alpha = _STYLE[key]
        err = pred[:T_plot] - true_full[:T_plot]
        ax.plot(t_ax[:T_plot], _rpm(err), color=color, ls=ls, lw=lw, alpha=alpha)
    for key in ("GRU_NOVOLT", "CNN", "TCN", "GRU"):
        if key in model_preds:
            label, color, ls, lw, alpha = _STYLE[key]
            t0 = WINDOW - 1
            err = model_preds[key][:T_plot - t0] - true_full[t0:T_plot]
            ax.plot(t_ax[t0:T_plot], _rpm(err), color=color, ls=ls, lw=lw, alpha=alpha)
    ax.axhline(0, color="k", lw=0.8, ls="--")
    ax.set_ylabel("Error (RPM)", fontsize=12)
    ax.set_xlabel("Time (s)", fontsize=12)
    ax.grid(True, alpha=0.4)

    # Single legend below both panels
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=4, fontsize=10,
               bbox_to_anchor=(0.5, 0.0), framealpha=0.95)
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.14)

    out = Path(args.out)
    plt.savefig(out, dpi=180, bbox_inches="tight")
    print(f"\nSaved → {out}")
    plt.show()


if __name__ == "__main__":
    main()
