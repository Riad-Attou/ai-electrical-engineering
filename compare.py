"""
Multi-method comparison for the motor + rod speed filter.

Produces two figures:
  figures/comparison_rmse.png        — test-set RMSE bar chart, all methods
  figures/comparison_all_methods.png — one test trajectory, speed + error panels

Baselines (EMA, Kalman) are tuned on the validation split, then evaluated on
the test split — the same protocol as train.py.

Usage
-----
    python compare.py                 # traj 0, full trajectory
    python compare.py --traj 2 --t-end 3.0
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
from utils.baselines import _ema, optimize_ema, optimize_kalman, kalman_predict_one
from utils.dataset import NormStats
from utils.motor import BDCMotorParams
from utils.traj import MotorSplit

# Shared team motor+rod plant — the Kalman's nominal (linear) model.
_BASE_PARAMS = BDCMotorParams(R=3.0, L=4e-3, Kt=0.05, Kb=0.05, J=7.04e-5, B=0.005, V_max=12.0)
WINDOW = 64

# key -> (label, color, linestyle, linewidth, alpha)
_STYLE: dict[str, tuple] = {
    "Noisy":  ("Noisy measurement", "#aaaaaa", "-", 1.0, 0.45),
    "EMA":    ("EMA (tuned)",       "#e07b00", "-", 1.6, 0.85),
    "Kalman": ("Kalman (tuned)",    "#9b59b6", "-", 1.8, 1.00),
    "CNN":    ("CNN",               "#3498db", "-", 1.6, 0.85),
    "TCN":    ("TCN",               "#1abc9c", "-", 2.0, 1.00),
    "GRU":    ("GRU",               "#2ecc71", "-", 2.2, 1.00),
    "True":   ("True speed",        "#1a1a1a", "--", 1.6, 1.00),
}
_MODEL_ORDER = ("CNN", "TCN", "GRU")


def _rpm(v: np.ndarray) -> np.ndarray:
    return v * 60 / (2 * np.pi)


def load_models(device: str) -> dict[str, torch.nn.Module]:
    """Load whichever model checkpoints exist (all use the voltage feature)."""
    ckpt = Path("checkpoints")
    candidates = {
        "GRU": (ckpt / "best_gru.pt", GRUFilter(input_size=2, hidden_size=32, num_layers=1)),
        "CNN": (ckpt / "best_cnn.pt", CNNFilter(input_size=2, channels=32, kernel_size=8, depth=2)),
        "TCN": (ckpt / "best_tcn.pt", TCNFilter(input_size=2, channels=32, kernel_size=4, n_levels=4)),
    }
    loaded = {}
    for key, (path, model) in candidates.items():
        if path.exists():
            model.load_state_dict(torch.load(path, weights_only=True, map_location=device))
            loaded[key] = model.to(device).eval()
            print(f"  loaded {key:4s} <- {path}")
        else:
            print(f"  skip   {key:4s} (no checkpoint at {path})")
    return loaded


def _model_predict_traj(model, noisy, volt, stats, device) -> np.ndarray:
    """Sliding-window inference on one trajectory. Returns (T-W+1,) rad/s."""
    win_noisy = sliding_window_view(stats.norm_noisy(noisy), WINDOW)
    win_volt = sliding_window_view(stats.norm_volt(volt), WINDOW)
    x = np.stack([win_noisy, win_volt], axis=-1).astype(np.float32)
    with torch.no_grad():
        pred_n = model(torch.from_numpy(x).to(device)).cpu().numpy()
    return stats.denorm_true(pred_n)


def _model_test_rmse(model, split, stats, device) -> float:
    """Full test-set RMSE (rad/s), causal windows, aligned to window end."""
    errs = []
    for n in range(split.test_noisy.shape[0]):
        pred = _model_predict_traj(model, split.test_noisy[n], split.test_voltage[n], stats, device)
        true = split.test_true[n, WINDOW - 1 :]
        errs.append(pred - true)
    e = np.concatenate(errs)
    return float(np.sqrt(np.mean(e**2)))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", default="data/rod_split.npz")
    parser.add_argument("--traj", type=int, default=0)
    parser.add_argument("--t-end", type=float, default=None)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    figures = Path("figures")
    figures.mkdir(exist_ok=True)

    split = MotorSplit.load(args.split)
    stats = NormStats.from_split(split)
    dt = split.dt

    print("Tuning baselines on val …")
    best_alpha, _ = optimize_ema(split)
    q_diag, r_var, _ = optimize_kalman(split, _BASE_PARAMS)
    print(f"  EMA alpha={best_alpha:.3f}   Kalman R={r_var:.2g} Qw={q_diag[1]:.2g}")

    print("Loading models …")
    models = load_models(device)

    # ------------------------------------------------------------------ RMSE
    raw_rmse = float(np.sqrt(np.mean((split.test_noisy - split.test_true) ** 2)))
    ema_rmse = float(np.sqrt(np.mean((_ema(split.test_noisy, best_alpha) - split.test_true) ** 2)))
    # Kalman over all test trajectories
    kf_pred = np.stack([kalman_predict_one(split.test_noisy[n], split.test_voltage[n],
                                           _BASE_PARAMS, dt, q_diag, r_var)
                        for n in range(split.test_noisy.shape[0])])
    kf_rmse = float(np.sqrt(np.mean((kf_pred - split.test_true) ** 2)))

    rmse = {"Raw": raw_rmse, "EMA": ema_rmse, "Kalman": kf_rmse}
    for key in _MODEL_ORDER:
        if key in models:
            rmse[key] = _model_test_rmse(models[key], split, stats, device)

    print("\nTest-set RMSE [rad/s]:")
    for k, v in rmse.items():
        print(f"  {k:8s} {v:6.3f}   ({100 * (1 - v / raw_rmse):+5.1f}% vs Raw)")

    # ------------------------------------------------------------- bar chart
    _bar_chart(rmse, raw_rmse, figures / "comparison_rmse.png")

    # --------------------------------------------------------- traj overlay
    _trajectory_overlay(args, split, stats, models, best_alpha, q_diag, r_var,
                        dt, device, figures / "comparison_all_methods.png")


def _bar_chart(rmse: dict[str, float], raw: float, out: Path) -> None:
    order = [k for k in ("Raw", "MA", "EMA", "Kalman", "CNN", "TCN", "GRU") if k in rmse]
    vals = [rmse[k] for k in order]
    colors = []
    palette = {"Raw": "#bbbbbb", "EMA": "#e07b00", "Kalman": "#9b59b6",
               "CNN": "#3498db", "TCN": "#1abc9c", "GRU": "#2ecc71"}
    for k in order:
        colors.append(palette.get(k, "#888888"))

    fig, ax = plt.subplots(figsize=(9, 5))
    bars = ax.bar(order, vals, color=colors, edgecolor="#333", linewidth=0.6)
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.03, f"{v:.2f}",
                ha="center", va="bottom", fontsize=10)
    ax.axhline(raw, color="#bbbbbb", ls="--", lw=1.0, zorder=0)
    ax.set_ylabel("Test RMSE [rad/s]", fontsize=12)
    ax.set_title("Speed-filter comparison — motor + rod (lower is better)", fontsize=13)
    ax.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(out, dpi=160)
    plt.close(fig)
    print(f"saved -> {out}")


def _trajectory_overlay(args, split, stats, models, best_alpha, q_diag, r_var,
                        dt, device, out: Path) -> None:
    n = args.traj
    noisy, true, volt = split.test_noisy[n], split.test_true[n], split.test_voltage[n]
    T = len(noisy)
    t_ax = np.arange(T) * dt
    T_plot = min(T, int(args.t_end / dt)) if args.t_end else T

    ema_pred = lfilter([1.0 - best_alpha], [1.0, -best_alpha], noisy)
    kalman_pred = kalman_predict_one(noisy, volt, _BASE_PARAMS, dt, q_diag, r_var)
    model_preds = {k: _model_predict_traj(m, noisy, volt, stats, device)
                   for k, m in models.items()}

    fig, axes = plt.subplots(2, 1, figsize=(14, 7), sharex=True,
                             gridspec_kw={"height_ratios": [3, 2]})

    def _plot(ax, y, key, t_offset=0):
        label, color, ls, lw, alpha = _STYLE[key]
        ax.plot(t_ax[t_offset:T_plot], _rpm(y[: T_plot - t_offset]),
                color=color, ls=ls, lw=lw, alpha=alpha, label=label)

    ax = axes[0]
    _plot(ax, noisy, "Noisy")
    _plot(ax, true, "True")
    _plot(ax, ema_pred, "EMA")
    _plot(ax, kalman_pred, "Kalman")
    for key in _MODEL_ORDER:
        if key in model_preds:
            _plot(ax, model_preds[key], key, t_offset=WINDOW - 1)
    ax.set_ylabel("Speed (RPM)", fontsize=12)
    ax.grid(True, alpha=0.4)
    ax.set_title(f"Test trajectory {n} — speed filter comparison (dt = {dt * 1e3:.0f} ms)", fontsize=13)

    ax = axes[1]
    for key, pred in [("EMA", ema_pred), ("Kalman", kalman_pred)]:
        label, color, ls, lw, alpha = _STYLE[key]
        ax.plot(t_ax[:T_plot], _rpm(pred[:T_plot] - true[:T_plot]),
                color=color, ls=ls, lw=lw, alpha=alpha)
    for key in _MODEL_ORDER:
        if key in model_preds:
            label, color, ls, lw, alpha = _STYLE[key]
            t0 = WINDOW - 1
            err = model_preds[key][: T_plot - t0] - true[t0:T_plot]
            ax.plot(t_ax[t0:T_plot], _rpm(err), color=color, ls=ls, lw=lw, alpha=alpha)
    ax.axhline(0, color="k", lw=0.8, ls="--")
    ax.set_ylabel("Error (RPM)", fontsize=12)
    ax.set_xlabel("Time (s)", fontsize=12)
    ax.grid(True, alpha=0.4)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=4, fontsize=10,
               bbox_to_anchor=(0.5, 0.0), framealpha=0.95)
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.14)
    plt.savefig(out, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"saved -> {out}")


if __name__ == "__main__":
    main()
