"""
Out-of-distribution (OOD) generalisation test for the motor + rod speed filter.

The models and baselines are configured on the in-distribution data
(step/ramp/random/mixed excitation, tuned on its val split), then evaluated on
chirp-excited trajectories they have never seen (data/rod_ood.npz). This probes
whether the learned filter generalises to an unseen input family.

Usage
-----
    python eval_ood.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from compare import _BASE_PARAMS, WINDOW, _model_predict_traj, _rpm, load_models
from utils.baselines import _ema, kalman_predict_one, optimize_ema, optimize_kalman
from utils.dataset import NormStats
from utils.traj import MotorSplit


def _ood_bar(rmse: dict[str, float], raw: float, out: Path) -> None:
    """Bar chart of OOD RMSE per method — same style as the in-distribution one."""
    order = [k for k in ("Raw", "EMA", "Kalman", "CNN", "TCN", "GRU") if k in rmse]
    vals = [rmse[k] for k in order]
    palette = {"Raw": "#bbbbbb", "EMA": "#e07b00", "Kalman": "#9b59b6",
               "CNN": "#3498db", "TCN": "#1abc9c", "GRU": "#2ecc71"}
    colors = [palette.get(k, "#888888") for k in order]

    fig, ax = plt.subplots(figsize=(9, 5))
    bars = ax.bar(order, vals, color=colors, edgecolor="#333", linewidth=0.6)
    for b, v in zip(bars, vals):
        pct = 100 * (1 - v / raw)
        tag = f"{v:.2f}" if b is bars[0] else f"{v:.2f}\n{pct:+.0f}%"
        ax.text(b.get_x() + b.get_width() / 2, v + 0.04, tag,
                ha="center", va="bottom", fontsize=9.5, linespacing=1.0)
    ax.axhline(raw, color="#bbbbbb", ls="--", lw=1.0, zorder=0)
    ax.set_ylim(0, raw * 1.18)
    ax.set_ylabel("OOD test RMSE [rad/s]", fontsize=12)
    ax.set_title("Out-of-distribution (chirp) — classical filters collapse, neural hold",
                 fontsize=12.5)
    ax.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(out, dpi=160)
    plt.close(fig)
    print(f"saved -> {out}")


def main() -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    figures = Path("figures")

    split = MotorSplit.load("data/rod_split.npz")           # for stats + tuning
    stats = NormStats.from_split(split)
    ood = np.load("data/rod_ood.npz")
    noisy, true, volt = ood["noisy"], ood["true"], ood["voltage"]
    dt = float(ood["dt"])
    N = noisy.shape[0]

    print("Tuning baselines on in-distribution val …")
    best_alpha, _ = optimize_ema(split)
    q_diag, r_var, _ = optimize_kalman(split, _BASE_PARAMS)

    models = load_models(device)

    raw = float(np.sqrt(np.mean((noisy - true) ** 2)))
    ema = float(np.sqrt(np.mean((_ema(noisy, best_alpha) - true) ** 2)))
    kf = np.stack([kalman_predict_one(noisy[n], volt[n], _BASE_PARAMS, dt, q_diag, r_var)
                   for n in range(N)])
    kf_rmse = float(np.sqrt(np.mean((kf - true) ** 2)))

    rmse = {"Raw": raw, "EMA": ema, "Kalman": kf_rmse}
    for key in ("CNN", "TCN", "GRU"):
        if key in models:
            errs = []
            for n in range(N):
                pred = _model_predict_traj(models[key], noisy[n], volt[n], stats, device)
                errs.append(pred - true[n, WINDOW - 1 :])
            rmse[key] = float(np.sqrt(np.mean(np.concatenate(errs) ** 2)))

    print("\nOOD (chirp) test-set RMSE [rad/s]:")
    for k, v in rmse.items():
        print(f"  {k:8s} {v:6.3f}   ({100 * (1 - v / raw):+5.1f}% vs Raw)")

    _ood_bar(rmse, raw, figures / "ood_rmse.png")

    # ---- overlay plot for one OOD trajectory -----------------------------
    n = 0
    T = noisy.shape[1]
    t_ax = np.arange(T) * dt
    fig, ax = plt.subplots(figsize=(13, 5))
    ax.plot(t_ax, _rpm(noisy[n]), color="#bbbbbb", lw=1.0, alpha=0.5, label="noisy")
    ax.plot(t_ax, _rpm(true[n]), color="#1a1a1a", ls="--", lw=1.5, label="true")
    kf0 = kalman_predict_one(noisy[n], volt[n], _BASE_PARAMS, dt, q_diag, r_var)
    ax.plot(t_ax, _rpm(kf0), color="#9b59b6", lw=1.5, label=f"Kalman ({kf_rmse:.2f})")
    if "GRU" in models:
        g = _model_predict_traj(models["GRU"], noisy[n], volt[n], stats, device)
        ax.plot(t_ax[WINDOW - 1 :], _rpm(g), color="#2ecc71", lw=2.0,
                label=f"GRU ({rmse['GRU']:.2f})")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Speed (RPM)")
    ax.set_title("OOD generalisation — chirp excitation (unseen in training)")
    ax.legend(ncol=4)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(figures / "ood_chirp.png", dpi=160)
    plt.close(fig)
    print("\nsaved -> figures/ood_chirp.png")


if __name__ == "__main__":
    main()
