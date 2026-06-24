"""
Multi-method comparison figure — all models on the test set.

Usage
-----
python compare.py                    # full 30 s test set
python compare.py --t-start 5 --t-end 8   # zoom to a reversal-rich window
python compare.py --out figures/my_fig.png
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from numpy.lib.stride_tricks import sliding_window_view

from models.cnn_filter import CNNFilter
from models.gru_filter import GRUFilter
from models.tcn_filter import TCNFilter
from utils.dataset import NormStats
from utils.servo import ServoSplit

WINDOW = 64

_STYLE: dict[str, tuple] = {
    # key: (label, color, linestyle, linewidth, alpha)
    "True":   ("True θ_l",              "#1a1a1a", "--", 1.6, 1.00),
    "Rigid":  ("Rigid (enc_m/N)",       "#888888", "-",  1.2, 0.85),
    "EncO":   ("Output encoder (enc_o)","#e07b00", "-",  1.2, 0.85),
    "GRU":    ("GRU",                   "#2ecc71", "-",  2.0, 1.00),
    "CNN":    ("CNN  (RF = 15 ms)",     "#3498db", "-",  1.6, 0.85),
    "TCN":    ("TCN  (RF = 91 ms)",     "#1abc9c", "-",  1.8, 0.95),
}


def _model_predict(model: torch.nn.Module, enc_m: np.ndarray, pwm: np.ndarray,
                   stats: NormStats, N: float, device: str) -> np.ndarray:
    """
    Sliding-window inference → (T - W + 1,) theta_l estimate in radians.
    Uses per-window velocity scaling: pred_error = model(x) × (local_vel / N).
    """
    enc_f    = enc_m.astype(np.float32)
    pwm_n    = stats.norm_pwm(pwm).astype(np.float32)
    wins_enc = sliding_window_view(enc_f,  WINDOW)
    wins_pwm = sliding_window_view(pwm_n,  WINDOW)
    diff_w   = np.diff(wins_enc, axis=1, prepend=wins_enc[:, :1])
    local_vel = diff_w.std(axis=1) + 1e-6              # (T-W+1,) motor-side vel
    enc_rel  = (wins_enc - wins_enc[:, :1]) / local_vel[:, None]
    x = np.stack([enc_rel, wins_pwm], axis=-1).astype(np.float32)
    model.eval()
    with torch.no_grad():
        pred_err_n = model(torch.from_numpy(x).to(device)).cpu().numpy()
    pred_err = pred_err_n * (local_vel / N)            # per-window denorm
    rigid    = enc_m[WINDOW - 1:] / N
    return rigid + pred_err


def load_models(device: str) -> dict[str, torch.nn.Module]:
    ckpt_dir = Path("checkpoints")
    candidates = {
        "GRU": (ckpt_dir / "best_gru.pt",
                GRUFilter(input_size=2, hidden_size=32, num_layers=1)),
        "CNN": (ckpt_dir / "best_cnn.pt",
                CNNFilter(input_size=2, channels=32, kernel_size=8, depth=2)),
        "TCN": (ckpt_dir / "best_tcn.pt",
                TCNFilter(input_size=2, channels=32, kernel_size=4, n_levels=4)),
    }
    loaded = {}
    for key, (path, model) in candidates.items():
        if path.exists():
            model.load_state_dict(torch.load(path, weights_only=True, map_location=device))
            model.to(device)
            loaded[key] = model
            print(f"  Loaded {key:<6s} ← {path}")
        else:
            print(f"  Skipped {key:<5s} (not found: {path})")
    return loaded


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split",   default="data/servo_split.npz")
    parser.add_argument("--t-start", type=float, default=None)
    parser.add_argument("--t-end",   type=float, default=None)
    parser.add_argument("--out",     default="figures/comparison_backlash.png")
    args = parser.parse_args()

    device  = "cuda" if torch.cuda.is_available() else "cpu"
    figures = Path("figures"); figures.mkdir(exist_ok=True)

    split  = ServoSplit.load(args.split)
    stats  = NormStats.from_split(split)

    t       = split.test_t
    enc_m   = split.test_enc_m
    enc_o   = split.test_enc_o
    theta_l = split.test_theta_l
    pwm     = split.test_pwm
    T       = len(t)

    # Time slice for plotting
    i0 = int(args.t_start / split.dt) if args.t_start is not None else 0
    i1 = int(args.t_end   / split.dt) if args.t_end   is not None else T

    print("Loading models …")
    models = load_models(device)
    model_preds: dict[str, np.ndarray] = {}
    for key, model in models.items():
        model_preds[key] = _model_predict(model, enc_m, pwm, stats, split.N, device)

    # ------------------------------------------------------------------
    # Compute and print RMSE table
    # ------------------------------------------------------------------
    t0     = WINDOW - 1  # first valid model prediction index
    rigid  = enc_m / split.N

    print(f"\n{'Method':<32s}  {'mrad':>7}  {'°':>8}")
    print("-" * 50)

    def _rmse(pred, ref, sl):
        return float(np.sqrt(np.mean((pred[sl] - ref[sl]) ** 2)))

    sl = slice(t0, i1)
    ref = theta_l

    baselines = {
        "Rigid (enc_m/N)":        rigid,
        "Output encoder (enc_o)": enc_o,
    }
    rigid_rmse = _rmse(rigid, ref, sl)
    for name, pred in baselines.items():
        r = _rmse(pred, ref, sl)
        print(f"  {name:<30s}  {r*1e3:7.3f}  {np.degrees(r):8.4f}")
    for key, pred in model_preds.items():
        pred_full = np.full(T, np.nan)
        pred_full[t0:] = pred
        r = _rmse(pred_full, ref, sl)
        impr = 100 * (1 - r / rigid_rmse)
        print(f"  {_STYLE[key][0]:<30s}  {r*1e3:7.3f}  {np.degrees(r):8.4f}  {impr:+6.1f}%")

    # ------------------------------------------------------------------
    # Plot — top panel: backlash residual (mrad); bottom: residual error (mrad)
    # All methods predict theta_l; showing (pred - rigid) makes differences visible.
    # Rigid baseline is the reference line at 0.
    # ------------------------------------------------------------------
    fig, axes = plt.subplots(2, 1, figsize=(14, 7), sharex=True,
                             gridspec_kw={"height_ratios": [3, 2]})

    sl_plot = slice(max(i0, t0), i1)
    t_sl    = t[sl_plot]

    # Backlash residual = pred - rigid = what each method adds on top of enc_m/N
    true_residual = theta_l - rigid  # ground-truth backlash error

    def _mrad(v): return v * 1e3

    ax = axes[0]
    # Ground truth backlash error
    ax.plot(t[sl_plot], _mrad(true_residual[sl_plot]),
            color=_STYLE["True"][1], ls="--", lw=1.4,
            alpha=0.9, label="True backlash error  (θ_l − enc_m/N)")
    # Output encoder residual
    ax.plot(t_sl, _mrad((enc_o - rigid)[sl_plot]),
            color=_STYLE["EncO"][1], ls="-", lw=0.9, alpha=0.6,
            label=_STYLE["EncO"][0])
    # Model residuals
    for key, pred in model_preds.items():
        label, color, ls, lw, alpha = _STYLE[key]
        pred_full = np.full(T, np.nan)
        pred_full[t0:] = pred
        residual = pred_full - rigid
        ax.plot(t_sl, _mrad(residual[sl_plot]), color=color, ls=ls,
                lw=lw, alpha=alpha, label=label)
    ax.axhline(0, color="#888888", lw=0.8, ls=":")
    ax.set_ylabel("Backlash residual (mrad)", fontsize=12)
    ax.grid(True, alpha=0.4)
    ax.set_title("Servo backlash compensation — all methods (test set, multisine)", fontsize=13)

    # Error panel — estimation error = pred - theta_l, in mrad
    ax = axes[1]
    ref_sl = theta_l[sl_plot]
    ax.plot(t_sl, _mrad((rigid - theta_l)[sl_plot]),
            color=_STYLE["Rigid"][1], lw=_STYLE["Rigid"][3],
            alpha=_STYLE["Rigid"][4], label=_STYLE["Rigid"][0])
    ax.plot(t_sl, _mrad((enc_o - theta_l)[sl_plot]),
            color=_STYLE["EncO"][1], lw=_STYLE["EncO"][3],
            alpha=_STYLE["EncO"][4], label=_STYLE["EncO"][0])
    for key, pred in model_preds.items():
        label, color, ls, lw, alpha = _STYLE[key]
        pred_full = np.full(T, np.nan)
        pred_full[t0:] = pred
        ax.plot(t_sl, _mrad((pred_full - theta_l)[sl_plot]),
                color=color, ls=ls, lw=lw, alpha=alpha, label=label)
    ax.axhline(0, color="k", lw=0.8, ls="--")
    ax.set_ylabel("Estimation error (mrad)", fontsize=12)
    ax.set_xlabel("Time (s)", fontsize=12)
    ax.grid(True, alpha=0.4)

    # Shared legend below both panels — merge handles from both axes, no duplicates
    h0, l0 = axes[0].get_legend_handles_labels()
    h1, l1 = axes[1].get_legend_handles_labels()
    seen = set()
    handles, labels = [], []
    for h, l in zip(h0 + h1, l0 + l1):
        if l not in seen:
            handles.append(h); labels.append(l); seen.add(l)
    fig.legend(handles, labels, loc="lower center", ncol=3, fontsize=10,
               bbox_to_anchor=(0.5, 0.0), framealpha=0.95)
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.12)

    out = Path(args.out)
    plt.savefig(out, dpi=180, bbox_inches="tight")
    print(f"\nSaved → {out}")
    plt.show()


if __name__ == "__main__":
    main()
