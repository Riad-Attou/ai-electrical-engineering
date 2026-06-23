"""
Servo backlash compensation — training pipeline (ML2).

Usage
-----
# Prepare dataset first (once):
#   python BDCmotor.py

# Train GRU:
python train.py
python train.py --model gru --window 128 --hidden 64 --layers 2

# CNN / TCN:
python train.py --model cnn
python train.py --model tcn
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from numpy.lib.stride_tricks import sliding_window_view
from torch.utils.data import DataLoader

from models.cnn_filter import CNNFilter
from models.gru_filter import GRUFilter
from models.tcn_filter import TCNFilter
from utils.baselines import run_all_baselines
from utils.dataset import NormStats, ServoDataset
from utils.servo import ServoSplit

# input_size is always 2 for ML2: (enc_m, pwm)
_INPUT_SIZE = 2


# ---------------------------------------------------------------------------
# Model factory
# ---------------------------------------------------------------------------

def build_model(args: argparse.Namespace) -> nn.Module:
    if args.model == "gru":
        return GRUFilter(input_size=_INPUT_SIZE, hidden_size=args.hidden,
                         num_layers=args.layers)
    if args.model == "cnn":
        return CNNFilter(input_size=_INPUT_SIZE, channels=args.channels,
                         kernel_size=args.kernel, depth=args.depth)
    if args.model == "tcn":
        return TCNFilter(input_size=_INPUT_SIZE, channels=args.tcn_channels,
                         kernel_size=args.tcn_kernel, n_levels=args.tcn_levels)
    raise ValueError(f"Unknown model '{args.model}'. Choose gru / cnn / tcn.")


# ---------------------------------------------------------------------------
# Training curves
# ---------------------------------------------------------------------------

def plot_training_curves(history: dict, model_name: str,
                         save_dir: Path = Path("figures")) -> None:
    epochs     = np.arange(1, len(history["train"]) + 1)
    train_rmse = np.sqrt(history["train"])
    val_rmse   = np.sqrt(history["val"])

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(epochs, train_rmse, label="train")
    ax.plot(epochs, val_rmse,   label="val")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Normalised RMSE")
    ax.set_title(f"{model_name} — training curves")
    ax.legend()
    ax.grid(True)
    plt.tight_layout()
    plt.savefig(save_dir / f"curves_{model_name.lower()}.png", dpi=150)
    plt.show()


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def _run_epoch(model: nn.Module, loader: DataLoader, loss_fn: nn.Module,
               device: str, opt: torch.optim.Optimizer | None = None) -> float:
    is_train = opt is not None
    model.train(is_train)
    total = 0.0
    with torch.set_grad_enabled(is_train):
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            loss  = loss_fn(model(x), y)
            if is_train:
                opt.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
            total += loss.item() * x.size(0)
    return total / len(loader.dataset)


def train_model(model: nn.Module, tr_loader: DataLoader, va_loader: DataLoader,
                epochs: int, lr: float, patience: int, device: str,
                ckpt_path: Path = Path("checkpoints/best_filter.pt")) -> dict:
    opt     = torch.optim.Adam(model.parameters(), lr=lr)
    sched   = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, patience=patience // 2,
                                                          factor=0.5)
    loss_fn  = nn.MSELoss()
    best_val = float("inf")
    wait     = 0
    history: dict[str, list[float]] = {"train": [], "val": []}

    for epoch in range(1, epochs + 1):
        tr = _run_epoch(model, tr_loader, loss_fn, device, opt=opt)
        va = _run_epoch(model, va_loader, loss_fn, device, opt=None)
        history["train"].append(tr)
        history["val"].append(va)
        sched.step(va)

        if va < best_val:
            best_val = va
            wait = 0
            torch.save(model.state_dict(), ckpt_path)
        else:
            wait += 1
            if wait >= patience:
                print(f"  Early stop at epoch {epoch}")
                break

        if epoch % 5 == 0 or epoch == 1:
            print(f"  Epoch {epoch:3d}  train {np.sqrt(tr):.4f}  "
                  f"val {np.sqrt(va):.4f}  (normalised RMSE)")

    model.load_state_dict(torch.load(ckpt_path, weights_only=True, map_location=device))
    return history


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate(model: nn.Module, test_ds: ServoDataset, stats: NormStats,
             device: str, batch: int = 4096) -> tuple[float, float]:
    """Return test RMSE on backlash error in (rad, degrees)."""
    loader = DataLoader(test_ds, batch_size=batch)
    preds, targets = [], []
    model.eval()
    with torch.no_grad():
        for x, y in loader:
            preds.append(model(x.to(device)).cpu())
            targets.append(y)
    pred     = stats.denorm_error(torch.cat(preds).numpy())
    true     = stats.denorm_error(torch.cat(targets).numpy())
    rmse_rad = float(np.sqrt(np.mean((pred - true) ** 2)))
    return rmse_rad, float(np.degrees(rmse_rad))


# ---------------------------------------------------------------------------
# Visualisation
# ---------------------------------------------------------------------------

def plot_test_result(model: nn.Module, split: ServoSplit, stats: NormStats,
                     window: int, model_name: str, device: str = "cpu",
                     save_dir: Path = Path("figures")) -> None:
    t       = split.test_t
    enc_m   = split.test_enc_m
    enc_o   = split.test_enc_o
    theta_l = split.test_theta_l
    pwm     = split.test_pwm
    T       = len(t)

    # Sliding-window inference using window-relative enc_m
    enc_f  = enc_m.astype(np.float32)
    pwm_n  = stats.norm_pwm(pwm).astype(np.float32)
    wins_enc = sliding_window_view(enc_f,  window)   # (T-W+1, W)
    wins_pwm = sliding_window_view(pwm_n,  window)
    enc_rel  = ((wins_enc - wins_enc[:, :1]) / stats.enc_rel_std)
    x = np.stack([enc_rel, wins_pwm], axis=-1).astype(np.float32)

    model.eval()
    with torch.no_grad():
        pred_err_n = model(torch.from_numpy(x).to(device)).cpu().numpy()
    pred_err = stats.denorm_error(pred_err_n)               # predicted backlash error
    rigid    = enc_m[window - 1:] / split.N                 # enc_m/N baseline
    pred     = rigid + pred_err                              # reconstructed theta_l

    t_pred  = t[window - 1:]
    true_a  = theta_l[window - 1:]
    rigid   = enc_m[window - 1:] / split.N
    enc_o_a = enc_o[window - 1:]

    def deg(v): return np.degrees(v)

    fig, axes = plt.subplots(2, 1, figsize=(13, 7), sharex=True)

    axes[0].plot(t,      deg(theta_l), lw=1.2, ls="--", color="k",
                 label="True θ_l")
    axes[0].plot(t_pred, deg(rigid),   lw=1.0, color="#888888",
                 label="Rigid (enc_m/N)")
    axes[0].plot(t_pred, deg(enc_o_a), lw=1.0, color="#e07b00",
                 label="enc_o (output encoder)")
    axes[0].plot(t_pred, deg(pred),    lw=1.8, color="#2ecc71",
                 label=model_name)
    axes[0].set_ylabel("Output angle (°)")
    axes[0].legend(ncol=4, fontsize=9)
    axes[0].grid(True, alpha=0.4)
    axes[0].set_title(f"Backlash compensation — {model_name}  (test, multisine)")

    axes[1].plot(t_pred, deg(pred    - true_a), color="#2ecc71", lw=1.5,
                 label=f"{model_name} error")
    axes[1].plot(t_pred, deg(rigid   - true_a), color="#888888", lw=1.0, alpha=0.7,
                 label="Rigid error")
    axes[1].plot(t_pred, deg(enc_o_a - true_a), color="#e07b00", lw=1.0, alpha=0.7,
                 label="enc_o error")
    axes[1].axhline(0, color="k", lw=0.8, ls="--")
    axes[1].set_ylabel("Error (°)")
    axes[1].set_xlabel("Time (s)")
    axes[1].legend(ncol=3, fontsize=9)
    axes[1].grid(True, alpha=0.4)

    plt.tight_layout()
    plt.savefig(save_dir / f"backlash_{model_name.lower()}.png", dpi=150)
    plt.show()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Servo backlash compensation training")
    parser.add_argument("--model",        default="gru", choices=["gru", "cnn", "tcn"])
    parser.add_argument("--hidden",       type=int,   default=32)
    parser.add_argument("--layers",       type=int,   default=1)
    parser.add_argument("--channels",     type=int,   default=32)
    parser.add_argument("--kernel",       type=int,   default=8)
    parser.add_argument("--depth",        type=int,   default=2)
    parser.add_argument("--tcn-channels", type=int,   default=32)
    parser.add_argument("--tcn-kernel",   type=int,   default=4)
    parser.add_argument("--tcn-levels",   type=int,   default=4)
    parser.add_argument("--window",       type=int,   default=64)
    parser.add_argument("--epochs",       type=int,   default=50)
    parser.add_argument("--lr",           type=float, default=1e-3)
    parser.add_argument("--batch",        type=int,   default=512)
    parser.add_argument("--patience",     type=int,   default=10)
    parser.add_argument("--workers",      type=int,   default=2)
    parser.add_argument("--split",        default="data/servo_split.npz")
    args = parser.parse_args()

    figures  = Path("figures");     figures.mkdir(exist_ok=True)
    ckpt_dir = Path("checkpoints"); ckpt_dir.mkdir(exist_ok=True)
    ckpt_path = ckpt_dir / f"best_{args.model}.pt"

    device  = "cuda" if torch.cuda.is_available() else "cpu"
    pin_mem = device == "cuda"
    print(f"Device : {device}")

    # ------------------------------------------------------------------
    # Load data
    # ------------------------------------------------------------------
    split = ServoSplit.load(args.split)
    print(f"Split  : train {len(split.train_t):,}  "
          f"val {len(split.val_t):,}  test {len(split.test_t):,}  steps")

    stats = NormStats.from_split(split)

    # ------------------------------------------------------------------
    # Baselines (aligned to window offset)
    # ------------------------------------------------------------------
    W = args.window
    print(f"\nBaselines (window W = {W}) …")
    baselines = run_all_baselines(split, start_idx=W - 1)
    for name, rmse in baselines.items():
        print(f"  {name:<30s}: {rmse*1e3:7.3f} mrad  |  {np.degrees(rmse):8.4f} °")

    rigid_rmse = next(iter(baselines.values()))

    # ------------------------------------------------------------------
    # Datasets and loaders
    # ------------------------------------------------------------------
    def mk_ds(enc_m, pwm, theta_l):
        err = theta_l - enc_m / split.N          # backlash error (target)
        return ServoDataset(
            enc_m,
            stats.norm_pwm(pwm),
            stats.norm_error(err),
            stats.enc_rel_std,
            W,
        )

    tr_ds = mk_ds(split.train_enc_m, split.train_pwm, split.train_theta_l)
    va_ds = mk_ds(split.val_enc_m,   split.val_pwm,   split.val_theta_l)
    te_ds = mk_ds(split.test_enc_m,  split.test_pwm,  split.test_theta_l)

    loader_kw = dict(batch_size=args.batch, num_workers=args.workers,
                     pin_memory=pin_mem)
    tr_loader = DataLoader(tr_ds, shuffle=True,  **loader_kw)
    va_loader = DataLoader(va_ds, shuffle=False, **loader_kw)
    print(f"\nSamples: train {len(tr_ds):,}  val {len(va_ds):,}  test {len(te_ds):,}")

    # ------------------------------------------------------------------
    # Model
    # ------------------------------------------------------------------
    model    = build_model(args).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    label    = args.model.upper()
    print(f"\nModel  : {label}  params={n_params:,}")
    if args.model == "tcn":
        rf = model.receptive_field
        print(f"         receptive field = {rf} steps ({rf * split.dt * 1e3:.0f} ms)")

    # ------------------------------------------------------------------
    # Train
    # ------------------------------------------------------------------
    print("\nTraining …")
    history = train_model(model, tr_loader, va_loader, args.epochs, args.lr,
                          args.patience, device, ckpt_path)

    # ------------------------------------------------------------------
    # Evaluate
    # ------------------------------------------------------------------
    test_rmse_rad, test_rmse_deg = evaluate(model, te_ds, stats, device)

    # ------------------------------------------------------------------
    # Comparison table
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print(f"{'Method':<32s}  {'mrad':>7}  {'°':>8}  {'vs rigid':>9}")
    print("-" * 60)
    for name, rmse in baselines.items():
        impr = 100 * (1 - rmse / rigid_rmse)
        print(f"  {name:<30s}  {rmse*1e3:7.3f}  {np.degrees(rmse):8.4f}  {impr:+7.1f}%")
    impr_model = 100 * (1 - test_rmse_rad / rigid_rmse)
    print(f"  {label:<30s}  {test_rmse_rad*1e3:7.3f}  "
          f"{test_rmse_deg:8.4f}  {impr_model:+7.1f}%")
    print("=" * 60)

    # ------------------------------------------------------------------
    # Plots
    # ------------------------------------------------------------------
    plot_training_curves(history, label, save_dir=figures)
    plot_test_result(model, split, stats, W, label, device=device,
                     save_dir=figures)


if __name__ == "__main__":
    main()
