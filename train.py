"""
BDC motor speed filter — training pipeline.

Usage
-----
# Generate dataset first (if not done):
#   python BDCmotor.py

# Train default GRU (with voltage feature):
python train.py

# Ablation — speed-only input (no voltage):
python train.py --no-voltage

# CNN and TCN:
python train.py --model cnn
python train.py --model tcn

# Bigger GRU:
python train.py --model gru --window 128 --hidden 64 --layers 2
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
from utils.dataset import BDCFilterDataset, NormStats
from utils.motor import BDCMotorParams
from utils.traj import MotorSplit

# ---------------------------------------------------------------------------
# Model factory
# ---------------------------------------------------------------------------

# Base motor params — used by the Kalman filter baseline
_BASE_PARAMS = BDCMotorParams(R=1.0, L=0.5e-3, Kt=0.01, Kb=0.01, J=1e-5, B=1e-6, V_max=12.0)


def build_model(args: argparse.Namespace) -> nn.Module:
    in_size = 1 if args.no_voltage else 2
    if args.model == "gru":
        return GRUFilter(input_size=in_size, hidden_size=args.hidden, num_layers=args.layers)
    if args.model == "cnn":
        return CNNFilter(input_size=in_size, channels=args.channels,
                         kernel_size=args.kernel, depth=args.depth)
    if args.model == "tcn":
        return TCNFilter(input_size=in_size, channels=args.tcn_channels,
                         kernel_size=args.tcn_kernel, n_levels=args.tcn_levels)
    raise ValueError(f"Unknown model '{args.model}'. Choose gru / cnn / tcn.")


# ---------------------------------------------------------------------------
# Training curves
# ---------------------------------------------------------------------------


def plot_training_curves(
    history:    dict,
    model_name: str,
    save_dir:   Path = Path("figures"),
):
    epochs     = np.arange(1, len(history["train"]) + 1)
    train_rmse = np.sqrt(history["train"])
    val_rmse   = np.sqrt(history["val"])

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(epochs, train_rmse, label="train")
    ax.plot(epochs, val_rmse,   label="val")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Normalized RMSE")
    ax.set_title(f"{model_name} — training curves")
    ax.legend()
    ax.grid(True)
    plt.tight_layout()
    plt.savefig(save_dir / f"curves_{model_name.lower()}.png", dpi=150)
    plt.show()


# ---------------------------------------------------------------------------
# Baseline (MA — kept here; EMA + Kalman live in utils/baselines.py)
# ---------------------------------------------------------------------------


def moving_avg_rmse(split: MotorSplit, window: int) -> float:
    """Causal moving-average baseline RMSE on the test set [rad/s]."""
    win  = sliding_window_view(split.test_noisy, window, axis=1)
    pred = win.mean(axis=-1)
    err  = pred - split.test_true[:, window - 1:]
    return float(np.sqrt(np.mean(err ** 2)))


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------


def _run_epoch(
    model: nn.Module,
    loader: DataLoader,
    loss_fn: nn.Module,
    device: str,
    opt: torch.optim.Optimizer | None = None,
) -> float:
    is_train = opt is not None
    model.train(is_train)
    total = 0.0
    with torch.set_grad_enabled(is_train):
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            loss = loss_fn(model(x), y)
            if is_train:
                opt.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
            total += loss.item() * x.size(0)
    return total / len(loader.dataset)


def train_model(
    model: nn.Module,
    tr_loader: DataLoader,
    va_loader: DataLoader,
    epochs: int,
    lr: float,
    patience: int,
    device: str,
    ckpt_path: Path = Path("checkpoints/best_filter.pt"),
) -> dict:
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(
        opt, patience=patience // 2, factor=0.5
    )
    loss_fn = nn.MSELoss()

    best_val = float("inf")
    wait = 0
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
            print(
                f"  Epoch {epoch:3d}  train {np.sqrt(tr):.4f}  val {np.sqrt(va):.4f}  (normalized RMSE)"
            )

    model.load_state_dict(
        torch.load(ckpt_path, weights_only=True, map_location=device)
    )
    return history


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------


def evaluate(
    model: nn.Module,
    test_ds: BDCFilterDataset,
    stats: NormStats,
    device: str,
    batch: int = 4096,
) -> tuple[float, float]:
    """Return test RMSE in (rad/s, RPM)."""
    loader = DataLoader(test_ds, batch_size=batch)
    preds, targets = [], []
    model.eval()
    with torch.no_grad():
        for x, y in loader:
            preds.append(model(x.to(device)).cpu())
            targets.append(y)
    pred = stats.denorm_true(torch.cat(preds).numpy())
    true = stats.denorm_true(torch.cat(targets).numpy())
    rmse = float(np.sqrt(np.mean((pred - true) ** 2)))
    return rmse, rmse * 60 / (2 * np.pi)


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------


def plot_test_trajectory(
    model: nn.Module,
    split: MotorSplit,
    stats: NormStats,
    window: int,
    model_name: str,
    traj_idx: int = 0,
    device: str = "cpu",
    save_dir: Path = Path("figures"),
):
    """Plot noisy / moving-avg baseline / model / true for one test trajectory."""
    n = traj_idx
    T = split.test_noisy.shape[1]
    t_ax = np.arange(T) * split.dt
    noisy = split.test_noisy[n]
    true = split.test_true[n]
    volt = split.test_voltage[n]

    # Build all input windows for this trajectory efficiently
    win_noisy = sliding_window_view(stats.norm_noisy(noisy), window)  # (T-W+1, W)
    win_volt = sliding_window_view(stats.norm_volt(volt), window)
    x = np.stack([win_noisy, win_volt], axis=-1).astype(np.float32)  # (T-W+1, W, 2)

    model.eval()
    with torch.no_grad():
        pred_n = model(torch.from_numpy(x).to(device)).cpu().numpy()
    pred = stats.denorm_true(pred_n)

    baseline = sliding_window_view(noisy, window).mean(axis=-1)
    t_pred = t_ax[window - 1 :]
    true_align = true[window - 1 :]

    def rpm(v):
        return v * 60 / (2 * np.pi)

    fig, axes = plt.subplots(2, 1, figsize=(12, 6), sharex=True)

    axes[0].plot(t_ax, rpm(noisy), alpha=0.3, color="C0", label="noisy measurement")
    axes[0].plot(t_pred, rpm(baseline), lw=1.2, color="C3", label="MA baseline")
    axes[0].plot(t_pred, rpm(pred), lw=1.8, color="C2", label=model_name)
    axes[0].plot(t_ax, rpm(true), lw=1.5, ls="--", color="C1", label="true speed")
    axes[0].set_ylabel("Speed (RPM)")
    axes[0].legend(ncol=4)
    axes[0].grid(True)
    axes[0].set_title(f"Test trajectory {n} — speed filter comparison")

    axes[1].plot(
        t_pred, rpm(pred - true_align), color="C2", label=f"{model_name} error"
    )
    axes[1].plot(
        t_pred, rpm(baseline - true_align), color="C3", alpha=0.6, label="MA error"
    )
    axes[1].axhline(0, color="k", lw=0.8, ls="--")
    axes[1].set_ylabel("Error (RPM)")
    axes[1].set_xlabel("Time (s)")
    axes[1].legend()
    axes[1].grid(True)

    plt.tight_layout()
    plt.savefig(save_dir / f"filter_{model_name.lower()}_test.png", dpi=150)
    plt.show()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(description="BDC motor speed filter training")
    # Model
    parser.add_argument("--model", default="gru", choices=["gru", "cnn", "tcn"])
    parser.add_argument("--no-voltage", action="store_true",
                        help="Ablation: use only noisy speed as input (no voltage)")
    # GRU
    parser.add_argument("--hidden",  type=int, default=32)
    parser.add_argument("--layers",  type=int, default=1)
    # CNN
    parser.add_argument("--channels", type=int, default=32)
    parser.add_argument("--kernel",   type=int, default=8)
    parser.add_argument("--depth",    type=int, default=2)
    # TCN
    parser.add_argument("--tcn-channels", type=int, default=32)
    parser.add_argument("--tcn-kernel",   type=int, default=4)
    parser.add_argument("--tcn-levels",   type=int, default=4,
                        help="Number of dilation levels (RF = 1 + 2*(k-1)*(2^n - 1))")
    # Training
    parser.add_argument("--window",   type=int,   default=64)
    parser.add_argument("--epochs",   type=int,   default=50)
    parser.add_argument("--lr",       type=float, default=1e-3)
    parser.add_argument("--batch",    type=int,   default=512)
    parser.add_argument("--patience", type=int,   default=10)
    parser.add_argument("--workers",  type=int,   default=2)
    parser.add_argument("--split",    default="data/motor_split.npz")
    args = parser.parse_args()

    figures  = Path("figures")
    ckpt_dir = Path("checkpoints")
    figures.mkdir(exist_ok=True)
    ckpt_dir.mkdir(exist_ok=True)

    suffix    = f"{args.model}" + ("_novolt" if args.no_voltage else "")
    ckpt_path = ckpt_dir / f"best_{suffix}.pt"

    device  = "cuda" if torch.cuda.is_available() else "cpu"
    pin_mem = device == "cuda"
    print(f"Device : {device}")

    # ------------------------------------------------------------------
    # Load data
    # ------------------------------------------------------------------
    split = MotorSplit.load(args.split)
    print(f"Split  : train {split.train_noisy.shape}  "
          f"val {split.val_noisy.shape}  test {split.test_noisy.shape}")

    stats = NormStats.from_split(split)

    # ------------------------------------------------------------------
    # Non-learned baselines (MA + EMA + Kalman) — computed once
    # ------------------------------------------------------------------
    print("\nComputing baselines (EMA + Kalman may take a few seconds) …")
    baselines = run_all_baselines(split, _BASE_PARAMS, ma_window=args.window)
    for name, rmse in baselines.items():
        print(f"  {name:<28s}: {rmse:6.2f} rad/s  |  {rmse*60/(2*np.pi):7.1f} RPM")

    ma_rmse = next(v for k, v in baselines.items() if k.startswith("MA"))

    # ------------------------------------------------------------------
    # Datasets and loaders
    # ------------------------------------------------------------------
    W = args.window

    def mk_ds(noisy, true, volt):
        return BDCFilterDataset(
            stats.norm_noisy(noisy), stats.norm_true(true), stats.norm_volt(volt),
            W, use_voltage=not args.no_voltage,
        )

    tr_ds = mk_ds(split.train_noisy, split.train_true, split.train_voltage)
    va_ds = mk_ds(split.val_noisy,   split.val_true,   split.val_voltage)
    te_ds = mk_ds(split.test_noisy,  split.test_true,  split.test_voltage)

    loader_kw = dict(batch_size=args.batch, num_workers=args.workers, pin_memory=pin_mem)
    tr_loader = DataLoader(tr_ds, shuffle=True,  **loader_kw)
    va_loader = DataLoader(va_ds, shuffle=False, **loader_kw)
    feat_str  = "omega_noisy only" if args.no_voltage else "omega_noisy + voltage"
    print(f"\nSamples: train {len(tr_ds):,}  val {len(va_ds):,}  test {len(te_ds):,}")
    print(f"Input  : {feat_str}")

    # ------------------------------------------------------------------
    # Model
    # ------------------------------------------------------------------
    model    = build_model(args).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    label    = suffix.upper()
    print(f"\nModel  : {label}  params={n_params:,}")
    if args.model == "tcn":
        rf = model.receptive_field
        print(f"         receptive field = {rf} steps ({rf * split.dt * 1e3:.0f} ms)")

    # ------------------------------------------------------------------
    # Train
    # ------------------------------------------------------------------
    print("\nTraining …")
    history = train_model(
        model, tr_loader, va_loader, args.epochs, args.lr, args.patience, device, ckpt_path
    )

    # ------------------------------------------------------------------
    # Evaluate
    # ------------------------------------------------------------------
    test_rmse, test_rmse_rpm = evaluate(model, te_ds, stats, device)

    # ------------------------------------------------------------------
    # Comparison table
    # ------------------------------------------------------------------
    print("\n" + "=" * 56)
    print(f"{'Method':<30s}  {'rad/s':>6}  {'RPM':>7}  {'vs MA':>7}")
    print("-" * 56)
    for name, rmse in baselines.items():
        impr = 100 * (1 - rmse / ma_rmse)
        print(f"  {name:<28s}  {rmse:6.2f}  {rmse*60/(2*np.pi):7.1f}  {impr:+6.1f}%")
    impr_model = 100 * (1 - test_rmse / ma_rmse)
    print(f"  {label:<28s}  {test_rmse:6.2f}  {test_rmse_rpm:7.1f}  {impr_model:+6.1f}%")
    print("=" * 56)

    # ------------------------------------------------------------------
    # Plots
    # ------------------------------------------------------------------
    plot_training_curves(history, label, save_dir=figures)
    plot_test_trajectory(model, split, stats, W, label,
                         device=device, save_dir=figures)


if __name__ == "__main__":
    main()
