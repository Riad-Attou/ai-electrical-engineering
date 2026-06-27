"""
Motor + rod speed-filter dataset generation — run once before training.

Generates, from the shared team plant (brushed DC motor with a vertical rod
load, matlab/param.m parameters):

  data/rod_split.npz : in-distribution train/val/test trajectories
                       (step / ramp / random / mixed excitation), split by
                       trajectory so there is no leakage.
  data/rod_ood.npz   : out-of-distribution test trajectories driven by a chirp
                       (swept-sine) voltage — an excitation family never seen
                       in training, used to probe generalisation.
  figures/data_overview.png : one representative trajectory (voltage, true vs
                       noisy speed, rod angle) plus a few sample trajectories.

Usage
-----
    python BDCmotor.py
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from utils.motor import BDCMotorParams, PendulumParams
from utils.traj import (
    TrajectoryConfig,
    generate_chirp_trajectories,
    generate_multi_trajectory_dataset,
    pack_split,
    split_trajectories,
)

FIGURES = Path("figures")
DATA = Path("data")
FIGURES.mkdir(exist_ok=True)
DATA.mkdir(exist_ok=True)

# Shared team motor+rod plant (matlab/param.m). The same params Alessandro's
# RL controller uses, so the whole project describes one physical system.
PARAMS = BDCMotorParams(R=3.0, L=4e-3, Kt=0.05, Kb=0.05, J=7.04e-5, B=0.005, V_max=12.0)
ROD = PendulumParams(m=0.05, l=0.1, g=9.81)


def _save_ood(path: Path, trajs) -> None:
    """Save OOD chirp trajectories as plain (N, T) arrays."""
    np.savez(
        path,
        noisy=np.stack([t.omega_noisy for t in trajs]),
        true=np.stack([t.omega_true for t in trajs]),
        voltage=np.stack([t.voltage for t in trajs]),
        theta=np.stack([t.theta_true for t in trajs]),
        dt=1e-3,
    )


def _overview_figure(split, config) -> None:
    """Presentation-ready overview of the generated data."""
    T = split.train_noisy.shape[1]
    t_axis = np.arange(T) * config.dt

    # Pick a representative trajectory with a good speed swing.
    swing = split.train_true.max(axis=1) - split.train_true.min(axis=1)
    n = int(np.argmax(swing))

    fig, axes = plt.subplots(3, 1, figsize=(11, 8), sharex=True)

    axes[0].plot(t_axis, split.train_voltage[n], color="C3")
    axes[0].set_ylabel("Voltage [V]")
    axes[0].grid(True, alpha=0.3)
    axes[0].set_title(
        "Motor + rod filter data — representative trajectory "
        f"(noise std ≈ {np.mean(config.noise_std_range):.1f} rad/s)"
    )

    axes[1].plot(t_axis, split.train_noisy[n], alpha=0.35, color="C0", label="noisy measurement")
    axes[1].plot(t_axis, split.train_true[n], lw=1.6, color="C1", label="true speed")
    axes[1].set_ylabel("Speed [rad/s]")
    axes[1].legend(loc="upper right")
    axes[1].grid(True, alpha=0.3)

    axes[2].plot(t_axis, split.train_noisy[n] - split.train_true[n], color="C2", lw=0.8)
    axes[2].set_ylabel("Sensor noise\n(noisy − true) [rad/s]")
    axes[2].set_xlabel("Time [s]")
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(FIGURES / "data_overview.png", dpi=150)
    plt.close(fig)


def main() -> None:
    config = TrajectoryConfig(
        n_trajectories=210,
        t_end=6.0,
        dt=1e-3,
        param_jitter=0.12,
        noise_std_range=(2.0, 4.0),
        noise_quant_range=(0.0, 1.0),
        train_frac=0.70,
        val_frac=0.15,
        rod=ROD,
    )

    print("Generating in-distribution trajectories …", flush=True)
    trajs = generate_multi_trajectory_dataset(base_params=PARAMS, config=config, seed=0)
    train, val, test = split_trajectories(trajs, config=config, seed=42)
    split = pack_split(train, val, test, dt=config.dt)
    split.save(DATA / "rod_split.npz")

    T = split.train_noisy.shape[1]
    print(
        f"  {config.n_trajectories} trajectories × {T} steps "
        f"({config.t_end} s @ {config.dt * 1e3:.0f} ms)"
    )
    print(f"  train {split.train_noisy.shape}  val {split.val_noisy.shape}  test {split.test_noisy.shape}")
    print("  saved → data/rod_split.npz")

    print("\nGenerating OOD chirp trajectories …", flush=True)
    ood = generate_chirp_trajectories(base_params=PARAMS, config=config, n=20, seed=9000)
    _save_ood(DATA / "rod_ood.npz", ood)
    print(f"  {len(ood)} chirp trajectories  saved → data/rod_ood.npz")

    # Quick signal report (helps confirm the noise SNR is sensible)
    sig = float(split.train_true.std())
    noi = float((split.train_noisy - split.train_true).std())
    print(
        f"\n  speed std ≈ {sig:.2f} rad/s   noise std ≈ {noi:.2f} rad/s   "
        f"SNR ≈ {sig / noi:.1f}×  ({100 * noi / sig:.0f}% noise)"
    )

    _overview_figure(split, config)
    print("  overview → figures/data_overview.png")


if __name__ == "__main__":
    main()
