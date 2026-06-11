"""
Brushed DC Motor — simulation and dataset generation for speed-filter training

Demos
-----
demo_raw_physics()   — single step-voltage run, shows current / speed / position
demo_dataset()       — original single-trajectory dataset (quick sanity check)
demo_ml_dataset()    — multi-trajectory dataset split by trajectory for AI training
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from utils.motor import BDCMotor, BDCMotorParams, SpeedSensorNoise, generate_dataset

FIGURES = Path("figures")
DATA    = Path("data")
FIGURES.mkdir(exist_ok=True)
DATA.mkdir(exist_ok=True)
from utils.traj import (
    TrajectoryConfig,
    generate_multi_trajectory_dataset,
    pack_split,
    split_trajectories,
)

PARAMS = BDCMotorParams(R=1.0, L=0.5e-3, Kt=0.01, Kb=0.01, J=1e-5, B=1e-6, V_max=12.0)
NOISE = SpeedSensorNoise(std=15.0, quantization=3, rng=np.random.default_rng(42))


# ---------------------------------------------------------------------------
# 1. Raw physics — step-voltage test
# ---------------------------------------------------------------------------


def demo_raw_physics():
    motor = BDCMotor(PARAMS)
    dt, t_end, voltage = 1e-3, 1.0, 12.0
    steps = int(t_end / dt)

    t_log = np.empty(steps)
    omega_log = np.empty(steps)
    i_log = np.empty(steps)
    theta_log = np.empty(steps)

    for k in range(steps):
        s = motor.step(dt, voltage)
        t_log[k] = s.t
        omega_log[k] = s.omega
        i_log[k] = s.i
        theta_log[k] = s.theta

    rpm = omega_log * 60 / (2 * np.pi)

    fig, axes = plt.subplots(3, 1, figsize=(9, 7), sharex=True)
    axes[0].plot(t_log, i_log)
    axes[0].set_ylabel("Current (A)")
    axes[0].grid(True)
    axes[1].plot(t_log, rpm, "C1")
    axes[1].set_ylabel("Speed (RPM)")
    axes[1].grid(True)
    axes[2].plot(t_log, np.degrees(theta_log), "C2")
    axes[2].set_ylabel("Position (deg)")
    axes[2].set_xlabel("Time (s)")
    axes[2].grid(True)
    axes[0].set_title(f"Step {voltage} V — raw physics")
    plt.tight_layout()
    plt.savefig(FIGURES / "demo_raw_physics.png", dpi=150)
    plt.show()

    ss = (
        PARAMS.Kt * (voltage / PARAMS.R) / (PARAMS.B + PARAMS.Kt * PARAMS.Kb / PARAMS.R)
    )
    print(f"[raw] Theoretical SS speed : {ss * 60 / (2 * np.pi):.1f} RPM")
    print(f"[raw] Simulated  SS speed  : {rpm[-1]:.1f} RPM")


# ---------------------------------------------------------------------------
# 2. Dataset generation — noisy vs true speed for filter training
# ---------------------------------------------------------------------------


def demo_dataset():
    dataset = generate_dataset(
        params=PARAMS, noise=NOISE, dt=1e-3, t_end=100.0, profile="mixed", seed=7
    )
    dataset.save(DATA / "motor_dataset.npz")

    print(f"[dataset] {len(dataset.t)} samples saved → data/motor_dataset.npz")
    print(f"  Filter INPUT  : dataset.omega_noisy  shape {dataset.omega_noisy.shape}")
    print(f"  Filter TARGET : dataset.omega_true   shape {dataset.omega_true.shape}")
    print(f"  Extra feature : dataset.voltage      shape {dataset.voltage.shape}")

    rpm_noisy = dataset.omega_noisy * 60 / (2 * np.pi)
    rpm_true = dataset.omega_true * 60 / (2 * np.pi)

    fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
    axes[0].plot(dataset.t, rpm_noisy, alpha=0.55, label="measured (noisy)")
    axes[0].plot(dataset.t, rpm_true, lw=1.8, label="true speed")
    axes[0].set_ylabel("Speed (RPM)")
    axes[0].legend()
    axes[0].grid(True)
    axes[0].set_title("Speed measurement noise — filter training data")

    axes[1].plot(dataset.t, dataset.voltage, "C3")
    axes[1].set_ylabel("Voltage (V)")
    axes[1].set_xlabel("Time (s)")
    axes[1].grid(True)

    plt.tight_layout()
    plt.savefig(FIGURES / "demo_dataset.png", dpi=150)
    plt.show()


# ---------------------------------------------------------------------------
# 3. Multi-trajectory dataset — proper AI training setup
# ---------------------------------------------------------------------------


def demo_ml_dataset():
    """
    Generate 200 independent trajectories with varied motor params, noise, and
    voltage profiles, then split them 70/15/15 by trajectory index.

    This is the recommended starting point for training a speed filter
    (TCN, 1D CNN, GRU, etc.).  The saved motor_split.npz has arrays shaped
    (N_trajectories, T_steps) — window along the T axis inside each trajectory
    when building a PyTorch / TF Dataset, never across trajectories.
    """
    config = TrajectoryConfig(
        n_trajectories=200,
        t_end=10.0,
        dt=1e-3,
        param_jitter=0.15,  # ±15 % on R, J, B → unit-to-unit tolerance
        noise_std_range=(5.0, 30.0),
        noise_quant_range=(0.0, 5.0),
        train_frac=0.70,
        val_frac=0.15,
    )

    print("Generating trajectories …", flush=True)
    trajectories = generate_multi_trajectory_dataset(
        base_params=PARAMS, config=config, seed=0
    )

    train, val, test = split_trajectories(trajectories, config=config, seed=42)
    split = pack_split(train, val, test, dt=config.dt)
    split.save(DATA / "motor_split.npz")

    T = split.train_noisy.shape[1]
    n_tr, n_va, n_te = len(train), len(val), len(test)
    print(
        f"\n[ml-dataset] {config.n_trajectories} trajectories × {T} steps  ({config.t_end} s @ {config.dt * 1e3:.0f} ms)"
    )
    print(f"  Train  {split.train_noisy.shape}   {n_tr} trajectories")
    print(f"  Val    {split.val_noisy.shape}    {n_va} trajectories")
    print(f"  Test   {split.test_noisy.shape}    {n_te} trajectories")
    print("  Saved → data/motor_split.npz")
    print()
    print("  Filter input  : split.train_noisy   (N, T)  noisy speed [rad/s]")
    print("  Filter target : split.train_true    (N, T)  true  speed [rad/s]")
    print("  Extra feature : split.train_voltage (N, T)  voltage     [V]")

    # Show 4 sample training trajectories
    t_axis = np.arange(T) * config.dt
    rpm_noisy = split.train_noisy[:4] * 60 / (2 * np.pi)
    rpm_true = split.train_true[:4] * 60 / (2 * np.pi)

    fig, axes = plt.subplots(4, 1, figsize=(10, 8), sharex=True)
    for ax, i in zip(axes, range(4)):
        ax.plot(
            t_axis,
            rpm_noisy[i],
            alpha=0.45,
            color="C0",
            label="noisy" if i == 0 else None,
        )
        ax.plot(
            t_axis, rpm_true[i], lw=1.5, color="C1", label="true" if i == 0 else None
        )
        ax.set_ylabel(f"Traj {i} (RPM)")
        ax.grid(True)
    axes[0].set_title("Sample training trajectories — noisy vs true speed")
    axes[0].legend()
    axes[-1].set_xlabel("Time (s)")
    plt.tight_layout()
    plt.savefig(FIGURES / "demo_ml_dataset.png", dpi=150)
    plt.show()


if __name__ == "__main__":
    demo_raw_physics()
    demo_dataset()
    demo_ml_dataset()
