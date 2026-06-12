"""
Multi-trajectory dataset utilities for speed-filter training.

Design rationale
----------------
A single long trajectory (the original approach) creates two problems:
  1. Windowed train/val/test splits overlap in time → data leakage.
  2. The filter only ever sees one motor, one noise level, one voltage profile
     → it can memorize rather than learn to denoise.

This module generates N short independent trajectories with controlled
variability across motor parameters, sensor noise, and voltage excitation.
Train/val/test are split at the trajectory level so there is zero leakage.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from utils.motor import BDCMotorParams, MotorDataset, SpeedSensorNoise, generate_dataset


@dataclass
class TrajectoryConfig:
    """Controls per-trajectory variability for multi-trajectory dataset generation."""

    n_trajectories: int = 200  # total independent trajectories
    t_end: float = 10.0  # seconds per trajectory
    dt: float = 1e-3
    # Motor params R, J, B are perturbed ±param_jitter fraction per trajectory
    # to simulate unit-to-unit manufacturing tolerances.
    param_jitter: float = 0.15
    # Per-trajectory Gaussian noise std drawn uniformly from this range [rad/s]
    noise_std_range: tuple = (5.0, 30.0)
    # Per-trajectory encoder quantization step [rad/s], 0 = continuous
    noise_quant_range: tuple = (0.0, 5.0)
    train_frac: float = 0.70
    val_frac: float = 0.15
    # test_frac = 1 - train_frac - val_frac  (~0.15 for defaults)


@dataclass
class MotorSplit:
    """
    Stacked train/val/test arrays, each shaped (N_trajectories, T_steps).

    The (N, T) layout is designed for sequence models (TCN, 1D CNN, GRU):
      - Iterate over N to get individual trajectories.
      - Window along T within each trajectory. Never window across trajectories.

    Arrays
    ------
    {train,val,test}_noisy   : noisy speed measurement [rad/s]  — filter input
    {train,val,test}_true    : true  speed             [rad/s]  — filter target
    {train,val,test}_voltage : applied voltage         [V]      — optional extra feature
    dt                       : simulation time step    [s]
    """

    train_noisy: np.ndarray  # (N_train, T)
    train_true: np.ndarray
    train_voltage: np.ndarray
    val_noisy: np.ndarray  # (N_val,   T)
    val_true: np.ndarray
    val_voltage: np.ndarray
    test_noisy: np.ndarray  # (N_test,  T)
    test_true: np.ndarray
    test_voltage: np.ndarray
    dt: float

    def save(self, path: str):
        np.savez(
            path,
            train_noisy=self.train_noisy,
            train_true=self.train_true,
            train_voltage=self.train_voltage,
            val_noisy=self.val_noisy,
            val_true=self.val_true,
            val_voltage=self.val_voltage,
            test_noisy=self.test_noisy,
            test_true=self.test_true,
            test_voltage=self.test_voltage,
            dt=self.dt,
        )

    @staticmethod
    def load(path: str) -> "MotorSplit":
        d = np.load(path)
        return MotorSplit(
            train_noisy=d["train_noisy"],
            train_true=d["train_true"],
            train_voltage=d["train_voltage"],
            val_noisy=d["val_noisy"],
            val_true=d["val_true"],
            val_voltage=d["val_voltage"],
            test_noisy=d["test_noisy"],
            test_true=d["test_true"],
            test_voltage=d["test_voltage"],
            dt=float(d["dt"]),
        )


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

_PROFILES = ("step", "ramp", "random", "mixed")


def _perturb_params(
    base: BDCMotorParams, jitter: float, rng: np.random.Generator
) -> BDCMotorParams:
    """Return a copy of base_params with R, J, B perturbed by ±jitter fraction."""

    def p(v: float) -> float:
        return float(v * (1.0 + rng.uniform(-jitter, jitter)))

    return BDCMotorParams(
        R=p(base.R),
        L=base.L,
        Kt=base.Kt,
        Kb=base.Kb,
        J=p(base.J),
        B=p(base.B),
        V_max=base.V_max,
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def generate_multi_trajectory_dataset(
    base_params: BDCMotorParams | None = None,
    config: TrajectoryConfig | None = None,
    seed: int = 0,
) -> list[MotorDataset]:
    """
    Generate N independent trajectories with varied motor params, noise, and
    voltage profiles.

    Variability sources
    -------------------
    - Motor params (R, J, B) are perturbed ±param_jitter per trajectory.
    - Noise std and quantization are drawn from their respective ranges.
    - Voltage profile cycles through step / ramp / random / mixed.
    - Every trajectory uses a fully independent random seed.

    Returns
    -------
    List of MotorDataset, one per trajectory.
    Pass to split_trajectories() to get train/val/test.
    """
    base_params = base_params or BDCMotorParams()
    config = config or TrajectoryConfig()
    master_rng = np.random.default_rng(seed)

    trajectories: list[MotorDataset] = []
    for k in range(config.n_trajectories):
        traj_seed = int(master_rng.integers(0, 2**31))
        traj_rng = np.random.default_rng(traj_seed)

        params = _perturb_params(base_params, config.param_jitter, traj_rng)

        noise = SpeedSensorNoise(
            std=float(traj_rng.uniform(*config.noise_std_range)),
            quantization=float(traj_rng.uniform(*config.noise_quant_range)),
            # Separate seed so noise is independent from param perturbation RNG.
            rng=np.random.default_rng(traj_seed + 100_000),
        )

        traj = generate_dataset(
            params=params,
            noise=noise,
            dt=config.dt,
            t_end=config.t_end,
            profile=_PROFILES[k % len(_PROFILES)],
            seed=traj_seed,
        )
        trajectories.append(traj)

    return trajectories


def split_trajectories(
    trajectories: list[MotorDataset],
    config: TrajectoryConfig | None = None,
    seed: int = 42,
) -> tuple[list[MotorDataset], list[MotorDataset], list[MotorDataset]]:
    """
    Split trajectories into train/val/test at the trajectory level.

    Trajectories are shuffled before splitting so every split contains a
    representative mix of voltage profiles and noise levels.

    This guarantees zero data leakage: each split is a disjoint set of
    trajectories, so no time-step from a test trajectory ever appears in train.
    """
    config = config or TrajectoryConfig()
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(trajectories)).tolist()
    n = len(idx)
    n_tr = int(n * config.train_frac)
    n_va = int(n * config.val_frac)

    train = [trajectories[i] for i in idx[:n_tr]]
    val = [trajectories[i] for i in idx[n_tr : n_tr + n_va]]
    test = [trajectories[i] for i in idx[n_tr + n_va :]]
    return train, val, test


def pack_split(
    train: list[MotorDataset],
    val: list[MotorDataset],
    test: list[MotorDataset],
    dt: float,
) -> MotorSplit:
    """Stack trajectory lists into (N_traj, T_steps) arrays."""

    def stack(split: list[MotorDataset], attr: str) -> np.ndarray:
        return np.stack([getattr(t, attr) for t in split])

    return MotorSplit(
        train_noisy=stack(train, "omega_noisy"),
        train_true=stack(train, "omega_true"),
        train_voltage=stack(train, "voltage"),
        val_noisy=stack(val, "omega_noisy"),
        val_true=stack(val, "omega_true"),
        val_voltage=stack(val, "voltage"),
        test_noisy=stack(test, "omega_noisy"),
        test_true=stack(test, "omega_true"),
        test_voltage=stack(test, "voltage"),
        dt=dt,
    )
