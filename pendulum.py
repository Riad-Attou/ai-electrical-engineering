"""
Pendulum dataset generation — motor + vertical rod system.
Saves pendulum_dataset.npz and plots noisy vs true speed and rod angle.
"""

import numpy as np
import matplotlib.pyplot as plt
import scipy.io

from utils.motor import (
    BDCMotorParams, PendulumParams, SpeedSensorNoise,
    generate_pendulum_dataset,
)

PARAMS = BDCMotorParams(
R=3.0, L=4e-3, Kt=0.05, Kb=0.05, J=7.04e-5, B=0.005, V_max=3.5)
ROD = PendulumParams(m=0.05, l=0.1, g=9.81)
NOISE = SpeedSensorNoise(std=0.0, quantization=0,
                         rng=np.random.default_rng(42))

dataset = generate_pendulum_dataset(
    params=PARAMS, pendulum=ROD, noise=NOISE,
    dt=0.001, t_end=10.0, profile="step", seed=7,
)
dataset.save("dataset/pendulum_dataset.npz")

print(f"Saved {len(dataset.t)} samples → dataset/pendulum_dataset.npz")
print(f"  Filter INPUT  : dataset.omega_noisy  {dataset.omega_noisy.shape}")
print(f"  Filter TARGET : dataset.omega_true   {dataset.omega_true.shape}")
print(f"  Extra feature : dataset.theta_true   {dataset.theta_true.shape}  (rod angle)")
print(f"  Extra feature : dataset.voltage      {dataset.voltage.shape}")

_mat = scipy.io.loadmat(
    "matlab_simple.mat", squeeze_me=True)

_, axes = plt.subplots(4, 1, figsize=(10, 10), sharex=True)

axes[0].plot(dataset.t, dataset.omega_noisy, alpha=0.5, label="measured (noisy)")
axes[0].plot(dataset.t, dataset.omega_true,  lw=1.5,    label="true speed")
axes[0].set_ylabel("Speed (rad/s)"); axes[0].legend(); axes[0].grid(True)
axes[0].set_title(f"Pendulum dataset  (m={ROD.m} kg, l={ROD.l} m, noise std={NOISE.std} rad/s)")

axes[1].plot(dataset.t, np.degrees(dataset.theta_true), "C2")
axes[1].set_ylabel("Rod angle (deg)"); axes[1].grid(True)

axes[2].plot(dataset.t, dataset.voltage, "C3")
axes[2].set_ylabel("Voltage (V)"); axes[2].set_xlabel("Time (s)"); axes[2].grid(True)

axes[3].plot(_mat["t"],   _mat["vel"],        "C4",  lw=1.2, label="MATLAB / Simulink")
axes[3].plot(dataset.t,  dataset.omega_true, "C0",  lw=1.2, label="math model (Python)", ls="--")
axes[3].set_ylabel("Speed (rad/s)")
axes[3].set_xlabel("Time (s)")
axes[3].set_title("Velocity: MATLAB simulation vs math model")
axes[3].legend()
axes[3].grid(True)

plt.tight_layout()
plt.show()
