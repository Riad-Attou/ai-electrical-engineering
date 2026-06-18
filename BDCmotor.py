"""
Brushed DC Motor — simulation and dataset generation for speed-filter training
"""

import numpy as np
import matplotlib.pyplot as plt

from utils.motor import BDCMotor, BDCMotorParams, PendulumParams, SpeedSensorNoise, generate_dataset
from utils.visualizer import MotorVisualizer


PARAMS = BDCMotorParams(
R=0.4, L=1e-3, Kt=0.1, Kb=0.1, J=0.04, B=1e-9, V_max=12.0
)
NOISE = SpeedSensorNoise(std=15.0, quantization=3,
                         rng=np.random.default_rng(42))


# ---------------------------------------------------------------------------
# 1. Raw physics — step-voltage test
# ---------------------------------------------------------------------------

def demo_raw_physics():
    motor = BDCMotor(PARAMS)
    dt, t_end, voltage = 1e-3, 1.0, 12.0
    steps = int(t_end / dt)

    t_log = np.empty(steps); omega_log = np.empty(steps)
    i_log = np.empty(steps); theta_log = np.empty(steps)

    for k in range(steps):
        s = motor.step(dt, voltage)
        t_log[k] = s.t; omega_log[k] = s.omega
        i_log[k] = s.i; theta_log[k] = s.theta

    fig, axes = plt.subplots(3, 1, figsize=(9, 7), sharex=True)
    axes[0].plot(t_log, i_log);         axes[0].set_ylabel("Current (A)");    axes[0].grid(True)
    axes[1].plot(t_log, omega_log, "C1"); axes[1].set_ylabel("Speed (rad/s)"); axes[1].grid(True)
    axes[2].plot(t_log, np.degrees(theta_log), "C2")
    axes[2].set_ylabel("Position (deg)"); axes[2].set_xlabel("Time (s)"); axes[2].grid(True)
    axes[0].set_title(f"Step {voltage} V — raw physics")
    plt.tight_layout()
    # plt.savefig("demo_raw_physics.png", dpi=150)
    plt.show()

    ss = PARAMS.Kt * (voltage / PARAMS.R) / (PARAMS.B + PARAMS.Kt * PARAMS.Kb / PARAMS.R)
    print(f"[raw] Theoretical SS speed : {ss:.3f} rad/s")
    print(f"[raw] Simulated  SS speed  : {omega_log[-1]:.3f} rad/s")


# ---------------------------------------------------------------------------
# 2. Dataset generation — noisy vs true speed for filter training
# ---------------------------------------------------------------------------

def demo_dataset():
    dataset = generate_dataset(params=PARAMS, noise=NOISE,
                               dt=1e-3, t_end=100.0, profile="mixed", seed=7)
    dataset.save("dataset/motor_dataset.npz")

    print(f"[dataset] {len(dataset.t)} samples saved → dataset/motor_dataset.npz")
    print(f"  Filter INPUT  : dataset.omega_noisy  shape {dataset.omega_noisy.shape}")
    print(f"  Filter TARGET : dataset.omega_true   shape {dataset.omega_true.shape}")
    print(f"  Extra feature : dataset.voltage      shape {dataset.voltage.shape}")

    fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
    axes[0].plot(dataset.t, dataset.omega_noisy, alpha=0.55, label="measured (noisy)")
    axes[0].plot(dataset.t, dataset.omega_true,  lw=1.8,     label="true speed")
    axes[0].set_ylabel("Speed (rad/s)"); axes[0].legend(); axes[0].grid(True)
    axes[0].set_title("Speed measurement noise — filter training data")

    axes[1].plot(dataset.t, dataset.voltage, "C3")
    axes[1].set_ylabel("Voltage (V)"); axes[1].set_xlabel("Time (s)"); axes[1].grid(True)

    plt.tight_layout()
    # plt.savefig("demo_dataset.png", dpi=150)
    plt.show()


# ---------------------------------------------------------------------------
# 3. Real-time pygame visualizer
# ---------------------------------------------------------------------------

def demo_visualizer():
    rod   = PendulumParams(m=0.1, l=0.3, g=9.81)
    motor = BDCMotor(PARAMS, pendulum=rod)
    noise = NOISE

    dt      = 1e-3
    voltage = 5.0

    vis = MotorVisualizer(title="BDC Motor + Rod", history_len=600)
    vis.open()

    running = True
    while running:
        state        = motor.step(dt, voltage)
        omega_noisy  = noise.measure(state.omega)
        running      = vis.update(state, voltage, omega_noisy=omega_noisy)

    vis.close()


if __name__ == "__main__":
    demo_visualizer()

