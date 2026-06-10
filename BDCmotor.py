"""
Brushed DC Motor — simulation and dataset generation for speed-filter training
"""

import numpy as np
import matplotlib.pyplot as plt

from utils.motor import BDCMotor, BDCMotorParams, SpeedSensorNoise, generate_dataset


PARAMS = BDCMotorParams(
    R=1.0, L=0.5e-3, Kt=0.01, Kb=0.01, J=1e-5, B=1e-6, V_max=12.0
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

    rpm = omega_log * 60 / (2 * np.pi)

    fig, axes = plt.subplots(3, 1, figsize=(9, 7), sharex=True)
    axes[0].plot(t_log, i_log);   axes[0].set_ylabel("Current (A)");  axes[0].grid(True)
    axes[1].plot(t_log, rpm, "C1"); axes[1].set_ylabel("Speed (RPM)"); axes[1].grid(True)
    axes[2].plot(t_log, np.degrees(theta_log), "C2")
    axes[2].set_ylabel("Position (deg)"); axes[2].set_xlabel("Time (s)"); axes[2].grid(True)
    axes[0].set_title(f"Step {voltage} V — raw physics")
    plt.tight_layout()
    # plt.savefig("demo_raw_physics.png", dpi=150)
    plt.show()

    ss = PARAMS.Kt * (voltage / PARAMS.R) / (PARAMS.B + PARAMS.Kt * PARAMS.Kb / PARAMS.R)
    print(f"[raw] Theoretical SS speed : {ss * 60/(2*np.pi):.1f} RPM")
    print(f"[raw] Simulated  SS speed  : {rpm[-1]:.1f} RPM")


# ---------------------------------------------------------------------------
# 2. Dataset generation — noisy vs true speed for filter training
# ---------------------------------------------------------------------------

def demo_dataset():
    dataset = generate_dataset(params=PARAMS, noise=NOISE,
                               dt=1e-3, t_end=100.0, profile="mixed", seed=7)
    dataset.save("motor_dataset.npz")

    print(f"[dataset] {len(dataset.t)} samples saved → motor_dataset.npz")
    print(f"  Filter INPUT  : dataset.omega_noisy  shape {dataset.omega_noisy.shape}")
    print(f"  Filter TARGET : dataset.omega_true   shape {dataset.omega_true.shape}")
    print(f"  Extra feature : dataset.voltage      shape {dataset.voltage.shape}")

    rpm_noisy = dataset.omega_noisy * 60 / (2 * np.pi)
    rpm_true  = dataset.omega_true  * 60 / (2 * np.pi)

    fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
    axes[0].plot(dataset.t, rpm_noisy, alpha=0.55, label="measured (noisy)")
    axes[0].plot(dataset.t, rpm_true,  lw=1.8,     label="true speed")
    axes[0].set_ylabel("Speed (RPM)"); axes[0].legend(); axes[0].grid(True)
    axes[0].set_title("Speed measurement noise — filter training data")

    axes[1].plot(dataset.t, dataset.voltage, "C3")
    axes[1].set_ylabel("Voltage (V)"); axes[1].set_xlabel("Time (s)"); axes[1].grid(True)

    plt.tight_layout()
    # plt.savefig("demo_dataset.png", dpi=150)
    plt.show()



if __name__ == "__main__":
    demo_raw_physics()
    demo_dataset()
