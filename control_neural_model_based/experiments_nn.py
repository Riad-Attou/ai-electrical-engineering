"""
Compare NNSpeedController vs PIGravityController on standard benchmark.

Requires trained model:  control_neural_model_based/results/nn_ctrl.pt
Run training first:      python -m control_neural_model_based.train_nn_ctrl

Then run this:           python -m control_neural_model_based.experiments_nn

Output: control_neural_model_based/results/nn_comparison.png
"""

from __future__ import annotations
from pathlib import Path

import numpy as np
import torch
import matplotlib.pyplot as plt

from utils.motor import BDCMotor, BDCMotorParams, PendulumParams, SpeedSensorNoise
from utils.controller import PIGravityController
from utils.motor_torch import NNSpeedController
from utils.traj import QuinticTrajectory

# ---------------------------------------------------------------------------
# System parameters (match experiments.py exactly)
# ---------------------------------------------------------------------------
PARAMS = BDCMotorParams(R=3.0, L=4e-3, Kt=0.05, Kb=0.05,
                        J=7.4e-5, B=0.005, V_max=12.0)
ROD    = PendulumParams(m=0.05, l=0.1, g=9.81)
NOISE  = SpeedSensorNoise(std=0.0, quantization=0.0)

DT     = 1e-3
T_END  = 5.0
KP     = 5.0
KI     = 2.0

TRAJ_Q0, TRAJ_QF = 0.0, 10.0
TRAJ_T0, TRAJ_TF = 0.5, 3.5
CONST_REF = 10.0

MODEL_PATH = Path(__file__).parent / "results" / "nn_ctrl.pt"
OUT_DIR    = Path(__file__).parent / "results"


# ---------------------------------------------------------------------------
# NNSpeedController wrapped as a stateful numpy-compatible controller
# ---------------------------------------------------------------------------
class NNControllerWrapper:
    """Wraps NNSpeedController for step-by-step numpy simulation."""

    def __init__(self, model: NNSpeedController):
        self.model      = model
        self._integral  = 0.0
        self._prev_e    = 0.0
        self._V_max     = model.V_max

    def reset(self):
        self._integral = 0.0
        self._prev_e   = 0.0

    def step(self, omega_meas: float, omega_target: float,
             theta: float, dt: float) -> float:
        e             = omega_target - omega_meas
        delta_e       = e - self._prev_e          # raw Δe, matches training
        self._integral = np.clip(
            self._integral + e * dt, -15.0, 15.0
        )

        with torch.no_grad():
            V = self.model(
                torch.tensor([e],             dtype=torch.float32),
                torch.tensor([delta_e],        dtype=torch.float32),
                torch.tensor([self._integral], dtype=torch.float32),
                torch.tensor([omega_target],   dtype=torch.float32),
                torch.tensor([theta],          dtype=torch.float32),
            ).item()

        self._prev_e = e
        return float(np.clip(V, -self._V_max, self._V_max))


# ---------------------------------------------------------------------------
# Simulation helper
# ---------------------------------------------------------------------------
def _simulate(ctrl, ref_fn) -> dict:
    motor = BDCMotor(PARAMS, pendulum=ROD)
    motor.reset(theta0=0.0)
    ctrl.reset()

    steps     = int(T_END / DT)
    t_arr     = np.empty(steps)
    omega_arr = np.empty(steps)
    ref_arr   = np.empty(steps)
    volt_arr  = np.empty(steps)
    theta_arr = np.empty(steps)

    omega_noisy = 0.0

    for k in range(steps):
        t            = k * DT
        omega_target = float(ref_fn(t))
        voltage      = ctrl.step(omega_meas=omega_noisy,
                                 omega_target=omega_target,
                                 theta=motor.state.theta,
                                 dt=DT)
        state        = motor.step(dt=DT, voltage=voltage)
        omega_noisy  = NOISE.measure(state.omega)

        t_arr[k]     = t
        omega_arr[k] = state.omega
        ref_arr[k]   = omega_target
        volt_arr[k]  = voltage
        theta_arr[k] = state.theta

    return dict(t=t_arr, omega=omega_arr, ref=ref_arr,
                voltage=volt_arr, theta=theta_arr)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def run_comparison():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"No trained model at {MODEL_PATH}\n"
            "Run:  python -m nn.train_nn_ctrl"
        )

    nn_model = NNSpeedController(V_max=PARAMS.V_max)
    nn_model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))
    nn_model.eval()

    ctrl_pi = PIGravityController(Kp=KP, Ki=KI, params=PARAMS, pendulum=ROD)
    ctrl_nn = NNControllerWrapper(nn_model)

    traj = QuinticTrajectory(TRAJ_Q0, TRAJ_QF, TRAJ_T0, TRAJ_TF)

    experiments = [
        ("PI + grav FFW\nconstant ref",   ctrl_pi, lambda t: CONST_REF),
        ("NN controller\nconstant ref",   ctrl_nn, lambda t: CONST_REF),
        ("PI + grav FFW\ntrajectory",     ctrl_pi, lambda t: float(traj.position(t))),
        ("NN controller\ntrajectory",     ctrl_nn, lambda t: float(traj.position(t))),
    ]

    results = []
    for title, ctrl, ref_fn in experiments:
        print(f"Running: {title.replace(chr(10),' ')} ...", end="  ", flush=True)
        data = _simulate(ctrl, ref_fn)
        results.append((title, data))
        err  = data["omega"] - data["ref"]
        rmse = np.sqrt(np.mean(err ** 2))
        print(f"RMSE = {rmse:.4f} rad/s")

    _plot(results)
    return results


def _plot(results):
    fig, axes = plt.subplots(2, 2, figsize=(14, 10), sharex=True)
    axes = axes.flatten()

    for ax, (title, data) in zip(axes, results):
        ax2 = ax.twinx()
        ax2.plot(data["t"], data["voltage"], color="C2", lw=0.8,
                 alpha=0.4, label="voltage (V)")
        ax2.set_ylabel("Voltage (V)", color="C2", fontsize=8)
        ax2.tick_params(axis="y", labelcolor="C2", labelsize=7)
        ax2.set_ylim(-PARAMS.V_max * 1.2, PARAMS.V_max * 1.2)

        ax.plot(data["t"], data["ref"],   "C3--", lw=1.2, label="ref")
        ax.plot(data["t"], data["omega"], "C0",   lw=1.2, label="omega")

        err  = data["omega"] - data["ref"]
        rmse = np.sqrt(np.mean(err ** 2))
        ax.set_title(f"{title}\nRMSE = {rmse:.4f} rad/s", fontsize=9)
        ax.set_ylabel("Speed (rad/s)", fontsize=8)
        ax.set_xlabel("Time (s)", fontsize=8)
        ax.grid(True, lw=0.4)

        lines1, labels1 = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax.legend(lines1 + lines2, labels1 + labels2,
                  loc="lower right", fontsize=7)

    fig.suptitle("NN Controller vs PI+Gravity FFW — BDC Motor + Rod", fontsize=13)
    plt.tight_layout()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUT_DIR / "nn_comparison.png"
    plt.savefig(out, dpi=150)
    print(f"\nFigure saved: {out}")
    plt.show()


if __name__ == "__main__":
    run_comparison()
