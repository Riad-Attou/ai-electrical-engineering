"""
Compare RL (REINFORCE) vs model-based NN vs PIGravityController.

Requires trained models:
    control_neural_model_based/results/nn_ctrl.pt       — from train_nn_ctrl.py (BPTT)
    control_reinforcement_learning/results/rl_policy.pt — from train_reinforce.py (REINFORCE)

Run:
    /opt/anaconda3/envs/ai-electrical/bin/python -m control_reinforcement_learning.experiments_rl
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
from control_reinforcement_learning.train_reinforce import GaussianPolicy

# ---------------------------------------------------------------------------
# System parameters
# ---------------------------------------------------------------------------
PARAMS = BDCMotorParams(R=3.0, L=4e-3, Kt=0.05, Kb=0.05,
                        J=7.4e-5, B=0.005, V_max=12.0)
ROD    = PendulumParams(m=0.05, l=0.1, g=9.81)
NOISE  = SpeedSensorNoise(std=0.0, quantization=0.0)

DT, T_END = 1e-3, 5.0
KP, KI    = 5.0, 2.0

TRAJ_Q0, TRAJ_QF = 0.0, 10.0
TRAJ_T0, TRAJ_TF = 0.5, 3.5
CONST_REF = 10.0

BPTT_MODEL = Path(__file__).parent.parent / "control_neural_model_based" / "results" / "nn_ctrl.pt"
RL_MODEL   = Path(__file__).parent / "results" / "rl_policy.pt"
OUT_DIR    = Path(__file__).parent / "results"


# ---------------------------------------------------------------------------
# Controller wrappers (all share same .step() interface)
# ---------------------------------------------------------------------------
class BPTTControllerWrapper:
    """Wraps NNSpeedController (model-based BPTT) for step-by-step numpy sim."""

    def __init__(self, model: NNSpeedController):
        self.model      = model
        self._integral  = 0.0
        self._prev_e    = 0.0

    def reset(self):
        self._integral = 0.0
        self._prev_e   = 0.0

    def step(self, omega_meas, omega_target, theta, dt):
        e             = omega_target - omega_meas
        delta_e       = e - self._prev_e
        self._integral = float(np.clip(self._integral + e * dt, -15.0, 15.0))
        with torch.no_grad():
            V = self.model(
                torch.tensor([e],              dtype=torch.float32),
                torch.tensor([delta_e],         dtype=torch.float32),
                torch.tensor([self._integral],  dtype=torch.float32),
                torch.tensor([omega_target],    dtype=torch.float32),
                torch.tensor([theta],           dtype=torch.float32),
            ).item()
        self._prev_e = e
        return float(np.clip(V, -PARAMS.V_max, PARAMS.V_max))


class RLControllerWrapper:
    """Wraps GaussianPolicy (REINFORCE) — uses mean action (deterministic at test)."""

    def __init__(self, policy: GaussianPolicy):
        self.policy     = policy
        self._integral  = 0.0
        self._prev_e    = 0.0

    def reset(self):
        self._integral = 0.0
        self._prev_e   = 0.0

    def step(self, omega_meas, omega_target, theta, dt):
        e             = omega_target - omega_meas
        delta_e       = e - self._prev_e
        self._integral = float(np.clip(self._integral + e * dt, -15.0, 15.0))

        obs = np.array([
            e             / 15.0,
            delta_e       / 1.0,
            self._integral / 15.0,
            omega_target  / 15.0,
            np.sin(theta),
            np.cos(theta),
        ], dtype=np.float32)

        V = self.policy.mean_action(obs)
        self._prev_e = e
        return float(np.clip(V, -PARAMS.V_max, PARAMS.V_max))


# ---------------------------------------------------------------------------
# Simulation helper
# ---------------------------------------------------------------------------
def _simulate(ctrl, ref_fn) -> dict:
    motor = BDCMotor(PARAMS, pendulum=ROD)
    motor.reset(theta0=0.0)
    ctrl.reset()

    steps = int(T_END / DT)
    t_arr = np.empty(steps); omega_arr = np.empty(steps)
    ref_arr = np.empty(steps); volt_arr = np.empty(steps)

    omega_noisy = 0.0
    for k in range(steps):
        t            = k * DT
        omega_target = float(ref_fn(t))
        voltage      = ctrl.step(omega_noisy, omega_target, motor.state.theta, DT)
        state        = motor.step(DT, voltage)
        omega_noisy  = NOISE.measure(state.omega)

        t_arr[k] = t; omega_arr[k] = state.omega
        ref_arr[k] = omega_target; volt_arr[k] = voltage

    return dict(t=t_arr, omega=omega_arr, ref=ref_arr, voltage=volt_arr)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def run():
    for path, name in [(BPTT_MODEL, "BPTT"), (RL_MODEL, "RL")]:
        if not path.exists():
            raise FileNotFoundError(f"{name} model not found: {path}")

    bptt_model = NNSpeedController(V_max=PARAMS.V_max)
    bptt_model.load_state_dict(torch.load(BPTT_MODEL, map_location="cpu"))
    bptt_model.eval()

    rl_policy = GaussianPolicy(V_max=PARAMS.V_max)
    rl_policy.load_state_dict(torch.load(RL_MODEL, map_location="cpu"))
    rl_policy.eval()

    ctrl_pi   = PIGravityController(Kp=KP, Ki=KI, params=PARAMS, pendulum=ROD)
    ctrl_bptt = BPTTControllerWrapper(bptt_model)
    ctrl_rl   = RLControllerWrapper(rl_policy)

    traj = QuinticTrajectory(TRAJ_Q0, TRAJ_QF, TRAJ_T0, TRAJ_TF)

    experiments = [
        ("PI + grav FFW\nconstant ref",  ctrl_pi,   lambda t: CONST_REF),
        ("NN BPTT\nconstant ref",        ctrl_bptt, lambda t: CONST_REF),
        ("NN REINFORCE\nconstant ref",   ctrl_rl,   lambda t: CONST_REF),
        ("PI + grav FFW\ntrajectory",    ctrl_pi,   lambda t: float(traj.position(t))),
        ("NN BPTT\ntrajectory",          ctrl_bptt, lambda t: float(traj.position(t))),
        ("NN REINFORCE\ntrajectory",     ctrl_rl,   lambda t: float(traj.position(t))),
    ]

    results = []
    for title, ctrl, ref_fn in experiments:
        print(f"Running: {title.replace(chr(10),' ')} ...", end="  ", flush=True)
        data = _simulate(ctrl, ref_fn)
        results.append((title, data))
        rmse = np.sqrt(np.mean((data["omega"] - data["ref"]) ** 2))
        print(f"RMSE = {rmse:.4f} rad/s")

    _plot(results)
    return results


def _plot(results):
    n   = len(results)
    fig, axes = plt.subplots(2, 3, figsize=(18, 10), sharex=True)
    axes = axes.flatten()

    for ax, (title, data) in zip(axes, results):
        ax2 = ax.twinx()
        ax2.plot(data["t"], data["voltage"], "C2", lw=0.7, alpha=0.4, label="voltage")
        ax2.set_ylabel("Voltage (V)", color="C2", fontsize=7)
        ax2.tick_params(axis="y", labelcolor="C2", labelsize=6)
        ax2.set_ylim(-PARAMS.V_max * 1.2, PARAMS.V_max * 1.2)

        ax.plot(data["t"], data["ref"],   "C3--", lw=1.2, label="ref")
        ax.plot(data["t"], data["omega"], "C0",   lw=1.2, label="ω")

        rmse = np.sqrt(np.mean((data["omega"] - data["ref"]) ** 2))
        ax.set_title(f"{title}\nRMSE = {rmse:.4f} rad/s", fontsize=9)
        ax.set_ylabel("Speed (rad/s)", fontsize=8)
        ax.set_xlabel("Time (s)", fontsize=8)
        ax.grid(True, lw=0.4)

        lines1, l1 = ax.get_legend_handles_labels()
        lines2, l2 = ax2.get_legend_handles_labels()
        ax.legend(lines1 + lines2, l1 + l2, loc="lower right", fontsize=6)

    fig.suptitle("PI vs NN BPTT vs NN REINFORCE — BDC Motor + Rod", fontsize=13)
    plt.tight_layout()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUT_DIR / "rl_comparison.png"
    plt.savefig(out, dpi=150)
    print(f"\nFigure: {out}")
    plt.show()


if __name__ == "__main__":
    run()
