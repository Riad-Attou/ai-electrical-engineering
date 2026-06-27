"""
12-experiment speed-control comparison for BDC motor + vertical rod.

Experiments
-----------
  1. PI only                                — constant reference  (10 rad/s)
  2. PI only                                — quintic trajectory  (0 -> 10 rad/s)
  3. PI + gravity FFW                       — constant reference
  4. PI + gravity FFW                       — quintic trajectory
  5. PI + polynomial FFW                    — constant reference
  6. PI + polynomial FFW                    — quintic trajectory
  7. PI + polynomial FFW + gravity FFW      — constant reference
  8. PI + polynomial FFW + gravity FFW      — quintic trajectory
  9. PI + full model FFW + gravity FFW      — constant reference
 10. PI + full model FFW + gravity FFW      — quintic trajectory
 11. PI + NN FFW                            — constant reference
 12. PI + NN FFW                            — quintic trajectory

Run:  python experiments.py   (from project root)
Output: poly/results/experiments.png
"""

from __future__ import annotations
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from utils.motor import BDCMotor, BDCMotorParams, PendulumParams, SpeedSensorNoise
from utils.traj import QuinticTrajectory
from poly.poly_controller import PolyFeedforwardController
from nn.nn_controller import NNFeedforwardController

# ---------------------------------------------------------------------------
# System parameters  (shared across all experiments)
# ---------------------------------------------------------------------------
PARAMS = BDCMotorParams(R=3.0, L=4e-3, Kt=0.05, Kb=0.05,
                        J=7.4e-5, B=0.005, V_max=12.0)
ROD    = PendulumParams(m=0.05, l=0.1, g=9.81)
NOISE  = SpeedSensorNoise(std=0.0, quantization=0.0)

DT     = 1e-3
T_END  = 5.0
KP     = 5.0
KI     = 2.0

# Trajectory: 0 -> 10 rad/s, ramps between t=0.5 s and t=3.5 s
TRAJ_Q0 = 0.0
TRAJ_QF = 10.0
TRAJ_T0 = 0.5
TRAJ_TF = 3.5

CONST_REF = 10.0   # rad/s for constant-reference experiments

OUT_DIR = Path(__file__).parent / "poly" / "results"

# ---------------------------------------------------------------------------
# Self-contained controller classes
# (defined here so experiments are independent of utils/controller.py)
# ---------------------------------------------------------------------------

class _PurePI:
    """Proportional-Integral controller, no feedforward."""

    def __init__(self, Kp: float, Ki: float, V_max: float):
        self.Kp = Kp
        self.Ki = Ki
        self._V_max = V_max
        self._integral = 0.0

    def reset(self) -> None:
        self._integral = 0.0

    def step(self, omega_meas: float, omega_target: float,
             theta: float, dt: float) -> float:
        e = omega_target - omega_meas
        self._integral += e * dt
        if self.Ki != 0.0:
            self._integral = np.clip(
                self._integral, -self._V_max / self.Ki, self._V_max / self.Ki)
        V = self.Kp * e + self.Ki * self._integral
        return float(np.clip(V, -self._V_max, self._V_max))


class _GravPI:
    """PI + analytical gravity feedforward only (no poly FFW)."""

    def __init__(self, Kp: float, Ki: float,
                 params: BDCMotorParams, pendulum: PendulumParams):
        self.Kp = Kp
        self.Ki = Ki
        self._p = params
        self._rod = pendulum
        self._integral = 0.0

    def reset(self) -> None:
        self._integral = 0.0

    def step(self, omega_meas: float, omega_target: float,
             theta: float, dt: float) -> float:
        p, rod, V_max = self._p, self._rod, self._p.V_max

        V_ff_grav = rod.m * rod.g * rod.l_cm * np.sin(theta) * p.R / p.Kt

        e = omega_target - omega_meas
        self._integral += e * dt
        if self.Ki != 0.0:
            self._integral = np.clip(
                self._integral, -V_max / self.Ki, V_max / self.Ki)

        V = V_ff_grav + self.Kp * e + self.Ki * self._integral
        return float(np.clip(V, -V_max, V_max))


class _PolyGravPI:
    """PI + polynomial FFW + analytical gravity feedforward."""

    def __init__(self, Kp: float, Ki: float,
                 params: BDCMotorParams, pendulum: PendulumParams,
                 coeffs: np.ndarray, omega_min: float, omega_max: float):
        self.Kp = Kp
        self.Ki = Ki
        self._p = params
        self._rod = pendulum
        self._coeffs = coeffs
        self._omega_min = omega_min
        self._omega_max = omega_max
        self._integral = 0.0

    def reset(self) -> None:
        self._integral = 0.0

    def step(self, omega_meas: float, omega_target: float,
             theta: float, dt: float) -> float:
        p, rod, V_max = self._p, self._rod, self._p.V_max

        omega_c = np.clip(omega_target, self._omega_min, self._omega_max)
        V_ff_poly =  float(np.clip(np.polyval(self._coeffs, omega_c), -V_max, V_max))
        V_ff_grav = rod.m * rod.g * rod.l_cm * np.sin(theta) * p.R / p.Kt

        e = omega_target - omega_meas
        self._integral += e * dt
        if self.Ki != 0.0:
            self._integral = np.clip(
                self._integral, -V_max / self.Ki, V_max / self.Ki)

        V = V_ff_poly + V_ff_grav + self.Kp * e + self.Ki * self._integral
        return float(np.clip(V, -V_max, V_max))


class _ModelGravPI:
    """PI + full analytical motor model FFW + analytical gravity FFW."""

    def __init__(self, Kp: float, Ki: float,
                 params: BDCMotorParams, pendulum: PendulumParams):
        self.Kp = Kp
        self.Ki = Ki
        self._p = params
        self._rod = pendulum
        self._integral = 0.0

    def reset(self) -> None:
        self._integral = 0.0

    def step(self, omega_meas: float, omega_target: float,
             theta: float, dt: float) -> float:
        p, rod, V_max = self._p, self._rod, self._p.V_max

        V_ff_model = omega_target * (p.R * p.B + p.Kb * p.Kt) / p.Kt
        V_ff_grav  = rod.m * rod.g * rod.l_cm * np.sin(theta) * p.R / p.Kt

        e = omega_target - omega_meas
        self._integral += e * dt
        if self.Ki != 0.0:
            self._integral = np.clip(
                self._integral, -V_max / self.Ki, V_max / self.Ki)

        V = V_ff_model + V_ff_grav + self.Kp * e + self.Ki * self._integral
        return float(np.clip(V, -V_max, V_max))


# ---------------------------------------------------------------------------
# Load polynomial coefficients (required for experiments 5-8)
# ---------------------------------------------------------------------------
_COEFFS_PATH = Path(__file__).parent / "poly" / "data" / "poly_coeffs.npz"

def _load_poly():
    if not _COEFFS_PATH.exists():
        raise FileNotFoundError(
            f"Polynomial coefficients not found: {_COEFFS_PATH}\n"
            "Run:  python -m poly.poly_collect\n"
            "      python -m poly.poly_fit"
        )
    d = np.load(_COEFFS_PATH)
    return d["coeffs"], float(d["omega_min"]), float(d["omega_max"])


# ---------------------------------------------------------------------------
# Load NN weights (required for experiments 11-12)
# ---------------------------------------------------------------------------
_NN_WEIGHTS_PATH = Path(__file__).parent / "nn" / "data" / "nn_weights.npz"

def _load_nn():
    if not _NN_WEIGHTS_PATH.exists():
        raise FileNotFoundError(
            f"NN weights not found: {_NN_WEIGHTS_PATH}\n"
            "Run:  python -m nn.nn_collect\n"
            "      python -m nn.nn_fit"
        )
    return NNFeedforwardController.from_file(
        Kp=KP, Ki=KI, params=PARAMS, pendulum=ROD, path=_NN_WEIGHTS_PATH
    )


# ---------------------------------------------------------------------------
# Simulation helper
# ---------------------------------------------------------------------------

def _simulate(ctrl, ref_fn) -> dict:
    """
    Run one experiment for T_END seconds.

    ctrl   : controller with .step(omega_meas, omega_target, theta, dt) -> float
             and .reset()
    ref_fn : callable t -> omega_ref (scalar float)
    """
    motor = BDCMotor(PARAMS, pendulum=ROD)
    motor.reset(theta0=0.0)
    ctrl.reset()

    steps = int(T_END / DT)
    t_arr     = np.empty(steps)
    omega_arr = np.empty(steps)
    ref_arr   = np.empty(steps)
    volt_arr  = np.empty(steps)
    theta_arr = np.empty(steps)

    omega_noisy = 0.0
    voltage     = 0.0

    for k in range(steps):
        t = k * DT
        omega_target = float(ref_fn(t))

        voltage     = ctrl.step(omega_meas=omega_noisy,
                                omega_target=omega_target,
                                theta=motor.state.theta,
                                dt=DT)
        state       = motor.step(dt=DT, voltage=voltage)
        omega_noisy = NOISE.measure(state.omega)

        t_arr[k]     = t
        omega_arr[k] = state.omega
        ref_arr[k]   = omega_target
        volt_arr[k]  = voltage
        theta_arr[k] = state.theta

    return dict(t=t_arr, omega=omega_arr, ref=ref_arr,
                voltage=volt_arr, theta=theta_arr)


# ---------------------------------------------------------------------------
# Experiment definitions
# ---------------------------------------------------------------------------

def run_experiments():
    poly_coeffs, omega_min, omega_max = _load_poly()
    ctrl_nn = _load_nn()

    traj = QuinticTrajectory(TRAJ_Q0, TRAJ_QF, TRAJ_T0, TRAJ_TF)

    def const_ref(t):
        return CONST_REF

    def traj_ref(t):
        return float(traj.position(t))

    # -- controllers --------------------------------------------------------
    ctrl_pi = _PurePI(KP, KI, PARAMS.V_max)

    ctrl_grav = _GravPI(KP, KI, PARAMS, ROD)

    ctrl_poly = PolyFeedforwardController(
        Kp=KP, Ki=KI, params=PARAMS,
        coeffs=poly_coeffs, omega_min=omega_min, omega_max=omega_max,
    )

    ctrl_poly_grav = _PolyGravPI(
        Kp=KP, Ki=KI, params=PARAMS, pendulum=ROD,
        coeffs=poly_coeffs, omega_min=omega_min, omega_max=omega_max,
    )

    ctrl_model_grav = _ModelGravPI(Kp=KP, Ki=KI, params=PARAMS, pendulum=ROD)

    experiments = [
        ("PI only\nconstant ref",                       ctrl_pi,         const_ref),
        ("PI only\ntrajectory",                         ctrl_pi,         traj_ref),
        ("PI + grav FFW\nconstant ref",                 ctrl_grav,       const_ref),
        ("PI + grav FFW\ntrajectory",                   ctrl_grav,       traj_ref),
        ("PI + poly FFW\nconstant ref",                 ctrl_poly,       const_ref),
        ("PI + poly FFW\ntrajectory",                   ctrl_poly,       traj_ref),
        ("PI + poly FFW + grav FFW\nconstant ref",      ctrl_poly_grav,  const_ref),
        ("PI + poly FFW + grav FFW\ntrajectory",        ctrl_poly_grav,  traj_ref),
        ("PI + full model FFW + grav FFW\nconstant ref", ctrl_model_grav, const_ref),
        ("PI + full model FFW + grav FFW\ntrajectory",   ctrl_model_grav, traj_ref),
        ("PI + NN FFW\nconstant ref",                    ctrl_nn,         const_ref),
        ("PI + NN FFW\ntrajectory",                      ctrl_nn,         traj_ref),
    ]

    results = []
    for title, ctrl, ref_fn in experiments:
        print(f"Running: {title.replace(chr(10), ' ')} ...", end="  ", flush=True)
        data = _simulate(ctrl, ref_fn)
        results.append((title, data))
        err = data["omega"] - data["ref"]
        print(f"RMSE = {np.sqrt(np.mean(err**2)):.4f} rad/s")

    return results


# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------

def plot_results(results: list):
    n = len(results)
    fig, axes = plt.subplots(n // 2, 2, figsize=(14, n // 2 * 4), sharex=True)
    axes = axes.flatten()

    colors = {"omega": "C0", "ref": "C3", "voltage": "C2"}

    for ax, (title, data) in zip(axes, results):
        t = data["t"]
        ax2 = ax.twinx()

        ax2.plot(t, data["voltage"], color=colors["voltage"],
                 lw=0.8, alpha=0.45, label="voltage (V)")
        ax2.set_ylabel("Voltage (V)", color=colors["voltage"], fontsize=8)
        ax2.tick_params(axis="y", labelcolor=colors["voltage"], labelsize=7)
        ax2.set_ylim(-PARAMS.V_max * 1.2, PARAMS.V_max * 1.2)

        ax.plot(t, data["ref"],   color=colors["ref"],   lw=1.2,
                ls="--", label="ref")
        ax.plot(t, data["omega"], color=colors["omega"], lw=1.2,
                label="omega true")

        err = data["omega"] - data["ref"]
        rmse = np.sqrt(np.mean(err**2))
        ax.set_title(f"{title}\nRMSE = {rmse:.3f} rad/s", fontsize=9)
        ax.set_ylabel("Speed (rad/s)", fontsize=8)
        ax.grid(True, lw=0.4)
        ax.axhline(0, color="gray", lw=0.5, ls=":")

        if ax in axes[-2:]:
            ax.set_xlabel("Time (s)", fontsize=8)

        lines1, labels1 = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax.legend(lines1 + lines2, labels1 + labels2,
                  loc="lower right", fontsize=7)

    fig.suptitle("BDC Motor + Rod  —  Speed Control Experiments", fontsize=13)
    plt.tight_layout()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / "experiments.png"
    plt.savefig(out_path, dpi=150)
    print(f"\nFigure saved: {out_path}")
    plt.show()


# ---------------------------------------------------------------------------

if __name__ == "__main__":
    results = run_experiments()
    plot_results(results)
