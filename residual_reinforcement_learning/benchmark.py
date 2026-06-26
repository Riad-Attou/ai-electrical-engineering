"""Deterministic RMSE benchmark: classical baselines vs residual RL controllers."""
import sys
import pathlib

ROOT = pathlib.Path(__file__).resolve().parent.parent
HERE = pathlib.Path(__file__).resolve().parent
for _p in (str(HERE), str(ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from utils.motor import BDCMotor, BDCMotorParams, PendulumParams, SpeedSensorNoise
from baselines import PurePI, PolyGravPI, load_poly_coeffs
from residual_controller import ResidualController
from reference import constant_reference, quintic_reference

# ---------------------------------------------------------------------------
# Plant, noise model, timing and PI gains (shared across the RL package)
# ---------------------------------------------------------------------------
PARAMS = BDCMotorParams(R=3.0, L=4e-3, Kt=0.05, Kb=0.05, J=7.4e-5, B=0.005, V_max=12.0)
ROD = PendulumParams(m=0.05, l=0.1, g=9.81)
NOISE = SpeedSensorNoise(std=0.0, quantization=0.0)
DT, T_END, KP, KI = 1e-3, 5.0, 5.0, 2.0
MODELS_DIR = HERE / "models"
RESULTS_DIR = HERE / "results"


def rmse(omega, ref) -> float:
    e = np.asarray(omega) - np.asarray(ref)
    return float(np.sqrt(np.mean(e ** 2)))


def simulate(ctrl, ref_fn) -> dict:
    motor = BDCMotor(PARAMS, pendulum=ROD)
    motor.reset(theta0=0.0)
    if hasattr(ctrl, "reset"):
        ctrl.reset()
    steps = int(T_END / DT)
    omega_arr = np.empty(steps); ref_arr = np.empty(steps)
    volt_arr = np.empty(steps); t_arr = np.empty(steps)
    omega_meas = 0.0
    for k in range(steps):
        t = k * DT
        omega_target = float(ref_fn(t))
        V = ctrl.step(omega_meas, omega_target, motor.state.theta, DT)
        state = motor.step(DT, V)
        omega_meas = NOISE.measure(state.omega)
        omega_arr[k] = state.omega; ref_arr[k] = omega_target
        volt_arr[k] = V; t_arr[k] = t
    return dict(t=t_arr, omega=omega_arr, ref=ref_arr, voltage=volt_arr)


def _load_policy(variant: str):
    try:
        from stable_baselines3 import SAC
    except Exception:
        return None
    path = MODELS_DIR / f"sac_residual_{variant}.zip"
    return SAC.load(path) if path.exists() else None


# ---------------------------------------------------------------------------
# Controller set under test: classical baselines + residual-PI (if trained)
# ---------------------------------------------------------------------------
def build_controllers() -> dict:
    coeffs, omin, omax = load_poly_coeffs()
    ctrls = {
        "PI": PurePI(KP, KI, PARAMS.V_max),
        "PI+poly+grav": PolyGravPI(KP, KI, PARAMS, ROD, coeffs, omin, omax),
    }
    pol_pi = _load_policy("pi")
    if pol_pi is not None:
        ctrls["residual-PI"] = ResidualController(
            PurePI(KP, KI, PARAMS.V_max), pol_pi, PARAMS)
    return ctrls


def run() -> dict:
    refs = {"const 10": constant_reference(10.0),
            "quintic 0->10": quintic_reference(0.0, 10.0, 0.5, 3.5)}
    ctrls = build_controllers()
    rows, data = {}, {}
    for cname, ctrl in ctrls.items():
        for rname, ref_fn in refs.items():
            d = simulate(ctrl, ref_fn)
            rows[(cname, rname)] = rmse(d["omega"], d["ref"])
            data[(cname, rname)] = d
    _print_table(ctrls, refs, rows)
    _save_latex(ctrls, refs, rows)
    _save_figure(ctrls, refs, data)
    return rows


def _print_table(ctrls, refs, rows) -> None:
    print("\nRMSE (rad/s)")
    print("  ".join(f"{h:>16}" for h in ["controller", *refs]))
    for c in ctrls:
        print("  ".join([f"{c:>16}", *[f"{rows[(c, r)]:>16.4f}" for r in refs]]))
    if "residual-PI" not in ctrls:
        print("\n[note] residual rows skipped: run train.py and install stable-baselines3")


def _save_latex(ctrls, refs, rows) -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    lines = []
    for c in ctrls:
        cells = " & ".join(f"{rows[(c, r)]:.3f}" for r in refs)
        lines.append(f"{c} & {cells} \\\\")
    (RESULTS_DIR / "rmse_table.tex").write_text("\n".join(lines) + "\n")


def _save_figure(ctrls, refs, data) -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(len(ctrls), len(refs),
                             figsize=(12, 3 * len(ctrls)), squeeze=False)
    for i, c in enumerate(ctrls):
        for j, r in enumerate(refs):
            ax, d = axes[i][j], data[(c, r)]
            ax.plot(d["t"], d["ref"], "C3--", lw=1)
            ax.plot(d["t"], d["omega"], "C0", lw=1)
            ax.set_title(f"{c} | {r}  RMSE={rmse(d['omega'], d['ref']):.3f}", fontsize=8)
            ax.grid(True, lw=0.3)
    fig.tight_layout()
    out = RESULTS_DIR / "benchmark.png"
    fig.savefig(out, dpi=150)
    print(f"figure -> {out}")


# ---------------------------------------------------------------------------
# Command-line entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    run()
