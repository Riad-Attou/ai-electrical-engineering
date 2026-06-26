"""
Motor input-voltage comparison: PI vs PI+poly+grav vs residual-PI, plus the
isolated RL contribution ΔV.

Runs the three controllers on the same speed reference and plots:
  - top panel    : the total commanded voltage V(t) for each controller, so the
                   different effort profiles are directly comparable;
  - bottom panel : the residual-PI decomposition  V = clip(V_base + ΔV)  with
                   V_base (the PI baseline) and ΔV (the learned correction)
                   drawn separately, so the RL contribution is visible on its own.

Uses the same plant and controllers as benchmark.py. The residual split is read
from the ResidualController state after each step (V_base and the held ΔV).

Run:
    python residual_reinforcement_learning/voltage_plot.py
    python residual_reinforcement_learning/voltage_plot.py --ref "const 10"
"""
import sys
import argparse
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

from utils.motor import BDCMotor, SpeedSensorNoise
from benchmark import PARAMS, ROD, DT, T_END, RESULTS_DIR, build_controllers
from residual_controller import ResidualController
from reference import constant_reference, quintic_reference

REFERENCES = {
    "const 10":      constant_reference(10.0),
    "quintic 0->10": quintic_reference(0.0, 10.0, 0.5, 3.5),
}


# ---------------------------------------------------------------------------
# Rollout: record total voltage, plus the V_base / ΔV split for residual ctrls
# ---------------------------------------------------------------------------
def _rollout(ctrl, ref_fn) -> dict:
    motor = BDCMotor(PARAMS, pendulum=ROD)
    motor.reset(theta0=0.0)
    if hasattr(ctrl, "reset"):
        ctrl.reset()
    noise = SpeedSensorNoise(std=0.0, quantization=0.0)
    steps = int(T_END / DT)
    t = np.empty(steps)
    V = np.empty(steps)                 # total applied voltage
    V_base = np.full(steps, np.nan)     # baseline (PI) part      — residual only
    dV = np.full(steps, np.nan)         # ΔV (RL) part, held @100 Hz — residual only
    is_residual = isinstance(ctrl, ResidualController)

    omega_meas = 0.0
    for k in range(steps):
        tk = k * DT
        omega_target = float(ref_fn(tk))
        v = ctrl.step(omega_meas, omega_target, motor.state.theta, DT)
        state = motor.step(DT, v)
        omega_meas = noise.measure(state.omega)
        t[k] = tk
        V[k] = v
        if is_residual:
            V_base[k] = ctrl._V_base    # set by ResidualController.step()
            dV[k] = ctrl._dv
    return dict(t=t, V=V, V_base=V_base, dV=dV)


# ---------------------------------------------------------------------------
# Figure: total-voltage comparison (top) + residual decomposition (bottom)
# ---------------------------------------------------------------------------
def plot_voltage_contributions(ref_name: str = "quintic 0->10",
                               out: pathlib.Path | None = None) -> pathlib.Path:
    if ref_name not in REFERENCES:
        raise ValueError(f"unknown reference {ref_name!r}; choose from {list(REFERENCES)}")
    ref_fn = REFERENCES[ref_name]
    ctrls = build_controllers()
    data = {name: _rollout(ctrl, ref_fn) for name, ctrl in ctrls.items()}

    V_max = PARAMS.V_max
    colors = {"PI": "C0", "PI+poly+grav": "C2", "residual-PI": "C3"}

    fig, (ax_tot, ax_dec) = plt.subplots(2, 1, figsize=(9, 7), sharex=True)

    # --- top: total motor input voltage for every controller ---------------
    for name, d in data.items():
        ax_tot.plot(d["t"], d["V"], color=colors.get(name), lw=1.3, label=name)
    # focus the y-axis on the actual signals (they stay far from the ±V_max rails)
    vmin = min(float(np.min(d["V"])) for d in data.values())
    vmax = max(float(np.max(d["V"])) for d in data.values())
    pad = 0.15 * (vmax - vmin + 1e-6)
    ax_tot.set_ylim(vmin - pad, vmax + pad)
    ax_tot.text(0.99, 0.04,
                f"|V| stays well below the ±{V_max:.0f} V rails (no saturation)",
                transform=ax_tot.transAxes, ha="right", va="bottom",
                fontsize=7, color="gray")
    ax_tot.set_ylabel("motor input voltage V (V)")
    ax_tot.set_title(f"Total motor input voltage — reference: {ref_name}")
    ax_tot.legend(fontsize=8, ncol=2, loc="best")
    ax_tot.grid(True, lw=0.3)

    # --- bottom: residual-PI split  V = clip(V_base + ΔV) ------------------
    if "residual-PI" in data:
        d = data["residual-PI"]
        ax_dec.plot(d["t"], d["V_base"], color="C0", lw=1.1, label="V_base  (PI baseline)")
        ax_dec.plot(d["t"], d["V"], color="C3", lw=1.4,
                    label="V applied = clip(V_base + ΔV)")
        ax_dec.plot(d["t"], d["dV"], color="C1", lw=1.4, label="ΔV  (RL contribution)")
        ax_dec.fill_between(d["t"], 0.0, d["dV"], color="C1", alpha=0.25)
        ax_dec.axhline(0.0, color="gray", lw=0.6)
        dv_lim = 0.3 * V_max  # ΔV is bounded to ±0.3·V_max by construction
        ax_dec.axhline(dv_lim, color="C1", ls=":", lw=0.7)
        ax_dec.axhline(-dv_lim, color="C1", ls=":", lw=0.7, label="±ΔV limit")
        ax_dec.set_title("residual-PI decomposition:  V = V_base (PI) + ΔV (RL)")
    else:
        ax_dec.text(0.5, 0.5, "residual-PI not available (no trained model)",
                    ha="center", va="center", transform=ax_dec.transAxes)
    ax_dec.set_ylabel("voltage (V)")
    ax_dec.set_xlabel("time (s)")
    ax_dec.legend(fontsize=8, ncol=2, loc="best")
    ax_dec.grid(True, lw=0.3)

    fig.tight_layout()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out = out or (RESULTS_DIR / "voltage_contributions.png")
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"figure -> {out}")
    return out


# ---------------------------------------------------------------------------
# Command-line entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref", default="quintic 0->10", choices=list(REFERENCES),
                    help="speed reference to drive the comparison")
    args = ap.parse_args()
    plot_voltage_contributions(args.ref)
