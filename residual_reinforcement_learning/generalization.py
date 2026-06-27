"""
Held-out generalisation check.

Evaluates the already-trained policies on references that differ from the two
benchmark references used to report the results -- including negative
setpoints, multi-step profiles, and one setpoint *beyond* the training range
[-12, 12] (extrapolation). This probes for overfitting: a genuine improvement
should hold across these unseen references.
"""
import sys
import pathlib

ROOT = pathlib.Path(__file__).resolve().parent.parent
HERE = pathlib.Path(__file__).resolve().parent
for _p in (str(HERE), str(ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import numpy as np
from benchmark import simulate, rmse, build_controllers
from reference import constant_reference, quintic_reference


def step_reference(levels, edges):
    levels = list(map(float, levels))
    edges = list(map(float, edges))

    def f(t):
        return levels[int(np.searchsorted(edges, t))]

    return f


# ---------------------------------------------------------------------------
# Held-out reference set (unseen during training; one extrapolation point)
# ---------------------------------------------------------------------------
HELDOUT = {
    "const 6":           constant_reference(6.0),
    "const -5":          constant_reference(-5.0),
    "const 12":          constant_reference(12.0),
    "quintic 2->9":      quintic_reference(2.0, 9.0, 1.0, 4.0),
    "quintic 0->-8":     quintic_reference(0.0, -8.0, 0.5, 4.0),
    "step 8/3":          step_reference([8.0, 3.0], [2.5]),
    "const 14 (extrap)": constant_reference(14.0),
}


# ---------------------------------------------------------------------------
# Command-line entry point
# ---------------------------------------------------------------------------
def run():
    ctrls = build_controllers()
    names = list(ctrls)
    print("\nHeld-out RMSE (rad/s)")
    print("  ".join(f"{h:>18}" for h in ["reference", *names]))
    totals = {n: 0.0 for n in names}
    for rname, ref_fn in HELDOUT.items():
        cells = []
        for n in names:
            d = simulate(ctrls[n], ref_fn)
            val = rmse(d["omega"], d["ref"])
            totals[n] += val
            cells.append(f"{val:>18.4f}")
        print("  ".join([f"{rname:>18}", *cells]))
    print("  ".join([f"{'MEAN':>18}",
                     *[f"{totals[n] / len(HELDOUT):>18.4f}" for n in names]]))


if __name__ == "__main__":
    run()
