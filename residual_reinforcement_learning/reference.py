"""Randomised and fixed speed-reference generators for residual RL."""
from __future__ import annotations
from typing import Callable
import numpy as np

from utils.traj import QuinticTrajectory


# ---------------------------------------------------------------------------
# Randomised training reference (constant | quintic ramp | multi-step)
# ---------------------------------------------------------------------------
class RandomReference:
    """
    Samples one reference profile (constant | quintic ramp | multi-step) at
    construction time; __call__(t) returns omega_ref [rad/s] for a scalar t.
    Deterministic given the rng.
    """

    def __init__(self, rng: np.random.Generator, t_end: float,
                 omega_range: tuple[float, float] = (-12.0, 12.0)):
        self.t_end = float(t_end)
        self.lo, self.hi = omega_range
        kind = int(rng.integers(0, 3))
        if kind == 0:                                  # constant
            c = float(rng.uniform(self.lo, self.hi))
            self._fn = lambda t, c=c: c
        elif kind == 1:                                # quintic ramp q0 -> qf
            q0 = float(rng.uniform(self.lo, self.hi))
            qf = float(rng.uniform(self.lo, self.hi))
            t0 = float(rng.uniform(0.0, 0.3 * t_end))
            tf = float(rng.uniform(0.5 * t_end, 0.9 * t_end))
            traj = QuinticTrajectory(q0, qf, t0, tf)
            self._fn = lambda t, traj=traj: float(traj.position(t))
        else:                                          # piecewise-constant steps
            n = int(rng.integers(2, 5))
            edges = np.sort(rng.uniform(0.0, t_end, n - 1))
            levels = rng.uniform(self.lo, self.hi, n)

            def _fn(t, edges=edges, levels=levels):
                return float(levels[int(np.searchsorted(edges, t))])

            self._fn = _fn

    def __call__(self, t: float) -> float:
        return float(np.clip(self._fn(t), self.lo, self.hi))


# ---------------------------------------------------------------------------
# Fixed references for deterministic benchmarking / held-out evaluation
# ---------------------------------------------------------------------------
def constant_reference(value: float) -> Callable[[float], float]:
    return lambda t, v=float(value): v


def quintic_reference(q0: float, qf: float, t0: float, tf: float) -> Callable[[float], float]:
    traj = QuinticTrajectory(q0, qf, t0, tf)
    return lambda t, traj=traj: float(traj.position(t))
