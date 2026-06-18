"""
Quintic polynomial trajectory planner.

Generates a smooth motion profile between two positions with zero velocity
and acceleration at both endpoints (default), satisfying 6 boundary conditions.

Usage:
    traj = QuinticTrajectory(q0=0, qf=10, t0=0.5, tf=3.5)
    omega_ref  = traj.position(t)   # scalar or array
    domega_ref = traj.velocity(t)
"""

from __future__ import annotations
import numpy as np


class QuinticTrajectory:
    """
    Quintic polynomial  q(τ) = Σ cₙ τⁿ,  τ = t − t₀, clamped outside [t₀, t_f].

    Boundary conditions (6 equations → 6 coefficients):
        q(t₀)  = q₀,  q(t_f)  = q_f
        q'(t₀) = v₀,  q'(t_f) = v_f     (default 0)
        q''(t₀)= a₀,  q''(t_f)= a_f     (default 0)
    """

    def __init__(
        self,
        q0: float,
        qf: float,
        t0: float,
        tf: float,
        v0: float = 0.0,
        vf: float = 0.0,
        a0: float = 0.0,
        af: float = 0.0,
    ):
        if tf <= t0:
            raise ValueError("tf must be strictly greater than t0")

        self.t0 = float(t0)
        self.tf = float(tf)
        self.q0 = float(q0)
        self.qf = float(qf)

        T = tf - t0
        A = np.array(
            [
                [1, 0,   0,       0,        0,         0      ],
                [0, 1,   0,       0,        0,         0      ],
                [0, 0,   2,       0,        0,         0      ],
                [1, T,   T**2,    T**3,     T**4,      T**5   ],
                [0, 1,   2*T,     3*T**2,   4*T**3,    5*T**4 ],
                [0, 0,   2,       6*T,      12*T**2,   20*T**3],
            ],
            dtype=float,
        )
        b = np.array([q0, v0, a0, qf, vf, af], dtype=float)
        self._c = np.linalg.solve(A, b)
        self._T = T

    def position(self, t: float | np.ndarray) -> np.ndarray:
        """Trajectory position; clamped to [q0, qf] boundary outside [t0, tf]."""
        tau = np.clip(np.asarray(t, dtype=float) - self.t0, 0.0, self._T)
        c = self._c
        return (
            c[0]
            + c[1] * tau
            + c[2] * tau**2
            + c[3] * tau**3
            + c[4] * tau**4
            + c[5] * tau**5
        )

    def velocity(self, t: float | np.ndarray) -> np.ndarray:
        """Trajectory velocity; zero outside [t0, tf] (clamped τ derivative = 0)."""
        t_arr = np.asarray(t, dtype=float)
        tau = np.clip(t_arr - self.t0, 0.0, self._T)
        inside = (t_arr >= self.t0) & (t_arr <= self.tf)
        c = self._c
        vel = (
            c[1]
            + 2 * c[2] * tau
            + 3 * c[3] * tau**2
            + 4 * c[4] * tau**3
            + 5 * c[5] * tau**4
        )
        return np.where(inside, vel, 0.0)
