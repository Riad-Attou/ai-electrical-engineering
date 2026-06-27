"""
Self-contained classical baseline controllers for the residual RL benchmark.

Shared interface:
    .reset()
    .step(omega_meas, omega_target, theta, dt) -> float    # voltage command [V]

Mirrors the private controllers in the repo-root experiments.py, kept local so
this package stays independent of root files.
"""
from __future__ import annotations
from pathlib import Path
import numpy as np

from utils.motor import BDCMotorParams, PendulumParams

_COEFFS_PATH = Path(__file__).resolve().parent.parent / "poly" / "data" / "poly_coeffs.npz"


# ---------------------------------------------------------------------------
# Fitted inverse-model polynomial (shared steady-state feedforward term)
# ---------------------------------------------------------------------------
def load_poly_coeffs(path: Path = _COEFFS_PATH):
    """Return (coeffs, omega_min, omega_max) from the fitted inverse polynomial."""
    d = np.load(path)
    return d["coeffs"], float(d["omega_min"]), float(d["omega_max"])


# ---------------------------------------------------------------------------
# Pure PI — feedback only (the weak baseline the residual policy assists)
# ---------------------------------------------------------------------------
class PurePI:
    """Proportional-Integral speed controller, no feedforward."""

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
            self._integral = float(np.clip(
                self._integral, -self._V_max / self.Ki, self._V_max / self.Ki))
        V = self.Kp * e + self.Ki * self._integral
        return float(np.clip(V, -self._V_max, self._V_max))


# ---------------------------------------------------------------------------
# PI + polynomial + gravity feedforward (the strong classical baseline)
# ---------------------------------------------------------------------------
class PolyGravPI:
    """PI + polynomial feedforward + analytical gravity feedforward."""

    def __init__(self, Kp: float, Ki: float,
                 params: BDCMotorParams, pendulum: PendulumParams,
                 coeffs, omega_min: float, omega_max: float):
        self.Kp = Kp
        self.Ki = Ki
        self._p = params
        self._rod = pendulum
        self._coeffs = np.asarray(coeffs, dtype=float)
        self._omega_min = float(omega_min)
        self._omega_max = float(omega_max)
        self._integral = 0.0

    def reset(self) -> None:
        self._integral = 0.0

    def step(self, omega_meas: float, omega_target: float,
             theta: float, dt: float) -> float:
        p, rod, V_max = self._p, self._rod, self._p.V_max
        omega_c = np.clip(omega_target, self._omega_min, self._omega_max)
        V_ff_poly = float(np.clip(np.polyval(self._coeffs, omega_c), -V_max, V_max))
        V_ff_grav = rod.m * rod.g * rod.l_cm * np.sin(theta) * p.R / p.Kt
        e = omega_target - omega_meas
        self._integral += e * dt
        if self.Ki != 0.0:
            self._integral = float(np.clip(
                self._integral, -V_max / self.Ki, V_max / self.Ki))
        V = V_ff_poly + V_ff_grav + self.Kp * e + self.Ki * self._integral
        return float(np.clip(V, -V_max, V_max))
