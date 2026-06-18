"""
PI speed controller with data-driven polynomial feedforward.

The feedforward lookup comes from the inverse steady-state map  w -> V
fitted by poly/poly_fit.py and stored in poly/data/poly_coeffs.npz.

Usage:
    ctrl = PolyFeedforwardController.from_file(Kp=5.0, Ki=2.0, params=PARAMS)
    voltage = ctrl.step(omega_meas, omega_target, theta=0.0, dt=1e-3)
"""

from __future__ import annotations
from pathlib import Path

import numpy as np
from utils.motor import BDCMotorParams

_DEFAULT_COEFFS = Path(__file__).parent / "data" / "poly_coeffs.npz"


class PolyFeedforwardController:
    """
    PI controller whose feedforward is a zero-bias polynomial  V_ff = f(omega_target).

    Parameters
    ----------
    Kp, Ki    : proportional / integral gains
    params    : motor params (only V_max is used for clipping)
    coeffs    : np.polyval-compatible coefficient array (highest power first)
    omega_min, omega_max : clamp range for omega_target before evaluating the polynomial
    """

    def __init__(
        self,
        Kp: float,
        Ki: float,
        params: BDCMotorParams,
        coeffs: np.ndarray,
        omega_min: float,
        omega_max: float,
    ):
        self.Kp = Kp
        self.Ki = Ki
        self._p = params
        self._coeffs = np.asarray(coeffs, dtype=float)
        self._omega_min = float(omega_min)
        self._omega_max = float(omega_max)
        self._integral = 0.0

    @classmethod
    def from_file(
        cls,
        Kp: float,
        Ki: float,
        params: BDCMotorParams,
        path: str | Path = _DEFAULT_COEFFS,
    ) -> "PolyFeedforwardController":
        d = np.load(path)
        return cls(
            Kp=Kp,
            Ki=Ki,
            params=params,
            coeffs=d["coeffs"],
            omega_min=float(d["omega_min"]),
            omega_max=float(d["omega_max"]),
        )

    def reset(self) -> None:
        self._integral = 0.0

    def step(
        self,
        omega_meas: float,
        omega_target: float,
        theta: float,
        dt: float,
    ) -> float:
        V_max = self._p.V_max

        omega_clamped = np.clip(omega_target, self._omega_min, self._omega_max)
        V_ff = float(np.clip(np.polyval(self._coeffs, omega_clamped), -V_max, V_max))

        e = omega_target - omega_meas
        self._integral += e * dt
        if self.Ki != 0.0:
            self._integral = np.clip(
                self._integral, -V_max / self.Ki, V_max / self.Ki
            )

        V = V_ff + self.Kp * e + self.Ki * self._integral
        return float(np.clip(V, -V_max, V_max))

    @property
    def integral(self) -> float:
        return self._integral
