"""
PI speed controller with neural-network feedforward.

The feedforward lookup comes from the inverse steady-state map  ω → V
trained by nn/nn_fit.py and stored in nn/data/nn_weights.npz.
Inference is pure numpy — no PyTorch required at runtime.

Usage:
    ctrl = NNFeedforwardController.from_file(Kp=5.0, Ki=2.0, params=PARAMS)
    voltage = ctrl.step(omega_meas, omega_target, theta=0.0, dt=1e-3)
"""

from __future__ import annotations
from pathlib import Path

import numpy as np
from utils.motor import BDCMotorParams, PendulumParams

_DEFAULT_WEIGHTS = Path(__file__).parent / "data" / "nn_weights.npz"


def _tanh(x: np.ndarray) -> np.ndarray:
    return np.tanh(x)


def _nn_forward(data: dict, omega: float) -> float:
    """Pure-numpy forward pass matching the 1→16→16→1 Tanh architecture."""
    h = np.array([omega], dtype=np.float32)
    # layer indices that have weights: 0 (Linear), 2 (Linear), 4 (Linear, no bias)
    for i in [0, 2, 4]:
        W = data[f"W{i}"]          # (out, in)
        h = W @ h
        if f"b{i}" in data:
            h = h + data[f"b{i}"]
        if i < 4:                   # hidden layers use Tanh; output is linear
            h = _tanh(h)
    return float(h[0])


class NNFeedforwardController:
    """
    PI controller whose feedforward is a trained MLP  V_ff = f(omega_target).

    Parameters
    ----------
    Kp, Ki      : proportional / integral gains
    params      : motor params (V_max, R, Kt used)
    pendulum    : pendulum params (m, g, l_cm for gravity FFW)
    weights     : dict loaded from nn_weights.npz
    omega_min, omega_max : clamp range for omega_target before NN evaluation
    """

    def __init__(
        self,
        Kp: float,
        Ki: float,
        params: BDCMotorParams,
        pendulum: PendulumParams,
        weights: dict,
        omega_min: float,
        omega_max: float,
    ):
        self.Kp = Kp
        self.Ki = Ki
        self._p   = params
        self._rod = pendulum
        self._weights   = weights
        self._omega_min = float(omega_min)
        self._omega_max = float(omega_max)
        self._integral  = 0.0

    @classmethod
    def from_file(
        cls,
        Kp: float,
        Ki: float,
        params: BDCMotorParams,
        pendulum: PendulumParams,
        path: str | Path = _DEFAULT_WEIGHTS,
    ) -> "NNFeedforwardController":
        raw = np.load(path)
        weights = {k: raw[k] for k in raw.files
                   if k not in ("omega_min", "omega_max")}
        return cls(
            Kp=Kp,
            Ki=Ki,
            params=params,
            pendulum=pendulum,
            weights=weights,
            omega_min=float(raw["omega_min"]),
            omega_max=float(raw["omega_max"]),
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
        p, rod, V_max = self._p, self._rod, self._p.V_max

        omega_clamped = float(np.clip(omega_target, self._omega_min, self._omega_max))
        V_ff_nn   = float(np.clip(_nn_forward(self._weights, omega_clamped), -V_max, V_max))
        V_ff_grav = rod.m * rod.g * rod.l_cm * np.sin(theta) * p.R / p.Kt

        e = omega_target - omega_meas
        self._integral += e * dt
        if self.Ki != 0.0:
            self._integral = np.clip(
                self._integral, -V_max / self.Ki, V_max / self.Ki
            )

        V = V_ff_nn + V_ff_grav + self.Kp * e + self.Ki * self._integral
        return float(np.clip(V, -V_max, V_max))

    @property
    def integral(self) -> float:
        return self._integral
