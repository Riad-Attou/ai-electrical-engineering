"""Drop-in residual controller: V = clip(V_base + ΔV, ±V_max), ΔV from a policy."""
from __future__ import annotations
import numpy as np

from utils.motor import BDCMotorParams


# ---------------------------------------------------------------------------
# Residual controller: classical baseline + bounded learned correction ΔV
# ---------------------------------------------------------------------------
class ResidualController:
    """
    Wraps a baseline controller + a trained policy into the standard
    .step(omega_meas, omega_target, theta, dt) -> float interface.

    The baseline runs every call (1 kHz); the policy is queried once per
    `control_decimation` calls and ΔV is held constant in between, matching the
    training-time control rate.

    Set `saturation_aware=True` to match a policy trained with the env's
    saturation-aware observation (adds the previous applied voltage as a 7th
    feature). `policy` is any object exposing
        .predict(obs, deterministic=True) -> (action_array, state).
    """

    def __init__(self, baseline, policy, params: BDCMotorParams,
                 dv_limit_frac: float = 0.3, control_decimation: int = 10,
                 e_scale: float = 5.0, omega_scale: float = 15.0,
                 saturation_aware: bool = False):
        self.baseline = baseline
        self.policy = policy
        self._p = params
        self.dv_limit = float(dv_limit_frac) * params.V_max
        self.decimation = int(control_decimation)
        self.e_scale = float(e_scale)
        self.omega_scale = float(omega_scale)
        self.saturation_aware = bool(saturation_aware)
        self._counter = 0
        self._dv = 0.0
        self._V_base = 0.0
        self._V_applied_prev = 0.0

    def reset(self) -> None:
        if hasattr(self.baseline, "reset"):
            self.baseline.reset()
        self._counter = 0
        self._dv = 0.0
        self._V_base = 0.0
        self._V_applied_prev = 0.0

    def _obs(self, omega_meas, omega_target, theta) -> np.ndarray:
        e = omega_target - omega_meas
        V_max = self._p.V_max
        feats = [
            e / self.e_scale,
            omega_meas / self.omega_scale,
            np.sin(theta),
            np.cos(theta),
            omega_target / self.omega_scale,
            self._V_base / V_max,
        ]
        if self.saturation_aware:
            feats.append(self._V_applied_prev / V_max)
        return np.array(feats, dtype=np.float32)

    def step(self, omega_meas: float, omega_target: float,
             theta: float, dt: float) -> float:
        V_max = self._p.V_max
        self._V_base = self.baseline.step(omega_meas, omega_target, theta, dt)
        if self._counter % self.decimation == 0:
            obs = self._obs(omega_meas, omega_target, theta)
            action, _ = self.policy.predict(obs, deterministic=True)
            a = float(np.clip(np.asarray(action).reshape(-1)[0], -1.0, 1.0))
            self._dv = a * self.dv_limit
        self._counter += 1
        V = float(np.clip(self._V_base + self._dv, -V_max, V_max))
        self._V_applied_prev = V
        return V
