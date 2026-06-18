"""
PI controller with steady-state gravity compensation for BDC motor + vertical rod.

Gravity feedforward (steady-state only, ignores inertia):
    V_ff = m*g*l_cm*sin(theta) * R / Kt

Full control law:
    e        = omega_target - omega_measured
    integral += e * dt
    V        = V_ff + Kp*e + Ki*integral
    V        = clip(V, -V_max, V_max)

Anti-windup: integral is clamped so that Ki*integral alone never exceeds V_max.
"""

from __future__ import annotations
import numpy as np
from utils.motor import BDCMotorParams, PendulumParams


class PIGravityController:
    """
    PI speed controller with steady-state gravity compensation.

    Parameters
    ----------
    Kp, Ki    : proportional and integral gains
    params    : motor electrical parameters (for R, Kt, V_max)
    pendulum  : rod parameters (for m, g, l_cm used in feedforward)
    """

    def __init__(
        self,
        Kp:       float,
        Ki:       float,
        params:   BDCMotorParams,
        pendulum: PendulumParams,
    ):
        self.Kp       = Kp
        self.Ki       = Ki
        self.p        = params
        self.rod      = pendulum
        self._integral = 0.0

    def reset(self):
        self._integral = 0.0

    def step(
        self,
        omega_meas:   float,   # measured (possibly noisy) speed [rad/s]
        omega_target: float,   # speed setpoint                  [rad/s]
        theta:        float,   # current rod angle               [rad]
        dt:           float,   # time step                       [s]
    ) -> float:
        """
        Compute voltage command.

        Returns voltage clipped to [-V_max, V_max].
        """
        p, rod = self.p, self.rod
        V_max  = p.V_max

        # ---- gravity feedforward (steady-state) ----
        V_ff = rod.m * rod.g * rod.l_cm * np.sin(theta) * p.R / p.Kt

        # ---- PI feedback ----
        e = omega_target - omega_meas
        self._integral += e * dt

        # anti-windup: keep Ki*integral within [-V_max, V_max]
        if self.Ki != 0.0:
            self._integral = np.clip(
                self._integral, -V_max / self.Ki, V_max / self.Ki
            )
        Vref_ff = 0.436175*omega_target - -0.002772*omega_target*omega_target
        V = Vref_ff  + V_ff + self.Kp * e + self.Ki * self._integral
        return float(np.clip(V, -V_max, V_max))

    @property
    def integral(self) -> float:
        return self._integral
