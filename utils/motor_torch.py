"""
Differentiable BDC motor + pendulum dynamics in PyTorch.

Mirrors BDCMotor from motor.py but uses torch tensors so gradients
flow through the simulation for backprop-through-time training.

Supports batched rollouts: state shape (B, 3) = [i, omega, theta].
"""

from __future__ import annotations
import torch
import torch.nn as nn
from utils.motor import BDCMotorParams, PendulumParams


class MotorDynamicsTorch:
    """
    Differentiable BDC motor + optional pendulum.

    All parameters stored as plain Python floats (no grad needed).
    Integration is Euler (fast, less memory) for training;
    call step_rk4 for higher accuracy.
    """

    def __init__(
        self,
        params:   BDCMotorParams,
        pendulum: PendulumParams | None = None,
        device:   str = "cpu",
    ):
        p = params
        self.R     = p.R
        self.L     = p.L
        self.Kt    = p.Kt
        self.Kb    = p.Kb
        self.B     = p.B
        self.V_max = p.V_max
        self.J_eff = p.J + (pendulum.I_rod if pendulum else 0.0)
        self.T_grav_coeff = (
            pendulum.m * pendulum.g * pendulum.l_cm if pendulum else 0.0
        )
        self.device = device

    def derivatives(self, x: torch.Tensor, voltage: torch.Tensor) -> torch.Tensor:
        """
        x       : (..., 3)  [i, omega, theta]
        voltage : (...,)    applied voltage
        returns : (..., 3)  dx/dt
        """
        i     = x[..., 0]
        omega = x[..., 1]
        theta = x[..., 2]

        T_grav = self.T_grav_coeff * torch.sin(theta)

        di_dt     = (voltage - self.R * i - self.Kb * omega) / self.L
        domega_dt = (self.Kt * i - self.B * omega - T_grav) / self.J_eff
        dtheta_dt = omega

        return torch.stack([di_dt, domega_dt, dtheta_dt], dim=-1)

    def step_euler(self, x: torch.Tensor, voltage: torch.Tensor, dt: float) -> torch.Tensor:
        return x + dt * self.derivatives(x, voltage)

    def step_rk4(self, x: torch.Tensor, voltage: torch.Tensor, dt: float) -> torch.Tensor:
        k1 = self.derivatives(x,                voltage)
        k2 = self.derivatives(x + dt / 2 * k1, voltage)
        k3 = self.derivatives(x + dt / 2 * k2, voltage)
        k4 = self.derivatives(x + dt * k3,      voltage)
        return x + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)


class NNSpeedController(nn.Module):
    """
    MLP speed controller.

    Inputs (6, pre-normalized to ~[-1, 1]):
        e / OMEGA_MAX          — speed error
        delta_e / DELTA_E_MAX  — one-step change in error (not /dt, avoids 1000x blow-up)
        integral_e / OMEGA_MAX — clamped integral
        omega_target / OMEGA_MAX
        sin(theta)             — gravity component, bounded [-1,1]
        cos(theta)             — needed to disambiguate theta

    Output (1): voltage V via tanh * V_max (smooth saturation).
    """

    OMEGA_MAX   = 15.0   # rad/s   — normalisation constant
    DELTA_E_MAX = 1.0    # rad/s   — max expected |Δe| per step

    def __init__(self, V_max: float = 12.0, hidden: int = 64):
        super().__init__()
        self.V_max = V_max
        self.net = nn.Sequential(
            nn.Linear(6, hidden),
            nn.Tanh(),
            nn.Linear(hidden, hidden),
            nn.Tanh(),
            nn.Linear(hidden, 1),
        )
        nn.init.uniform_(self.net[-1].weight, -0.01, 0.01)
        nn.init.zeros_(self.net[-1].bias)

    def forward(
        self,
        e:            torch.Tensor,  # (...,)  raw rad/s
        delta_e:      torch.Tensor,  # (...,)  e[k] - e[k-1], raw rad/s
        integral_e:   torch.Tensor,  # (...,)  clamped integral
        omega_target: torch.Tensor,  # (...,)  raw rad/s
        theta:        torch.Tensor,  # (...,)  raw rad — unbounded, use sin/cos
    ) -> torch.Tensor:
        inp = torch.stack([
            e            / self.OMEGA_MAX,
            delta_e      / self.DELTA_E_MAX,
            integral_e   / self.OMEGA_MAX,
            omega_target / self.OMEGA_MAX,
            torch.sin(theta),
            torch.cos(theta),
        ], dim=-1)
        raw = self.net(inp).squeeze(-1)
        return torch.tanh(raw) * self.V_max
