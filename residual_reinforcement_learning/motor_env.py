"""Gymnasium environment: residual ΔV action on top of a baseline controller."""
from __future__ import annotations
from dataclasses import dataclass
from typing import Callable
import numpy as np
import gymnasium as gym
from gymnasium import spaces

from utils.motor import BDCMotor, BDCMotorParams, PendulumParams, SpeedSensorNoise
from reference import RandomReference


# ---------------------------------------------------------------------------
# Reward weights (see step(): L1 tracking error + effort / rate / saturation)
# ---------------------------------------------------------------------------
@dataclass
class RewardConfig:
    e_scale: float = 5.0      # rad/s, tracking-error normaliser
    alpha: float = 1.0        # tracking-error (absolute) weight
    lam: float = 0.01         # residual-magnitude (effort) penalty weight
    mu: float = 0.05          # residual-rate (smoothness) penalty weight
    sat_penalty: float = 0.0  # clip-excess penalty weight (saturation-aware mode)


# ---------------------------------------------------------------------------
# Gymnasium environment: residual ΔV action on top of a baseline controller
# ---------------------------------------------------------------------------
class MotorResidualEnv(gym.Env):
    """
    Observation (float32): [e/e_scale, omega/omega_scale, sin th, cos th,
                            omega*/omega_scale, V_base/V_max]
        + (saturation_aware) V_applied_prev/V_max  -> a 7th feature.
    Action (float32, 1):    a in [-1, 1]  ->  ΔV = a * dv_limit.
    The motor integrates at dt (1 kHz, RK4); the policy acts once per
    `control_decimation` sim steps and ΔV is held constant in between.

    saturation_aware mode (for operating near the ±V_max rail, e.g. high speeds):
      - the observation gains the previous applied (clipped) voltage, so the
        policy can see when it sits at the rail;
      - the effort/rate penalties use the *effective* (post-clip) residual, and
        an optional penalty (`sat_penalty`) discourages commanding beyond the rail.
    With saturation_aware=False the behaviour is identical to the original env.
    """

    metadata = {"render_modes": []}

    def __init__(self,
                 baseline_factory: Callable[[], object],
                 params: BDCMotorParams | None = None,
                 rod: PendulumParams | None = None,
                 dt: float = 1e-3,
                 t_end: float = 5.0,
                 control_decimation: int = 10,
                 dv_limit_frac: float = 0.3,
                 reward_cfg: RewardConfig | None = None,
                 omega_scale: float = 15.0,
                 omega_range: tuple[float, float] = (-12.0, 12.0),
                 noise_std: float = 1.0,
                 randomize: bool = False,
                 saturation_aware: bool = False):
        super().__init__()
        self._make_baseline = baseline_factory
        self._params0 = params or BDCMotorParams()
        self._rod = rod or PendulumParams()
        self.dt = float(dt)
        self.t_end = float(t_end)
        self.decimation = int(control_decimation)
        self.dv_limit = float(dv_limit_frac) * self._params0.V_max
        self.rcfg = reward_cfg or RewardConfig()
        self.omega_scale = float(omega_scale)
        self.omega_range = omega_range
        self.noise_std = float(noise_std)
        self.randomize = bool(randomize)
        self.saturation_aware = bool(saturation_aware)

        obs_dim = 7 if self.saturation_aware else 6
        self.observation_space = spaces.Box(-10.0, 10.0, shape=(obs_dim,), dtype=np.float32)
        self.action_space = spaces.Box(-1.0, 1.0, shape=(1,), dtype=np.float32)

        self.motor = None
        self.baseline = None
        self.ref = None
        self.noise = None
        self.params = None
        self.t = 0.0
        self.omega_meas = 0.0
        self.V_base = 0.0
        self.prev_dv = 0.0
        self.V_applied_prev = 0.0

    def _sample_params(self) -> BDCMotorParams:
        if not self.randomize:
            return self._params0
        p0 = self._params0
        j = lambda x: x * float(self.np_random.uniform(0.85, 1.15))
        return BDCMotorParams(R=j(p0.R), L=j(p0.L), Kt=j(p0.Kt), Kb=j(p0.Kb),
                              J=j(p0.J), B=j(p0.B), V_max=p0.V_max)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.params = self._sample_params()
        self.motor = BDCMotor(self.params, pendulum=self._rod)
        self.motor.reset(theta0=0.0)
        self.baseline = self._make_baseline()
        if hasattr(self.baseline, "reset"):
            self.baseline.reset()
        self.noise = SpeedSensorNoise(std=self.noise_std, quantization=0.0,
                                      rng=self.np_random)
        self.ref = RandomReference(self.np_random, self.t_end, self.omega_range)
        self.t = 0.0
        self.omega_meas = self.noise.measure(self.motor.state.omega)
        self.V_base = 0.0
        self.prev_dv = 0.0
        self.V_applied_prev = 0.0
        return self._obs(), {}

    def _obs(self) -> np.ndarray:
        omega_target = self.ref(self.t)
        e = omega_target - self.omega_meas
        theta = self.motor.state.theta
        V_max = self.params.V_max
        feats = [
            e / self.rcfg.e_scale,
            self.omega_meas / self.omega_scale,
            np.sin(theta),
            np.cos(theta),
            omega_target / self.omega_scale,
            self.V_base / V_max,
        ]
        if self.saturation_aware:
            feats.append(self.V_applied_prev / V_max)
        return np.clip(np.array(feats, dtype=np.float32), -10.0, 10.0).astype(np.float32)

    def step(self, action):
        a = float(np.clip(np.asarray(action, dtype=np.float32).reshape(-1)[0], -1.0, 1.0))
        dv = a * self.dv_limit
        V_max = self.params.V_max
        err_abs_sum = 0.0
        eff_dv_sum = 0.0
        clip_excess_sq_sum = 0.0
        for _ in range(self.decimation):
            omega_target = self.ref(self.t)
            self.V_base = self.baseline.step(self.omega_meas, omega_target,
                                             self.motor.state.theta, self.dt)
            V_unclipped = self.V_base + dv
            V = float(np.clip(V_unclipped, -V_max, V_max))
            eff_dv_sum += (V - self.V_base)
            clip_excess_sq_sum += (max(0.0, abs(V_unclipped) - V_max) / V_max) ** 2
            state = self.motor.step(self.dt, V)
            self.omega_meas = self.noise.measure(state.omega)
            err = omega_target - state.omega
            err_abs_sum += abs(err)
            self.V_applied_prev = V
            self.t += self.dt
        mean_err_abs = err_abs_sum / self.decimation
        eff_dv = eff_dv_sum / self.decimation

        # In saturation-aware mode penalise the *effective* (post-clip) residual
        # and the clip excess; otherwise reproduce the original commanded-dv reward.
        residual = eff_dv if self.saturation_aware else dv
        mean_clip_excess_sq = (clip_excess_sq_sum / self.decimation) if self.saturation_aware else 0.0

        # Tracking term uses the mean ABSOLUTE error (L1), weighted by alpha.
        r = (-self.rcfg.alpha * mean_err_abs / self.rcfg.e_scale
             - self.rcfg.lam * (residual / V_max) ** 2
             - self.rcfg.mu * ((residual - self.prev_dv) / V_max) ** 2
             - self.rcfg.sat_penalty * mean_clip_excess_sq)
        self.prev_dv = residual

        truncated = self.t >= self.t_end - 1e-9
        info = {"dv": dv, "eff_dv": eff_dv, "V_base": self.V_base,
                "omega": float(self.motor.state.omega), "V_applied": self.V_applied_prev}
        return self._obs(), float(r), False, truncated, info
