"""
Brushed DC Motor Model  (optionally with vertical rod load)
======================
State variables: [current i (A), angular velocity omega (rad/s), position theta (rad)]

Electrical:  L * di/dt = V - R*i - Kb*omega
Mechanical (motor only):
    J * domega/dt = Kt*i - B*omega - T_load

Mechanical (motor + vertical rod, uniform rod about pivot):
    I_rod  = (1/3) * m * l²          (rod inertia about shaft)
    l_cm   = l / 2                   (centre of mass distance)
    T_grav = m * g * l_cm * sin(θ)   (gravity torque, restoring when |θ| < π/2)

    (J + I_rod) * domega/dt = Kt*i - B*omega - T_grav

Kinematic:   dtheta/dt = omega
"""

from __future__ import annotations
from dataclasses import dataclass, field
import numpy as np


@dataclass
class BDCMotorParams:
    R:     float = 1.0      # Armature resistance   [Ohm]
    L:     float = 0.5e-3   # Armature inductance   [H]
    Kt:    float = 0.01     # Torque constant       [N·m/A]
    Kb:    float = 0.01     # Back-EMF constant     [V·s/rad]
    J:     float = 1e-5     # Rotor inertia         [kg·m²]
    B:     float = 1e-6     # Viscous friction      [N·m·s/rad]
    V_max: float = 24.0     # Maximum supply voltage [V]


@dataclass
class PendulumParams:
    """
    Uniform vertical rod attached to the motor shaft.

    m : mass          [kg]
    l : total length  [m]
    g : gravity       [m/s²]

    Derived quantities (read-only properties):
      I_rod = (1/3) * m * l²     moment of inertia about pivot
      l_cm  = l / 2              centre-of-mass distance
    """
    m: float = 0.1     # [kg]
    l: float = 0.3     # [m]
    g: float = 9.81    # [m/s²]

    @property
    def I_rod(self) -> float:
        return (1.0 / 3.0) * self.m * self.l ** 2

    @property
    def l_cm(self) -> float:
        return self.l / 2.0


@dataclass
class SpeedSensorNoise:
    """
    Speed measurement noise only.

    std          : Gaussian noise standard deviation  [rad/s]
    quantization : Encoder resolution                 [rad/s per count], 0 = off
    """
    std:          float = 1.0    # [rad/s]  ≈ ~10 RPM
    quantization: float = 0.0   # [rad/s per count], 0 = disabled
    rng: np.random.Generator = field(
        default_factory=lambda: np.random.default_rng(42)
    )

    def measure(self, omega_true: float) -> float:
        omega_meas = omega_true + self.rng.normal(0.0, self.std)
        if self.quantization > 0:
            omega_meas = round(omega_meas / self.quantization) * self.quantization
        return float(omega_meas)


@dataclass
class BDCMotorState:
    i:     float = 0.0
    omega: float = 0.0
    theta: float = 0.0
    t:     float = 0.0

    def as_array(self) -> np.ndarray:
        return np.array([self.i, self.omega, self.theta])


class BDCMotor:
    """Brushed DC motor with RK4 integrator, optional vertical rod load."""

    def __init__(
        self,
        params:   BDCMotorParams  | None = None,
        pendulum: PendulumParams  | None = None,
    ):
        self.p        = params   or BDCMotorParams()
        self.pendulum = pendulum             # None → no rod
        self.state    = BDCMotorState()

    def _J_eff(self) -> float:
        """Effective inertia: motor rotor + rod (if attached)."""
        if self.pendulum is None:
            return self.p.J
        return self.p.J + self.pendulum.I_rod

    def _derivatives(self, x: np.ndarray, voltage: float, T_load: float) -> np.ndarray:
        i, omega, theta = x
        p = self.p

        # gravity torque from vertical rod (zero when no pendulum)
        T_grav = 0.0
        if self.pendulum is not None:
            rod = self.pendulum
            T_grav = rod.m * rod.g * rod.l_cm * np.sin(theta)

        di_dt     = (voltage - p.R * i - p.Kb * omega) / p.L
        domega_dt = (p.Kt * i - p.B * omega - T_load - T_grav) / self._J_eff()
        dtheta_dt = omega
        return np.array([di_dt, domega_dt, dtheta_dt])

    def step(self, dt: float, voltage: float, T_load: float = 0.0) -> BDCMotorState:
        x  = self.state.as_array()
        k1 = self._derivatives(x,           voltage, T_load)
        k2 = self._derivatives(x + dt/2*k1, voltage, T_load)
        k3 = self._derivatives(x + dt/2*k2, voltage, T_load)
        k4 = self._derivatives(x + dt*k3,   voltage, T_load)
        xn = x + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)
        self.state = BDCMotorState(i=float(xn[0]), omega=float(xn[1]),
                                   theta=float(xn[2]), t=self.state.t + dt)
        return self.state

    def reset(self, i0: float = 0.0, omega0: float = 0.0, theta0: float = 0.0):
        self.state = BDCMotorState(i=i0, omega=omega0, theta=theta0, t=0.0)


@dataclass
class MotorDataset:
    """
    Arrays aligned on the same time axis.

    omega_noisy : (N,)  measured (noisy) speed  [rad/s]  — filter INPUT
    omega_true  : (N,)  true speed              [rad/s]  — filter TARGET
    voltage     : (N,)  applied voltage         [V]      — optional extra feature
    t           : (N,)  time stamps             [s]
    """
    omega_noisy: np.ndarray
    omega_true:  np.ndarray
    voltage:     np.ndarray
    t:           np.ndarray

    def save(self, path: str):
        np.savez(path, omega_noisy=self.omega_noisy,
                 omega_true=self.omega_true,
                 voltage=self.voltage, t=self.t)

    @staticmethod
    def load(path: str) -> "MotorDataset":
        d = np.load(path)
        return MotorDataset(omega_noisy=d["omega_noisy"], omega_true=d["omega_true"],
                            voltage=d["voltage"], t=d["t"])


@dataclass
class PendulumDataset:
    """
    Arrays aligned on the same time axis — motor + vertical rod system.

    omega_noisy : (N,)  measured (noisy) speed   [rad/s]  — filter INPUT
    omega_true  : (N,)  true angular velocity     [rad/s]  — filter TARGET
    theta_true  : (N,)  true rod angle            [rad]
    voltage     : (N,)  applied voltage           [V]      — optional feature
    t           : (N,)  time stamps               [s]
    """
    omega_noisy: np.ndarray
    omega_true:  np.ndarray
    theta_true:  np.ndarray
    voltage:     np.ndarray
    t:           np.ndarray

    def save(self, path: str):
        np.savez(path,
                 omega_noisy=self.omega_noisy,
                 omega_true=self.omega_true,
                 theta_true=self.theta_true,
                 voltage=self.voltage,
                 t=self.t)

    @staticmethod
    def load(path: str) -> "PendulumDataset":
        d = np.load(path)
        return PendulumDataset(
            omega_noisy=d["omega_noisy"],
            omega_true=d["omega_true"],
            theta_true=d["theta_true"],
            voltage=d["voltage"],
            t=d["t"],
        )


def generate_pendulum_dataset(
    params:   BDCMotorParams  | None = None,
    pendulum: PendulumParams  | None = None,
    noise:    SpeedSensorNoise | None = None,
    dt:       float = 1e-3,
    t_end:    float = 10.0,
    profile:  str   = "mixed",   # "step" | "ramp" | "random" | "mixed"
    seed:     int   = 0,
) -> PendulumDataset:
    """
    Simulate the motor + vertical rod under a chosen voltage profile.
    Records noisy speed measurement alongside true speed and angle.

    Filter training:
        X = dataset.omega_noisy   (or stack with dataset.voltage / sin(theta_true))
        y = dataset.omega_true
    """
    motor = BDCMotor(params, pendulum=pendulum or PendulumParams())
    noise = noise or SpeedSensorNoise(rng=np.random.default_rng(seed))
    rng   = np.random.default_rng(seed + 1)
    V_max = motor.p.V_max

    steps           = int(t_end / dt)
    change_interval = max(1, int(0.5 / dt))

    omega_noisy = np.empty(steps, dtype=np.float32)
    omega_true  = np.empty(steps, dtype=np.float32)
    theta_true  = np.empty(steps, dtype=np.float32)
    voltage_log = np.empty(steps, dtype=np.float32)
    t_log       = np.empty(steps, dtype=np.float32)

    voltage = 0.0
    for k in range(steps):
        if profile == "step":
            voltage = V_max
        elif profile == "ramp":
            # -V_max → +V_max over the episode
            voltage = V_max * (2.0 * k / steps - 1.0)
        elif profile == "random":
            if k % change_interval == 0:
                voltage = rng.uniform(-V_max, V_max)
        else:  # mixed: constant → bipolar ramp → random segments
            seg = (k // (steps // 3)) % 3
            if seg == 0:
                voltage = V_max * 0.5
            elif seg == 1:
                frac = (k % (steps // 3)) / (steps // 3)
                voltage = V_max * (2.0 * frac - 1.0)
            else:
                if k % change_interval == 0:
                    voltage = rng.uniform(-V_max, V_max)

        state = motor.step(dt, voltage)

        omega_noisy[k] = noise.measure(state.omega)
        omega_true[k]  = state.omega
        theta_true[k]  = state.theta
        voltage_log[k] = voltage
        t_log[k]       = state.t

    return PendulumDataset(
        omega_noisy=omega_noisy,
        omega_true=omega_true,
        theta_true=theta_true,
        voltage=voltage_log,
        t=t_log,
    )


def generate_dataset(
    params:  BDCMotorParams   | None = None,
    noise:   SpeedSensorNoise | None = None,
    dt:      float = 1e-3,
    t_end:   float = 10.0,
    profile: str   = "mixed",   # "step" | "ramp" | "random" | "mixed"
    seed:    int   = 0,
) -> MotorDataset:
    """
    Simulate the motor under a chosen voltage profile and record both
    the true speed and the noisy speed measurement.

    Filter training:
        X = dataset.omega_noisy   (or stack with dataset.voltage)
        y = dataset.omega_true
    """
    motor = BDCMotor(params)
    noise = noise or SpeedSensorNoise(rng=np.random.default_rng(seed))
    rng   = np.random.default_rng(seed + 1)
    V_max = motor.p.V_max

    steps           = int(t_end / dt)
    change_interval = max(1, int(0.5 / dt))   # new voltage every 0.5 s

    omega_noisy = np.empty(steps, dtype=np.float32)
    omega_true  = np.empty(steps, dtype=np.float32)
    voltage_log = np.empty(steps, dtype=np.float32)
    t_log       = np.empty(steps, dtype=np.float32)

    voltage = 0.0
    for k in range(steps):
        if profile == "step":
            voltage = V_max
        elif profile == "ramp":
            # -V_max → +V_max over the episode
            voltage = V_max * (2.0 * k / steps - 1.0)
        elif profile == "random":
            if k % change_interval == 0:
                voltage = rng.uniform(-V_max, V_max)
        else:  # mixed: constant → bipolar ramp → random segments
            seg = (k // (steps // 3)) % 3
            if seg == 0:
                voltage = V_max * 0.5
            elif seg == 1:
                frac = (k % (steps // 3)) / (steps // 3)
                voltage = V_max * (2.0 * frac - 1.0)
            else:
                if k % change_interval == 0:
                    voltage = rng.uniform(-V_max, V_max)

        state = motor.step(dt, voltage)

        omega_noisy[k] = noise.measure(state.omega)
        omega_true[k]  = state.omega
        voltage_log[k] = voltage
        t_log[k]       = state.t

    return MotorDataset(omega_noisy=omega_noisy, omega_true=omega_true,
                        voltage=voltage_log, t=t_log)
