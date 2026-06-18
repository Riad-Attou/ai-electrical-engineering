"""
Motor + rod visualizer with PI gravity-compensation controller.
Run: python run_viz.py

Keys:
  UP   / DOWN  — increase / decrease target speed by 1 rad/s
  R            — reset motor and controller
  ESC / close  — quit
"""

import pygame
from utils.motor import BDCMotor, BDCMotorParams, PendulumParams, SpeedSensorNoise
from utils.visualizer import MotorVisualizer
from utils.controller import PIGravityController

# ---------------------------------------------------------------------------
# System parameters
# ---------------------------------------------------------------------------
PARAMS = BDCMotorParams(R=3.0, L=4e-3, Kt=0.05, Kb=0.05,
                        J=7.04e-5, B=0.005, V_max=12.0)
ROD    = PendulumParams(m=0.05, l=0.1, g=9.81)
NOISE  = SpeedSensorNoise(std=0.0, quantization=0.0)

# ---------------------------------------------------------------------------
# Controller
# ---------------------------------------------------------------------------
ctrl = PIGravityController(Kp=5.0, Ki=2.0, params=PARAMS, pendulum=ROD)

# ---------------------------------------------------------------------------
# Simulation timing
# ---------------------------------------------------------------------------
FPS             = 60
DT              = 1e-3
STEPS_PER_FRAME = max(1, int(round(1.0 / (FPS * DT))))  # ≈ 17

# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------
def make_motor():
    m = BDCMotor(params=PARAMS, pendulum=ROD)
    m.reset(theta0=0.0)
    return m

motor        = make_motor()
omega_target = 10.0          # [rad/s] — adjust with UP/DOWN
TARGET_STEP  = 1.0           # rad/s per key press

vis = MotorVisualizer(title="BDC Motor + Rod  |  ↑↓ speed  R reset", fps=FPS)
vis.open()

omega_noisy = 0.0
voltage     = 0.0
running     = True

while running:
    # ---- keyboard: read once per frame after event pump (inside vis.update) ----
    running = vis.update(
        state=motor.state,
        voltage=voltage,
        omega_noisy=omega_noisy,
        omega_target=omega_target,
    )

    keys = pygame.key.get_pressed()
    if keys[pygame.K_UP]:
        omega_target += TARGET_STEP / FPS * 50   # smooth hold-down ramp
    if keys[pygame.K_DOWN]:
        omega_target -= TARGET_STEP / FPS * 50
    if keys[pygame.K_r]:
        motor = make_motor()
        ctrl.reset()

    # ---- simulation steps ----
    for _ in range(STEPS_PER_FRAME):
        voltage     = ctrl.step(omega_meas=omega_noisy,
                                omega_target=omega_target,
                                theta=motor.state.theta,
                                dt=DT)
        state       = motor.step(dt=DT, voltage=5) # voltage)
        omega_noisy = NOISE.measure(state.omega)

vis.close()
