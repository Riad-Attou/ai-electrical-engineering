"""
Real-time pygame visualizer for the BDC motor + rod, with switchable
controllers — including the trained residual-RL policies.

Controllers come straight from the residual-RL benchmark
(`build_controllers`), so the plant parameters and policies match exactly
what was trained and benchmarked. Self-contained in this package; the root
`run_viz.py` is left untouched.

Run from the repo root:
    python residual_reinforcement_learning/run_viz_rl.py

Keys:
  1 .. 3       select controller (PI / PI+poly+grav / residual-PI,
               whichever are available)
  UP / DOWN    increase / decrease target speed
  R            reset motor and controller
  ESC / close  quit
"""
import sys
import pathlib

import pygame

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parent
for _p in (str(HERE), str(ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from utils.motor import BDCMotor, SpeedSensorNoise
from utils.visualizer import MotorVisualizer
from benchmark import build_controllers, PARAMS, ROD   # same plant + controllers as the benchmark

# ---------------------------------------------------------------------------
# Simulation timing and noise model
# ---------------------------------------------------------------------------
NOISE           = SpeedSensorNoise(std=0.0, quantization=0.0)
FPS             = 60
DT              = 1e-3
STEPS_PER_FRAME = max(1, int(round(1.0 / (FPS * DT))))   # ≈ 17
TARGET_STEP     = 1.0                                     # rad/s per key press


def make_motor() -> BDCMotor:
    m = BDCMotor(params=PARAMS, pendulum=ROD)
    m.reset(theta0=0.0)
    return m


# ---------------------------------------------------------------------------
# Real-time loop
# ---------------------------------------------------------------------------
def main(max_frames: int | None = None) -> None:
    controllers = build_controllers()          # {name: controller}; residual ones if models present
    names = list(controllers)

    # default to the first residual controller if available, else the last classical one
    current = next((i for i, n in enumerate(names) if n.startswith("residual")),
                   len(names) - 1)

    motor        = make_motor()
    ctrl         = controllers[names[current]]
    ctrl.reset()
    omega_target = 10.0
    omega_noisy  = 0.0
    voltage      = 0.0

    base_title = f"BDC Motor + Rod  |  1-{len(names)} controller   UP/DOWN speed   R reset"

    vis = MotorVisualizer(title=f"{base_title}   |   active: {names[current]}", fps=FPS)
    vis.open()

    num_keys = [pygame.K_1, pygame.K_2, pygame.K_3, pygame.K_4][:len(names)]
    prev_num = [False] * len(num_keys)

    running = True
    frame   = 0
    while running:
        running = vis.update(state=motor.state, voltage=voltage,
                             omega_noisy=omega_noisy, omega_target=omega_target)

        keys = pygame.key.get_pressed()
        if keys[pygame.K_UP]:
            omega_target += TARGET_STEP / FPS * 50   # smooth hold-down ramp
        if keys[pygame.K_DOWN]:
            omega_target -= TARGET_STEP / FPS * 50
        if keys[pygame.K_r]:
            motor = make_motor()
            ctrl.reset()

        # edge-detected controller selection (1..N)
        for i, kcode in enumerate(num_keys):
            if keys[kcode] and not prev_num[i]:
                current = i
                ctrl = controllers[names[current]]
                ctrl.reset()
                pygame.display.set_caption(f"{base_title}   |   active: {names[current]}")
            prev_num[i] = keys[kcode]

        # ---- simulation steps ----
        for _ in range(STEPS_PER_FRAME):
            voltage     = ctrl.step(omega_meas=omega_noisy,
                                    omega_target=omega_target,
                                    theta=motor.state.theta,
                                    dt=DT)
            state       = motor.step(dt=DT, voltage=voltage)
            omega_noisy = NOISE.measure(state.omega)

        frame += 1
        if max_frames is not None and frame >= max_frames:
            running = False

    vis.close()


if __name__ == "__main__":
    main()
