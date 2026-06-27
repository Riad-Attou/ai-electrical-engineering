"""
Step 1 — Collect raw speed data at steady state.

Identical data-collection pipeline to poly/poly_collect.py.
Output: nn/data/nn_data.npz  with keys  voltage, omega  (flat arrays)
Run:    python -m nn.nn_collect   (from project root)
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
from utils.motor import BDCMotor, BDCMotorParams, PendulumParams, SpeedSensorNoise

# ---------------------------------------------------------------------------
PARAMS = BDCMotorParams(R=3.0, L=4e-3, Kt=0.05, Kb=0.05,
                        J=7.4e-5, B=0.005, V_max=12.0)
ROD    = PendulumParams(m=0.05, l=0.1, g=9.81)
NOISE  = SpeedSensorNoise(std=0.0, quantization=0,
                          rng=np.random.default_rng(42))

DT           = 1e-3
SS_WINDOW    = 1.0    # seconds — rolling window to check for steady state
SS_STD_TOL   = 0.5    # rad/s  — std of window must be below this
SS_DRIFT_TOL = 0.3    # rad/s  — |mean_now - mean_prev| must be below this
MAX_WAIT     = 5.0    # seconds — give up and record anyway
N_SAMPLES    = 100    # raw samples to collect after steady state
VOLT_STEP    = 0.5    # voltage increment [V]
OUT_PATH     = Path(__file__).parent / "data" / "nn_data.npz"
# ---------------------------------------------------------------------------


def collect_at_voltage(voltage: float) -> tuple[np.ndarray, float]:
    motor = BDCMotor(PARAMS, pendulum=ROD)
    motor.reset(theta0=0.0)

    window_steps = int(SS_WINDOW / DT)
    max_steps    = int(MAX_WAIT  / DT)

    window    = []
    prev_mean = None
    t         = 0

    while t < max_steps:
        s = motor.step(DT, voltage)
        window.append(s.omega)
        t += 1

        if len(window) > window_steps:
            window.pop(0)

        if len(window) == window_steps:
            std  = float(np.std(window))
            mean = float(np.mean(window))

            if std < SS_STD_TOL:
                if prev_mean is None:
                    prev_mean = mean
                elif abs(mean - prev_mean) < SS_DRIFT_TOL:
                    break
                else:
                    prev_mean = mean

    wait_s = t * DT

    omega_samples = []
    for _ in range(N_SAMPLES):
        s = motor.step(DT, voltage)
        omega_samples.append(NOISE.measure(s.omega))

    return np.array(omega_samples, dtype=np.float32), wait_s


def collect():
    voltages = np.arange(-PARAMS.V_max, PARAMS.V_max + VOLT_STEP, VOLT_STEP)

    all_voltage = []
    all_omega   = []

    for v in voltages:
        omega_raw, waited = collect_at_voltage(v)
        all_voltage.append(np.full(len(omega_raw), v, dtype=np.float32))
        all_omega.append(omega_raw)
        print(f"  V = {v:+6.2f} V  →  {len(omega_raw)} samples  "
              f"ω ∈ [{omega_raw.min():+.2f}, {omega_raw.max():+.2f}] rad/s  "
              f"(waited {waited:.1f} s)")

    voltage_arr = np.concatenate(all_voltage)
    omega_arr   = np.concatenate(all_omega)

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    np.savez(OUT_PATH, voltage=voltage_arr, omega=omega_arr)
    print(f"\nSaved {len(omega_arr)} raw samples ({len(voltages)} voltage levels) → {OUT_PATH}")


if __name__ == "__main__":
    print("Sweeping voltage -V_max → +V_max, collecting raw steady-state data...")
    collect()
