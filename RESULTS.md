# Experimental Results — Servo Backlash Compensation (ML2)

## Setup

**Task (ML2):** Estimate the true output (load) angle θ_l of a servo drive from the motor-side encoder position enc_m and the PWM command. The gearbox has a backlash dead zone; a rigid-coupling observer fails at every direction reversal.

**Physical model — two-inertia servo with gearbox**

| Parameter | Value |
|---|---|
| R, L, Ke, Kt | 2.0 Ω, 0.5 mH, 0.05 V·s/rad, 0.05 N·m/A |
| Motor inertia Jm | 2×10⁻⁵ kg·m² |
| Load inertia Jl | 5×10⁻³ kg·m² |
| Gear ratio N | 50 |
| Gear efficiency η | 0.85 |
| Output-side backlash half-gap | 0.02 rad (1.15°) |
| Motor-side backlash half-gap | 1.00 rad (57.3°) |
| Mesh stiffness / damping | 5 N·m/rad / 0.02 N·m·s/rad |
| Friction model | Stribeck + cogging (12 per rev) + thermal drift |
| Sensors | Motor encoder 12-bit + output encoder 14-bit (with eccentricity) |
| Load time constant τ_l = Jl / (N²·η·cg) | ≈ 0.12 ms |

**Dataset**

| Property | Value |
|---|---|
| Source | MATLAB/Simulink simulation (pre-recorded CSV) |
| Training | 30 s chirp, 1→20 Hz, A = 0.9 — resampled to dt = 1 ms |
| Test | 30 s multisine (6 tones, 1.3–12.8 Hz) — unseen excitation |
| Split | first 70 % of chirp → train (21 001 steps), last 30 % → val (9 000 steps) |
| Total training windows | ~20 938 (window W = 64 steps = 64 ms) |

**Feature engineering — per-window velocity normalization**

The chirp excitation (train) has 2.4× higher peak motor speed than the multisine (test). A fixed global normalization of the backlash error (by its training std) causes a 2.4× scale mismatch on test, making models over-correct and score worse than rigid coupling. The fix is per-window velocity scaling:

| Signal | Formula | Notes |
|---|---|---|
| `enc_m_rel` | enc_m[t] − enc_m[window_start] | Window-relative; eliminates DC drift |
| `local_vel` | std(diff(enc_m_win)) | Per-window motor velocity scale |
| Input feature 1 | enc_m_rel / local_vel | Scale-invariant motion pattern |
| Input feature 2 | (pwm − pwm_mean) / pwm_std | Z-scored PWM |
| Target | (θ_l − enc_m/N) / (local_vel / N) | Velocity-relative backlash error |
| Inference denorm | model_output × (local_vel / N) | Back to radians, per window |

---

## Results

All RMSE values are on the **test set** (30 s multisine, never seen during training or model selection).

| Method | RMSE (mrad) | RMSE (°) | vs rigid coupling |
|---|---:|---:|---:|
| **Rigid coupling** — enc_m/N | 0.102 | 0.0058 | — |
| Output encoder — enc_o direct | 0.323 | 0.0185 | −217.5 % |
| **TCN** — dilated conv, RF = 91 ms | **0.052** | **0.0030** | **+48.4 %** |
| **GRU** — enc_m + pwm | **0.072** | **0.0041** | **+29.4 %** |
| **CNN** — valid conv, RF = 15 ms | 0.117 | 0.0067 | −14.8 % |

**Model details**

| Model | Params | Input | Context |
|---|---:|---|---|
| GRU | 3 489 | (W, 2) | full W = 64 ms via hidden state |
| CNN | 8 801 | (W, 2) | RF = 15 ms (valid conv, depth 2, k = 8) |
| TCN | 33 153 | (W, 2) | RF = 91 ms (dilated, k = 4, 4 levels) |

---

## Analysis

### 1. Why the backlash error is small — the spring never engages

The physical backlash gap at the output is 0.02 rad (20 mrad). However, the observable backlash error (θ_l − enc_m/N) reaches only 0.6 mrad peak. This is 3% of the theoretical gap.

Root cause: in the simulation model, the mesh torque is

    T_mesh = k_g × dz + c_g × (ω_m − N × ω_l)

where dz = clamp-residual of (θ_m − N θ_l) w.r.t. ±gap_motor. The **damper** c_g × (ω_m − N ω_l) is active at all times (even inside the dead zone), while the **spring** k_g × dz only activates when |φ| > gap_motor = 1.0 rad.

In the data, φ = θ_m − N × θ_l stays within ±0.03 rad throughout both the chirp and multisine runs — never reaching the 1.0 rad threshold. Consequently:

- Spring force = 0 at all times (gear never engages through the spring)
- The load is coupled to the motor **only through the damper** (c_g = 0.02 N·m·s/rad)
- The observable error is **damper-induced position lag**, not dead-zone hysteresis

This lag is proportional to motor velocity and acceleration, with characteristic time τ_l = J_l / (N²·η·c_g) ≈ 0.12 ms.

### 2. Distribution shift and the per-window normalization fix

The chirp (train) and multisine (test) differ not only in excitation pattern but also in motor speed amplitude:

| Split | Motor speed std (rad/s) | Backlash error std (mrad) | Ratio |
|---|---:|---:|---:|
| Chirp (train) | 98 | 0.23 | — |
| Multisine (test) | 41 | 0.10 | 0.44 |

A model trained with fixed error_std normalisation from the chirp would apply 2.3× too-large corrections on the test set, making predictions worse than rigid coupling. Per-window velocity normalisation removes this shift: both the input (enc_m_rel / local_vel) and the target (backlash_error / (local_vel / N)) scale with the same local motor velocity, keeping their ratio nearly constant across train and test.

### 3. TCN outperforms GRU on this task

TCN (RF = 91 ms) achieves the best RMSE (0.052 mrad, +48.4%). GRU (full 64 ms context via hidden state) is second at 0.072 mrad (+29.4%). CNN (RF = 15 ms) underperforms rigid coupling (0.117 mrad, −14.8%).

The damper-lag correction requires knowledge of the **recent velocity history** to estimate the current lag term. TCN's 91 ms dilated-convolution window covers ~6 motor electrical cycles at the highest test frequency (12.8 Hz → period 78 ms), which is enough to estimate instantaneous velocity and acceleration reliably. CNN's 15 ms window is too short to average out encoder quantisation and estimate velocity well enough.

GRU's hidden state accumulates context across the full 64 ms window and generalises well, slightly behind TCN.

### 4. Context window vs backlash dynamics

| Time scale | Value |
|---|---|
| Motor τ_e = L/R | 0.25 ms |
| Load τ_l = Jl / (N²·η·c_g) | ≈ 0.12 ms |
| Damper coupling settling (4τ_l) | ≈ 0.5 ms |
| Window W = 64 ms | covers ~128 settling times |
| TCN RF = 91 ms | ~2× the window size via dilation |

The settling time is only 0.5 ms, far shorter than the 64 ms window. The window is not needed for "memory of the dead zone" (which never activates), but for accurate velocity/acceleration estimation from the noisy quantised encoder signal.

### 5. Output encoder is worse than rigid coupling

The output encoder (enc_o) scores 0.323 mrad RMSE — 3× worse than rigid coupling. This is because enc_o carries a periodic eccentricity error (h₁ = 0.8 mrad, h₂ = 0.3 mrad peak) plus read latency (0.5 ms). These systematic errors dominate the tiny (0.1 mrad) backlash signal.

---

## Summary

TCN achieves the best test RMSE (0.052 mrad, +48.4% vs rigid coupling) by capturing the velocity-dependent damper lag over its 91 ms dilated-convolution receptive field. GRU (0.072 mrad, +29.4%) follows; CNN (0.117 mrad, −14.8%) underperforms due to its short 15 ms context.

The key methodological insight is **per-window velocity normalisation**: since the backlash error in this dataset is dominated by damper lag (proportional to motor speed), normalising by the window's own velocity scale eliminates the 2.4× speed mismatch between the chirp training set and multisine test set, enabling models to generalise.

The gear spring (main backlash nonlinearity) never engages because the motor-side transmission error φ remains 33× smaller than the motor-side half-gap (0.03 rad vs 1.0 rad). In a scenario with a smaller gap or larger oscillation amplitude, the dead-zone hysteresis would dominate and recurrent architectures with longer context would likely show a larger advantage over the rigid baseline.
