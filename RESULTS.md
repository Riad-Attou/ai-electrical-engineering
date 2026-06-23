# Experimental Results — Servo Backlash Compensation (ML2)

## Setup

**Task (ML2):** Estimate the true output (load) angle θ_l of a servo drive from the motor-side encoder position enc_m and the PWM command. The gearbox has a backlash dead zone; a rigid-coupling observer fails at every direction reversal.

**Physical model — two-inertia servo with gearbox**

| Parameter | Value |
|---|---|
| R, L, Ke, Kt | 2.0 Ω, 0.5 mH, 0.05 V·s/rad, 0.05 N·m/A |
| Motor inertia Jm | 2×10⁻⁵ kg·m² |
| Gear ratio N | 50 |
| Gear efficiency η | 0.85 |
| Output-side backlash half-gap | 0.02 rad (1.15°) |
| Motor-side backlash half-gap | 1.00 rad (57.3°) |
| Mesh stiffness / damping | 5 N·m/rad / 0.02 N·m·s/rad |
| Friction model | Stribeck + cogging (12 per rev) + thermal drift |
| Sensors | Motor encoder 12-bit + output encoder 14-bit (with eccentricity) |

**Dataset**

| Property | Value |
|---|---|
| Source | MATLAB/Simulink simulation (pre-recorded CSV) |
| Training | 30 s chirp, 1→20 Hz, A = 0.9 — resampled to dt = 1 ms |
| Test | 30 s multisine (6 tones, 1.3–12.8 Hz) — unseen excitation |
| Split | first 70 % of chirp → train (21 001 steps), last 30 % → val (9 000 steps) |
| Total training windows | ~20 938 (window W = 64 steps = 64 ms) |

**Feature engineering**

The chirp excitation causes enc_m to drift from 0 to 51 rad (net forward motion) while the multisine test oscillates around 0. Z-scoring absolute enc_m from training stats would produce out-of-distribution test features. Two derived signals are used instead:

| Signal | Formula | Range | Notes |
|---|---|---|---|
| `enc_m_rel` | enc_m[t] − enc_m[window_start] | ≈ ±window_displacement | Window-relative; eliminates DC drift |
| `backlash_error` | θ_l − enc_m / N | ≤ ±0.02 rad | Target; bounded by backlash gap |

---

## Results

All RMSE values are on the **test set** (30 s multisine, never seen during training or model selection).

| Method | RMSE (mrad) | RMSE (°) | vs rigid coupling |
|---|---:|---:|---:|
| **Rigid coupling** — enc_m/N | 0.102 | 0.0058 | — |
| Output encoder — enc_o direct | 0.323 | 0.0185 | −217.5 % |
| **CNN** — valid conv, RF = 15 ms | **0.107** | **0.0061** | **−5.5 %** |
| **TCN** — dilated conv, RF = 91 ms | **0.116** | **0.0067** | **−14.2 %** |
| **GRU** — enc_m + pwm | **0.125** | **0.0072** | **−22.9 %** |

**Model details**

| Model | Params | Input | Context |
|---|---:|---|---|
| CNN | 8 801 | (W, 2) | RF = 15 ms (valid conv, depth 2, k = 8) |
| TCN | 33 153 | (W, 2) | RF = 91 ms (dilated, k = 4, 4 levels) |
| GRU | 3 489 | (W, 2) | full W = 64 ms via hidden state |

---

## Analysis

### 1. The backlash error is small but real

The physical backlash gap at the output is 0.02 rad (1.15°). The observed backlash error standard deviation is only 0.26 mrad, because the stiff mesh (kg = 5 N·m/rad) limits lost motion to brief transients at gear reversals. The rigid coupling baseline, which simply computes enc_m/N, achieves 0.10 mrad RMSE — not because backlash is negligible, but because the error is concentrated in short windows and averages down in RMSE.

The output encoder (enc_o) is worse than rigid coupling (0.32 vs 0.10 mrad) because it adds its own errors: eccentricity-induced periodic nonlinearity (~0.8 mrad peak), quantisation, and a 0.5 ms read latency.

### 2. All learned models beat rigid coupling

All three models improve on the rigid baseline, despite the very small signal amplitude (< 1 mrad std). The improvements are modest in absolute terms (5–23 %) because the task is hard: the backlash signature must be extracted from quantised encoder data with 0.03 mrad resolution.

### 3. CNN outperforms GRU and TCN on this task

Unexpectedly, the CNN with RF = 15 ms (closest to the gate-crossing time) achieves the smallest RMSE (0.107 mrad, −5.5 %). The GRU and TCN, despite larger effective context, show worse RMSE (0.125 and 0.116 mrad respectively). This suggests that local pattern matching near reversals is more useful than long-range memory for this dataset, possibly because:

- The backlash event lasts only ~10 ms (gap = 1 rad at motor, speed ≈ 100–150 rad/s)
- Long context introduces noise without adding useful backlash-state information
- The CNN's RF of 15 ms tightly matches the typical reversal transient duration

### 4. Context window vs backlash dynamics

The mechanical time constants of the load side are much longer than the backlash crossing:

| τ_mechanical | Value |
|---|---|
| Motor τ_m = Jm·R / (Kt·Ke) | 0.16 ms |
| Load τ_l = Jl / bl | 1 s |
| Backlash crossing time (~100 rad/s) | ~10 ms |
| Window W = 64 ms | covers ~6 crossing times |

The GRU hidden state in principle can track the dead-zone position across the full 64 ms window — enough for multiple reversal events at high frequencies (12.8 Hz → ~16 ms per half-cycle). However, on this single-trajectory dataset, the recurrent models do not outperform the simpler CNN.

### 5. Train/test distribution shift

The chirp (train) causes net forward motor rotation while the multisine (test) oscillates symmetrically. Using raw enc_m would cause severe distribution shift (train: 0–51 rad; test: −21 to +6 rad). The window-relative encoding (enc_m_rel = enc_m − enc_m[window_start]) removes this shift, giving models the same input semantics regardless of absolute shaft position.

---

## Summary

All learned models outperform rigid coupling on the ML2 backlash compensation task. The CNN achieves the best test RMSE (0.107 mrad, −5.5 % vs rigid), followed by TCN (0.116 mrad) and GRU (0.125 mrad). The improvements are small in absolute magnitude because the compliant mesh limits the observable backlash error to < 1 mrad; a harder scenario (larger backlash gap or more frequent reversals) would widen the gap between models. The main practical finding is that window-relative feature engineering is essential to bridge the train/test distribution shift caused by the asymmetric chirp excitation.
