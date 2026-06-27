# Experimental Results — Speed Filtering for the Motor + Rod System (ML2)

## Task

Recover the true motor speed `omega_true` from a noisy speed sensor
`omega_noisy` (plus the known command `voltage`) on the shared team plant: a
brushed DC motor driving a **vertical rod load**. This is the filtering /
observer block of the project.

**Inputs (features):** `[omega_noisy, voltage]`.  **Target:** `omega_true`.
The rod angle is an unmeasured internal state and is never given to the filter.

## Physical model — brushed DC motor + vertical rod

| Parameter | Value |
|---|---|
| R, L | 3.0 Ω, 4 mH |
| Kt, Kb | 0.05 N·m/A, 0.05 V·s/rad |
| Rotor inertia J | 7.04×10⁻⁵ kg·m² |
| Viscous friction B | 5×10⁻³ N·m·s/rad |
| Supply voltage | ±12 V |
| Rod mass / length | 0.05 kg / 0.10 m |

Parameters match `matlab/param.m`, i.e. the same plant used by the RL control
section, so the whole project describes one physical system.

The rod contributes a gravity torque `T_grav = m·g·l_cm·sin(θ)`. This makes the
mechanical dynamics **nonlinear and state-dependent** — the key property that
separates a learned filter from a linear estimator.

## Dataset

Generated from the Python model (`python BDCmotor.py`):

- **210 in-distribution trajectories**, 6 s each at dt = 1 ms, split by
  trajectory **70 / 15 / 15** (147 / 31 / 32) — no time-step leaks between
  splits. Excitation cycles step / ramp / random / mixed bipolar voltage.
- **Per-trajectory variability:** motor R, J, B perturbed ±12 %; sensor noise
  std drawn from 2–4 rad/s with 0–1 rad/s encoder quantization.
- **OOD set:** 20 trajectories driven by a **chirp** (swept-sine) voltage — an
  excitation family never seen in training (`data/rod_ood.npz`).

Signal vs noise on the training set: speed std ≈ 21.5 rad/s, noise std ≈
3.1 rad/s → **SNR ≈ 7× (≈14 % noise)**.

## Methods

**Classical baselines** (tuned on the validation split, evaluated on test):
- **Raw** — the unfiltered noisy sensor (reference).
- **MA** — causal moving average, window = 64.
- **EMA** — exponential moving average, α tuned on val.
- **Kalman** — steady-state linear Kalman filter on the nominal `[i, ω]` motor
  model. Its linear model **omits the gravity term**; R and Q are tuned on val,
  so it is a fair, well-configured baseline rather than a strawman.

**Learned filters** (window W = 64 ms, input `[omega_noisy, voltage]`):
- **CNN** — stacked causal 1-D convolutions (8.8 k params).
- **GRU** — single-layer recurrent filter (3.5 k params).
- **TCN** — dilated causal convolutions, 91 ms receptive field (33 k params).

All learned filters: MSE loss, Adam (lr 1e-3), early stopping on val.

## Results — in-distribution test set

| Method | RMSE [rad/s] | RMSE [RPM] | vs Raw |
|---|---:|---:|---:|
| Raw (no filter) | 2.97 | 28.3 | — |
| MA (window 64) | 3.00 | 28.6 | −1.0 % |
| Kalman (tuned) | 1.32 | 12.6 | +55.6 % |
| EMA (tuned) | 1.07 | 10.2 | +64.0 % |
| CNN | 0.80 | 7.6 | +73.1 % |
| TCN | 0.65 | 6.2 | +78.0 % |
| **GRU** | **0.65** | **6.2** | **+78.1 %** |

**Two notable findings:**

1. The **model-based Kalman (1.32) is beaten by the model-free EMA (1.07).**
   The Kalman's linear `[i, ω]` model cannot represent the rod's gravity
   torque, so its model mismatch outweighs its model-based advantage — it
   degenerates toward trusting the noisy measurement.
2. The **learned filters win clearly.** GRU/TCN reach 0.65 rad/s, **~51 %
   lower error than the tuned Kalman**, by learning the nonlinear,
   state-dependent dynamics directly from data.

A 64-sample moving average barely helps (−1 %): the rod swings fast enough that
its lag cancels its smoothing.

## Results — out-of-distribution (chirp excitation, unseen in training)

| Method | RMSE [rad/s] | vs Raw | Δ vs in-distribution |
|---|---:|---:|---|
| Raw | 3.00 | — | — |
| EMA (tuned) | 2.64 | +12.1 % | collapses (was +64 %) |
| Kalman (tuned) | 2.29 | +23.6 % | collapses (was +56 %) |
| TCN | 0.98 | +67.3 % | holds |
| GRU | 0.93 | +68.9 % | holds |
| CNN | 0.81 | +73.0 % | holds |

On an excitation it never saw, the **classical filters fall apart** (EMA +64 %
→ +12 %, Kalman +56 % → +24 %) while the **neural filters stay robust**
(+67–73 %). The learned filters generalise to unseen inputs; the hand-tuned
classical filters were implicitly over-fit to the training excitation's
spectrum.

## Reproduce

```bash
python BDCmotor.py                 # generate data/rod_split.npz + rod_ood.npz
python train.py --model gru        # also: --model cnn / tcn
python compare.py                  # figures/comparison_rmse.png + overlay
python eval_ood.py                 # OOD chirp evaluation + figure
```

## Figures

- `figures/data_overview.png` — voltage, true vs noisy speed, sensor noise.
- `figures/comparison_rmse.png` — test RMSE bar chart, all methods.
- `figures/comparison_all_methods.png` — one test trajectory, speed + error.
- `figures/curves_{gru,cnn,tcn}.png` — training curves.
- `figures/filter_{gru,cnn,tcn}_test.png` — per-model test trajectory.
- `figures/ood_chirp.png` — OOD generalisation on chirp excitation.
