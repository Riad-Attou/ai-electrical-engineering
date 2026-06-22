# NN Model-Based Speed Controller

Neural network speed controller for BDC motor + vertical rod, trained via **backpropagation through differentiable physics simulation** (no real data required).

---

## Overview

Classical PI controllers require manual gain tuning and explicit analytical feedforward terms (gravity compensation, polynomial feedforward). This approach instead embeds the motor physics equations inside PyTorch and trains an MLP to output voltage commands by minimizing tracking error **through the simulation itself**.

```
NN weights → V → [motor ODEs in PyTorch] → ω → tracking error → backprop → NN weights
```

No recorded data. No manual gain selection. The network learns what voltage to apply by experiencing the simulated consequences of its own outputs.

---

## System

**Motor:** Brushed DC motor + uniform vertical rod (pendulum load)

```
Electrical:   L·di/dt  = V − R·i − Kb·ω
Mechanical:   J·dω/dt  = Kt·i − B·ω − m·g·(l/2)·sin(θ)
Kinematic:    dθ/dt    = ω
```

| Parameter | Value |
|-----------|-------|
| R (resistance) | 3.0 Ω |
| L (inductance) | 4×10⁻³ H |
| Kt (torque constant) | 0.05 N·m/A |
| Kb (back-EMF constant) | 0.05 V·s/rad |
| J (rotor inertia) | 7.4×10⁻⁵ kg·m² |
| B (viscous friction) | 0.005 N·m·s/rad |
| V_max | 12 V |
| Rod mass m | 0.05 kg |
| Rod length l | 0.1 m |

---

## Architecture

### Differentiable Motor (`utils/motor_torch.py` — `MotorDynamicsTorch`)

Rewrites the 3-state ODE in PyTorch tensors. Supports batched rollouts: state shape `(B, 3)` = `[i, ω, θ]`.

**Integration:** Runge-Kutta 4 (RK4)

> Euler integration is unstable during backprop for this system.  
> Electrical time constant τ = L/R = 4e-3/3 ≈ 1.33 ms ≈ dt.  
> RK4 evaluates derivatives at 4 intermediate points per step — numerically stable and more accurate.

### Neural Network Controller (`utils/motor_torch.py` — `NNSpeedController`)

**Architecture:** MLP with 2 hidden layers

```
[6 inputs] → Linear(6→64) → Tanh → Linear(64→64) → Tanh → Linear(64→1) → tanh × V_max
```

**Inputs (6):**

| Input | Formula | Normalization | Why |
|-------|---------|---------------|-----|
| Speed error | e = ω_ref − ω | ÷ 15 | Proportional signal |
| Error change | Δe = e[k] − e[k−1] | ÷ 1 | Derivative signal — NOT divided by dt (÷1e-3 would amplify by 1000×, causing NaN) |
| Integral error | ∫e·dt (clamped ±15) | ÷ 15 | Steady-state correction |
| Target speed | ω_ref | ÷ 15 | Different speeds require different feedforward V |
| Gravity direction (sin) | sin(θ) | already ∈ [−1,1] | Gravity torque component |
| Gravity direction (cos) | cos(θ) | already ∈ [−1,1] | Disambiguates θ quadrant |

> **Why sin/cos instead of θ?** Motor spinning → θ grows unboundedly. NN would encounter out-of-distribution inputs at test time. sin(θ) and cos(θ) always stay in [−1, 1] regardless of how many rotations.

**Output:** `tanh(raw) × V_max`

> Hard clamp (`torch.clamp`) kills gradients at the voltage limit — network cannot learn to reduce voltage. `tanh` saturates smoothly, gradients remain nonzero everywhere.

**Weight initialization:** Last layer weights ~ Uniform(−0.01, 0.01), bias = 0. Ensures early outputs near zero so the motor starts from rest without violent voltage spikes.

---

## Training

### Method: Backpropagation Through Time (BPTT) with Truncation

Each training episode:
1. Sample random target speed `ω_ref ∈ [−15, 15]` rad/s
2. Sample random initial rod angle `θ₀ ∈ [−π, π]` (v2)
3. Roll out 300 simulation steps (0.3 s)
4. Loss = MSE on the **last 75%** of steps (skip startup transient)
5. Backprop through simulation → update NN

**Truncated BPTT:** Gradients are detached every 50 steps.

> Without truncation, gradients travel 300 steps back through RK4 → gradient norm explodes. 50 steps ≈ 4–5 electrical time constants, enough to capture the relevant dynamics.

### Hyperparameters

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| `DT` | 1×10⁻³ s | Matches experiment timestep |
| `T_STEPS` | 300 | 0.3 s — long enough to observe steady-state behavior |
| `TBPTT` | 50 | Gradient truncation depth (~5τ_L) |
| `BATCH` | 64 | Parallel episodes per update — diversity stabilizes training |
| `N_EPISODES` | 4000 | Loss plateaus ~2000; extra 2000 for fine convergence |
| `LR` | 1×10⁻³ | Standard Adam. Higher (3×10⁻³) caused early instability |
| `LR schedule` | Cosine decay → 5×10⁻⁵ | Large steps early (fast learning), small steps late (fine-tune) |
| `HIDDEN` | 64 | Sufficient for 6-input problem. Larger adds no measurable gain |
| `OMEGA_MAX` | 15 rad/s | Above test reference (10 rad/s) for generalization margin |
| `LOSS_START` | 0.25 | Ignore first 25% (75 steps) — avoid rewarding fast rise at cost of steady-state |
| `grad_clip` | 0.5 | Cap gradient norm — prevents single divergent episode from corrupting weights |
| `theta0` (v1) | 0 | Rod starts vertical |
| `theta0` (v2) | Uniform(−π, π) | Forces NN to learn gravity rejection at all rod angles |

### Optimizer

Adam with cosine annealing LR schedule.

```
LR: 1e-3  →  (cosine)  →  5e-5   over 4000 episodes
```

---

## Results

### Training Convergence

Loss converged from ~10² to ~2×10⁻⁵ over 4000 episodes (4 orders of magnitude).

### Comparison vs PI Controllers (5 s simulation, no noise)

| Controller | Constant Ref RMSE | Trajectory RMSE |
|---|---|---|
| PI only | 0.495 rad/s | 0.322 rad/s |
| PI + gravity FFW | 0.450 rad/s | 0.279 rad/s |
| PI + poly FFW | 0.376 rad/s | 0.171 rad/s |
| PI + poly FFW + gravity FFW | 0.322 rad/s | 0.033 rad/s |
| **NN model-based (ours)** | **0.350 rad/s** | **0.018 rad/s** |

**NN wins on trajectory tracking by ~2× over the best engineered PI.**  
Constant-ref performance is competitive (within 9% of best PI).

### Why Constant Ref Has Remaining Oscillations

At constant ω, rod angle θ grows continuously → gravity torque `m·g·(l/2)·sin(θ)` oscillates periodically. Both PI and NN controllers produce similar oscillation amplitude (~±0.1 rad/s). This is a **fundamental disturbance rejection limit**, not a training failure. Eliminating it would require an active disturbance observer (e.g., extended state observer or RL-based approach).

### v1 vs v2 (Effect of Random θ₀)

| | Constant Ref | Trajectory |
|---|---|---|
| v1 (θ₀ = 0) | 0.346 rad/s | 0.024 rad/s |
| v2 (θ₀ ~ Uniform) | 0.350 rad/s | 0.018 rad/s |

v2 marginally worse on constant ref but better on trajectory. Random θ₀ forces NN to encounter and reject gravity disturbances from the start of each episode.

---

## File Structure

```
AI-Electrical-2026/
├── utils/
│   ├── motor.py           — numpy BDC motor simulation (original)
│   ├── motor_torch.py     — differentiable PyTorch motor + NNSpeedController
│   └── controller.py      — PIGravityController (baseline)
├── nn/
│   ├── train_nn_ctrl.py   — training script (BPTT)
│   ├── experiments_nn.py  — comparison vs PI
│   └── results/
│       ├── nn_ctrl.pt         — saved model weights
│       ├── train_loss.png     — training loss curve
│       └── nn_comparison.png  — comparison plots
```

---

## Usage

```bash
cd AI-Electrical-2026

# Train (≈ 3–5 min on CPU)
/opt/anaconda3/envs/ai-electrical/bin/python -m nn.train_nn_ctrl

# Compare vs PI
/opt/anaconda3/envs/ai-electrical/bin/python -m nn.experiments_nn
```

---

## Key Design Decisions

### Why Model-Based (not RL)?

RL requires thousands of environment interactions and a reward signal. Model-based BPTT uses the analytical motor equations directly — gradients flow through physics, convergence is much faster (~minutes vs hours).

Tradeoff: model-based requires accurate physics knowledge. RL can handle model uncertainty and real hardware. See `nn-rl/` for the RL approach.

### Why Not Pure Supervised Learning?

Supervised learning needs labeled data: `(state, correct_voltage)` pairs. Generating "correct voltages" requires solving an optimal control problem. BPTT skips this — the network discovers the correct voltages by minimizing tracking error end-to-end.
