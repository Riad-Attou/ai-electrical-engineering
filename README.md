# BDC Motor Speed Filter — AI in Electrical Engineering

Course project: train a neural network to denoise speed measurements from a simulated brushed DC motor.

The motor is simulated in Python (RK4 integrator). A noisy encoder reading is generated from the true speed. The goal is to learn a causal filter that recovers the true speed from the noisy measurement, optionally using the voltage as an extra feature.

---

## Project structure

```
BDCmotor.py            Entry point: physics demo + dataset generation
train.py               Full training pipeline (run this to train a model)

utils/
  motor.py             BDC motor model (RK4), noise model, single-trajectory dataset
  traj.py              Multi-trajectory generation, train/val/test split, MotorSplit
  dataset.py           PyTorch Dataset (sliding windows) + NormStats

models/
  gru_filter.py        GRUFilter  — GRU hidden state → linear head
  cnn_filter.py        CNNFilter  — stacked 1D convolutions → linear head

test_simulated_dataset/   Separate MATLAB/Simulink project (different assignment)
```

---

## Install

Create a virtual environment first (required on Arch Linux and recommended everywhere):

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
```

Install PyTorch for your hardware — use the [official selector](https://pytorch.org/get-started/locally/) to get the right command (CPU / CUDA / ROCm). Then install the remaining dependencies:

```bash
pip install numpy matplotlib
```

---

## Workflow

### 1. Generate the dataset

```bash
python BDCmotor.py
```

This runs three demos:
- `demo_raw_physics()` — step-voltage response (current, speed, position)
- `demo_dataset()` — single-trajectory noisy vs. true speed plot, saves `motor_dataset.npz`
- `demo_ml_dataset()` — generates **200 independent trajectories**, splits them 70/15/15 by trajectory, saves `motor_split.npz`

`motor_split.npz` is what `train.py` consumes.

### 2. Train

```bash
# Default: GRU, window=64 steps (64 ms), hidden=32
python train.py

# CNN instead
python train.py --model cnn

# Larger GRU with longer context
python train.py --model gru --window 128 --hidden 64 --layers 2
```

The script:
1. Prints a moving-average baseline RMSE (the reference to beat)
2. Trains with Adam + ReduceLROnPlateau + early stopping
3. Saves the best checkpoint to `best_filter.pt`
4. Prints test RMSE in rad/s and RPM with % improvement over baseline
5. Saves a comparison plot to `filter_gru_test.png` (or `filter_cnn_test.png`)

### 3. Tune

All key hyperparameters are CLI flags:

| Flag | Default | Description |
|---|---|---|
| `--model` | `gru` | `gru` or `cnn` |
| `--window` | `64` | Context length in timesteps (1 step = 1 ms) |
| `--hidden` | `32` | GRU hidden size |
| `--layers` | `1` | GRU number of layers |
| `--channels` | `32` | CNN channels per layer |
| `--kernel` | `8` | CNN kernel size |
| `--depth` | `2` | CNN number of conv layers |
| `--epochs` | `50` | Max training epochs |
| `--lr` | `1e-3` | Adam learning rate |
| `--batch` | `512` | Batch size |
| `--patience` | `10` | Early stopping patience |

---

## Dataset design

**Why 200 short trajectories instead of one long one?**

With a single trajectory, any train/val/test split by time window creates overlapping context — the model can see information close to the test region during training (data leakage). With independent trajectories, the split is at the trajectory level: train/val/test are completely disjoint.

**Variability across trajectories** prevents the filter from memorizing a single operating condition:
- Motor parameters R, J, B are perturbed ±15% per trajectory (manufacturing tolerance)
- Noise standard deviation drawn uniformly from 5–30 rad/s per trajectory
- Encoder quantization step drawn from 0–5 rad/s per trajectory
- Voltage profile cycles through: constant step / linear ramp / random steps / mixed

Each trajectory is 10 s at 1 ms timestep (10 000 samples). Total: ~1.4 M training windows.

**Output shape:** `(N_trajectories, T_steps)` — window along T inside each trajectory, never across boundaries.

---

## Model input/output

| Signal | Role | Shape |
|---|---|---|
| `omega_noisy` | Filter input (noisy speed) | `(B, W)` |
| `voltage` | Optional extra feature | `(B, W)` |
| `omega_true` | Filter target (true speed) | `(B,)` |

Both inputs are z-normalized using training-set statistics. The model predicts the normalized true speed at the last timestep of the window; the output is denormalized for evaluation and plotting.

---

## Adding a new model

Create `models/my_filter.py` with a class that takes `(B, W, 2)` and returns `(B,)`, then add it to the `build_model()` function in `train.py`:

```python
# models/my_filter.py
class MyFilter(nn.Module):
    def forward(self, x):   # x: (B, W, 2)
        ...
        return y             # (B,)
```

```python
# train.py — build_model()
if args.model == "myfilter":
    return MyFilter(...)
```

Then run `python train.py --model myfilter`.
