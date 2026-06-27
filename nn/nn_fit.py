"""
Steps 2-5 — Plot, invert, and fit a neural network regression.

  Step 2: plot  x=voltage, y=speed          (forward map)
  Step 3: plot  x=speed,   y=voltage        (inverted map)
  Step 4: MLP regression on averaged (ω → V) inverse map
  Step 5: print model summary and fit quality

Architecture: 1 → 64 → 64 → 1  (Tanh activations, no bias on output for V(0)≈0)
Weights saved as numpy arrays in nn_weights.npz for zero-dependency inference.

Output: nn/data/nn_weights.npz
Run:    python -m nn.nn_fit   (from project root)
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn

DATA_PATH    = Path(__file__).parent / "data" / "nn_data.npz"
WEIGHTS_PATH = Path(__file__).parent / "data" / "nn_weights.npz"

HIDDEN       = 16
EPOCHS       = 10000
LR           = 1e-3
WEIGHT_DECAY = 1e-3   # L2 regularisation — main lever against overfitting
SEED         = 0


class _MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(1, HIDDEN),
            nn.ReLU(),
            nn.Linear(HIDDEN, HIDDEN),
            nn.ReLU(),
            nn.Linear(HIDDEN, 1, bias=True),  # no bias → V(0) = 0
        )

    def forward(self, x):
        return self.net(x)


def fit():
    torch.manual_seed(SEED)

    d       = np.load(DATA_PATH)
    voltage = d["voltage"].astype(np.float64)
    omega   = d["omega"].astype(np.float64)

    # ---- average omega per voltage level (same as poly pipeline) ----
    unique_voltages = np.unique(voltage)
    omega_mean      = np.array([omega[voltage == v].mean() for v in unique_voltages])

    omega_min = float(omega_mean.min())
    omega_max = float(omega_mean.max())

    # ---- Step 2: forward map V → ω ----
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    axes[0].scatter(voltage, omega, color="C0", s=8, alpha=0.5)
    axes[0].set_xlabel("Voltage (V)")
    axes[0].set_ylabel("Speed (rad/s)")
    axes[0].set_title("Step 2 — Forward map  V → ω")
    axes[0].axhline(0, color="gray", lw=0.8, ls="--")
    axes[0].axvline(0, color="gray", lw=0.8, ls="--")
    axes[0].grid(True)

    # ---- Step 3 & 4: invert, train MLP on averaged data ----
    X_np = omega_mean.reshape(-1, 1).astype(np.float32)
    y_np = unique_voltages.reshape(-1, 1).astype(np.float32)

    X = torch.from_numpy(X_np)
    y = torch.from_numpy(y_np)

    model     = _MLP()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    loss_fn   = nn.MSELoss()

    for epoch in range(1, EPOCHS + 1):
        optimizer.zero_grad()
        loss = loss_fn(model(X), y)
        loss.backward()
        optimizer.step()
        if epoch % 1000 == 0:
            print(f"  epoch {epoch:5d}  loss = {loss.item():.6f}")

    model.eval()
    with torch.no_grad():
        omega_dense_np = np.linspace(omega_min, omega_max, 300).astype(np.float32)
        omega_dense_t  = torch.from_numpy(omega_dense_np.reshape(-1, 1))
        v_fit_np       = model(omega_dense_t).numpy().ravel()

    axes[1].scatter(omega_mean, unique_voltages, color="C1", zorder=3, label="averaged data")
    axes[1].plot(omega_dense_np, v_fit_np, color="C3", lw=2, label="NN fit  (1→64→64→1, Tanh)")
    axes[1].set_xlabel("Speed (rad/s)")
    axes[1].set_ylabel("Voltage (V)")
    axes[1].set_title("Step 3 & 4 — Inverse map  ω → V  (NN fit)")
    axes[1].axhline(0, color="gray", lw=0.8, ls="--")
    axes[1].axvline(0, color="gray", lw=0.8, ls="--")
    axes[1].legend()
    axes[1].grid(True)

    plt.tight_layout()
    fig_path = Path(__file__).parent / "data" / "nn_fit.png"
    plt.savefig(fig_path, dpi=150)
    plt.show()
    print(f"Plot saved → {fig_path}")

    # ---- Step 5: fit quality ----
    with torch.no_grad():
        v_pred = model(X).numpy().ravel()
    ss_res = np.sum((unique_voltages - v_pred) ** 2)
    ss_tot = np.sum((unique_voltages - unique_voltages.mean()) ** 2)
    r2     = 1.0 - ss_res / ss_tot
    print(f"\n  R² = {r2:.6f}   (1.0 = perfect)")

    # ---- save weights as numpy arrays for zero-dependency inference ----
    weights = {}
    for i, layer in enumerate(model.net):
        if isinstance(layer, nn.Linear):
            weights[f"W{i}"] = layer.weight.detach().numpy()
            if layer.bias is not None:
                weights[f"b{i}"] = layer.bias.detach().numpy()

    WEIGHTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    np.savez(WEIGHTS_PATH,
             omega_min=np.array(omega_min),
             omega_max=np.array(omega_max),
             **weights)
    print(f"Weights saved → {WEIGHTS_PATH}")


if __name__ == "__main__":
    fit()
