"""
Train NNSpeedController via backprop through differentiable motor simulation.

Fixes vs v1
-----------
- RK4 integration (Euler unstable for L/R stiffness during backprop)
- Truncated BPTT: detach state every TBPTT steps (caps gradient depth)
- Inputs normalised inside NNSpeedController (sin/cos θ, Δe not Δe/dt)
- Smaller LR, tighter grad clip

Run:
    cd /Users/pasorn/Desktop/Xian-Jiaotong/class/AI-Electrical-2026
    /opt/anaconda3/envs/ai-electrical/bin/python -m control_neural_model_based.train_nn_ctrl
"""

from __future__ import annotations
from pathlib import Path

import torch
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt

from utils.motor import BDCMotorParams, PendulumParams
from utils.motor_torch import MotorDynamicsTorch, NNSpeedController

# ---------------------------------------------------------------------------
# System parameters (match experiments.py)
# ---------------------------------------------------------------------------
PARAMS = BDCMotorParams(R=3.0, L=4e-3, Kt=0.05, Kb=0.05,
                        J=7.4e-5, B=0.005, V_max=12.0)
ROD    = PendulumParams(m=0.05, l=0.1, g=9.81)

# ---------------------------------------------------------------------------
# Training hyperparameters
# ---------------------------------------------------------------------------
DT          = 1e-3
T_STEPS     = 300        # steps per episode
TBPTT       = 50         # truncated-BPTT chunk length (detach state every N steps)
BATCH       = 64
N_EPISODES  = 4000
LR          = 1e-3
HIDDEN      = 64
OMEGA_MAX   = 15.0       # rad/s training range
LOSS_START  = 0.25       # skip first 25% of episode (startup transient)

SAVE_DIR   = Path(__file__).parent / "results"
MODEL_PATH = SAVE_DIR / "nn_ctrl.pt"
DEVICE     = "cpu"

INT_CLAMP  = OMEGA_MAX   # integral anti-windup clamp


def rollout_tbptt(
    ctrl:      NNSpeedController,
    motor:     MotorDynamicsTorch,
    omega_ref: torch.Tensor,   # (B,)
    T:         int,
    dt:        float,
    chunk:     int,
    theta0:    torch.Tensor | None = None,  # (B,) initial rod angle
) -> torch.Tensor:
    """
    Truncated-BPTT rollout.  Accumulates loss over chunks of `chunk` steps.
    Returns total loss (scalar, already backprop'd through each chunk).
    """
    B = omega_ref.shape[0]
    x          = torch.zeros(B, 3, device=DEVICE)
    if theta0 is not None:
        x[:, 2] = theta0                        # set random initial rod angle
    integral_e = torch.zeros(B, device=DEVICE)
    prev_e     = torch.zeros(B, device=DEVICE)

    loss_start = int(T * LOSS_START)
    total_loss = torch.tensor(0.0, device=DEVICE)
    n_loss     = 0

    for t in range(T):
        # chunk boundary: detach to truncate gradient
        if t > 0 and t % chunk == 0:
            x          = x.detach()
            integral_e = integral_e.detach()

        omega = x[:, 1]
        theta = x[:, 2]

        e         = omega_ref - omega
        delta_e   = e - prev_e                          # raw Δe, NOT /dt
        integral_e = torch.clamp(
            integral_e + e * dt, -INT_CLAMP, INT_CLAMP
        )

        voltage = ctrl(e, delta_e, integral_e, omega_ref, theta)
        x       = motor.step_rk4(x, voltage, dt)

        if t >= loss_start:
            err = x[:, 1] - omega_ref
            total_loss = total_loss + (err ** 2).mean()
            n_loss += 1

        prev_e = e.detach()   # finite-diff only, no gradient through time

    return total_loss / max(n_loss, 1)


def train():
    SAVE_DIR.mkdir(parents=True, exist_ok=True)

    motor = MotorDynamicsTorch(PARAMS, pendulum=ROD, device=DEVICE)
    ctrl  = NNSpeedController(V_max=PARAMS.V_max, hidden=HIDDEN).to(DEVICE)

    opt       = optim.Adam(ctrl.parameters(), lr=LR)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=N_EPISODES, eta_min=5e-5)

    losses = []
    print(f"Training {N_EPISODES} episodes | batch={BATCH} | T={T_STEPS} | TBPTT={TBPTT}")
    print(f"RK4 integration | LR={LR} cosine → 5e-5")

    for ep in range(N_EPISODES):
        omega_ref = torch.FloatTensor(BATCH).uniform_(-OMEGA_MAX, OMEGA_MAX)
        theta0    = torch.FloatTensor(BATCH).uniform_(-torch.pi, torch.pi)

        loss = rollout_tbptt(ctrl, motor, omega_ref, T_STEPS, DT, TBPTT, theta0=theta0)

        opt.zero_grad()
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(ctrl.parameters(), 0.5)
        opt.step()
        scheduler.step()

        loss_val = loss.item()
        losses.append(loss_val if np.isfinite(loss_val) else float("nan"))

        if (ep + 1) % 100 == 0:
            recent = [v for v in losses[-100:] if np.isfinite(v)]
            rmse   = np.sqrt(np.mean(recent)) if recent else float("nan")
            print(f"  ep {ep+1:4d}/{N_EPISODES}  loss={loss_val:.4f}  "
                  f"100-ep RMSE={rmse:.4f}  grad={grad_norm:.3f}  "
                  f"lr={scheduler.get_last_lr()[0]:.2e}")

    torch.save(ctrl.state_dict(), MODEL_PATH)
    print(f"\nModel saved: {MODEL_PATH}")

    # loss curve
    valid = [(i, v) for i, v in enumerate(losses) if np.isfinite(v)]
    if valid:
        xs, ys = zip(*valid)
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.semilogy(xs, ys, lw=0.8)
        ax.set_xlabel("Episode")
        ax.set_ylabel("MSE loss")
        ax.set_title("NN Controller Training Loss")
        ax.grid(True, which="both", lw=0.4)
        fig.tight_layout()
        fig.savefig(SAVE_DIR / "train_loss.png", dpi=150)
        print(f"Loss curve: {SAVE_DIR / 'train_loss.png'}")

    return ctrl, losses


if __name__ == "__main__":
    train()
