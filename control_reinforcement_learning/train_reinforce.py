"""
REINFORCE (vanilla policy gradient) speed controller for BDC motor + rod.

Key difference from model-based BPTT
--------------------------------------
BPTT  : gradients flow THROUGH the simulation (requires differentiable physics).
REINFORCE : simulation is a BLACK BOX — only log π(a|s) needs gradients.
            Can be used on real hardware without knowing the physics.

Algorithm (per update)
----------------------
1. Collect B episodes with stochastic policy π_θ(V | s)
2. Compute discounted returns G_t = Σ γ^k r_{t+k}
3. Subtract baseline b (mean return) to reduce variance
4. Policy gradient: ∇θ J = E[ Σ_t ∇θ log π_θ(a_t|s_t) · (G_t − b) ]
5. Adam step on policy parameters

Run:
    cd /Users/pasorn/Desktop/Xian-Jiaotong/class/AI-Electrical-2026
    /opt/anaconda3/envs/ai-electrical/bin/python -m control_reinforcement_learning.train_reinforce
"""

from __future__ import annotations
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Normal
import torch.optim as optim
import matplotlib.pyplot as plt

from utils.motor import BDCMotor, BDCMotorParams, PendulumParams

# ---------------------------------------------------------------------------
# System parameters (match experiments.py)
# ---------------------------------------------------------------------------
PARAMS = BDCMotorParams(R=3.0, L=4e-3, Kt=0.05, Kb=0.05,
                        J=7.4e-5, B=0.005, V_max=12.0)
ROD    = PendulumParams(m=0.05, l=0.1, g=9.81)

# ---------------------------------------------------------------------------
# Hyperparameters
# ---------------------------------------------------------------------------
DT          = 1e-3
T_STEPS     = 300       # steps per episode
B_EPISODES  = 16        # episodes per gradient update (reduces variance)
N_UPDATES   = 3000      # total gradient updates
LR          = 3e-4      # Adam LR (lower than BPTT — RL is noisier)
GAMMA       = 0.99      # discount factor
HIDDEN      = 64
OMEGA_MAX   = 15.0      # rad/s training range
LOSS_START  = 0.25      # fraction of episode to skip (startup transient)
ENTROPY_COEF = 0.01     # entropy bonus — prevents premature policy collapse
INT_CLAMP   = 15.0

SAVE_DIR   = Path(__file__).parent / "results"
MODEL_PATH = SAVE_DIR / "rl_policy.pt"


# ---------------------------------------------------------------------------
# Environment (numpy black box — no PyTorch, no grad)
# ---------------------------------------------------------------------------
class MotorEnv:
    """
    Single-episode BDC motor environment.

    Observation (6): [e/15, Δe/1, ∫e/15, ω_ref/15, sin(θ), cos(θ)]
    Action      (1): voltage V ∈ [−V_max, V_max]
    Reward          : −e²  (negative squared tracking error)
    """

    def __init__(
        self,
        omega_ref: float,
        theta0:    float = 0.0,
    ):
        self.motor     = BDCMotor(PARAMS, pendulum=ROD)
        self.omega_ref = omega_ref
        self.theta0    = theta0
        self._integral = 0.0
        self._prev_e   = 0.0
        self._t        = 0

    def reset(self) -> np.ndarray:
        self.motor.reset(theta0=self.theta0)
        self._integral = 0.0
        self._prev_e   = 0.0
        self._t        = 0
        return self._obs()

    def step(self, voltage: float):
        voltage = float(np.clip(voltage, -PARAMS.V_max, PARAMS.V_max))
        state   = self.motor.step(DT, voltage)

        e              = self.omega_ref - state.omega
        self._integral = float(np.clip(self._integral + e * DT, -INT_CLAMP, INT_CLAMP))
        reward         = -(e ** 2)
        self._prev_e   = e
        self._t       += 1
        done           = self._t >= T_STEPS

        return self._obs(), float(reward), done

    def _obs(self) -> np.ndarray:
        s       = self.motor.state
        e       = self.omega_ref - s.omega
        delta_e = e - self._prev_e
        return np.array([
            e            / 15.0,
            delta_e      / 1.0,
            self._integral / 15.0,
            self.omega_ref / 15.0,
            np.sin(s.theta),
            np.cos(s.theta),
        ], dtype=np.float32)


# ---------------------------------------------------------------------------
# Stochastic policy: Gaussian π(V | s) = N(μ(s), σ)
# ---------------------------------------------------------------------------
class GaussianPolicy(nn.Module):
    """
    MLP outputs mean voltage μ(s).
    Shared learnable log_std (not state-dependent — simpler, good for control).

    At training: sample V ~ N(μ, σ) for exploration.
    At test:     use μ directly (deterministic greedy).
    """

    def __init__(self, V_max: float = 12.0, hidden: int = 64):
        super().__init__()
        self.V_max = V_max
        self.net = nn.Sequential(
            nn.Linear(6, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden), nn.Tanh(),
            nn.Linear(hidden, 1),
        )
        # log_std: start at 0.5 → std ≈ 1.6 V (reasonable exploration)
        self.log_std = nn.Parameter(torch.tensor(0.5))

        nn.init.uniform_(self.net[-1].weight, -0.01, 0.01)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, obs: torch.Tensor):
        """Returns (mean, std) — both tensors."""
        mu  = torch.tanh(self.net(obs)) * self.V_max
        std = torch.exp(self.log_std).clamp(0.05, 3.0)
        return mu, std

    def sample(self, obs: torch.Tensor):
        """Sample action and return (action, log_prob, entropy)."""
        mu, std  = self(obs)
        dist     = Normal(mu, std)
        action   = dist.sample()
        log_prob = dist.log_prob(action).squeeze(-1)   # (B,)
        entropy  = dist.entropy().squeeze(-1)           # (B,)
        action   = torch.clamp(action.squeeze(-1), -self.V_max, self.V_max)
        return action, log_prob, entropy

    def mean_action(self, obs: np.ndarray) -> float:
        """Deterministic inference (no sampling)."""
        with torch.no_grad():
            t   = torch.FloatTensor(obs).unsqueeze(0)
            mu, _ = self(t)
            return float(torch.clamp(mu, -self.V_max, self.V_max).item())


# ---------------------------------------------------------------------------
# Episode collection
# ---------------------------------------------------------------------------
def collect_episodes(policy: GaussianPolicy, n: int):
    """
    Run n episodes with stochastic policy.
    Returns lists of per-step tensors across all episodes.
    """
    all_log_probs = []
    all_returns   = []
    all_entropies = []
    all_rewards   = []   # for logging

    loss_start_step = int(T_STEPS * LOSS_START)

    for _ in range(n):
        omega_ref = float(np.random.uniform(-OMEGA_MAX, OMEGA_MAX))
        theta0    = float(np.random.uniform(-np.pi, np.pi))

        env = MotorEnv(omega_ref, theta0)
        obs = env.reset()

        ep_log_probs = []
        ep_entropies = []
        ep_rewards   = []

        for t in range(T_STEPS):
            obs_t              = torch.FloatTensor(obs).unsqueeze(0)
            action, lp, ent    = policy.sample(obs_t)
            obs, reward, done  = env.step(action.item())

            ep_log_probs.append(lp.squeeze())
            ep_entropies.append(ent.squeeze())
            ep_rewards.append(reward if t >= loss_start_step else 0.0)

        # discounted returns G_t
        G        = 0.0
        returns  = []
        for r in reversed(ep_rewards):
            G = r + GAMMA * G
            returns.insert(0, G)

        all_log_probs.append(torch.stack(ep_log_probs))     # (T,)
        all_entropies.append(torch.stack(ep_entropies))     # (T,)
        all_returns.append(torch.tensor(returns, dtype=torch.float32))
        all_rewards.append(sum(ep_rewards))

    log_probs = torch.cat(all_log_probs)    # (n*T,)
    entropies = torch.cat(all_entropies)
    returns   = torch.cat(all_returns)

    # normalise returns (variance reduction)
    returns = (returns - returns.mean()) / (returns.std() + 1e-8)

    return log_probs, returns, entropies, float(np.mean(all_rewards))


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------
def train():
    SAVE_DIR.mkdir(parents=True, exist_ok=True)

    policy = GaussianPolicy(V_max=PARAMS.V_max, hidden=HIDDEN)
    opt    = optim.Adam(policy.parameters(), lr=LR)

    reward_hist = []
    loss_hist   = []

    print(f"REINFORCE | updates={N_UPDATES} | B={B_EPISODES} ep/update "
          f"| T={T_STEPS} steps | γ={GAMMA}")
    print(f"Total episodes: {N_UPDATES * B_EPISODES:,}")

    for update in range(N_UPDATES):
        log_probs, returns, entropies, mean_reward = collect_episodes(policy, B_EPISODES)

        # REINFORCE loss with entropy bonus
        pg_loss      = -(log_probs * returns).mean()
        entropy_loss = -ENTROPY_COEF * entropies.mean()
        loss         = pg_loss + entropy_loss

        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(policy.parameters(), 0.5)
        opt.step()

        reward_hist.append(mean_reward)
        loss_hist.append(loss.item())

        if (update + 1) % 100 == 0:
            avg_r = np.mean(reward_hist[-100:])
            print(f"  update {update+1:4d}/{N_UPDATES}  "
                  f"avg_reward={avg_r:.4f}  "
                  f"loss={loss.item():.4f}  "
                  f"std={policy.log_std.exp().item():.3f}")

    torch.save(policy.state_dict(), MODEL_PATH)
    print(f"\nPolicy saved: {MODEL_PATH}")

    _plot_training(reward_hist)
    return policy, reward_hist


def _plot_training(reward_hist):
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(reward_hist, lw=0.6, alpha=0.5, color="C0", label="per-update reward")

    # smoothed
    w = 50
    smooth = np.convolve(reward_hist, np.ones(w)/w, mode="valid")
    ax.plot(range(w-1, len(reward_hist)), smooth, lw=1.5, color="C1", label=f"{w}-update avg")

    ax.set_xlabel("Update")
    ax.set_ylabel("Mean episode reward (−MSE·steps)")
    ax.set_title("REINFORCE Training — BDC Motor Speed Control")
    ax.legend(fontsize=8)
    ax.grid(True, lw=0.4)
    fig.tight_layout()
    fig.savefig(SAVE_DIR / "rl_reward.png", dpi=150)
    print(f"Reward curve: {SAVE_DIR / 'rl_reward.png'}")


if __name__ == "__main__":
    train()
