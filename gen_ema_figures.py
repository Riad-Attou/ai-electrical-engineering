#!/usr/bin/env python3
"""Generate EMA variance and phase-shift figures for the presentation slide."""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from utils.baselines import optimize_ema
from utils.traj import MotorSplit

FIG = Path("figures")
FIG.mkdir(exist_ok=True)

dt = 1e-3  # 1 ms — shared project sampling interval

# Load split and find the tuned alpha
split = MotorSplit.load("data/rod_split.npz")
best_alpha, _ = optimize_ema(split)
print(f"Tuned EMA alpha = {best_alpha:.3f}")

# Representative alphas to compare
ALPHAS  = [0.80, 0.90, 0.95, best_alpha]
COLORS  = ["#3498db", "#e07b00", "#1abc9c", "#e74c3c"]
LABELS  = [f"α = {a:.2f}" if a != best_alpha else f"α = {a:.3f}  (tuned)" for a in ALPHAS]

# Frequency axis (0 → Nyquist)
f = np.linspace(0, 500, 8000)
omega = 2 * np.pi * f * dt          # digital frequency [rad]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
fig.patch.set_facecolor("#FBFAF6")
for ax in (ax1, ax2):
    ax.set_facecolor("#FBFAF6")

for a, c, lbl in zip(ALPHAS, COLORS, LABELS):
    lw = 2.6 if a == best_alpha else 1.6
    zord = 5 if a == best_alpha else 3

    # Magnitude response |H(e^{jω})|
    denom = 1 + a**2 - 2 * a * np.cos(omega)
    H_mag = (1 - a) / np.sqrt(denom)

    # Output variance = |H|² × input variance → relative std = |H|
    ax1.plot(f, H_mag, color=c, lw=lw, label=lbl, zorder=zord)

    # Group delay in ms  τ(ω) = α(1−α)/(1+α²−2α·cos ω)  ×  dt × 1000
    gd_ms = a * (1 - a) / denom * dt * 1000
    ax2.plot(f, gd_ms, color=c, lw=lw, label=lbl, zorder=zord)

# DC delay annotation for tuned alpha
dc_delay = best_alpha / (1 - best_alpha) * dt * 1000
ax2.axhline(dc_delay, color="#e74c3c", ls=":", lw=1.2, zorder=2)
ax2.text(105, dc_delay + 0.4, f"{dc_delay:.0f} ms", color="#e74c3c", fontsize=10)

# -3 dB reference line
ax1.axhline(1 / np.sqrt(2), color="#888", ls="--", lw=1.0, zorder=1, label="−3 dB threshold")

# Panel 1 — variance / magnitude
ax1.set_xlabel("Frequency  [Hz]", fontsize=12)
ax1.set_ylabel("Output / input amplitude  |H(f)|", fontsize=12)
ax1.set_title("Variance reduction\nHigher α → stronger low-pass, less noise", fontsize=12, fontweight="bold")
ax1.set_xlim(0, 200)
ax1.set_ylim(0, 1.05)
ax1.legend(fontsize=10, framealpha=0.9)
ax1.grid(True, alpha=0.35, color="#ccc")

# Panel 2 — group delay
ax2.set_xlabel("Frequency  [Hz]", fontsize=12)
ax2.set_ylabel("Group delay  [ms]", fontsize=12)
ax2.set_title("Phase shift (group delay)\nHigher α → more lag at every frequency", fontsize=12, fontweight="bold")
ax2.set_xlim(0, 200)
ax2.legend(fontsize=10, framealpha=0.9)
ax2.grid(True, alpha=0.35, color="#ccc")

plt.tight_layout(pad=1.8)
out = FIG / "ema_properties.png"
plt.savefig(out, dpi=160, bbox_inches="tight")
plt.close(fig)
print(f"saved -> {out}")
