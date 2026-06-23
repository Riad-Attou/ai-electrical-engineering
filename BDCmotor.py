"""
Servo dataset preparation — run this once before training.

Reads the pre-recorded Simulink CSVs (chirp + multisine), resamples to a
uniform 1 ms grid, and saves a ready-to-use split to data/servo_split.npz.

Also generates a quick overview figure.

Usage
-----
python BDCmotor.py
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from utils.servo import GEAR_RATIO, build_servo_split

TRAIN_CSV = Path("test_simulated_dataset/ml2_train_set.csv")
TEST_CSV  = Path("test_simulated_dataset/ml2_test_set.csv")
OUT_NPZ   = Path("data/servo_split.npz")
FIGURES   = Path("figures")
FIGURES.mkdir(exist_ok=True)
Path("data").mkdir(exist_ok=True)


def main() -> None:
    print("Loading and resampling CSVs …")
    split = build_servo_split(TRAIN_CSV, TEST_CSV, val_frac=0.30, dt=1e-3)
    split.save(OUT_NPZ)

    T_tr = len(split.train_t)
    T_va = len(split.val_t)
    T_te = len(split.test_t)

    print(f"\nSplit saved → {OUT_NPZ}")
    print(f"  Train  : {T_tr:6d} steps  ({split.train_t[-1]:.1f} s)  "
          f"  enc_m ∈ [{split.train_enc_m.min():.2f}, {split.train_enc_m.max():.2f}] rad")
    print(f"  Val    : {T_va:6d} steps  ({split.val_t[-1] - split.val_t[0]:.1f} s)  "
          f"  enc_m ∈ [{split.val_enc_m.min():.2f}, {split.val_enc_m.max():.2f}] rad")
    print(f"  Test   : {T_te:6d} steps  ({split.test_t[-1]:.1f} s)  "
          f"  enc_m ∈ [{split.test_enc_m.min():.2f}, {split.test_enc_m.max():.2f}] rad")

    # Rigid-coupling sanity check on test set
    rigid_err = split.test_enc_m / GEAR_RATIO - split.test_theta_l
    rigid_rmse = float(np.sqrt(np.mean(rigid_err ** 2)))
    print(f"\nRigid-coupling baseline (enc_m/N) RMSE on test : "
          f"{rigid_rmse*1e3:.3f} mrad  ({np.degrees(rigid_rmse):.4f} °)")
    print(f"Max backlash gap (motor side) : {GEAR_RATIO * 0.02:.2f} rad = "
          f"{np.degrees(GEAR_RATIO * 0.02):.1f} °")

    # Overview figure
    _plot_overview(split)


def _plot_overview(split) -> None:
    fig, axes = plt.subplots(3, 1, figsize=(13, 8), sharex=False)

    for ax, t, pwm, enc_m, theta_l, title in [
        (axes[0], split.train_t, split.train_pwm,
         split.train_enc_m, split.train_theta_l, "Training data (chirp)"),
        (axes[1], split.val_t,   split.val_pwm,
         split.val_enc_m,   split.val_theta_l,   "Validation data (chirp, tail)"),
        (axes[2], split.test_t,  split.test_pwm,
         split.test_enc_m,  split.test_theta_l,  "Test data (multisine — unseen)"),
    ]:
        ax2 = ax.twinx()
        ax.plot(t, np.degrees(enc_m / split.N), color="C0", lw=0.8, alpha=0.7,
                label="enc_m/N (rigid)")
        ax.plot(t, np.degrees(theta_l), color="C1", lw=1.2, ls="--",
                label="theta_l (true)")
        ax2.plot(t, pwm, color="C3", lw=0.6, alpha=0.5, label="pwm")
        ax.set_ylabel("Angle (°)")
        ax2.set_ylabel("PWM", color="C3")
        ax2.tick_params(axis="y", labelcolor="C3")
        ax.set_title(title)
        ax.legend(loc="upper left", fontsize=8)
        ax.grid(True, alpha=0.3)

    axes[-1].set_xlabel("Time (s)")
    plt.tight_layout()
    out = FIGURES / "data_overview.png"
    plt.savefig(out, dpi=150)
    print(f"\nFigure saved → {out}")
    plt.show()


if __name__ == "__main__":
    main()
