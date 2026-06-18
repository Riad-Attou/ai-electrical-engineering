"""
Steps 2-5 — Plot, invert and fit polynomial regression.

  Step 2: plot  x=voltage, y=speed          (forward map)
  Step 3: plot  x=speed,   y=voltage        (inverted map)
  Step 4: polynomial regression on inverted data
  Step 5: print equation  V = c0 + c1·ω + c2·ω² + ... + cn·ωⁿ

Output: nn/data/poly_coeffs.npz  (coefficients, highest power first)
Run:    python -m nn.poly_fit   (from project root)
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import matplotlib.pyplot as plt

DATA_PATH   = Path(__file__).parent / "data" / "poly_data.npz"
COEFFS_PATH = Path(__file__).parent / "data" / "poly_coeffs.npz"
DEGREE      = 2  # polynomial degree


def fit():
    d       = np.load(DATA_PATH)
    voltage = d["voltage"].astype(np.float64)
    omega   = d["omega"].astype(np.float64)

    # ---- average omega per voltage level ----
    unique_voltages = np.unique(voltage)
    omega_mean      = np.array([omega[voltage == v].mean() for v in unique_voltages])

    # ---- Step 2: forward map V → ω  (scatter only, no line) ----
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    axes[0].scatter(voltage, omega, color="C0", s=8, alpha=0.5)
    axes[0].set_xlabel("Voltage (V)")
    axes[0].set_ylabel("Speed (rad/s)")
    axes[0].set_title("Step 2 — Forward map  V → ω")
    axes[0].axhline(0, color="gray", lw=0.8, ls="--")
    axes[0].axvline(0, color="gray", lw=0.8, ls="--")
    axes[0].grid(True)

    # ---- Step 3 & 4: invert, fit polynomial ω → V  (no intercept) ----
    # fit on averaged points, not raw scatter
    X = np.column_stack([omega_mean ** p for p in range(1, DEGREE + 1)])
    c_low, _, _, _ = np.linalg.lstsq(X, unique_voltages, rcond=None)  # [c1, c2, ..., cN]

    # Store as np.polyval-compatible array (highest power first, constant = 0)
    coeffs = np.append(c_low[::-1], 0.0)

    def poly(w):
        return np.polyval(coeffs, w)

    omega_dense = np.linspace(omega_mean.min(), omega_mean.max(), 300)
    v_fit       = poly(omega_dense)

    axes[1].scatter(omega_mean, unique_voltages, color="C1", zorder=3, label="averaged data")
    axes[1].plot(omega_dense, v_fit, color="C3", lw=2,
                 label=f"poly degree {DEGREE}, zero-bias")
    axes[1].set_xlabel("Speed (rad/s)")
    axes[1].set_ylabel("Voltage (V)")
    axes[1].set_title("Step 3 & 4 — Inverse map  ω → V  (zero-bias fit)")
    axes[1].axhline(0, color="gray", lw=0.8, ls="--")
    axes[1].axvline(0, color="gray", lw=0.8, ls="--")
    axes[1].legend()
    axes[1].grid(True)

    plt.tight_layout()
    plt.savefig(Path(__file__).parent / "data" / "poly_fit.png", dpi=150)
    plt.show()

    # ---- Step 5: print equation ----
    print(f"\nPolynomial degree {DEGREE}, zero-bias  (V = f(ω), no constant):")
    terms = []
    for power, c in enumerate(c_low, start=1):
        if power == 1:
            terms.append(f"{c:+.6f}·ω")
        else:
            terms.append(f"{c:+.6f}·ω^{power}")
    print("  V = " + "  ".join(terms))

    # goodness of fit (on averaged points)
    v_pred = poly(omega_mean)
    ss_res = np.sum((unique_voltages - v_pred) ** 2)
    ss_tot = np.sum((unique_voltages - unique_voltages.mean()) ** 2)
    r2     = 1 - ss_res / ss_tot
    print(f"\n  R² = {r2:.6f}   (1.0 = perfect)")

    # ---- save ----
    COEFFS_PATH.parent.mkdir(parents=True, exist_ok=True)
    np.savez(COEFFS_PATH,
             coeffs=coeffs,
             degree=np.array(DEGREE),
             omega_min=np.array(omega_mean.min()),
             omega_max=np.array(omega_mean.max()))
    print(f"\nCoefficients saved → {COEFFS_PATH}")
    return coeffs


if __name__ == "__main__":
    fit()
