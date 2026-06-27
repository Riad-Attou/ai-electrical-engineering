"""
Machine-learning validation metrics for the residual controllers.

This is a *regression / continuous-control* problem (the policy outputs a
continuous action; performance is the tracking error e = omega - omega_ref).
Classification metrics (accuracy, F1) therefore do not apply. We report MAE,
MSE, RMSE, a tracking-R^2 (defined only for time-varying references), and an
in-tolerance-band fraction, plus two diagnostics:
  - a parity plot (achieved vs reference speed), the regression analogue of an
    "accuracy plot";
  - the tracking-error autocorrelation (the "covariance function"): a whiteness
    check -- structure left in the error means exploitable bias/lag remains.

Everything is computed on the already-trained policies (no retraining).
Run: python residual_reinforcement_learning/metrics.py
"""
import sys
import pathlib

ROOT = pathlib.Path(__file__).resolve().parent.parent
HERE = pathlib.Path(__file__).resolve().parent
for _p in (str(HERE), str(ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from benchmark import simulate, build_controllers, RESULTS_DIR, DT
from reference import constant_reference, quintic_reference
from generalization import HELDOUT

TOL = 0.5   # rad/s, in-tolerance-band threshold


# ---------------------------------------------------------------------------
# Pure metric functions (regression metrics on the tracking error)
# ---------------------------------------------------------------------------
def mae(omega, ref) -> float:
    return float(np.mean(np.abs(np.asarray(omega) - np.asarray(ref))))


def mse(omega, ref) -> float:
    e = np.asarray(omega) - np.asarray(ref)
    return float(np.mean(e ** 2))


def rmse(omega, ref) -> float:
    return float(np.sqrt(mse(omega, ref)))


def r2_tracking(omega, ref) -> float:
    """
    Coefficient of determination of the tracked signal vs the reference.
    Returns nan for (near-)constant references, where target variance ~ 0 and
    R^2 is undefined.
    """
    omega = np.asarray(omega, dtype=float)
    ref = np.asarray(ref, dtype=float)
    ss_res = float(np.sum((omega - ref) ** 2))
    ss_tot = float(np.sum((ref - ref.mean()) ** 2))
    if ss_tot < 1e-9:
        return float("nan")
    return 1.0 - ss_res / ss_tot


def in_band(omega, ref, tol: float = TOL) -> float:
    return float(np.mean(np.abs(np.asarray(omega) - np.asarray(ref)) < tol))


def error_autocorr(e, max_lag: int = 400) -> np.ndarray:
    """Normalised autocovariance of the de-meaned error, lags 0..max_lag."""
    e = np.asarray(e, dtype=float) - float(np.mean(e))
    n = len(e)
    denom = float(np.dot(e, e))
    if denom < 1e-12:
        return np.zeros(max_lag + 1)
    ac = np.array([float(np.dot(e[:n - k], e[k:])) for k in range(max_lag + 1)])
    return ac / denom


# ---------------------------------------------------------------------------
# References, scorecard and diagnostic plots (parity + error autocorrelation)
# ---------------------------------------------------------------------------
def all_references() -> dict:
    refs = {
        "const 10": constant_reference(10.0),
        "quintic 0->10": quintic_reference(0.0, 10.0, 0.5, 3.5),
    }
    refs.update(HELDOUT)
    return refs


def scorecard(ctrls, refs) -> dict:
    print(f"\nML scorecard (mean over {len(refs)} references; tol={TOL} rad/s)")
    hdr = ["controller", "MAE", "MSE", "RMSE", "R2(var)", "in-band%"]
    print("  ".join(f"{h:>14}" for h in hdr))
    rows = {}
    for cname, ctrl in ctrls.items():
        maes, mses, rmses, r2s, bands = [], [], [], [], []
        for ref_fn in refs.values():
            d = simulate(ctrl, ref_fn)
            maes.append(mae(d["omega"], d["ref"]))
            mses.append(mse(d["omega"], d["ref"]))
            rmses.append(rmse(d["omega"], d["ref"]))
            r2 = r2_tracking(d["omega"], d["ref"])
            if not np.isnan(r2):
                r2s.append(r2)
            bands.append(in_band(d["omega"], d["ref"]))
        r = dict(MAE=float(np.mean(maes)), MSE=float(np.mean(mses)),
                 RMSE=float(np.mean(rmses)),
                 R2=float(np.mean(r2s)) if r2s else float("nan"),
                 band=100.0 * float(np.mean(bands)))
        rows[cname] = r
        print("  ".join([f"{cname:>14}", f"{r['MAE']:>14.4f}", f"{r['MSE']:>14.4f}",
                         f"{r['RMSE']:>14.4f}", f"{r['R2']:>14.4f}", f"{r['band']:>13.1f}%"]))
    return rows


def parity_plot(ctrls, refs, ref_name: str = "quintic 0->10") -> None:
    ref_fn = refs[ref_name]
    n = len(ctrls)
    fig, axes = plt.subplots(1, n, figsize=(4 * n, 4), squeeze=False)
    for ax, (cname, ctrl) in zip(axes[0], ctrls.items()):
        d = simulate(ctrl, ref_fn)
        ax.scatter(d["ref"], d["omega"], s=3, alpha=0.3)
        lim = [float(min(d["ref"].min(), d["omega"].min())),
               float(max(d["ref"].max(), d["omega"].max()))]
        ax.plot(lim, lim, "k--", lw=1)
        ax.set_title(f"{cname}\nR2={r2_tracking(d['omega'], d['ref']):.3f}", fontsize=9)
        ax.set_xlabel("reference omega* (rad/s)")
        ax.set_ylabel("achieved omega (rad/s)")
        ax.grid(True, lw=0.3)
    fig.suptitle(f"Parity plot — {ref_name}")
    fig.tight_layout()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out = RESULTS_DIR / "metrics_parity.png"
    fig.savefig(out, dpi=150)
    print(f"parity plot -> {out}")


def autocorr_plot(ctrls, refs, ref_name: str = "quintic 0->10", max_lag: int = 400) -> None:
    ref_fn = refs[ref_name]
    lags = np.arange(max_lag + 1) * DT   # seconds
    fig, ax = plt.subplots(figsize=(8, 4))
    for cname, ctrl in ctrls.items():
        d = simulate(ctrl, ref_fn)
        ac = error_autocorr(d["omega"] - d["ref"], max_lag)
        ax.plot(lags, ac, lw=1, label=cname)
    ax.axhline(0.0, color="gray", lw=0.5)
    ax.set_xlabel("lag (s)")
    ax.set_ylabel("error autocorrelation")
    ax.set_title(f"Tracking-error autocorrelation — {ref_name}")
    ax.legend(fontsize=8)
    ax.grid(True, lw=0.3)
    fig.tight_layout()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out = RESULTS_DIR / "metrics_autocorr.png"
    fig.savefig(out, dpi=150)
    print(f"autocorr plot -> {out}")


# ---------------------------------------------------------------------------
# Command-line entry point
# ---------------------------------------------------------------------------
def run() -> None:
    ctrls = build_controllers()
    refs = all_references()
    scorecard(ctrls, refs)
    parity_plot(ctrls, refs)
    autocorr_plot(ctrls, refs)


if __name__ == "__main__":
    run()
