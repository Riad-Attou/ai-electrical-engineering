# Kalman view: filtered curve + its "training"

**Date:** 2026-06-29
**Branch:** `riad/ai-filter-training`
**Status:** Approved (design)

## Goal

Add a standalone visualization that shows the Kalman baseline *alone* — not
buried among the neural models in `compare.py`. Two things to see:

1. **The filtered curve** — Kalman speed estimate vs noisy measurement vs true
   speed on one test trajectory, plus its error.
2. **Its "training"** — the Kalman has no gradient training; its only tuning is
   the grid search over `(Q_omega, R_var)` on the validation set
   (`optimize_kalman`). Visualize that search: a val-RMSE heatmap over the grid
   with the chosen optimum marked, plus a 1-D slice.

## Context / constraints

- Plant: shared motor + rod. Kalman uses the nominal *linear* motor model
  (no gravity term) — see `utils/baselines.py` docstring.
- Tuning protocol: tune on the **val** split, evaluate on **test** (same as
  `train.py` / `compare.py`).
- Follow existing repo conventions: argparse, `_BASE_PARAMS`, save PNGs to
  `figures/` at high DPI, reuse `_STYLE` palette and `_rpm` helper.

## Bug fix (prerequisite)

`utils/baselines.py` has an uncommitted working-tree line in `optimize_kalman`:

```python
            if best is None or rmse < best[2]:
                best = (q_diag, float(r), rmse)
                best = (q_diag * 1, float(1e-6), 0.01)   # <-- delete this
```

The second assignment overwrites every grid-search winner with a hardcoded
`R_var = 1e-6` and a fake val RMSE of `0.01`, so `optimize_kalman` never returns
the real optimum. **Delete the override line** to restore genuine tuning
(expected Kalman test RMSE ~1.32 rad/s, per the deck).

## Supporting refactor in `utils/baselines.py`

Extract the grid evaluation so the figure and the tuner share one source of
truth (no recomputed/forked grids):

```python
def kalman_val_grid(
    split: MotorSplit,
    params: BDCMotorParams,
    r_grid: np.ndarray | None = None,
    q_grid: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Validation RMSE over the (R_var x Q_omega) grid.

    Returns (r_grid, q_grid, rmse) where rmse has shape (len(q_grid), len(r_grid))
    and rmse[j, k] is the val RMSE for Q_omega=q_grid[j], R_var=r_grid[k]
    (with Q_i = 0.1 * Q_omega, matching optimize_kalman's coupling).
    """
```

- Default grids identical to today's `optimize_kalman`:
  `r_grid = np.logspace(-2.0, 1.8, 10)`, `q_grid = np.logspace(-3.0, 0.5, 8)`.
- `optimize_kalman` is rewritten to call `kalman_val_grid`, take the argmin, and
  return `((Q_i, Q_omega), best_R_var, val_rmse)` — same signature and return
  shape as today. No other caller changes (`compare.py`, `eval_ood.py`,
  `run_all_baselines` keep working unchanged).

## New file: `kalman_view.py`

Top-level script, mirroring `compare.py`'s structure.

### CLI

```
python kalman_view.py                      # traj 0, full trajectory
python kalman_view.py --traj 2 --t-end 3.0
```

- `--split` (default `data/rod_split.npz`)
- `--traj` (int, default 0)
- `--t-end` (float, optional; clip the curve in seconds)

### Flow (`main`)

1. Load `MotorSplit`, get `dt`.
2. `r_grid, q_grid, rmse = kalman_val_grid(split, _BASE_PARAMS)`.
3. `q_diag, r_var, val_rmse = optimize_kalman(split, _BASE_PARAMS)` (the optimum;
   consistent with the grid because both use the same defaults).
4. Build figure 1 (curve) and figure 2 (training). Print the chosen
   `(R_var, Q_omega)` and the test RMSE.

### Figure 1 — `figures/kalman_curve.png`

Two stacked panels sharing the x-axis (time, s), like
`_trajectory_overlay` but Kalman-only:

- Top: `Noisy`, `True`, `Kalman` in RPM (reuse `_STYLE` keys + `_rpm`).
  Title notes traj index and dt.
- Bottom: Kalman error `(kalman - true)` in RPM with a zero reference line.
- DPI 180, `bbox_inches="tight"`.

### Figure 2 — `figures/kalman_training.png`

Two side-by-side panels:

- Left: `pcolormesh`/`imshow` heatmap of `rmse` over `R_var` (x, log) and
  `Q_omega` (y, log). Colorbar labeled "Val RMSE [rad/s]". Mark the optimum
  `(r_var, q_diag[1])` with a star + annotation.
- Right: 1-D slice — val RMSE vs `R_var` at the best `Q_omega` row (log x),
  minimum marked with the same star color. Title: "Tuning slice at
  Q_omega = {best}".
- DPI 180.

## Out of scope (YAGNI)

- No wiring into `make_slides.py` (offer as a follow-up after review).
- No changes to neural / EMA / MA paths or to `compare.py`.
- No new dependencies (matplotlib + numpy + scipy already used).

## Testing / verification

- `python kalman_view.py` runs clean and writes both PNGs.
- Printed optimum is a genuine interior grid point (not pinned `R_var=1e-6`),
  and the printed test RMSE is ~1.3 rad/s — confirming the override is gone.
- Heatmap optimum marker lands on the argmin cell of the displayed grid.
