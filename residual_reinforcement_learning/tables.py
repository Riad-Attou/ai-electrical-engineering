"""
Multi-metric comparison tables (Markdown + CSV).

Defines a TRAIN reference set (sampled from the training distribution) and a
TEST reference set (the held-out references) and reports the selected
regression metrics -- MAE, MSE, RMSE, R2, in-band% (not just RMSE) -- for:
  - the base residual-PI network on train vs test  -> table_base_nn.{md,csv};
  - residual-PI at different network depths on the test set, mean +/- std over
    seeds, read from results/sweep_metrics.json  -> table_depths.{md,csv}.

Run: python residual_reinforcement_learning/tables.py
"""
import sys
import csv
import json
import pathlib

ROOT = pathlib.Path(__file__).resolve().parent.parent
HERE = pathlib.Path(__file__).resolve().parent
for _p in (str(HERE), str(ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import numpy as np
from benchmark import build_controllers, simulate, RESULTS_DIR
from reference import RandomReference
from generalization import HELDOUT
from metrics import mae, mse, rmse, r2_tracking, in_band

METRIC_KEYS = ["MAE", "MSE", "RMSE", "R2", "in_band%"]


# ---------------------------------------------------------------------------
# Train / test reference sets and per-controller metric evaluation
# ---------------------------------------------------------------------------
def train_references(n: int = 12, seed: int = 123) -> dict:
    """References sampled from the training distribution (proxy for 'train set')."""
    rng = np.random.default_rng(seed)
    return {f"train_{i}": RandomReference(rng, 5.0) for i in range(n)}


def test_references() -> dict:
    """The held-out references ('test set')."""
    return dict(HELDOUT)


def eval_controller(ctrl, refs: dict) -> dict:
    """Mean of each metric over a reference set (R2 over time-varying refs only)."""
    maes, mses, rmses, r2s, bands = [], [], [], [], []
    for ref in refs.values():
        d = simulate(ctrl, ref)
        maes.append(mae(d["omega"], d["ref"]))
        mses.append(mse(d["omega"], d["ref"]))
        rmses.append(rmse(d["omega"], d["ref"]))
        r2 = r2_tracking(d["omega"], d["ref"])
        if not np.isnan(r2):
            r2s.append(r2)
        bands.append(in_band(d["omega"], d["ref"]))
    return {"MAE": float(np.mean(maes)), "MSE": float(np.mean(mses)),
            "RMSE": float(np.mean(rmses)),
            "R2": float(np.mean(r2s)) if r2s else float("nan"),
            "in_band%": 100.0 * float(np.mean(bands))}


# ---------------------------------------------------------------------------
# Markdown / CSV writers
# ---------------------------------------------------------------------------
def to_markdown(headers: list, rows: list) -> str:
    out = "| " + " | ".join(headers) + " |\n"
    out += "| " + " | ".join("---" for _ in headers) + " |\n"
    for r in rows:
        out += "| " + " | ".join(str(c) for c in r) + " |\n"
    return out


def write_csv(path: pathlib.Path, headers: list, rows: list) -> None:
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(headers)
        w.writerows(rows)


def _fmt(m: dict) -> list:
    return [f"{m['MAE']:.4f}", f"{m['MSE']:.4f}", f"{m['RMSE']:.4f}",
            f"{m['R2']:.4f}", f"{m['in_band%']:.1f}"]


# ---------------------------------------------------------------------------
# Comparison tables (base-NN train-vs-test, depth sweep)
# ---------------------------------------------------------------------------
def base_nn_table(ctrl_name: str = "residual-PI") -> None:
    """Base NN evaluated on train vs test set, all metrics -> md + csv."""
    ctrls = build_controllers()
    if ctrl_name not in ctrls:
        print(f"[skip] {ctrl_name} not available (no trained model)")
        return
    ctrl = ctrls[ctrl_name]
    headers = ["set"] + METRIC_KEYS
    rows = []
    for set_name, refs in [("train", train_references()), ("test", test_references())]:
        rows.append([set_name] + _fmt(eval_controller(ctrl, refs)))
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    md = f"# Base NN ({ctrl_name}) — train vs test set\n\n" + to_markdown(headers, rows)
    (RESULTS_DIR / "table_base_nn.md").write_text(md)
    write_csv(RESULTS_DIR / "table_base_nn.csv", headers, rows)
    print(md)
    print(f"written -> {RESULTS_DIR / 'table_base_nn.md'} (+ .csv)")


def base_nn_table_from_sweep(sweep_json: pathlib.Path | None = None,
                             base_depth: str = "[256,256]") -> None:
    """Base NN (a chosen depth) on train vs test from the sweep JSON, mean over seeds."""
    sweep_json = sweep_json or (RESULTS_DIR / "sweep_metrics.json")
    if not sweep_json.exists():
        print(f"[skip] {sweep_json} not found — run the depth sweep first")
        return
    data = json.loads(sweep_json.read_text())
    headers = ["set"] + METRIC_KEYS
    rows = []
    for set_name in ("train", "test"):
        recs = [r for r in data if r["depth"] == base_depth and r["set"] == set_name]
        if not recs:
            continue
        m = {k: float(np.nanmean([r[k] for r in recs])) for k in METRIC_KEYS}
        rows.append([set_name] + _fmt(m))
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    md = (f"# Base NN ({base_depth}, saturation-aware) — train vs test set\n\n"
          + to_markdown(headers, rows))
    (RESULTS_DIR / "table_base_nn.md").write_text(md)
    write_csv(RESULTS_DIR / "table_base_nn.csv", headers, rows)
    print(md)
    print(f"written -> {RESULTS_DIR / 'table_base_nn.md'} (+ .csv)")


def depth_table(sweep_json: pathlib.Path | None = None) -> None:
    """Depth comparison on the test set, mean +/- std over seeds, from a sweep JSON."""
    sweep_json = sweep_json or (RESULTS_DIR / "sweep_metrics.json")
    if not sweep_json.exists():
        print(f"[skip] {sweep_json} not found — run the depth sweep first")
        return
    data = [r for r in json.loads(sweep_json.read_text()) if r["set"] == "test"]
    depths = []
    for r in data:
        if r["depth"] not in depths:
            depths.append(r["depth"])
    headers = ["net_arch"] + [f"{k} (mean±std)" for k in METRIC_KEYS]
    rows = []
    for d in depths:
        cells = [d]
        for k in METRIC_KEYS:
            vals = [r[k] for r in data if r["depth"] == d and not np.isnan(r[k])]
            cells.append(f"{np.mean(vals):.4f}±{np.std(vals):.4f}" if vals else "nan")
        rows.append(cells)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    md = "# Depth comparison (residual-PI) — test set\n\n" + to_markdown(headers, rows)
    (RESULTS_DIR / "table_depths.md").write_text(md)
    write_csv(RESULTS_DIR / "table_depths.csv", headers, rows)
    print(md)
    print(f"written -> {RESULTS_DIR / 'table_depths.md'} (+ .csv)")


# ---------------------------------------------------------------------------
# Command-line entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    # base table: prefer the saturation-aware sweep if present, else the live model
    if (RESULTS_DIR / "sweep_metrics.json").exists():
        base_nn_table_from_sweep()
    else:
        base_nn_table()
    depth_table()
