"""
Saturation-aware depth sweep for residual-PI.

Trains residual-PI (saturation-aware) at three net_arch depths over three seeds,
then evaluates every model on the train and test reference sets with all metrics
(MAE/MSE/RMSE/R2/in-band), writing results/sweep_metrics.json. tables.py turns
that into the base-NN (train vs test) and depth-comparison (test) tables.

Resumable: models are cached under models/ and results accumulate after every
run. Run in background:
    python residual_reinforcement_learning/sweep.py --no-progress
Quick pipeline check:
    python residual_reinforcement_learning/sweep.py --smoke
"""
import sys
import json
import argparse
import pathlib

ROOT = pathlib.Path(__file__).resolve().parent.parent
HERE = pathlib.Path(__file__).resolve().parent
for _p in (str(HERE), str(ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import train
from motor_env import RewardConfig
from baselines import PurePI
from residual_controller import ResidualController
from benchmark import PARAMS, KP, KI, RESULTS_DIR
import tables

DEPTHS = {"[64]": [64], "[256,256]": [256, 256], "[256,256,256]": [256, 256, 256]}
SEEDS = [0, 1, 2]
TIMESTEPS = 300_000
SAT_PENALTY = 0.05

_RESULTS = RESULTS_DIR / "sweep_metrics.json"


# ---------------------------------------------------------------------------
# Train every (depth, seed), evaluate on train + test, write metrics & tables
# ---------------------------------------------------------------------------
def _controller(model):
    return ResidualController(PurePI(KP, KI, PARAMS.V_max), model, PARAMS,
                              saturation_aware=True)


def run(timesteps: int = TIMESTEPS, depths: dict = DEPTHS, seeds=SEEDS,
        progress_bar: bool = False) -> list:
    from stable_baselines3 import SAC

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    results = json.loads(_RESULTS.read_text()) if _RESULTS.exists() else []
    done = {(r["depth"], r["seed"]) for r in results}
    train_refs = tables.train_references()
    test_refs = tables.test_references()
    rc = RewardConfig(sat_penalty=SAT_PENALTY)

    for dname, depth in depths.items():
        for seed in seeds:
            if (dname, seed) in done:
                print(f"skip {dname} seed {seed} (already done)")
                continue
            tag = f"sat_{dname.strip('[]').replace(',', '_')}_s{seed}"
            model_path = train.MODELS_DIR / f"sac_residual_pi_{tag}.zip"
            if not model_path.exists():
                train.train("pi", timesteps, seed, False, progress_bar=progress_bar,
                            net_arch=depth, tag=tag, saturation_aware=True, reward_cfg=rc)
            model = SAC.load(model_path)
            ctrl = _controller(model)
            for set_name, refs in [("train", train_refs), ("test", test_refs)]:
                m = tables.eval_controller(ctrl, refs)
                results.append(dict(depth=dname, seed=seed, set=set_name, **m))
            _RESULTS.write_text(json.dumps(results, indent=2))
            te = next(r for r in results if r["depth"] == dname and r["seed"] == seed
                      and r["set"] == "test")
            print(f"{dname} seed {seed}: test RMSE = {te['RMSE']:.4f}")

    print(f"\nsweep complete -> {_RESULTS}")
    tables.base_nn_table_from_sweep(_RESULTS)
    tables.depth_table(_RESULTS)
    _depth_figure(results, depths)
    return results


# ---------------------------------------------------------------------------
# Depth-ablation figure: held-out (test) RMSE, mean +/- std over seeds
# ---------------------------------------------------------------------------
def _depth_figure(results: list, depths: dict,
                  classical_rmse: float = 0.222) -> None:
    import numpy as np
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    labels, means, stds = [], [], []
    for dname in depths:
        vals = [r["RMSE"] for r in results
                if r["depth"] == dname and r["set"] == "test"]
        if not vals:
            continue
        labels.append(dname)
        means.append(float(np.mean(vals)))
        stds.append(float(np.std(vals)))
    if not labels:
        return

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(labels, means, yerr=stds, capsize=6, color="C0", alpha=0.85)
    ax.axhline(classical_rmse, color="C3", ls="--", lw=1,
               label="PI+poly+grav (classical)")
    ax.set_ylabel("held-out RMSE (rad/s)")
    ax.set_xlabel("net_arch")
    ax.set_title("residual-PI: network-depth ablation (mean +/- std over seeds)")
    ax.legend(fontsize=8)
    ax.grid(True, axis="y", lw=0.3)
    fig.tight_layout()
    out = RESULTS_DIR / "ablation.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"figure -> {out}")


# ---------------------------------------------------------------------------
# Command-line entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--timesteps", type=int, default=TIMESTEPS)
    ap.add_argument("--no-progress", action="store_true")
    ap.add_argument("--smoke", action="store_true",
                    help="tiny pipeline check: 1 depth, 1 seed, 200 steps")
    args = ap.parse_args()
    if args.smoke:
        run(timesteps=200, depths={"[64]": [64]}, seeds=[0], progress_bar=False)
    else:
        run(timesteps=args.timesteps, progress_bar=not args.no_progress)
