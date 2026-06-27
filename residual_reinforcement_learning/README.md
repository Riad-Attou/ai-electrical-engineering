# Residual reinforcement learning — BDC motor speed control

A residual SAC policy (`V = clip(V_base + ΔV, ±V_max)`) learns a bounded
correction `ΔV` on top of a classical **PI** speed controller for the
brushed-DC motor + rod plant. The policy is trained with a weighted **L1**
(absolute) tracking-error reward and benchmarked against the classical PI and
PI+poly+grav (feedforward + gravity) controllers.

All code is self-contained here; `utils/` and `poly/data/` in the repo root are
reused read-only.

## Install

```
pip install -r residual_reinforcement_learning/requirements.txt
```

## Train

```
python residual_reinforcement_learning/train.py --timesteps 300000
```

Saves the policy to `models/sac_residual_pi.zip` plus a JSON run config.
Reward weights are CLI flags (`--alpha`, `--lam`, `--mu`); `--saturation-aware`
adds the rail-aware observation/reward, `--randomize` enables domain
randomisation of the motor parameters.

## Benchmark

```
python residual_reinforcement_learning/benchmark.py
```

Prints the RMSE table and writes `results/benchmark.png`. If no trained model
(or Stable-Baselines3) is present it runs the two classical rows alone.

## ML validation

```
python residual_reinforcement_learning/metrics.py          # scorecard + parity / autocorrelation plots
python residual_reinforcement_learning/generalization.py   # held-out references (incl. extrapolation)
python residual_reinforcement_learning/sweep.py             # depth × seed sweep -> tables + ablation figure
```

`sweep.py` is resumable and writes `results/sweep_metrics.json`, the train/test
and depth tables, and `results/ablation.png`.

```
python residual_reinforcement_learning/voltage_plot.py   # input-voltage comparison + isolated RL ΔV
```

Compares the motor input voltage `V(t)` for PI / PI+poly+grav / residual-PI and
shows the residual decomposition `V = V_base (PI) + ΔV (RL)`, writing
`results/voltage_contributions.png` (use `--ref "const 10"` for the constant case).

## Report

`report.tex` is a standalone two-column report; compile it with `pdflatex`. It
`\includegraphics` the figures in `results/` and embeds the metric tables.

## Visualise (real-time)

```
python residual_reinforcement_learning/run_viz_rl.py
```

Pygame window; switch controllers live with keys `1`–`3`
(PI / PI+poly+grav / residual-PI), `UP`/`DOWN` to change the target speed, `R`
to reset. Requires `pygame`; residual-PI appears only if the trained model is
present. The repo-root `run_viz.py` is left untouched.

## Layout

| File | Role |
|---|---|
| `baselines.py` | self-contained PI and PI+poly+grav controllers |
| `reference.py` | randomised + fixed speed-reference generators |
| `motor_env.py` | Gymnasium env, residual ΔV action at 100 Hz, L1 reward |
| `residual_controller.py` | drop-in `.step()` wrapper (baseline + policy) |
| `train.py` | SAC training of the residual-PI policy |
| `benchmark.py` | deterministic RMSE table + figure |
| `metrics.py` | regression scorecard (MAE/MSE/RMSE/R²/in-band) + diagnostic plots |
| `generalization.py` | held-out reference evaluation (overfitting check) |
| `tables.py` | train-vs-test and depth-comparison tables (Markdown + CSV) |
| `sweep.py` | depth × seed training sweep + depth-ablation figure |
| `voltage_plot.py` | input-voltage comparison + isolated RL ΔV contribution |
| `run_viz_rl.py` | real-time pygame viz; switch controllers live (keys 1–3) |
