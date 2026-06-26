"""Train residual SAC policies on top of the PI and PI+poly+grav baselines."""
import sys
import json
import argparse
import pathlib

ROOT = pathlib.Path(__file__).resolve().parent.parent
HERE = pathlib.Path(__file__).resolve().parent
for _p in (str(HERE), str(ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from utils.motor import BDCMotorParams, PendulumParams
from baselines import PurePI
from motor_env import MotorResidualEnv, RewardConfig

# ---------------------------------------------------------------------------
# Plant parameters and PI gains (must match benchmark.py / run_viz_rl.py)
# ---------------------------------------------------------------------------
PARAMS = BDCMotorParams(R=3.0, L=4e-3, Kt=0.05, Kb=0.05, J=7.4e-5, B=0.005, V_max=12.0)
ROD = PendulumParams(m=0.05, l=0.1, g=9.81)
KP, KI = 5.0, 2.0
MODELS_DIR = HERE / "models"


def make_baseline_factory(variant: str):
    if variant == "pi":
        return lambda: PurePI(KP, KI, PARAMS.V_max)
    raise ValueError(f"unknown variant: {variant!r} (only 'pi' is supported)")


def make_env(variant: str, seed: int, randomize: bool,
             reward_cfg: RewardConfig | None = None,
             noise_std: float = 1.0, saturation_aware: bool = False) -> MotorResidualEnv:
    env = MotorResidualEnv(make_baseline_factory(variant), params=PARAMS, rod=ROD,
                           reward_cfg=reward_cfg or RewardConfig(), randomize=randomize,
                           noise_std=noise_std, saturation_aware=saturation_aware)
    env.reset(seed=seed)
    return env


def train(variant: str, timesteps: int, seed: int, randomize: bool,
          progress_bar: bool = True, reward_cfg: RewardConfig | None = None,
          noise_std: float = 1.0, net_arch=(256, 256), tag: str | None = None,
          saturation_aware: bool = False) -> pathlib.Path:
    from stable_baselines3 import SAC
    from stable_baselines3.common.monitor import Monitor

    reward_cfg = reward_cfg or RewardConfig()
    env = Monitor(make_env(variant, seed, randomize,
                           reward_cfg=reward_cfg, noise_std=noise_std,
                           saturation_aware=saturation_aware))
    model = SAC("MlpPolicy", env, verbose=1, seed=seed,
                learning_rate=3e-4, gamma=0.99, buffer_size=200_000,
                batch_size=256, learning_starts=5_000, ent_coef="auto",
                policy_kwargs=dict(net_arch=list(net_arch)))
    model.learn(total_timesteps=timesteps, progress_bar=progress_bar)

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    name = f"sac_residual_{variant}" + (f"_{tag}" if tag else "")
    model_path = MODELS_DIR / f"{name}.zip"
    model.save(model_path)
    cfg = dict(variant=variant, timesteps=timesteps, seed=seed, randomize=randomize,
               Kp=KP, Ki=KI, params=vars(PARAMS), dv_limit_frac=0.3,
               control_decimation=10, alpha=reward_cfg.alpha, lam=reward_cfg.lam,
               mu=reward_cfg.mu, sat_penalty=reward_cfg.sat_penalty, noise_std=noise_std,
               net_arch=list(net_arch), saturation_aware=saturation_aware)
    (MODELS_DIR / f"{name}.json").write_text(json.dumps(cfg, indent=2))
    print(f"saved -> {model_path}")
    return model_path


# ---------------------------------------------------------------------------
# Command-line entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--timesteps", type=int, default=300_000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--randomize", action="store_true")
    ap.add_argument("--no-progress", action="store_true",
                    help="disable the tqdm progress bar (cleaner logs for background runs)")
    ap.add_argument("--lam", type=float, default=0.01,
                    help="residual-magnitude (effort) penalty weight")
    ap.add_argument("--mu", type=float, default=0.05,
                    help="residual-rate (smoothness) penalty weight")
    ap.add_argument("--alpha", type=float, default=1.0,
                    help="tracking-error (absolute) weight")
    ap.add_argument("--net-arch", type=int, nargs="+", default=[256, 256],
                    metavar="N", help="hidden layer sizes, e.g. --net-arch 64")
    ap.add_argument("--noise-std", type=float, default=1.0,
                    help="training measurement-noise std [rad/s] (0 = noise-free)")
    ap.add_argument("--saturation-aware", action="store_true",
                    help="saturation-aware observation + effective-residual reward")
    ap.add_argument("--sat-penalty", type=float, default=0.05,
                    help="clip-excess penalty weight (only with --saturation-aware)")
    args = ap.parse_args()
    rc = RewardConfig(alpha=args.alpha, lam=args.lam, mu=args.mu,
                      sat_penalty=args.sat_penalty)
    train("pi", args.timesteps, args.seed, args.randomize,
          progress_bar=not args.no_progress, reward_cfg=rc, noise_std=args.noise_std,
          net_arch=tuple(args.net_arch), saturation_aware=args.saturation_aware)
