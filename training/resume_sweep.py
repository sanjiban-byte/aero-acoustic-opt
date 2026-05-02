"""
training/resume_sweep.py — Resume interrupted training from best_model.zip
===========================================================================

Use after an interrupted training run. Loads best_model.zip for each λ
and continues training for the remaining steps.

Usage
-----
    python training/resume_sweep.py              # resumes all λ values
    python training/resume_sweep.py --lambda 0.0 # resumes single λ
"""

from __future__ import annotations

import argparse
import multiprocessing as mp
import sys
import time
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import config


def resume_one_lambda(
    lam             : float,
    steps_completed : int = 1_000_000,
    total_steps     : int = config.PPO_TOTAL_STEPS,
) -> None:
    """
    Load best_model.zip and continue training for remaining steps.

    SB3 PPO.load() restores network weights and optimizer state.
    The rollout buffer is fresh (acceptable — takes ~128 steps to refill).
    """
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor
    from stable_baselines3.common.callbacks import EvalCallback
    from environment.airfoil_env_acoustic import AirfoilEnvAcoustic

    tag       = f"lambda_{lam:.2f}"
    best_dir  = config.MODELS_DIR / f"{tag}_best"
    log_dir   = config.MODELS_DIR / f"{tag}_logs"
    model_path = best_dir / "best_model.zip"

    if not model_path.exists():
        print(f"[{tag}] ✗ No checkpoint found at {model_path}")
        print(f"[{tag}]   Run full training instead: python training/train_sweep.py --lambda {lam}")
        return

    remaining_steps = total_steps - steps_completed
    if remaining_steps <= 0:
        print(f"[{tag}] ✓ Already complete ({steps_completed:,} steps)")
        return

    print(f"[{tag}] ▶ Resuming from {steps_completed:,} steps — "
          f"{remaining_steps:,} remaining", flush=True)
    t0 = time.time()

    N_ENVS = 4

    def _make(l=lam):
        return AirfoilEnvAcoustic(acoustic_lambda=l, use_xfoil=False)

    train_env = VecMonitor(DummyVecEnv([_make] * N_ENVS))
    eval_env  = VecMonitor(DummyVecEnv([_make] * 1))

    # Load model — restores weights + optimizer state
    model = PPO.load(
        str(model_path),
        env    = train_env,
        device = "cpu",
    )

    print(f"[{tag}] Loaded model. "
          f"obs={train_env.observation_space.shape}  "
          f"act={train_env.action_space.shape}", flush=True)

    # EvalCallback — continues appending to existing log
    eval_cb = EvalCallback(
        eval_env             = eval_env,
        best_model_save_path = str(best_dir),
        log_path             = str(log_dir),
        eval_freq            = 50_000,
        n_eval_episodes      = 5,
        deterministic        = True,
        verbose              = 0,
        warn                 = False,
    )

    try:
        model.learn(
            total_timesteps     = remaining_steps,
            callback            = eval_cb,
            reset_num_timesteps = False,  # ← critical: don't reset step counter
            progress_bar        = False,
        )
    except KeyboardInterrupt:
        print(f"[{tag}] ⚠ Interrupted — saving partial model", flush=True)
    except Exception as exc:
        train_env.close()
        eval_env.close()
        raise RuntimeError(f"[{tag}] Resume failed: {exc}") from exc

    # Save final model
    out = config.MODELS_DIR / f"{tag}_final"
    model.save(str(out))
    elapsed = time.time() - t0
    print(f"[{tag}] ✓ Done in {elapsed/3600:.2f}h → {out}.zip", flush=True)

    train_env.close()
    eval_env.close()


def run_sequential(lambdas, steps_completed):
    config.MODELS_DIR.mkdir(parents=True, exist_ok=True)
    for lam in lambdas:
        try:
            resume_one_lambda(lam, steps_completed=steps_completed)
        except Exception as exc:
            print(f"  ✗ λ={lam:.2f} failed: {exc}", flush=True)
    print("\n✓ Resume sweep complete.")


def run_parallel(lambdas, steps_completed):
    print(f"\n{'='*60}")
    print(f"Resuming {len(lambdas)} parallel runs from {steps_completed:,} steps")
    print(f"  λ values    : {lambdas}")
    print(f"  Remaining   : {config.PPO_TOTAL_STEPS - steps_completed:,} steps each")
    print(f"{'='*60}\n")
    config.MODELS_DIR.mkdir(parents=True, exist_ok=True)

    ctx   = mp.get_context("spawn")
    procs = []
    for lam in lambdas:
        p = ctx.Process(
            target = resume_one_lambda,
            args   = (lam, steps_completed, config.PPO_TOTAL_STEPS),
            name   = f"resume_lambda_{lam:.2f}",
        )
        p.start()
        procs.append((lam, p))
        print(f"  Spawned PID {p.pid}  λ={lam:.2f}", flush=True)

    print(f"\nAll {len(procs)} workers running...\n")
    any_failed = False
    for lam, p in procs:
        p.join()
        ok = (p.exitcode == 0)
        print(f"  λ={lam:.2f} {'✓' if ok else f'✗ exit {p.exitcode}'}", flush=True)
        if not ok:
            any_failed = True

    if any_failed:
        print("\n⚠ One or more runs failed.")
        sys.exit(1)
    else:
        print("\n✓ All runs complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--lambda", dest="lam", type=float, default=None)
    parser.add_argument("--completed", type=int, default=1_000_000,
                        help="Steps already completed (default: 1,000,000)")
    parser.add_argument("--sequential", action="store_true")
    args = parser.parse_args()

    lambdas = [args.lam] if args.lam is not None else config.LAMBDA_VALUES

    # First check which models actually exist
    print("\nChecking saved checkpoints...")
    missing = []
    found   = []
    for lam in lambdas:
        tag  = f"lambda_{lam:.2f}"
        path = config.MODELS_DIR / f"{tag}_best" / "best_model.zip"
        if path.exists():
            size = path.stat().st_size / 1024
            print(f"  ✓ λ={lam:.2f}  {path}  ({size:.0f} KB)")
            found.append(lam)
        else:
            print(f"  ✗ λ={lam:.2f}  NOT FOUND: {path}")
            missing.append(lam)

    if missing:
        print(f"\n⚠ Missing checkpoints for λ={missing}")
        print("  These will need full retraining:")
        print(f"  python training/train_sweep.py --sequential", end="")
        for lam in missing:
            print(f" --lambda {lam}", end="")
        print()

    if not found:
        print("\n✗ No checkpoints found. Run full training instead:")
        print("  python training/train_sweep.py")
        sys.exit(1)

    if args.sequential or len(found) == 1:
        run_sequential(found, args.completed)
    else:
        run_parallel(found, args.completed)
