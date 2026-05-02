"""
training/train_sweep.py — PPO acoustic sweep launcher
======================================================

Compatible with:
  stable-baselines3 >= 2.3.0
  gymnasium         >= 0.29.0

Usage
-----
    python training/train_sweep.py --smoke-test     # 50k steps, λ=0, ~2 min
    python training/train_sweep.py                  # full parallel sweep
    python training/train_sweep.py --sequential     # one λ at a time
    python training/train_sweep.py --lambda 1.5     # single λ value
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


# ════════════════════════════════════════════════════════════════════════════
# SINGLE LAMBDA TRAINING FUNCTION
# ════════════════════════════════════════════════════════════════════════════

def train_one_lambda(
    lam         : float,
    total_steps : int  = config.PPO_TOTAL_STEPS,
    smoke_test  : bool = False,
) -> None:
    """
    Train a single PPO agent with acoustic_lambda = lam.

    All imports are inside this function so it is safe to spawn via
    multiprocessing on Windows (which uses the 'spawn' start method and
    re-executes the module in each child process).

    Key hyperparameter fix vs. original plan
    ----------------------------------------
    [OC2025] used n_envs=1, so their rollout buffer was n_steps=32 entries.
    We use n_envs=4, so the buffer is n_steps × n_envs entries.
    SB3 requires batch_size ≤ buffer_size, so:

        buffer_size = n_steps × n_envs = 128 × 4 = 512 = batch_size  ✓

    This preserves batch_size=512 from [OC2025] while satisfying SB3.
    """

    # ── Imports (inside function for multiprocessing safety) ─────────────
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor
    from stable_baselines3.common.callbacks import EvalCallback
    from environment.airfoil_env_acoustic import AirfoilEnvAcoustic

    N_ENVS       = 4
    actual_steps = 50_000 if smoke_test else total_steps
    tag          = f"lambda_{lam:.2f}"
    best_dir     = config.MODELS_DIR / f"{tag}_best"
    log_dir      = config.MODELS_DIR / f"{tag}_logs"
    best_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    print(f"[{tag}] ▶ Starting ({actual_steps:,} steps)", flush=True)
    t0 = time.time()

    # ── Environments ──────────────────────────────────────────────────────
    # Use DummyVecEnv directly — avoids make_vec_env compatibility shims.
    # VecMonitor wraps at the VecEnv level (not individual env level) which
    # is fully compatible with SB3 2.x + gymnasium 0.29.
    def _make(l=lam):
        return AirfoilEnvAcoustic(acoustic_lambda=l)

    train_env = VecMonitor(DummyVecEnv([_make] * N_ENVS))
    eval_env  = VecMonitor(DummyVecEnv([_make] * 1))

    # ── PPO model ─────────────────────────────────────────────────────────
    model = PPO(
        policy        = "MlpPolicy",
        env           = train_env,
        learning_rate = config.PPO_LEARNING_RATE,   # 0.000268
        n_steps       = 128,                        # buffer = 128×4 = 512 = batch_size
        batch_size    = 512,
        n_epochs      = config.PPO_N_EPOCHS,        # 20
        gamma         = config.PPO_GAMMA,           # 0.995
        gae_lambda    = config.PPO_GAE_LAMBDA,      # 0.98
        ent_coef      = config.PPO_ENT_COEF,        # 0.001
        vf_coef       = config.PPO_VF_COEF,         # 0.754843
        clip_range    = config.PPO_CLIP_RANGE,      # 0.3
        max_grad_norm = config.PPO_MAX_GRAD_NORM,   # 5.0
        policy_kwargs = dict(
            net_arch  = dict(pi=[128, 128], vf=[128, 128]),  # SB3 2.x format
        ),
        verbose       = 1,
        device        = "cpu",
        seed          = 42,
    )

    print(
        f"[{tag}] obs={train_env.observation_space.shape}  "
        f"act={train_env.action_space.shape}  "
        f"buffer={128 * N_ENVS}  batch={512}",
        flush=True,
    )

    # ── EvalCallback ──────────────────────────────────────────────────────
    # eval_freq is measured in *total env steps across all envs*.
    # 50 000 steps / 4 envs ≈ every 12 500 policy updates.
    # n_eval_episodes=5 keeps evaluation fast (each episode ≤ 10 steps).
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

    # ── Train ─────────────────────────────────────────────────────────────
    try:
        model.learn(
            total_timesteps     = actual_steps,
            callback            = eval_cb,
            reset_num_timesteps = True,
            progress_bar        = False,
        )
    except KeyboardInterrupt:
        print(f"[{tag}] ⚠ Interrupted — saving partial model", flush=True)
    except Exception as exc:
        train_env.close()
        eval_env.close()
        raise RuntimeError(f"[{tag}] Training failed: {exc}") from exc

    # ── Save ──────────────────────────────────────────────────────────────
    out = config.MODELS_DIR / f"{tag}_final"
    model.save(str(out))
    elapsed = time.time() - t0
    print(f"[{tag}] ✓ Finished in {elapsed/3600:.2f}h  →  {out}.zip", flush=True)

    train_env.close()
    eval_env.close()


# ════════════════════════════════════════════════════════════════════════════
# LAUNCHERS
# ════════════════════════════════════════════════════════════════════════════

def run_sequential(
    lambdas     : list[float],
    total_steps : int,
    smoke_test  : bool,
) -> None:
    print(f"\n{'='*60}")
    print(f"Sequential training  ({len(lambdas)} run{'s' if len(lambdas)>1 else ''})")
    print(f"  λ values : {lambdas}")
    print(f"  Steps    : {total_steps:,}")
    print(f"{'='*60}\n")
    config.MODELS_DIR.mkdir(parents=True, exist_ok=True)

    for lam in lambdas:
        try:
            train_one_lambda(lam, total_steps=total_steps, smoke_test=smoke_test)
        except Exception as exc:
            print(f"  ✗ λ={lam:.2f} failed: {exc}", flush=True)

    print("\n✓ Sequential sweep complete.")


def run_parallel(
    lambdas     : list[float],
    total_steps : int,
    smoke_test  : bool,
) -> None:
    print(f"\n{'='*60}")
    print(f"Launching {len(lambdas)} parallel training runs")
    print(f"  λ values    : {lambdas}")
    print(f"  Steps each  : {total_steps:,}")
    print(f"  Output dir  : {config.MODELS_DIR}")
    print(f"{'='*60}\n")
    config.MODELS_DIR.mkdir(parents=True, exist_ok=True)

    ctx   = mp.get_context("spawn")
    procs = []
    for lam in lambdas:
        p = ctx.Process(
            target = train_one_lambda,
            args   = (lam, total_steps, smoke_test),
            name   = f"train_lambda_{lam:.2f}",
        )
        p.start()
        procs.append((lam, p))
        print(f"  Spawned PID {p.pid}  λ={lam:.2f}", flush=True)

    print(f"\nAll {len(procs)} workers running...\n")

    any_failed = False
    for lam, p in procs:
        p.join()
        ok = (p.exitcode == 0)
        print(f"  λ={lam:.2f} {'✓' if ok else f'✗ (exit {p.exitcode})'}", flush=True)
        if not ok:
            any_failed = True

    if any_failed:
        print("\n⚠ One or more runs failed.")
        sys.exit(1)
    else:
        print("\n✓ All runs complete.")


# ════════════════════════════════════════════════════════════════════════════
# ENTRY POINT  — must be guarded for Windows multiprocessing (spawn method)
# ════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Launch PPO acoustic sweep. Run from repo root.",
    )
    parser.add_argument(
        "--lambda", dest="lam", type=float, default=None,
        metavar="LAMBDA",
        help="Train a single λ value only.",
    )
    parser.add_argument(
        "--smoke-test", action="store_true",
        help="Run λ=0 for 50k steps to verify pipeline (~2 min).",
    )
    parser.add_argument(
        "--sequential", action="store_true",
        help="Train all λ values one at a time (safer on low-RAM machines).",
    )
    parser.add_argument(
        "--steps", type=int, default=config.PPO_TOTAL_STEPS,
        help=f"Steps per run (default {config.PPO_TOTAL_STEPS:,}).",
    )
    args = parser.parse_args()

    if args.smoke_test:
        print("🔬 Smoke-test mode: λ=0 for 50k steps")
        run_sequential([0.0], 50_000, smoke_test=True)
    elif args.lam is not None:
        run_sequential([args.lam], args.steps, smoke_test=False)
    elif args.sequential:
        run_sequential(config.LAMBDA_VALUES, args.steps, smoke_test=False)
    else:
        run_parallel(config.LAMBDA_VALUES, args.steps, smoke_test=False)
