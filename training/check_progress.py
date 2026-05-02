"""
training/check_progress.py — Monitor overnight training runs
============================================================

Run this any time to see where each λ run stands.
Reads the evaluations.npz files written by EvalCallback.

Usage
-----
    python training/check_progress.py          # full table
    python training/check_progress.py --watch  # refresh every 60 s
"""

from __future__ import annotations

import sys
import time
import argparse
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import numpy as np
import config


def _load_eval(log_dir: Path) -> dict | None:
    """Load evaluations.npz written by SB3's EvalCallback."""
    npz_path = log_dir / "evaluations.npz"
    if not npz_path.exists():
        return None
    try:
        data = np.load(npz_path)
        return {
            "timesteps" : data["timesteps"],
            "results"   : data["results"],    # shape (n_evals, n_eval_eps)
        }
    except Exception:
        return None


def _summarise(lam: float) -> str:
    tag     = f"lambda_{lam:.2f}"
    log_dir = config.MODELS_DIR / f"{tag}_logs"
    best_dir= config.MODELS_DIR / f"{tag}_best"
    final   = config.MODELS_DIR / f"{tag}_final.zip"

    # Status
    if final.exists():
        status = "✓ DONE"
    elif (log_dir / "checkpoints").exists():
        ckpts = list((log_dir / "checkpoints").glob("*.zip"))
        status = f"⏳ running ({len(ckpts)} checkpoints)"
    elif log_dir.exists():
        status = "⏳ started (no checkpoints yet)"
    else:
        status = "⬜ not started"

    # Evaluation data
    data = _load_eval(log_dir)
    if data is None:
        return f"  λ={lam:.2f}  {status:35s}  no eval data yet"

    timesteps = data["timesteps"]
    results   = data["results"]             # (n_evals, n_eps)
    mean_r    = results.mean(axis=1)        # mean reward per eval checkpoint
    latest_ts = int(timesteps[-1])
    latest_r  = float(mean_r[-1])
    best_r    = float(mean_r.max())
    pct_done  = 100.0 * latest_ts / config.PPO_TOTAL_STEPS

    best_marker = "  ←best" if (best_dir / "best_model.zip").exists() else ""

    return (
        f"  λ={lam:.2f}  {status:35s}  "
        f"{latest_ts:>8,} steps ({pct_done:4.1f}%)  "
        f"latest_r={latest_r:+7.2f}  "
        f"best_r={best_r:+7.2f}"
        f"{best_marker}"
    )


def print_summary() -> None:
    print(f"\n{'='*90}")
    print(f"  Training progress — {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  Target: {config.PPO_TOTAL_STEPS:,} steps per run")
    print(f"{'='*90}")
    for lam in config.LAMBDA_VALUES:
        print(_summarise(lam))
    print(f"{'='*90}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--watch", action="store_true",
                        help="Refresh every 60 seconds until all runs complete.")
    parser.add_argument("--interval", type=int, default=60,
                        help="Refresh interval in seconds (default: 60).")
    args = parser.parse_args()

    if not args.watch:
        print_summary()
    else:
        print("Watching training progress (Ctrl-C to stop)...")
        try:
            while True:
                print_summary()
                time.sleep(args.interval)
        except KeyboardInterrupt:
            print("\nStopped.")
