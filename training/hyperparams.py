"""
training/hyperparams.py — PPO hyperparameters for the acoustic sweep
=====================================================================

All values sourced from Orgeira-Crespo et al. (2025) [OC2025] Table 10,
which reports the result of a 35-trial Optuna hyperparameter search using
Tree-structured Parzen Estimation across 21 million training steps.

We borrow these directly because:
  - Our environment structure is identical (CST + Gymnasium + PPO + SB3)
  - The reward function has the same scale as theirs (CL/CD ≈ 50–80)
  - The observation vector is only slightly larger (20 vs 19 dims)

The one deliberate departure: NET_ARCH = [128, 128] rather than the
[64, 64] in their Table 10. Section 3.3 of [OC2025] itself uses [128, 128]
for their "one-box" model because extra observations require more capacity.
Our SPL observation is the same type of extension.

Usage
-----
    from training.hyperparams import PPO_KWARGS, ENV_KWARGS, TRAIN_KWARGS
    model = PPO("MlpPolicy", env, **PPO_KWARGS)
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config


# ════════════════════════════════════════════════════════════════════════════
# PPO ALGORITHM HYPERPARAMETERS
# ════════════════════════════════════════════════════════════════════════════
# Passed directly to stable_baselines3.PPO.__init__()

PPO_KWARGS = dict(
    learning_rate  = config.PPO_LEARNING_RATE,   # 0.000268  — [OC2025] Table 10
    n_steps        = 128,                         # was 32 — buffer = 128*4 = 512 = batch_size 
    batch_size     = 512,                         # now valid: 512 ≤ 128*4 = 512
    n_epochs       = config.PPO_N_EPOCHS,         # 20        — gradient epochs per update
    gamma          = config.PPO_GAMMA,            # 0.995     — discount factor
    gae_lambda     = config.PPO_GAE_LAMBDA,       # 0.98      — GAE smoothing
    ent_coef       = config.PPO_ENT_COEF,         # 0.001     — entropy bonus
    vf_coef        = config.PPO_VF_COEF,          # 0.754843  — value function weight
    clip_range     = config.PPO_CLIP_RANGE,       # 0.3       — PPO clip epsilon
    max_grad_norm  = config.PPO_MAX_GRAD_NORM,    # 5.0       — gradient clipping
    policy_kwargs  = dict(
        net_arch   = dict(pi=config.PPO_NET_ARCH, vf=config.PPO_NET_ARCH),        # [128, 128] — [OC2025] Section 3.3
    ),
    verbose        = 0,    # suppress per-step prints; use TensorBoard for monitoring
    device         = "cpu", # PPO on 128×128 MLP trains faster on CPU than GPU
                            # (tensor transfer overhead > compute gain at this size)
)


# ════════════════════════════════════════════════════════════════════════════
# ENVIRONMENT CONSTRUCTOR KWARGS
# ════════════════════════════════════════════════════════════════════════════
# Passed to AirfoilEnvAcoustic() — acoustic_lambda is added per-run.
# These are the same for all λ values to ensure fair Pareto comparison.

ENV_KWARGS_BASE = dict(
    max_steps        = config.MAX_STEPS,          # 10  — [OC2025] Table 11
    n_params         = config.N_CST_PARAMS,       # 8   — CST weights per surface
    scale_actions    = config.SCALE_ACTIONS,      # 0.3 — max Δweight per step
    cl_target        = config.CL_TARGET,          # 0.6 — fixed for Pareto analysis
    re               = config.RE,                 # 500 000
    cl_wide          = config.CL_WIDE,            # 20  — Gaussian γ
    efficiency_param = config.EFFICIENCY_PARAM,   # 1.0 — α scaling
    use_xfoil        = False,                     # surrogate during training
    neuralfoil_size  = "small",                   # faster during training
    # acoustic_lambda added per-run in train_sweep.py
)


# ════════════════════════════════════════════════════════════════════════════
# TRAINING LOOP KWARGS
# ════════════════════════════════════════════════════════════════════════════

TRAIN_KWARGS = dict(
    total_timesteps = config.PPO_TOTAL_STEPS,     # 2 000 000
    # progress_bar added per-run to avoid pickling issues with multiprocessing
)


# ════════════════════════════════════════════════════════════════════════════
# EVAL CALLBACK KWARGS
# ════════════════════════════════════════════════════════════════════════════
# EvalCallback saves the best model and logs evaluation metrics.
# eval_freq is in *env steps* (not policy updates), so with n_envs=4:
#   50 000 env steps / 4 envs = 12 500 policy steps between evaluations
# This gives ~40 evaluation checkpoints over 2M steps — enough to plot
# smooth learning curves.

EVAL_KWARGS = dict(
    eval_freq       = 50_000,    # evaluate every 50k env steps
    n_eval_episodes = 10,        # 10 deterministic episodes per evaluation
    deterministic   = True,
    render          = False,
    verbose         = 0,
)


# ════════════════════════════════════════════════════════════════════════════
# PARALLEL TRAINING SETTINGS
# ════════════════════════════════════════════════════════════════════════════

N_ENVS          = config.PPO_N_ENVS    # 4 parallel envs per λ run
LAMBDA_VALUES   = config.LAMBDA_VALUES # [0.0, 0.5, 1.5, 3.0, 6.0]
