"""
Run this to get the FULL traceback of the training error.
    python training/debug_train.py
"""
import sys, traceback
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor
from stable_baselines3.common.callbacks import EvalCallback
from environment.airfoil_env_acoustic import AirfoilEnvAcoustic

print(f"SB3 version: ", end="")
import stable_baselines3; print(stable_baselines3.__version__)
print(f"gymnasium version: ", end="")
import gymnasium; print(gymnasium.__version__)

def _make(lam=0.0):
    return AirfoilEnvAcoustic(acoustic_lambda=lam)

train_env = VecMonitor(DummyVecEnv([_make] * 4))
eval_env  = VecMonitor(DummyVecEnv([_make] * 1))

model = PPO(
    "MlpPolicy", train_env,
    learning_rate = config.PPO_LEARNING_RATE,
    n_steps       = 128,
    batch_size    = 512,
    n_epochs      = config.PPO_N_EPOCHS,
    gamma         = config.PPO_GAMMA,
    gae_lambda    = config.PPO_GAE_LAMBDA,
    ent_coef      = config.PPO_ENT_COEF,
    vf_coef       = config.PPO_VF_COEF,
    clip_range    = config.PPO_CLIP_RANGE,
    max_grad_norm = config.PPO_MAX_GRAD_NORM,
    policy_kwargs = dict(net_arch=dict(pi=[128,128], vf=[128,128])),
    verbose       = 1,
    device        = "cpu",
)

eval_cb = EvalCallback(
    eval_env, best_model_save_path=None,
    log_path=None, eval_freq=50_000,
    n_eval_episodes=5, deterministic=True,
    verbose=0, warn=False,
)

print("\n--- Calling model.learn(5000) ---")
try:
    model.learn(total_timesteps=5000, callback=eval_cb)
    print("SUCCESS")
except Exception:
    print("\n=== FULL TRACEBACK ===")
    traceback.print_exc()

train_env.close()
eval_env.close()
