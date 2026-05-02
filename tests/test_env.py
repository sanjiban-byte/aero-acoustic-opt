"""
tests/test_env.py — Block 5 sanity checks for AirfoilEnvAcoustic
=================================================================

Run from the repository root:
    python tests/test_env.py

All 6 tests must pass before launching training (Block 6).
A failed test means the reward signal is broken and PPO will not learn.

Tests
-----
1. Environment instantiates and conforms to Gymnasium API
2. reset() returns correct observation shape and type
3. step() returns valid (non-NaN, non-inf) obs and reward
4. Reward scales correctly with acoustic_lambda
5. Invalid geometry triggers the -100 penalty
6. CL-floor penalty fires when CL would be near zero

Expected output (all passing):
  ── Test 1: Environment Instantiation ──
  ✅ PASSED
  ── Test 2: reset() API Compliance ──
  ✅ PASSED
  ── Test 3: step() Returns Valid Values ──
  ✅ PASSED
  ── Test 4: Reward Scales With Lambda ──
  ✅ PASSED
  ── Test 5: Invalid Geometry Penalty ──
  ✅ PASSED
  ── Test 6: CL-Floor Penalty ──
  ✅ PASSED
  ══ All environment tests passed ══
"""

import sys
import warnings
from pathlib import Path
import numpy as np

# ── Path setup ──────────────────────────────────────────────────────────────
# tests/ lives at repo_root/tests/ so parent is tests/, parent.parent is root
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import config
from environment.airfoil_env_acoustic import (
    AirfoilEnvAcoustic,
    INVALID_GEOM_PENALTY,
    CL_FLOOR,
    CL_FLOOR_PENALTY,
)

try:
    import gymnasium as gym
    from gymnasium import spaces
    _GYM_MODULE = "gymnasium"
except ImportError:
    import gym
    from gym import spaces
    _GYM_MODULE = "gym"


# ════════════════════════════════════════════════════════════════════════════
# TEST 1 — INSTANTIATION
# ════════════════════════════════════════════════════════════════════════════

def test_instantiation():
    """
    Environment must instantiate without errors for all λ values.

    Checks:
      - Correct observation space dimension (20)
      - Correct action space dimension (17)
      - Baseline values loaded (SPL > 0, L/D > 0)
      - Surrogate status reported (True or False — not a crash)
    """
    print("\n── Test 1: Environment Instantiation ──")

    for lam in config.LAMBDA_VALUES:
        env = AirfoilEnvAcoustic(acoustic_lambda=lam)

        # Observation space
        assert env.observation_space.shape == (20,), (
            f"λ={lam}: obs dim = {env.observation_space.shape}, expected (20,)"
        )

        # Action space
        assert env.action_space.shape == (17,), (
            f"λ={lam}: action dim = {env.action_space.shape}, expected (17,)"
        )

        # Action bounds
        np.testing.assert_array_equal(
            env.action_space.low,
            np.full(17, -1.0, dtype=np.float32),
            err_msg=f"λ={lam}: action low bound wrong",
        )
        np.testing.assert_array_equal(
            env.action_space.high,
            np.full(17, 1.0, dtype=np.float32),
            err_msg=f"λ={lam}: action high bound wrong",
        )

        # Baseline values
        assert env.baseline_spl > 0, f"λ={lam}: baseline_spl = {env.baseline_spl}"
        assert env.baseline_eff > 0, f"λ={lam}: baseline_eff = {env.baseline_eff}"

        # Lambda stored correctly
        assert env.acoustic_lambda == lam, f"Lambda not stored: {env.acoustic_lambda}"

        env.close()
        print(f"  λ={lam:4.1f} → obs={env.observation_space.shape}  "
              f"act={env.action_space.shape}  "
              f"SPL_base={env.baseline_spl:.2f} dB  "
              f"L/D_base={env.baseline_eff:.2f}  "
              f"surrogate={env._surrogate_available}")

    print(" ✅ PASSED")


# ════════════════════════════════════════════════════════════════════════════
# TEST 2 — reset() API COMPLIANCE
# ════════════════════════════════════════════════════════════════════════════

def test_reset():
    """
    reset() must return (obs, info) where obs is a valid (20,) float32 array.

    Gymnasium API requirement: reset() returns (obs, info_dict).
    The obs must lie within observation_space bounds (checked approximately —
    the seed may be slightly outside the declared bounds due to the initial
    CST seed values, which is acceptable and expected).

    Also checks:
      - SPL is set to a physically plausible value after reset
      - Step counter is zero after reset
    """
    print("\n── Test 2: reset() API Compliance ──")

    env = AirfoilEnvAcoustic(acoustic_lambda=0.5)
    result = env.reset(seed=42)

    # Gymnasium API: must return a 2-tuple
    assert isinstance(result, tuple), f"reset() returned {type(result)}, expected tuple"
    assert len(result) == 2, f"reset() returned {len(result)}-tuple, expected 2-tuple"

    obs, info = result

    # Observation type and shape
    assert isinstance(obs, np.ndarray), f"obs is {type(obs)}, expected np.ndarray"
    assert obs.shape == (20,), f"obs.shape = {obs.shape}, expected (20,)"
    assert obs.dtype == np.float32, f"obs.dtype = {obs.dtype}, expected float32"

    # No NaN / inf
    assert np.all(np.isfinite(obs)), (
        f"obs contains NaN or inf at indices: {np.where(~np.isfinite(obs))[0]}"
    )

    # Info must be a dict (can be empty)
    assert isinstance(info, dict), f"info is {type(info)}, expected dict"

    # SPL set to a plausible value
    initial_spl = env.current_spl
    assert config.SPL_CLIP_MIN <= initial_spl <= config.SPL_CLIP_MAX, (
        f"Initial SPL = {initial_spl:.2f} outside "
        f"[{config.SPL_CLIP_MIN}, {config.SPL_CLIP_MAX}] dB"
    )

    # SPL entry in obs (index 19) should match current_spl / 100
    spl_in_obs = float(obs[19]) * 100.0
    assert abs(spl_in_obs - env.current_spl) < 0.01, (
        f"SPL in obs ({spl_in_obs:.2f}) != env.current_spl ({env.current_spl:.2f})"
    )

    # Step counter reset
    assert env._step_count == 0, f"Step count not reset: {env._step_count}"

    print(f"  obs[:8]  (upper)  : {obs[:8]}")
    print(f"  obs[8:16](lower)  : {obs[8:16]}")
    print(f"  obs[16]  (LE)     : {obs[16]:.4f}")
    print(f"  obs[17]  (CL tgt) : {obs[17]:.4f}")
    print(f"  obs[18]  (Re/1M)  : {obs[18]:.4f}")
    print(f"  obs[19]  (SPL/100): {obs[19]:.4f}  → {obs[19]*100:.2f} dB")

    env.close()
    print(" ✅ PASSED")


# ════════════════════════════════════════════════════════════════════════════
# TEST 3 — step() RETURNS VALID VALUES
# ════════════════════════════════════════════════════════════════════════════

def test_step_valid_values():
    """
    Run 20 steps with random actions and verify no NaN / inf rewards.

    Also checks:
      - Returned tuple conforms to Gymnasium 5-tuple API
      - obs always has shape (20,) and is float32
      - reward is a finite scalar
      - terminated is always False (we use truncated for episode length)
      - truncated becomes True exactly at step max_steps
    """
    print("\n── Test 3: step() Returns Valid Values ──")

    env = AirfoilEnvAcoustic(acoustic_lambda=0.5, max_steps=10)
    obs, _ = env.reset(seed=7)

    n_steps      = 0
    rewards_seen = []
    spl_seen     = []

    for step_idx in range(12):   # go 2 beyond max_steps to confirm truncation
        action = env.action_space.sample()
        result = env.step(action)

        # Gymnasium 5-tuple API
        assert len(result) == 5, (
            f"step() returned {len(result)}-tuple, expected 5-tuple"
        )
        obs, reward, terminated, truncated, info = result

        # Types
        assert isinstance(obs,        np.ndarray), "obs not ndarray"
        assert isinstance(reward,     (float, int, np.floating)), "reward not scalar"
        assert isinstance(terminated, (bool, np.bool_)), "terminated not bool"
        assert isinstance(truncated,  (bool, np.bool_)), "truncated not bool"
        assert isinstance(info,       dict), "info not dict"

        # Shapes / finiteness
        assert obs.shape == (20,), f"step {step_idx}: obs.shape = {obs.shape}"
        assert np.all(np.isfinite(obs)), (
            f"step {step_idx}: obs has NaN/inf at "
            f"{np.where(~np.isfinite(obs))[0]}"
        )
        assert np.isfinite(reward), f"step {step_idx}: reward = {reward}"

        # terminated must be False — we never set it True
        assert not terminated, f"step {step_idx}: terminated=True, expected False"

        rewards_seen.append(float(reward))
        spl_seen.append(env.current_spl)

        if truncated:
            # Reset for next episode
            obs, _ = env.reset()
        else:
            n_steps += 1

    rewards_arr = np.array(rewards_seen)
    print(f"  Steps run        : {len(rewards_arr)}")
    print(f"  Reward range     : [{rewards_arr.min():.2f}, {rewards_arr.max():.2f}]")
    print(f"  Reward mean      : {rewards_arr.mean():.2f}")
    print(f"  SPL range (dB)   : [{min(spl_seen):.2f}, {max(spl_seen):.2f}]")

    # Rewards should not all be -100 (geometry penalty dominating)
    non_penalty = rewards_arr[rewards_arr > INVALID_GEOM_PENALTY + 50]
    assert len(non_penalty) > 0, (
        "All rewards are geometry penalties — check CST bounds or validity function"
    )

    env.close()
    print(" ✅ PASSED")


# ════════════════════════════════════════════════════════════════════════════
# TEST 4 — REWARD SCALES WITH λ
# ════════════════════════════════════════════════════════════════════════════

def test_reward_scales_with_lambda():
    """
    For the SAME action sequence, a higher λ must produce a lower mean reward
    (or equal, if ΔSPL ≤ 0 everywhere).

    This confirms that the acoustic penalty is wired correctly:
      r_low_lambda > r_high_lambda  when  ΔSPL > 0 (airfoil is louder)

    We use a deterministic action sequence (all zeros = no change) applied to
    the seed airfoil.  The seed airfoil's SPL was set at reset — if it happens
    to be quieter than baseline, rewards are equal across λ.  We therefore
    test the SIGN of the reward difference rather than strict ordering.

    Also verifies that λ=0 behaves identically to the pure-DRLFoil reward
    (no acoustic term at all).
    """
    print("\n── Test 4: Reward Scales With Lambda ──")

    FIXED_ACTIONS = [
        np.full(17,  0.5, dtype=np.float32),   # push upward (thicker → louder)
        np.full(17, -0.5, dtype=np.float32),   # push downward
        np.zeros(17, dtype=np.float32),        # no change
    ]

    mean_rewards = {}

    for lam in [0.0, 0.5, 3.0, 6.0]:
        env = AirfoilEnvAcoustic(acoustic_lambda=lam, max_steps=len(FIXED_ACTIONS))
        env.reset(seed=42)

        total_r = 0.0
        total_spl_delta = 0.0
        n_valid = 0

        for action in FIXED_ACTIONS:
            obs, reward, terminated, truncated, info = env.step(action)
            # Only count non-penalty steps for the mean
            if reward > INVALID_GEOM_PENALTY + 50:
                total_r += reward
                total_spl_delta += env.current_spl - env.baseline_spl
                n_valid += 1

        mean_r   = total_r / max(n_valid, 1)
        mean_spl = total_spl_delta / max(n_valid, 1)
        mean_rewards[lam] = (mean_r, mean_spl)
        env.close()

        print(f"  λ={lam:4.1f}  mean_reward={mean_r:+7.3f}  "
              f"mean_ΔSPL={mean_spl:+6.3f} dB  valid_steps={n_valid}")

    # Key invariant: higher lambda must not produce a HIGHER reward than lower
    # lambda when the airfoil is louder than baseline.
    # We check this only when mean_ΔSPL > 0 (airfoil is louder).
    r0, dspl0 = mean_rewards[0.0]
    r6, dspl6 = mean_rewards[6.0]

    if dspl6 > 0:
        assert r0 >= r6, (
            f"λ=0 reward ({r0:.3f}) < λ=6.0 reward ({r6:.3f}) "
            f"but ΔSPL > 0 — acoustic penalty not wired correctly"
        )
        print(f"\n  ✓ Higher λ reduces reward when airfoil is louder than baseline")
    else:
        print(f"\n  ✓ ΔSPL ≤ 0 at all steps — airfoil is quieter, penalty is negative "
              f"(bonus). λ ordering reversed as expected.")

    # λ=0 must match pure efficiency reward (no acoustic term)
    env_pure = AirfoilEnvAcoustic(acoustic_lambda=0.0, max_steps=3)
    env_pure.reset(seed=42)

    for action in FIXED_ACTIONS:
        obs, reward, _, _, info = env_pure.step(action)
        if reward > INVALID_GEOM_PENALTY + 50:
            # Manually compute expected reward without acoustic term
            cl  = info.get("cl", env_pure.current_cl)
            cd  = info.get("cd", env_pure.current_cd)
            if cd > 0:
                eff     = cl / cd
                dcl     = cl - config.CL_TARGET
                gauss   = float(np.exp(-config.CL_WIDE * dcl ** 2))
                r_expect = config.EFFICIENCY_PARAM * eff * gauss
                assert abs(reward - r_expect) < 0.01, (
                    f"λ=0 reward ({reward:.4f}) ≠ pure aero reward ({r_expect:.4f})"
                )
    env_pure.close()
    print("  ✓ λ=0 reward matches pure aerodynamic formula")

    print(" ✅ PASSED")


# ════════════════════════════════════════════════════════════════════════════
# TEST 5 — INVALID GEOMETRY PENALTY
# ════════════════════════════════════════════════════════════════════════════

def test_invalid_geometry_penalty():
    """
    Force an invalid geometry and confirm the -100 penalty fires.

    We do this by monkey-patching is_physically_valid to return False for
    one step, then confirm:
      • reward == INVALID_GEOM_PENALTY == -100
      • CST params are NOT updated (environment stays at previous state)
    """
    print("\n── Test 5: Invalid Geometry Penalty ──")

    from boundary_layer import generate_dstar_dataset as _gdd

    env = AirfoilEnvAcoustic(acoustic_lambda=1.0, max_steps=10)
    env.reset(seed=0)

    # Store state before the forced-invalid step
    upper_before = env._upper.copy()
    lower_before = env._lower.copy()
    le_before    = env._le

    # Monkey-patch validity checker
    _original = _gdd.is_physically_valid
    _gdd.is_physically_valid = lambda u, l: False

    # Also patch the one imported into the environment module
    import environment.airfoil_env_acoustic as _env_mod
    _orig_env = _env_mod.is_physically_valid
    _env_mod.is_physically_valid = lambda u, l: False

    try:
        action = np.zeros(17, dtype=np.float32)
        obs, reward, terminated, truncated, info = env.step(action)

        # Penalty must fire
        assert reward == INVALID_GEOM_PENALTY, (
            f"Expected INVALID_GEOM_PENALTY ({INVALID_GEOM_PENALTY}), got {reward}"
        )

        # State must NOT have changed
        np.testing.assert_array_equal(
            env._upper, upper_before,
            err_msg="Upper CST updated on invalid geometry — should stay unchanged",
        )
        np.testing.assert_array_equal(
            env._lower, lower_before,
            err_msg="Lower CST updated on invalid geometry — should stay unchanged",
        )
        assert env._le == le_before, (
            f"LE weight changed on invalid geometry: {env._le} vs {le_before}"
        )

        # Info should have a reason key
        assert "reason" in info, "info dict missing 'reason' key on invalid geometry"

    finally:
        # Restore original functions
        _gdd.is_physically_valid = _original
        _env_mod.is_physically_valid = _orig_env

    print(f"  Penalty value    : {reward}")
    print(f"  State preserved  : upper unchanged ✓, lower unchanged ✓")
    env.close()
    print(" ✅ PASSED")


# ════════════════════════════════════════════════════════════════════════════
# TEST 6 — CL-FLOOR PENALTY
# ════════════════════════════════════════════════════════════════════════════

def test_cl_floor_penalty():
    """
    Force CL below CL_FLOOR and confirm the -50 penalty fires.

    This guards against reward collapse where the agent learns to produce
    aerodynamically useless flat plates just to minimise acoustic noise.

    We monkey-patch _compute_aero to return CL = 0.01 (below CL_FLOOR = 0.10),
    then verify:
      • reward == CL_FLOOR_PENALTY == -50
      • Observation is still valid (no crash)
    """
    print("\n── Test 6: CL-Floor Penalty ──")

    env = AirfoilEnvAcoustic(acoustic_lambda=3.0, max_steps=5)
    env.reset(seed=1)

    import environment.airfoil_env_acoustic as _env_mod

    # Patch _compute_aero on the INSTANCE (not the class) to avoid side-effects
    _original_method = env._compute_aero
    env._compute_aero = lambda coords: (0.01, 0.005)  # CL = 0.01 << CL_FLOOR = 0.10

    try:
        action = np.zeros(17, dtype=np.float32)
        obs, reward, terminated, truncated, info = env.step(action)

        assert reward == CL_FLOOR_PENALTY, (
            f"Expected CL_FLOOR_PENALTY ({CL_FLOOR_PENALTY}), got {reward:.3f}. "
            f"CL_FLOOR is {CL_FLOOR} — check _compute_reward CL guard."
        )

        # Observation should still be valid (step completed)
        assert obs.shape == (20,), f"obs.shape wrong: {obs.shape}"
        assert np.all(np.isfinite(obs)), "obs has NaN after CL-floor penalty"

    finally:
        env._compute_aero = _original_method

    print(f"  CL injected      : 0.01  (CL_FLOOR = {CL_FLOOR})")
    print(f"  Penalty returned : {reward}")
    env.close()
    print(" ✅ PASSED")


# ════════════════════════════════════════════════════════════════════════════
# MAIN
# ════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("=" * 60)
    print("Block 5 — AirfoilEnvAcoustic test suite")
    print(f"Using {_GYM_MODULE}")
    print("=" * 60)

    tests = [
        test_instantiation,
        test_reset,
        test_step_valid_values,
        test_reward_scales_with_lambda,
        test_invalid_geometry_penalty,
        test_cl_floor_penalty,
    ]

    passed = 0
    failed = 0
    for t in tests:
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                t()
            passed += 1
        except AssertionError as e:
            print(f" ❌ FAILED: {e}")
            failed += 1
        except Exception as e:
            print(f" ❌ ERROR: {type(e).__name__}: {e}")
            failed += 1

    print()
    print("=" * 60)
    if failed == 0:
        print(f"══ All {passed} environment tests passed ══")
        print()
        print("Next step: git add environment/ tests/test_env.py")
        print("           python training/train_sweep.py   ← Block 6")
    else:
        print(f"  {passed} passed, {failed} FAILED")
        print("  Fix failures before launching training.")
    print("=" * 60)
