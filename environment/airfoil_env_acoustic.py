"""
AirfoilEnvAcoustic — Extended DRLFoil Gymnasium Environment
============================================================
Block 5 of the aero-acoustic optimisation project.

This class extends the DRLFoil environment with acoustic trailing-edge
noise (SPL) in both the observation space and the reward function.

Core novelty vs. base DRLFoil [OC2025]:
  1. Acoustic SPL is added to the observation vector (agent "hears" itself).
  2. A BPM-based acoustic penalty λ·ΔSPL_norm is subtracted from the reward.
  3. The δ* surrogate replaces XFOIL during training (2830× speedup).
  4. An optional use_xfoil flag reactivates real XFOIL during evaluation.

Observation space (dim = 20):
  [upper_cst(8), lower_cst(8), le_weight(1),
   cl_target(1), re_norm(1), spl_norm(1)]

Action space (dim = 17):
  [Δupper(8), Δlower(8), Δle(1)],  each in [-1, 1]
  Applied as: new_weight = old_weight + action × SCALE_ACTIONS
  then clipped to CST bounds.

Reward function:
  r = α · (CL/CD) · exp(−γ · ΔCL²)  −  λ · (SPL − SPL_baseline)/|SPL_baseline|

  where α = EFFICIENCY_PARAM, γ = CL_WIDE, λ = acoustic_lambda.
  Additional penalties:
    • −100  for geometrically invalid airfoils (prevents XFOIL/NeuralFoil crash)
    • −50   if CL < CL_FLOOR (prevents reward-collapse to near-flat plates)
    • −10   (normalised) if NeuralFoil or surrogate fail to converge

Key references:
  [OC2025] Orgeira-Crespo et al. (2025) — DRLFoil base framework
  [BPM89]  Brooks, Pope, Marcolini (1989) — trailing-edge noise model
"""

from __future__ import annotations

import sys
import time
import warnings
from pathlib import Path
from typing import Optional, Tuple, Dict, Any

import numpy as np

# ── Path setup ─────────────────────────────────────────────────────────────
# This file lives at  <repo>/environment/airfoil_env_acoustic.py
# repo root is therefore Path(__file__).parent.parent
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import config

# ── Gymnasium ───────────────────────────────────────────────────────────────
try:
    import gymnasium as gym
    from gymnasium import spaces
except ImportError:
    import gym
    from gym import spaces

# ── Project modules ─────────────────────────────────────────────────────────
from acoustics.bpm_model import compute_spl_tbl_te
from boundary_layer.dstar_inference import predict_dstar_fast, is_surrogate_loaded
from boundary_layer.xfoil_bl import get_delta_star, blasius_fallback
from boundary_layer.generate_dstar_dataset import cst_to_coordinates, is_physically_valid

# ── NeuralFoil / AeroSandbox for CL and CD ──────────────────────────────────
try:
    import aerosandbox as asb
    _HAS_NEURALFOIL = True
except ImportError:
    _HAS_NEURALFOIL = False
    warnings.warn(
        "[AirfoilEnvAcoustic] aerosandbox not found. "
        "CL/CD will fall back to a flat-plate approximation. "
        "Install with: pip install aerosandbox",
        stacklevel=2,
    )


# ════════════════════════════════════════════════════════════════════════════
# CONSTANTS
# ════════════════════════════════════════════════════════════════════════════

# Minimum acceptable lift coefficient.
# Below this the airfoil is aerodynamically useless and probably degenerate.
# DRLFoil does not have this check; we add it to prevent λ > 0 runs from
# collapsing to near-zero-lift flat plates.
CL_FLOOR       = 0.10
CL_FLOOR_PENALTY = -50.0

# Penalty for geometry that fails the physical validity check.
# Must be large enough to strongly discourage bad actions but not so large
# that early training is dominated by pure geometry avoidance.
INVALID_GEOM_PENALTY = -100.0

# Penalty (normalised ΔSPL fraction) when NeuralFoil/surrogate fail.
# Treat as 10 dB above baseline, expressed as normalised fraction.
# A failed NeuralFoil call is rare but must not crash the training run.
AERO_FAIL_PENALTY_NORM_DSPL = 10.0 / 43.681  # ≈ 0.229 (using baseline 43.681 dB)

# SPL normalisation anchor — baseline SPL for NACA 0012 at fixed condition.
# Loaded from disk at init; this default is used only if the file is missing.
_BASELINE_SPL_FALLBACK = 43.681  # dB — from compute_baseline.py
_BASELINE_EFF_FALLBACK = 58.70   # L/D — from compute_baseline.py


# ════════════════════════════════════════════════════════════════════════════
# HELPER: GEOMETRY THICKNESS CHECK
# ════════════════════════════════════════════════════════════════════════════

def _min_thickness_ok(upper: np.ndarray, lower: np.ndarray) -> bool:
    """
    Additional geometry check beyond is_physically_valid().

    Prevents near-flat plates that would fool the BPM model into predicting
    extremely low SPL (δ* → 0) while having CL ≈ 0.

    Rule: maximum upper y − minimum lower y must exceed 1% chord.
    At our chord of 0.5 m this means at least 5 mm thickness.
    """
    # Quick estimate using CST weights at mid-chord (x ≈ 0.5)
    # Upper surface midpoint y ≈ mean of weights × class function peak ≈ 0.35 × mean
    est_upper_thick = np.mean(upper) * 0.35
    est_lower_thick = np.mean(lower) * 0.35  # typically negative for real airfoil
    thickness_est   = est_upper_thick - est_lower_thick
    return thickness_est > 0.005  # at least 0.5% of chord in CST units


def _flatten_cst(
    upper: np.ndarray,
    lower: np.ndarray,
    le:    float,
) -> np.ndarray:
    """Concatenate CST params into the standard 17-element vector."""
    return np.concatenate([upper, lower, [le]]).astype(np.float32)


# ════════════════════════════════════════════════════════════════════════════
# MAIN ENVIRONMENT CLASS
# ════════════════════════════════════════════════════════════════════════════

class AirfoilEnvAcoustic(gym.Env):
    """
    Extended DRLFoil Gymnasium environment with BPM acoustic penalty.

    Parameters
    ----------
    acoustic_lambda : float
        Weight of the acoustic penalty term in the reward (λ).
        λ = 0.0 → pure DRLFoil behaviour (sanity-check mode).
        λ = 0.5 → light noise penalty.
        λ = 6.0 → aggressive noise penalty.
        See config.LAMBDA_VALUES for the full sweep set.

    max_steps : int, optional
        Steps per episode before automatic reset.
        Defaults to config.MAX_STEPS (= 10).

    n_params : int, optional
        CST weights per surface.
        Defaults to config.N_CST_PARAMS (= 8).

    scale_actions : float, optional
        Maximum absolute change per CST weight per step.
        Defaults to config.SCALE_ACTIONS (= 0.3).

    cl_target : float, optional
        Target lift coefficient (fixed operating condition).
        Defaults to config.CL_TARGET (= 0.6).

    re : float, optional
        Reynolds number.
        Defaults to config.RE (= 500 000).

    cl_wide : float, optional
        Gaussian width γ in the CL accuracy term of the reward.
        Defaults to config.CL_WIDE (= 20). 

    efficiency_param : float, optional
        Scale factor α on the aerodynamic efficiency reward.
        Defaults to config.EFFICIENCY_PARAM (= 1.0).

    use_xfoil : bool, optional
        If True, use real XFOIL for δ* (evaluation mode).
        If False (default), use the fast surrogate (training mode).

    neuralfoil_size : str, optional
        NeuralFoil model size passed to AeroSandbox.
        "small" (default during training), "large" or "xlarge" for evaluation.
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        acoustic_lambda  : float = 0.0,
        max_steps        : Optional[int]   = None,
        n_params         : Optional[int]   = None,
        scale_actions    : Optional[float] = None,
        cl_target        : Optional[float] = None,
        re               : Optional[float] = None,
        cl_wide          : Optional[float] = None,
        efficiency_param : Optional[float] = None, 
        use_xfoil        : bool  = False,
        neuralfoil_size  : str   = "small",
        seed             : Optional[int] = None,
    ):
        super().__init__()

        # ── Hyperparameters ───────────────────────────────────────────────
        self.acoustic_lambda  = float(acoustic_lambda)
        self.max_steps        = int(max_steps)        if max_steps        is not None else config.MAX_STEPS
        self.n_params         = int(n_params)         if n_params         is not None else config.N_CST_PARAMS
        self.scale_actions    = float(scale_actions)  if scale_actions    is not None else config.SCALE_ACTIONS
        self.cl_target        = float(cl_target)      if cl_target        is not None else config.CL_TARGET
        self.re_current       = float(re)             if re               is not None else float(config.RE)
        self.cl_wide          = float(cl_wide)        if cl_wide          is not None else float(config.CL_WIDE)
        self.efficiency_param = float(efficiency_param) if efficiency_param is not None else config.EFFICIENCY_PARAM
        self.use_xfoil        = bool(use_xfoil)
        self.neuralfoil_size  = neuralfoil_size

        # ── Derived constants ─────────────────────────────────────────────
        self.n_cst_total = 2 * self.n_params + 1   # upper(8) + lower(8) + le(1) = 17
        self.obs_dim     = self.n_cst_total + 3    # + cl_target + Re_norm + spl_norm = 20

        # ── Baseline values ───────────────────────────────────────────────
        # Loaded from disk (written by compute_baseline.py).
        # The reward uses ΔSPL_norm = (SPL - SPL_baseline) / |SPL_baseline|,
        # so baseline SPL is the anchor for the acoustic penalty.
        self.baseline_spl, self.baseline_eff = self._load_baseline()

        # ── Surrogate status ──────────────────────────────────────────────
        self._surrogate_available = is_surrogate_loaded()
        if not self._surrogate_available:
            warnings.warn(
                "[AirfoilEnvAcoustic] δ* surrogate not loaded — "
                "using Blasius fallback. Run train_dstar_surrogate.py first.",
                stacklevel=2,
            )

        # ── Episode state (initialised properly in reset()) ───────────────
        self._step_count    = 0
        self._episode_count = 0 

        # CST parameter vectors (raw, unnormalised)
        rng = np.random.default_rng(seed=self._episode_count)  # reproducible per episode
        self._upper = rng.uniform(0.05, 0.25, self.n_params).astype(np.float32)
        self._lower = rng.uniform(-0.25, -0.05, self.n_params).astype(np.float32)
        self._le    = float(rng.uniform(-0.05, 0.05))

        # Current acoustic and aerodynamic state (filled at reset/step)
        self.current_spl = self.baseline_spl
        self.current_cl  = 0.0
        self.current_cd  = 1e-3   # safe non-zero default
        self.current_ld  = 0.0

        # Diagnostics — inspectable from outside (e.g. evaluate_all.py)
        self.xfoil_fail_count     = 0
        self.neuralfoil_fail_count = 0
        self.total_step_count     = 0

        # ── Gymnasium spaces ──────────────────────────────────────────────
        # Action space: normalised delta in [-1, 1], applied as Δw = a × SCALE
        # One delta per CST weight: 8 upper + 8 lower + 1 LE = 17
        self.action_space = spaces.Box(
            low   = -1.0,
            high  =  1.0,
            shape = (self.n_cst_total,),
            dtype = np.float32,
        ) 

        # Observation space
        # Upper weights:  [CST_UPPER_MIN, CST_UPPER_MAX] × n_params
        # Lower weights:  [CST_LOWER_MIN, CST_LOWER_MAX] × n_params
        # LE weight:      [-0.1, 0.1]
        # CL target:      [0.0, 2.0]   (always 0.6 in our study)
        # Re_norm:        [0.1, 3.0]   (Re / 1e6, always 0.5 here)
        # SPL_norm:       [0.0, 1.5]   (SPL / 100.0, typical 0.4–0.7)
        low_upper = np.full(self.n_params, config.CST_UPPER_MIN, dtype=np.float32)
        high_upper= np.full(self.n_params, config.CST_UPPER_MAX, dtype=np.float32)
        low_lower = np.full(self.n_params, config.CST_LOWER_MIN, dtype=np.float32)
        high_lower= np.full(self.n_params, config.CST_LOWER_MAX, dtype=np.float32)

        obs_low  = np.concatenate([
            low_upper, low_lower,
            [-0.1],           # LE
            [0.0],            # cl_target
            [0.1],            # Re_norm
            [0.0],            # spl_norm
        ]).astype(np.float32)

        obs_high = np.concatenate([
            high_upper, high_lower,
            [0.1],            # LE
            [2.0],            # cl_target
            [3.0],            # Re_norm
            [1.5],            # spl_norm
        ]).astype(np.float32)

        self.observation_space = spaces.Box(
            low   = obs_low,
            high  = obs_high,
            shape = (self.obs_dim,),
            dtype = np.float32,
        )

    # ════════════════════════════════════════════════════════════════════════
    # GYMNASIUM INTERFACE
    # ════════════════════════════════════════════════════════════════════════

    def reset(
        self,
        seed    : Optional[int] = None,
        options : Optional[Dict[str, Any]] = None,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Reset the environment to the seed airfoil.

        The seed airfoil (config.AIRFOIL_SEED_*) is a mildly cambered starting
        point.  Its SPL is computed once here and stored as self.current_spl.
        All subsequent steps compute ΔSPL relative to self.baseline_spl.

        Returns
        -------
        obs   : np.ndarray, shape (20,)
        info  : dict  (empty — matches Gymnasium API)
        """
        super().reset(seed=seed)

        self._step_count    = 0
        self._episode_count += 1

        # Reinitialise CST params from seed
        self._upper = np.array(config.AIRFOIL_SEED_UPPER, dtype=np.float32).copy()
        self._lower = np.array(config.AIRFOIL_SEED_LOWER, dtype=np.float32).copy()
        self._le    = float(config.AIRFOIL_SEED_LE[0])

        # Compute initial SPL for the seed airfoil
        cst_vec = _flatten_cst(self._upper, self._lower, self._le)
        dsp, dss = self._compute_dstar(cst_vec)
        spl, _, _ = compute_spl_tbl_te(dsp, dss)
        self.current_spl = float(np.clip(spl, config.SPL_CLIP_MIN, config.SPL_CLIP_MAX))

        # Compute initial aero state
        coords = cst_to_coordinates(self._upper, self._lower, self._le)
        self.current_cl, self.current_cd = self._compute_aero(coords)
        self.current_ld = (self.current_cl / self.current_cd
                           if self.current_cd > 0 else 0.0)

        obs = self._get_observation()
        return obs, {}

    def step(
        self,
        action: np.ndarray,
    ) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """
        Apply an action (CST weight delta) and return the new state.

        Action processing:
          1. Scale: Δw = action × SCALE_ACTIONS
          2. Apply: new_weight = old_weight + Δw
          3. Clip to [CST_MIN, CST_MAX] bounds

        Returns
        -------
        obs        : np.ndarray (20,)
        reward     : float
        terminated : bool  — always False (episode length controlled by truncated)
        truncated  : bool  — True when step_count == max_steps
        info       : dict  with diagnostic keys
        """
        self._step_count   += 1
        self.total_step_count += 1

        # ── 1. Apply action ───────────────────────────────────────────────
        action = np.array(action, dtype=np.float32).clip(-1.0, 1.0)
        delta  = action * self.scale_actions      # shape (17,)

        new_upper = np.clip(
            self._upper + delta[:self.n_params],
            config.CST_UPPER_MIN,
            config.CST_UPPER_MAX,
        ).astype(np.float32)

        new_lower = np.clip(
            self._lower + delta[self.n_params : 2 * self.n_params],
            config.CST_LOWER_MIN,
            config.CST_LOWER_MAX,
        ).astype(np.float32)

        new_le = float(np.clip(
            self._le + delta[-1],
            -0.1,
            0.1,
        ))

        # ── 2. Geometry validity check ────────────────────────────────────
        if not is_physically_valid(new_upper, new_lower):
            # Do NOT update state — keep current geometry
            reward = INVALID_GEOM_PENALTY
            obs    = self._get_observation()
            truncated = (self._step_count >= self.max_steps)
            return obs, reward, False, truncated, {
                "reason": "invalid_geometry",
                "step":   self._step_count,
            }

        if not _min_thickness_ok(new_upper, new_lower):
            reward = INVALID_GEOM_PENALTY
            obs    = self._get_observation()
            truncated = (self._step_count >= self.max_steps)
            return obs, reward, False, truncated, {
                "reason": "invalid_geometry",
                "step":   self._step_count,
            }

        # ── 3. Accept geometry, compute physics ───────────────────────────
        self._upper = new_upper
        self._lower = new_lower
        self._le    = new_le

        # Convert CST to (x, y) coordinates
        coords  = cst_to_coordinates(self._upper, self._lower, self._le)
        cst_vec = _flatten_cst(self._upper, self._lower, self._le)

        # ── 4. Aerodynamics: CL and CD via NeuralFoil ────────────────────
        cl, cd = self._compute_aero(coords)
        self.current_cl = cl
        self.current_cd = cd
        self.current_ld = cl / cd if cd > 0 else 0.0

        # ── 5. Acoustics: δ* → SPL via surrogate (or XFOIL) ──────────────
        dsp, dss = self._compute_dstar(cst_vec)
        spl, _, _ = compute_spl_tbl_te(dsp, dss)
        self.current_spl = float(np.clip(spl, config.SPL_CLIP_MIN, config.SPL_CLIP_MAX))

        # ── 6. Reward ─────────────────────────────────────────────────────
        reward = self._compute_reward(cl, cd)

        # ── 7. Observation and termination ────────────────────────────────
        obs       = self._get_observation()
        truncated = (self._step_count >= self.max_steps)

        info = {
            "cl":          cl,
            "cd":          cd,
            "ld":          self.current_ld,
            "spl":         self.current_spl,
            "delta_spl":   self.current_spl - self.baseline_spl,
            "step":        self._step_count,
            "ep_num":     self._episode_count,
        }

        return obs, reward, False, truncated, info

    def render(self):
        """Minimal render — prints current state to stdout."""
        print(
            f"  Episode {self._episode_count:4d}  "
            f"Step {self._step_count:2d}/{self.max_steps}  "
            f"CL={self.current_cl:+.4f}  "
            f"CD={self.current_cd:.6f}  "
            f"L/D={self.current_ld:6.2f}  "
            f"SPL={self.current_spl:.2f} dB  "
            f"ΔSPL={self.current_spl - self.baseline_spl:+.2f} dB"
        )

    def close(self):
        pass

    # ════════════════════════════════════════════════════════════════════════
    # OBSERVATION CONSTRUCTION
    # ════════════════════════════════════════════════════════════════════════

    def _get_observation(self) -> np.ndarray:
        """
        Build the 20-element observation vector.

        Layout:
          [0  : 8 ] upper CST weights       (raw, in [-0.05, 0.5])
          [8  :16 ] lower CST weights       (raw, in [-0.5,  0.05])
          [16 ]     LE weight               (raw, in [-0.1,  0.1 ])
          [17 ]     CL target               (always config.CL_TARGET = 0.6)
          [18 ]     Reynolds number / 1e6   (= 0.5 here)
          [19 ]     current SPL / 100.0     (normalised; ~0.4 for NACA0012)

        The agent therefore knows:
          - current airfoil shape (CST params)
          - what CL it is trying to achieve
          - current noise level (so it can correlate geometry changes with SPL)
        """
        return np.concatenate([
            self._upper.astype(np.float32),
            self._lower.astype(np.float32),
            np.array([self._le], dtype=np.float32),
            np.array([self.cl_target], dtype=np.float32),
            np.array([self.re_current / 1e6], dtype=np.float32),
            np.array([self.current_spl / 100.0], dtype=np.float32),
        ]).astype(np.float32)

    # ════════════════════════════════════════════════════════════════════════
    # REWARD FUNCTION
    # ════════════════════════════════════════════════════════════════════════

    def _compute_reward(
        self,
        cl : float,
        cd : float,
    ) -> float:
        """
        Compute the combined aeroacoustic reward.

        r = α · E · exp(−γ · ΔCL²)  −  λ · ΔSPL_norm

        where:
          E             = CL / CD              (aerodynamic efficiency)
          ΔCL           = CL − CL_target       (CL accuracy error)
          ΔSPL_norm     = (SPL − SPL_baseline) / |SPL_baseline|
                          positive → louder than baseline (bad)
                          negative → quieter than baseline (good)

        The normalisation of ΔSPL ensures that the acoustic penalty is
        dimensionless and its scale is comparable to the efficiency reward.
        At our baseline (SPL ≈ 43.7 dB, L/D ≈ 58.7):
          aero reward ≈ 1.0 × 58.7 × exp(0) ≈ 58.7
          acoustic penalty ≈ λ × (ΔSPL / 43.7)
          At λ = 3.0 and ΔSPL = 5 dB: penalty ≈ 3.0 × 0.114 ≈ 0.34
        This keeps the penalty as a small but meaningful fraction of the
        aero reward, preventing reward collapse.

        Extra penalties:
          −50   if CL < CL_FLOOR (guards against degenerate flat-plate collapse)
          −100  for invalid geometry (handled in step() before this is called)
        """
        # Guard against aerodynamic degeneracy
        if cd <= 0 or np.isnan(cd) or np.isnan(cl) or np.isinf(cl) or np.isinf(cd):
            return float(INVALID_GEOM_PENALTY)

        # CL floor check — prevents λ > 0 reward-collapse to near-zero-lift shapes
        if cl < CL_FLOOR:
            return float(CL_FLOOR_PENALTY)

        # ── Aerodynamic term ──────────────────────────────────────────────
        efficiency = cl / cd
        delta_cl   = cl - self.cl_target
        cl_gauss   = float(np.exp(-self.cl_wide * delta_cl ** 2))
        aero_reward = self.efficiency_param * efficiency * cl_gauss

        # ── Acoustic penalty ──────────────────────────────────────────────
        # Skip for λ = 0 to avoid the tiny overhead on the pure-DRLFoil run.
        if self.acoustic_lambda > 0.0:
            delta_spl_norm = (
                (self.current_spl - self.baseline_spl)
                / abs(self.baseline_spl)
            )
            acoustic_penalty = self.acoustic_lambda * delta_spl_norm
        else:
            acoustic_penalty = 0.0

        reward = aero_reward - acoustic_penalty
        return float(reward)

    # ════════════════════════════════════════════════════════════════════════
    # PHYSICS HELPERS
    # ════════════════════════════════════════════════════════════════════════

    def _compute_dstar(
        self,
        cst_vec: np.ndarray,
    ) -> Tuple[float, float]:
        """
        Predict δ*_pressure and δ*_suction from CST parameters.

        Training mode (use_xfoil=False):
          Uses the trained MLP surrogate — ~0.177 ms per call.
          Falls back to Blasius if surrogate not loaded.

        Evaluation mode (use_xfoil=True):
          Uses real XFOIL via subprocess — 0.1–12 s per call.
          Falls back to surrogate if XFOIL fails to converge.

        Returns
        -------
        (delta_star_pressure, delta_star_suction) in metres, both clipped to
        [DSTAR_CLIP_MIN, DSTAR_CLIP_MAX].
        """
        if not self.use_xfoil:
            # ── Surrogate path (training) ─────────────────────────────────
            return predict_dstar_fast(cst_vec)

        # ── XFOIL path (evaluation) ───────────────────────────────────────
        coords = cst_to_coordinates(
            cst_vec[:self.n_params],
            cst_vec[self.n_params : 2 * self.n_params],
            float(cst_vec[-1]),
        )

        result = get_delta_star(
            coords,
            aoa_deg = config.AOA_DEG,
            Re      = self.re_current,
            chord   = config.CHORD,
        )

        if result is None:
            self.xfoil_fail_count += 1
            # Fallback to surrogate first, then Blasius
            if self._surrogate_available:
                return predict_dstar_fast(cst_vec)
            return blasius_fallback(Re=self.re_current)

        return result

    def _compute_aero(
        self,
        coords: np.ndarray,
    ) -> Tuple[float, float]:
        """
        Compute (CL, CD) for the current airfoil shape via NeuralFoil.

        NeuralFoil is ~10–50× faster than XFOIL and runs entirely in-process
        (no subprocess overhead), making it suitable for RL reward computation.
        We use model_size="small" during training for further speed, and
        "xlarge" during evaluation for accuracy.

        Returns
        -------
        (CL, CD) as floats.  On failure returns (0.0, 1.0) — zero efficiency,
        which the reward function will penalise appropriately.
        """
        if not _HAS_NEURALFOIL:
            # Flat-plate thin-airfoil approximation as last resort
            cl_approx = 2.0 * np.pi * np.radians(config.AOA_DEG)
            cd_approx = 0.012  # typical at Re = 500k
            return float(cl_approx), float(cd_approx)

        try:
            af   = asb.Airfoil(coordinates=coords)
            aero = af.get_aero_from_neuralfoil(
                alpha      = config.AOA_DEG,
                Re         = self.re_current,
                mach       = config.MACH,
                model_size = self.neuralfoil_size,
            )

            cl = float(np.atleast_1d(aero["CL"]).flatten()[0])
            cd = float(np.atleast_1d(aero["CD"]).flatten()[0])

            # Sanity bounds
            if not np.isfinite(cl) or not np.isfinite(cd):
                raise ValueError(f"NeuralFoil returned non-finite: CL={cl}, CD={cd}")
            if cd <= 0:
                raise ValueError(f"Non-positive CD = {cd}")

            return cl, cd

        except Exception as exc:
            self.neuralfoil_fail_count += 1
            # Log at most once per 1000 failures to avoid log spam during training
            if self.neuralfoil_fail_count % 1000 == 1:
                warnings.warn(
                    f"[AirfoilEnvAcoustic] NeuralFoil failed "
                    f"({self.neuralfoil_fail_count} times): {exc}",
                    stacklevel=2,
                )
            return 0.0, 1.0   # efficiency = 0 → reward ≈ 0

    # ════════════════════════════════════════════════════════════════════════
    # BASELINE LOADING
    # ════════════════════════════════════════════════════════════════════════

    def _load_baseline(self) -> Tuple[float, float]:
        """
        Load NACA 0012 baseline SPL and L/D from disk.

        These are written once by evaluation/compute_baseline.py.
        If not found, fall back to hard-coded values from the midterm report.

        Returns
        -------
        (baseline_spl, baseline_eff) as floats.
        """
        spl = _BASELINE_SPL_FALLBACK
        eff = _BASELINE_EFF_FALLBACK

        spl_path = config.BASELINE_SPL
        eff_path = config.BASELINE_EFF

        if Path(spl_path).exists():
            try:
                spl = float(np.load(spl_path))
            except Exception:
                pass

        if Path(eff_path).exists():
            try:
                eff = float(np.load(eff_path))
            except Exception:
                pass

        return float(spl), float(eff)

    # ════════════════════════════════════════════════════════════════════════
    # PUBLIC PROPERTIES / UTILITIES
    # ════════════════════════════════════════════════════════════════════════

    @property
    def current_airfoil_coords(self) -> np.ndarray:
        """Return (x, y) coordinates of the current airfoil. Useful for plotting."""
        return cst_to_coordinates(self._upper, self._lower, self._le)

    @property
    def current_cst_params(self) -> np.ndarray:
        """Return the 17-element CST parameter vector."""
        return _flatten_cst(self._upper, self._lower, self._le)

    def get_diagnostics(self) -> Dict[str, Any]:
        """
        Return a dict of diagnostic counters.
        Useful for monitoring convergence rates during training.
        """
        return {
            "xfoil_fail_count"     : self.xfoil_fail_count,
            "neuralfoil_fail_count": self.neuralfoil_fail_count,
            "total_steps"          : self.total_step_count,
            "episode"              : self._episode_count,
            "baseline_spl"         : self.baseline_spl,
            "baseline_eff"         : self.baseline_eff,
            "current_spl"          : self.current_spl,
            "current_ld"           : self.current_ld,
            "acoustic_lambda"      : self.acoustic_lambda,
            "surrogate_available"  : self._surrogate_available,
        }
