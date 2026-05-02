"""Single source of truth for all the constants."""

from pathlib import Path

# ── Fixed operating condition ────────────────────────────────────────────────
RE      = 500_000           # Reynolds number                [dimensionless]
CHORD   = 0.5               # airfoil chord length           [m]
NU_AIR  = 1.5e-5            # kinematic viscosity of air     [m²/s]
U_INF   = RE * NU_AIR / CHORD   # = 15.0 m/s — DERIVED 
RHO_AIR = 1.225             # air density at sea level       [kg/m³]
C_SOUND = 343.0             # speed of sound at 20°C         [m/s]
MACH    = U_INF / C_SOUND   # ≈ 0.044 — well within BPM's low-Mach validity

# AoA = 5° — moderate angle, within BPM calibration range, gives CL ≈ 0.6
# on NACA 0012 at Re = 500k, consistent with [OC2025] Table 15's CL_target=0.6
AOA_DEG = 5.0               # angle of attack                [degrees]

# BPM observer geometry — standard convention in aeroacoustics literature
SPAN    = 1.0               # spanwise extent for SPL calc   [m] (per-unit-span)
R_OBS   = 1.0               # observer distance from TE      [m]
THETA_OBS = 90.0            # polar angle: 90° = directly overhead [degrees]
PHI_OBS   = 90.0            # azimuthal angle: 90° = overhead     [degrees]


# ── CST parameterisation ─────────────────────────────────────────────────────
N_CST_PARAMS    = 8    # per surface (upper + lower)
CST_UPPER_MIN    = -0.05    # minimum value for upper surface weights
CST_UPPER_MAX    =  0.50    # maximum value for upper surface weights
CST_LOWER_MIN    = -0.50    # minimum value for lower surface weights
CST_LOWER_MAX    =  0.05    # maximum value for lower surface weights

# ── DRLFoil environment ──────────────────────────────────────────────────────
MAX_STEPS       = 10
SCALE_ACTIONS   = 0.05
CL_TARGET       = 0.6
CL_WIDE         = 20
EFFICIENCY_PARAM = 1.0
AIRFOIL_SEED_UPPER = [0.17] * N_CST_PARAMS
AIRFOIL_SEED_LOWER = [-0.17] * N_CST_PARAMS
AIRFOIL_SEED_LE    = [0.0]  # leading edge parameter

# ── PPO hyperparameters (Orgeira-Crespo Table 10) ────────────────────────────
PPO_LEARNING_RATE  = 0.000268
PPO_N_STEPS        = 32
PPO_BATCH_SIZE     = 512
PPO_N_EPOCHS       = 20
PPO_GAMMA          = 0.995
PPO_GAE_LAMBDA     = 0.98
PPO_ENT_COEF       = 0.001
PPO_VF_COEF        = 0.754843
PPO_CLIP_RANGE     = 0.3
PPO_MAX_GRAD_NORM  = 5.0
PPO_NET_ARCH       = [128, 128]
PPO_N_ENVS         = 4
PPO_TOTAL_STEPS    = 2_000_000

# ── λ sweep values ───────────────────────────────────────────────────────────
LAMBDA_VALUES = [0.0, 0.5, 1.5, 3.0, 6.0]

# ── δ* surrogate dataset ─────────────────────────────────────────────────────
DSTAR_N_SAMPLES   = 5_000
DSTAR_RANDOM_SEED = 42

# ── BPM validity bounds ──────────────────────────────────────────────────────
DSTAR_MIN = 1e-5   # m  — clip below this
DSTAR_MAX = 0.025  # m  — clip above this
SPL_MIN   = 20.0   # dB
SPL_MAX   = 120.0  # dB

# ════════════════════════════════════════════════════════════════════════════
# BPM MODEL PHYSICAL BOUNDS
# ════════════════════════════════════════════════════════════════════════════
# Why these bounds exist:
#
# DSTAR_CLIP: BPM contains log10(δ* × M^5 × ...) inside the SPL formula.
# If δ* is zero or negative (which can happen from XFOIL non-convergence
# or a degenerate CST shape), log10 returns -inf or crashes entirely.
# 1e-5 m (10 microns) is far below any real attached boundary layer at
# Re = 500,000 — if δ* is this small, the geometry is degenerate anyway.
# 0.025 m (5% of chord = 25 mm) is beyond any attached BL at this Re.
#
# SPL_CLIP: Prevents reward function from seeing +inf or -inf dB values
# from extreme geometries. 20–120 dB covers all physically meaningful
# aeroacoustic situations.

DSTAR_CLIP_MIN = 1e-5    # m  — minimum believable δ*
DSTAR_CLIP_MAX = 0.025   # m  — maximum believable δ* (5% chord)
SPL_CLIP_MIN   = 20.0    # dB — quieter than this is not physically real here
SPL_CLIP_MAX   = 120.0   # dB — louder than this means something is broken

# ── Success criteria ─────────────────────────────────────────────────────────
SPL_REDUCTION_TARGET_DB = 3.0   # minimum dB reduction vs NACA0012
LD_RETENTION_FRACTION   = 0.85  # must keep ≥85% of baseline L/D
XFOIL_CONVERGENCE_MIN   = 0.70  # reject runs with >30% XFOIL failures

# ── Paths ────────────────────────────────────────────────────────────────────
from pathlib import Path
ROOT          = Path(__file__).parent
DATA_RAW      = ROOT / "data" / "raw"
DATA_GEN      = ROOT / "data" / "generated"
DATA_SURR     = ROOT / "data" / "surrogate"
MODELS_DIR    = ROOT / "trained_models"
RESULTS_DIR   = ROOT / "results"
NASA_DATASET  = DATA_RAW / "nasa_airfoil_self_noise" / "airfoil_self_noise.dat"

DSTAR_X_PATH  = DATA_GEN / "dstar_X.npy"
DSTAR_Y_PATH  = DATA_GEN / "dstar_y.npy"
BASELINE_SPL  = DATA_GEN / "baseline_spl.npy"
BASELINE_EFF  = DATA_GEN / "baseline_eff.npy"

SURROGATE_WEIGHTS  = DATA_SURR / "dstar_surrogate.pt"
SURROGATE_SCALER_X = DATA_SURR / "dstar_scaler_X.pkl"
SURROGATE_SCALER_Y = DATA_SURR / "dstar_scaler_y.pkl"

import os, platform

# XFOIL binary path
if platform.system() == "Windows":
    XFOIL_BIN   = ROOT / "xfoil.exe"
    XFOIL_SHELL = True    # Windows subprocess needs shell=True
else:
    XFOIL_BIN   = "xfoil"
    XFOIL_SHELL = False