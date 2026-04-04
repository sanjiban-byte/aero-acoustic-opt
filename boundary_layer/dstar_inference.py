# boundary_layer/dstar_inference.py
"""
Fast delta* inference using the trained MLP surrogate.

This module is imported by the RL environment.
predict_dstar_fast() replaces get_delta_star() during training.

Speed comparison:
  get_delta_star (XFOIL)    : 0.1 – 12s per call
  predict_dstar_fast (MLP)  : ~0.1ms per call

The surrogate is loaded ONCE at module import time and kept in memory.
All subsequent calls are pure tensor operations — no subprocess overhead.
"""

import numpy as np
import torch
import joblib
import sys
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).parent.parent))
import config


# ════════════════════════════════════════════════════════════════════════════
# MODEL DEFINITION  (must match train_dstar_surrogate.py exactly)
# ════════════════════════════════════════════════════════════════════════════

import torch.nn as nn

class DStarMLP(nn.Module):
    def __init__(self, in_dim: int = 17):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 2),
            nn.Softplus(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ════════════════════════════════════════════════════════════════════════════
# LOAD SURROGATE AT MODULE IMPORT TIME
# ════════════════════════════════════════════════════════════════════════════
# Why at import time and not inside the function?
# The RL environment calls predict_dstar_fast() ~2,000,000 times during
# training. Loading the model inside the function would mean 2M file reads.
# Loading once at import keeps it in RAM for the entire training session.

_surrogate_loaded = False
_model            = None
_scaler_X         = None
_scaler_y         = None
_in_dim           = 17


def _load_surrogate():
    """Load surrogate weights and scalers from disk. Called once at import."""
    global _surrogate_loaded, _model, _scaler_X, _scaler_y, _in_dim

    if _surrogate_loaded:
        return True

    # Check all required files exist
    required = [
        config.SURROGATE_WEIGHTS,
        config.SURROGATE_SCALER_X,
        config.SURROGATE_SCALER_Y,
        config.DATA_SURR / "in_dim.pt",
    ]
    for path in required:
        if not Path(path).exists():
            return False

    try:
        _in_dim   = int(torch.load(
            config.DATA_SURR / "in_dim.pt",
            weights_only=True,
        ))
        _model    = DStarMLP(_in_dim)
        _model.load_state_dict(torch.load(
            config.SURROGATE_WEIGHTS,
            map_location="cpu",
            weights_only=True,
        ))
        _model.eval()

        _scaler_X = joblib.load(config.SURROGATE_SCALER_X)
        _scaler_y = joblib.load(config.SURROGATE_SCALER_Y)

        _surrogate_loaded = True
        return True

    except Exception as e:
        print(f"[dstar_inference] Failed to load surrogate: {e}")
        return False


# Attempt to load at import time
_load_surrogate()


# ════════════════════════════════════════════════════════════════════════════
# INFERENCE FUNCTION
# ════════════════════════════════════════════════════════════════════════════

def predict_dstar_fast(
    cst_params : np.ndarray,
) -> tuple[float, float]:
    """
    Predict delta*_pressure and delta*_suction from CST parameters.

    Parameters
    ----------
    cst_params : 1D numpy array of length 17
                 [upper_weights(8), lower_weights(8), le_weight(1)]
                 Same format as used in generate_dstar_dataset.py

    Returns
    -------
    (delta_star_pressure, delta_star_suction) in metres

    Falls back to blasius_fallback() if surrogate is not loaded.
    Never raises an exception — always returns two positive floats.
    """
    # Fallback if surrogate not available
    if not _surrogate_loaded:
        from boundary_layer.xfoil_bl import blasius_fallback
        return blasius_fallback()

    try:
        # Reshape to (1, 17) for scaler
        x_arr  = np.array(cst_params, dtype=np.float32).reshape(1, -1)

        # Normalise input
        x_norm = _scaler_X.transform(x_arr)
        x_t    = torch.tensor(x_norm, dtype=torch.float32)

        # Forward pass — no gradient needed at inference
        with torch.no_grad():
            y_norm = _model(x_t).numpy()

        # Inverse-transform to physical units
        y_phys = _scaler_y.inverse_transform(y_norm)[0]

        # Clip to physical bounds
        dstar_p = float(np.clip(y_phys[0],
                                config.DSTAR_CLIP_MIN,
                                config.DSTAR_CLIP_MAX))
        dstar_s = float(np.clip(y_phys[1],
                                config.DSTAR_CLIP_MIN,
                                config.DSTAR_CLIP_MAX))

        return dstar_p, dstar_s

    except Exception:
        # Never crash the RL environment
        from boundary_layer.xfoil_bl import blasius_fallback
        return blasius_fallback()


def is_surrogate_loaded() -> bool:
    """Check if surrogate is available. Used by environment at init."""
    return _surrogate_loaded