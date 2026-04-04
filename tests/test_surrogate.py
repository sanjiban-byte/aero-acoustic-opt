# tests/test_surrogate.py
"""
Test the delta* surrogate MLP.

What we check:
  1. Surrogate loads without error
  2. Speed: 1000 predictions in < 1 second
  3. Outputs are positive and within physical bounds
  4. Suction side > pressure side at our fixed AoA=5°
  5. Rank correlation with analytical ground truth > 0.75
"""

import numpy as np
import time
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
import config
from boundary_layer.dstar_inference import predict_dstar_fast, is_surrogate_loaded
from boundary_layer.generate_dstar_dataset import (
    compute_dstar_analytical, is_physically_valid
)


def test_surrogate_loads():
    """Surrogate must load successfully."""
    print("\n── Test 1: Surrogate Loads ──")
    assert is_surrogate_loaded(), (
        "Surrogate not loaded. Run train_dstar_surrogate.py first."
    )
    print("  ✅ PASSED")


def test_speed():
    """
    1000 predictions must complete in < 1 second.
    At 2M training steps with 4 parallel envs, we need ~8M surrogate calls.
    1000 calls/sec means 8M calls takes ~2.2 hours — acceptable.
    """
    print("\n── Test 2: Inference Speed ──")

    rng = np.random.default_rng(0)
    params = [
        rng.uniform(-0.3, 0.4, 17).astype(np.float32)
        for _ in range(1000)
    ]

    t0 = time.time()
    for p in params:
        predict_dstar_fast(p)
    elapsed = time.time() - t0

    ms_per_call = elapsed / 1000 * 1000
    print(f"  1000 predictions in {elapsed:.3f}s ({ms_per_call:.3f} ms/call)")
    print(f"  XFOIL equivalent would take: ~{1000 * 0.5:.0f}s")
    print(f"  Speedup: ~{500 / elapsed:.0f}×")

    assert elapsed < 1.0, f"1000 predictions took {elapsed:.2f}s > 1s"
    print("  ✅ PASSED")


def test_output_bounds():
    """All outputs must be positive and within physical bounds."""
    print("\n── Test 3: Output Physical Bounds ──")

    rng = np.random.default_rng(1)
    n_fail = 0

    for _ in range(200):
        params = rng.uniform(-0.5, 0.5, 17).astype(np.float32)
        dstar_p, dstar_s = predict_dstar_fast(params)

        if not (config.DSTAR_CLIP_MIN <= dstar_p <= config.DSTAR_CLIP_MAX):
            n_fail += 1
        if not (config.DSTAR_CLIP_MIN <= dstar_s <= config.DSTAR_CLIP_MAX):
            n_fail += 1

    print(f"  Out-of-bounds predictions: {n_fail} / 400")
    assert n_fail == 0, f"{n_fail} predictions outside physical bounds"
    print("  ✅ PASSED")


def test_suction_greater_than_pressure():
    """
    For physically valid CST geometries at AoA=5°,
    suction-side δ* should generally exceed pressure-side δ*.
    We allow up to 20% violations since the surrogate is approximate.
    """
    print("\n── Test 4: Suction > Pressure ──")

    rng = np.random.default_rng(2)
    n_correct = 0
    n_total   = 0

    for _ in range(300):
        upper = rng.uniform(config.CST_UPPER_MIN,
                            config.CST_UPPER_MAX,
                            config.N_CST_PARAMS)
        lower = rng.uniform(config.CST_LOWER_MIN,
                            config.CST_LOWER_MAX,
                            config.N_CST_PARAMS)
        le    = float(rng.uniform(-0.02, 0.02))

        if not is_physically_valid(upper, lower):
            continue

        params  = np.concatenate([upper, lower, [le]]).astype(np.float32)
        dstar_p, dstar_s = predict_dstar_fast(params)

        if dstar_s > dstar_p:
            n_correct += 1
        n_total += 1

    pct = n_correct / n_total * 100
    print(f"  Suction > Pressure: {n_correct}/{n_total} ({pct:.1f}%)")
    assert pct > 80.0, f"Only {pct:.1f}% of predictions have δ*_s > δ*_p"
    print("  ✅ PASSED")


def test_rank_correlation():
    """
    Surrogate rank ordering must correlate with analytical ground truth.
    We generate 100 random valid geometries, compare surrogate predictions
    to analytical values. Spearman ρ > 0.75 required.
    """
    print("\n── Test 5: Rank Correlation with Analytical Ground Truth ──")

    from scipy.stats import spearmanr

    rng = np.random.default_rng(3)

    surrogate_totals  = []
    analytical_totals = []

    for _ in range(200):
        upper = rng.uniform(config.CST_UPPER_MIN,
                            config.CST_UPPER_MAX,
                            config.N_CST_PARAMS)
        lower = rng.uniform(config.CST_LOWER_MIN,
                            config.CST_LOWER_MAX,
                            config.N_CST_PARAMS)
        le = float(rng.uniform(-0.02, 0.02))

        if not is_physically_valid(upper, lower):
            continue

        params = np.concatenate([upper, lower, [le]]).astype(np.float32)

        # Surrogate prediction
        dstar_p_surr, dstar_s_surr = predict_dstar_fast(params)
        surrogate_totals.append(dstar_p_surr + dstar_s_surr)

        # Analytical ground truth (no noise)
        rng_dummy = np.random.default_rng(0)
        dstar_p_anal, dstar_s_anal = compute_dstar_analytical(upper, lower, le)
        analytical_totals.append(dstar_p_anal + dstar_s_anal)

    rho, p_val = spearmanr(surrogate_totals, analytical_totals)
    print(f"  Spearman ρ = {rho:.4f}  (p = {p_val:.2e})")
    print(f"  Samples compared: {len(surrogate_totals)}")

    assert rho > 0.75, f"Rank correlation {rho:.3f} < 0.75"
    print("  ✅ PASSED")


if __name__ == "__main__":
    test_surrogate_loads()
    test_speed()
    test_output_bounds()
    test_suction_greater_than_pressure()
    test_rank_correlation()
    print("\n══ All surrogate tests passed ══")