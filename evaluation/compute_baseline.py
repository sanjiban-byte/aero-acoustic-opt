"""
Compute and save the NACA 0012 baseline at our fixed operating condition.

Saves two files:
  data/generated/baseline_spl.npy  — scalar SPL in dB
  data/generated/baseline_eff.npy  — scalar L/D efficiency

These are loaded by the reward function and evaluation scripts.
Run this ONCE before training. Do not re-run during training.
"""

import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
import config
from boundary_layer.xfoil_bl import get_delta_star, blasius_fallback
from acoustics.bpm_model import compute_spl_tbl_te

# Also need aerosandbox for NACA 0012 coordinates and NeuralFoil for L/D
try:
    import aerosandbox as asb
    import aerosandbox.numpy as anp
    HAS_ASB = True
except ImportError:
    HAS_ASB = False
    print("WARNING: aerosandbox not available — using coordinate file fallback")


def get_naca0012_coords() -> np.ndarray:
    """
    Get NACA 0012 coordinates in XFOIL-compatible format.

    Tries aerosandbox first (generates clean cosine-spaced coords).
    Falls back to reading the UIUC .dat file.
    """
    if HAS_ASB:
        af     = asb.Airfoil("naca0012")
        coords = af.coordinates   # Nx2, normalised, XFOIL ordering
        print(f"  Loaded NACA 0012 from aerosandbox: {len(coords)} points")
        return coords

    # Fallback: read from UIUC file
    uiuc_file = config.UIUC_DIR / "naca0012.dat"
    if not uiuc_file.exists():
        raise FileNotFoundError(
            f"NACA 0012 coordinates not found at {uiuc_file}. "
            f"Either install aerosandbox or download naca0012.dat "
            f"from https://m-selig.ae.illinois.edu/ads/coord/naca0012.dat"
        )
    # UIUC files have a name line then x y pairs
    coords = []
    with open(uiuc_file) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) == 2:
                try:
                    coords.append([float(parts[0]), float(parts[1])])
                except ValueError:
                    continue   # skip header
    coords = np.array(coords)
    print(f"  Loaded NACA 0012 from UIUC file: {len(coords)} points")
    return coords


def compute_naca0012_ld() -> float:
    """
    Compute NACA 0012 L/D at fixed operating condition using NeuralFoil.

    NeuralFoil API note: in newer AeroSandbox versions, the Kulfan
    parameter method changed. We use the direct coordinate-based approach
    which is version-stable.
    """
    if not HAS_ASB:
        print("  WARNING: aerosandbox not available — using hardcoded NACA0012 L/D")
        return 55.0

    try:
        # Method 1: Use AeroSandbox's built-in NeuralFoil wrapper
        # This is the most version-stable approach
        af = asb.Airfoil("naca0012")

        aero = af.get_aero_from_neuralfoil(
            alpha      = config.AOA_DEG,
            Re         = config.RE,
            mach       = config.MACH,
            model_size = "xlarge",
        )

        CL = float(np.atleast_1d(aero["CL"]).flatten()[0])
        CD = float(np.atleast_1d(aero["CD"]).flatten()[0])
        print(f"  NeuralFoil (via AeroSandbox): CL = {CL:.4f}, CD = {CD:.6f}")

    except AttributeError:
        try:
            # Method 2: Direct NeuralFoil import with coordinates
            import neuralfoil as nf
            af     = asb.Airfoil("naca0012")
            coords = af.coordinates

            # NeuralFoil can accept coordinates directly in some versions
            aero = nf.get_aero_from_coordinates(
                coordinates = coords,
                alpha       = config.AOA_DEG,
                Re          = config.RE,
                model_size  = "xlarge",
            )
            CL = float(np.atleast_1d(aero["CL"]).flatten()[0])
            CD = float(np.atleast_1d(aero["CD"]).flatten()[0])
            print(f"  NeuralFoil (direct coords): CL = {CL:.4f}, CD = {CD:.6f}")

        except Exception:
            # Method 3: Hardcoded fallback
            # NACA 0012 at Re=500k, AoA=5° is well-documented in literature
            # CL ≈ 0.60, CD ≈ 0.011 → L/D ≈ 55
            print("  NeuralFoil API unavailable — using literature value")
            print("  NACA 0012 at Re=500k AoA=5°: CL≈0.60, CD≈0.011, L/D≈55")
            return 55.0

    if CD <= 0 or np.isnan(CL) or np.isnan(CD):
        print("  WARNING: Invalid aero values — using fallback L/D = 55")
        return 55.0

    ld = CL / CD
    print(f"  L/D = {ld:.2f}")
    return float(ld)


def main():
    print("=" * 55)
    print("Computing NACA 0012 baseline at fixed operating condition")
    print("=" * 55)
    print(f"  AoA    = {config.AOA_DEG}°")
    print(f"  Re     = {config.RE:,}")
    print(f"  U_inf  = {config.U_INF:.2f} m/s")
    print(f"  chord  = {config.CHORD} m")
    print()

    # ── Step 1: Get NACA 0012 coordinates ────────────────────────────────
    print("Step 1: Loading NACA 0012 coordinates...")
    coords = get_naca0012_coords()

    # ── Step 2: Run XFOIL to get δ* ──────────────────────────────────────
    print("\nStep 2: Running XFOIL for boundary layer δ*...")
    bl_result = get_delta_star(coords)

    if bl_result is None:
        print("  XFOIL did not converge — using Blasius fallback")
        print("  (This is acceptable for the baseline but check XFOIL installation)")
        dstar_p, dstar_s = blasius_fallback()
        used_fallback = True
    else:
        dstar_p, dstar_s = bl_result
        used_fallback = False

    print(f"  δ*_pressure = {dstar_p*1000:.4f} mm")
    print(f"  δ*_suction  = {dstar_s*1000:.4f} mm")
    print(f"  Source      : {'Blasius fallback' if used_fallback else 'XFOIL'}")

    # ── Step 3: Compute BPM SPL ───────────────────────────────────────────
    print("\nStep 3: Computing baseline SPL via BPM...")
    spl_baseline, freqs, spectrum = compute_spl_tbl_te(dstar_p, dstar_s)
    print(f"  SPL = {spl_baseline:.3f} dB")

    # ── Step 4: Compute L/D ───────────────────────────────────────────────
    print("\nStep 4: Computing baseline L/D via NeuralFoil...")
    ld_baseline = compute_naca0012_ld()

    # ── Step 5: Save to disk ──────────────────────────────────────────────
    print("\nStep 5: Saving baseline values...")
    config.DATA_GEN.mkdir(parents=True, exist_ok=True)

    np.save(config.BASELINE_SPL, spl_baseline)
    np.save(config.BASELINE_EFF, ld_baseline)

    # Also save the δ* values — useful for the surrogate validation later
    np.save(config.DATA_GEN / "baseline_dstar_p.npy", dstar_p)
    np.save(config.DATA_GEN / "baseline_dstar_s.npy", dstar_s)

    print(f"  Saved baseline_spl.npy  → {spl_baseline:.3f} dB")
    print(f"  Saved baseline_eff.npy  → {ld_baseline:.3f}")

    # ── Step 6: Sanity checks ─────────────────────────────────────────────
    print("\nStep 6: Sanity checks...")

    assert 20 < spl_baseline < 90, \
        f"Baseline SPL = {spl_baseline:.1f} dB is outside plausible range [20, 90]"

    assert 30 < ld_baseline < 150, \
        f"Baseline L/D = {ld_baseline:.1f} is outside plausible range [30, 150]"

    assert 0 < dstar_p < 0.02, \
        f"δ*_pressure = {dstar_p:.6f} m outside plausible range"

    assert 0 < dstar_s < 0.02, \
        f"δ*_suction  = {dstar_s:.6f} m outside plausible range"

    print("  All sanity checks passed ✅")

    print()
    print("=" * 55)
    print("BASELINE SUMMARY")
    print("=" * 55)
    print(f"  SPL (NACA 0012) = {spl_baseline:.3f} dB")
    print(f"  L/D (NACA 0012) = {ld_baseline:.3f}")
    print(f"  δ*_pressure     = {dstar_p*1000:.4f} mm")
    print(f"  δ*_suction      = {dstar_s*1000:.4f} mm")
    print()
    print("  Success criteria thresholds:")
    print(f"    Noise target   : SPL ≤ {spl_baseline - config.SPL_REDUCTION_TARGET_DB:.3f} dB")
    print(f"    L/D floor      : L/D ≥ {ld_baseline * config.LD_RETENTION_FRACTION:.3f}")
    print()
    print("  These values are now saved and will be used by all")
    print("  training runs and evaluation scripts automatically.")


if __name__ == "__main__":
    main()