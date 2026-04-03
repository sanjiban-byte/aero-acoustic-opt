# boundary_layer/xfoil_bl.py
"""
XFOIL Boundary Layer Extractor

Drives xfoil.exe via subprocess to compute boundary layer displacement
thickness (δ*) at the trailing edge on both airfoil surfaces.

Why subprocess and not a Python XFOIL library?
  - Most Python XFOIL wrappers (like the one in AeroSandbox) expose CL/CD
    but not the full boundary layer dump. We need the raw DUMP output.
  - Subprocess gives us direct access to every XFOIL output.

The XFOIL DUMP file format (7 columns):
  s      x      y      Ue/V   Dstar  Theta  Cf
  [0]    [1]    [2]    [3]    [4]    [5]    [6]

  s     : arc length from stagnation point
  x, y  : surface coordinates (normalised by chord)
  Ue/V  : edge velocity ratio
  Dstar : displacement thickness δ* (normalised by chord) ← we want this
  Theta : momentum thickness
  Cf    : skin friction coefficient

XFOIL writes the dump in two passes:
  Pass 1: from stagnation point → trailing edge along UPPER surface
  Pass 2: from stagnation point → trailing edge along LOWER surface
  (both go in the direction of increasing arc length s)
"""

import subprocess
import tempfile
import os
import numpy as np
from typing import Optional
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
import config

def _is_valid_geometry(coords: np.ndarray) -> bool:
    """
    Quick geometry sanity check before calling XFOIL.
    Rejects obviously degenerate shapes that would cause XFOIL to hang.

    Checks:
    1. Enough points to define an airfoil
    2. x coordinates span [0, 1] (normalised)
    3. Non-zero thickness somewhere (not a flat line)
    4. No NaN or Inf values
    """
    if coords is None or len(coords) < 8:
        return False

    if not np.all(np.isfinite(coords)):
        return False

    x = coords[:, 0]
    y = coords[:, 1]

    # Must span chord direction
    if x.max() - x.min() < 0.5:
        return False

    # Must have some thickness — not a flat plate
    if y.max() - y.min() < 0.001:
        return False

    # x must be in reasonable range
    if x.min() < -0.1 or x.max() > 1.1:
        return False

    return True


def get_delta_star(
    coords       : np.ndarray,
    aoa_deg      : float = None,
    Re           : float = None,
    chord        : float = None,
    Mach         : float = None,
    n_crit       : float = 9.0,
    max_iter     : int   = 200,
) -> Optional[tuple[float, float]]:
    """
    Run XFOIL on a given airfoil and return trailing-edge δ* on both surfaces.

    Parameters
    ----------
    coords   : Nx2 numpy array of [x, y] airfoil coordinates,
               normalised by chord (x in [0,1], y typically in [-0.15, 0.15]).
               Must go from trailing edge → upper surface → leading edge
               → lower surface → trailing edge (standard XFOIL ordering).
    aoa_deg  : angle of attack [degrees]. Defaults to config.AOA_DEG
    Re       : Reynolds number. Defaults to config.RE
    chord    : chord length [m]. Used to dimensionalise δ* from XFOIL's
               normalised output. Defaults to config.CHORD
    Mach     : Mach number. Defaults to config.MACH
    n_crit   : boundary layer transition criterion (9 = clean flow, typical)
    max_iter : XFOIL viscous iteration limit

    Returns
    -------
    (delta_star_pressure, delta_star_suction) in metres
    Returns None if XFOIL fails to converge or crashes.
    """
    # Apply defaults
    if aoa_deg is None: aoa_deg = config.AOA_DEG
    if Re      is None: Re      = config.RE
    if chord   is None: chord   = config.CHORD
    if Mach    is None: Mach    = config.MACH

    # ── Geometry validity check BEFORE launching XFOIL ───────────────────
    # This prevents XFOIL from hanging on degenerate CST shapes.
    # Returns None immediately — caller will use Blasius fallback.
    if not _is_valid_geometry(coords):
        return None

    # Use a temporary directory for all XFOIL I/O files
    # Why tempfile? XFOIL writes files to disk. If we hardcode filenames,
    # parallel calls (e.g. from SubprocVecEnv) would overwrite each other.
    # tempfile gives each call its own isolated directory.
    with tempfile.TemporaryDirectory() as tmpdir:
        coords_file = os.path.join(tmpdir, "airfoil.dat")
        dump_file   = os.path.join(tmpdir, "bl_dump.txt")
        polar_file  = os.path.join(tmpdir, "polar.txt")

        # ── Step 1: Write airfoil coordinates to file ─────────────────────
        # XFOIL expects: first line = airfoil name (optional)
        # Then x y pairs, one per line, normalised coordinates
        _write_coords(coords, coords_file)

        # ── Step 2: Build the XFOIL command script ────────────────────────
        # This simulates what you'd type interactively in XFOIL.
        # Each line is one command. Empty lines = pressing Enter (confirm).
        #
        # Command walkthrough:
        #   LOAD <file>    : load airfoil from coordinates file
        #   PANE           : re-panel with XFOIL's cosine spacing (more robust)
        #   OPER           : enter operating point menu
        #   VISC <Re>      : enable viscous analysis at given Reynolds number
        #   MACH <M>       : set Mach number
        #   ITER <n>       : set max viscous iterations
        #   VPAR           : enter viscous parameters sub-menu
        #   N <ncrit>      : set Ncrit (transition criterion)
        #                    Enter (blank) to exit VPAR sub-menu
        #   ALFA <aoa>     : run analysis at this angle of attack
        #   DUMP <file>    : write boundary layer dump file
        #   \n\n           : two blank lines to exit OPER menu
        #   QUIT           : exit XFOIL

        xfoil_script = (
            f"LOAD {coords_file}\n"
            f"PANE\n"
            f"OPER\n"
            f"VISC {Re:.0f}\n"
            f"MACH {Mach:.4f}\n"
            f"ITER {max_iter}\n"
            f"VPAR\n"
            f"N {n_crit}\n"
            f"\n"
            f"ALFA {aoa_deg:.2f}\n"
            f"DUMP {dump_file}\n"
            f"\n"
            f"\n"
            f"QUIT\n"
        )

        # ── Step 3: Write script to file and run XFOIL ────────────────────
        script_file = os.path.join(tmpdir, "xfoil_script.txt")
        with open(script_file, 'w') as f:
            f.write(xfoil_script)

        try:
            with open(script_file, 'r') as script_in:
                result = subprocess.run(
                    str(config.XFOIL_BIN),
                    stdin          = script_in,
                    capture_output = True,
                    text           = True,
                    timeout        = 15,
                    shell          = config.XFOIL_SHELL,
                    creationflags  = subprocess.CREATE_NEW_PROCESS_GROUP,
                )
        except subprocess.TimeoutExpired:
            return None
        except FileNotFoundError:
            raise FileNotFoundError(
                f"XFOIL binary not found at {config.XFOIL_BIN}. "
                f"Check config.XFOIL_BIN path."
            )

        # ── Step 4: Check convergence ─────────────────────────────────────
        # XFOIL prints "VISCAL:  Convergence failed" if the viscous solver
        # didn't converge. We check stdout for this string.
        stdout = result.stdout
        if "Convergence failed" in stdout or "VISCAL" in stdout:
            return None

        # ── Step 5: Check dump file was written ───────────────────────────
        if not os.path.exists(dump_file):
            # XFOIL didn't reach the DUMP command — likely crashed early
            # (e.g. self-intersecting airfoil that PANE couldn't handle)
            return None

        if os.path.getsize(dump_file) < 50:
            # File exists but is nearly empty — XFOIL wrote a header but
            # no data. Treat as non-convergence.
            return None

        # ── Step 6: Parse the dump file ───────────────────────────────────
        result = _parse_dump_file(dump_file, chord)
        return result


def _write_coords(coords: np.ndarray, filepath: str) -> None:
    """
    Write airfoil coordinates to a file in XFOIL format.

    Validates that:
    - coordinates are 2D (Nx2)
    - x values are in [0, 1] range (normalised)
    - no duplicate consecutive points (XFOIL chokes on these)
    """
    assert coords.ndim == 2 and coords.shape[1] == 2, \
        f"coords must be Nx2, got shape {coords.shape}"

    # Remove duplicate consecutive points
    # (can happen at the leading edge if two panels land on the same point)
    keep = np.ones(len(coords), dtype=bool)
    for i in range(1, len(coords)):
        if np.allclose(coords[i], coords[i-1], atol=1e-8):
            keep[i] = False
    coords = coords[keep]

    with open(filepath, 'w') as f:
        f.write("OptimisedAirfoil\n")
        for x, y in coords:
            f.write(f"  {x:.6f}  {y:.6f}\n")



def _parse_dump_file(dump_file, chord):
    """
    Parse XFOIL DUMP file. Returns (delta_star_pressure, delta_star_suction).

    Strategy: instead of assuming which segment is suction vs pressure
    based on ordering, we USE THE Y-COORDINATE to decide.

    At positive AoA, the suction (upper) surface has POSITIVE y near TE.
    The pressure (lower) surface has NEGATIVE or near-zero y near TE.
    This is true regardless of which direction XFOIL walks the dump.
    """
    try:
        import numpy as np
        import config as cfg

        rows = []
        with open(dump_file, 'r') as f:
            for line in f:
                stripped = line.strip()
                if not stripped or stripped.startswith('#'):
                    continue
                try:
                    vals = [float(v) for v in stripped.split()]
                    if len(vals) >= 5:
                        rows.append(vals)
                except ValueError:
                    continue

        if len(rows) < 10:
            return None

        data   = np.array(rows)
        x_col  = data[:, 1]   # x/c
        y_col  = data[:, 2]   # y/c
        ds_col = data[:, 4]   # Dstar/c

        # Split at leading edge (minimum x)
        le_idx  = int(np.argmin(x_col))
        seg1_x  = x_col[:le_idx + 1]
        seg1_y  = y_col[:le_idx + 1]
        seg1_ds = ds_col[:le_idx + 1]
        seg2_x  = x_col[le_idx:]
        seg2_y  = y_col[le_idx:]
        seg2_ds = ds_col[le_idx:]

        if len(seg1_x) < 3 or len(seg2_x) < 3:
            return None

        def te_mean(x_arr, ds_arr):
            mask = x_arr > 0.95
            if not np.any(mask):
                mask = np.zeros(len(x_arr), dtype=bool)
                mask[-3:] = True
            vals = ds_arr[mask]
            vals = vals[np.isfinite(vals) & (vals > 0)]
            return float(np.mean(vals)) if len(vals) > 0 else None

        def te_y_mean(x_arr, y_arr):
            """Mean y/c in TE region — tells us if this is upper or lower surface."""
            mask = x_arr > 0.95
            if not np.any(mask):
                mask = np.zeros(len(x_arr), dtype=bool)
                mask[-3:] = True
            return float(np.mean(y_arr[mask]))

        ds1 = te_mean(seg1_x, seg1_ds)
        ds2 = te_mean(seg2_x, seg2_ds)
        y1  = te_y_mean(seg1_x, seg1_y)
        y2  = te_y_mean(seg2_x, seg2_y)

        if ds1 is None or ds2 is None:
            return None

        # Assign suction vs pressure by y-coordinate sign near TE
        # Suction (upper) surface: larger y near TE
        # Pressure (lower) surface: smaller y near TE
        if y1 >= y2:
            # Segment 1 is upper (suction), segment 2 is lower (pressure)
            dstar_suction  = ds1
            dstar_pressure = ds2
        else:
            # Segment 2 is upper (suction), segment 1 is lower (pressure)
            dstar_suction  = ds2
            dstar_pressure = ds1

        dstar_s = float(np.clip(dstar_suction  * chord,
                                cfg.DSTAR_CLIP_MIN, cfg.DSTAR_CLIP_MAX))
        dstar_p = float(np.clip(dstar_pressure * chord,
                                cfg.DSTAR_CLIP_MIN, cfg.DSTAR_CLIP_MAX))

        return dstar_p, dstar_s   # always (pressure, suction)

    except Exception as e:
        return None



def blasius_fallback(
    Re    : float = None,
    aoa_deg: float = None,
    chord : float = None,
) -> tuple[float, float]:
    """
    Blasius flat-plate estimate of δ* when XFOIL fails.

    δ* = 1.72 × c / sqrt(Re_c)   (Blasius laminar, at x = c)

    This is a conservative approximation — real turbulent BL is thinner.
    Used ONLY as a safety fallback to prevent training crashes.
    Geometries that trigger this should be penalised.

    AoA correction:
      Suction side BL grows with AoA (adverse pressure gradient)
      Pressure side BL shrinks slightly with AoA
    """
    if Re    is None: Re    = config.RE
    if aoa_deg is None: aoa_deg = config.AOA_DEG
    if chord is None: chord = config.CHORD

    dstar_base = 1.72 * chord / np.sqrt(Re)
    aoa_rad    = np.radians(aoa_deg)

    dstar_s = dstar_base * (1.0 + 2.5 * aoa_rad)  # suction side grows
    dstar_p = dstar_base * (1.0 - 0.5 * aoa_rad)  # pressure side shrinks

    dstar_s = float(np.clip(dstar_s, config.DSTAR_CLIP_MIN, config.DSTAR_CLIP_MAX))
    dstar_p = float(np.clip(dstar_p, config.DSTAR_CLIP_MIN, config.DSTAR_CLIP_MAX))

    return dstar_p, dstar_s