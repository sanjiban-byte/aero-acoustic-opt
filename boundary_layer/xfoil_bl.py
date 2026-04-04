"""
XFOIL Boundary Layer Extractor
Extracts displacement thickness delta* from XFOIL boundary layer dump.
Returns (delta_star_pressure, delta_star_suction) in metres.
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
    if coords is None or len(coords) < 8:
        return False
    if not np.all(np.isfinite(coords)):
        return False
    x = coords[:, 0]
    y = coords[:, 1]
    if x.max() - x.min() < 0.5:
        return False
    if y.max() - y.min() < 0.001:
        return False
    if x.min() < -0.1 or x.max() > 1.1:
        return False
    return True


def _write_coords(coords: np.ndarray, filepath: str) -> None:
    keep = np.ones(len(coords), dtype=bool)
    for i in range(1, len(coords)):
        if np.allclose(coords[i], coords[i-1], atol=1e-8):
            keep[i] = False
    coords = coords[keep]
    with open(filepath, 'w') as f:
        f.write("OptimisedAirfoil\n")
        for x, y in coords:
            f.write(f"  {x:.6f}  {y:.6f}\n")


def _parse_dump_file(dump_file: str, chord: float) -> Optional[tuple]:
    try:
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
        x_col  = data[:, 1]
        y_col  = data[:, 2]
        ds_col = data[:, 4]

        le_idx = int(np.argmin(x_col))
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

        if y1 >= y2:
            dstar_suction  = ds1
            dstar_pressure = ds2
        else:
            dstar_suction  = ds2
            dstar_pressure = ds1

        dstar_s = float(np.clip(dstar_suction  * chord, config.DSTAR_CLIP_MIN, config.DSTAR_CLIP_MAX))
        dstar_p = float(np.clip(dstar_pressure * chord, config.DSTAR_CLIP_MIN, config.DSTAR_CLIP_MAX))

        return dstar_p, dstar_s

    except Exception:
        return None


def get_delta_star(
    coords  : np.ndarray,
    aoa_deg : float = None,
    Re      : float = None,
    chord   : float = None,
    Mach    : float = None,
    n_crit  : float = 9.0,
    max_iter: int   = 200,
) -> Optional[tuple]:
    """
    Run XFOIL and return (delta_star_pressure, delta_star_suction) in metres.
    Returns None if XFOIL fails or geometry is invalid.

    Uses bare filenames + cwd=tmpdir to handle spaces in Windows paths.
    Uses psutil to force-kill hung XFOIL processes on timeout.
    """
    if aoa_deg is None: aoa_deg = config.AOA_DEG
    if Re      is None: Re      = config.RE
    if chord   is None: chord   = config.CHORD
    if Mach    is None: Mach    = config.MACH

    if not _is_valid_geometry(coords):
        return None

    with tempfile.TemporaryDirectory() as tmpdir:
        coords_file = os.path.join(tmpdir, "airfoil.dat")
        dump_file   = os.path.join(tmpdir, "bl_dump.txt")
        script_file = os.path.join(tmpdir, "script.txt")

        _write_coords(coords, coords_file)

        xfoil_script = (
            "PLOP\nG\n\n"
            "LOAD airfoil.dat\n"
            "PANE\n"
            "OPER\n"
            f"VISC {Re:.0f}\n"
            f"MACH {Mach:.4f}\n"
            f"ITER {max_iter}\n"
            "VPAR\nN 9\n\n"
            f"ALFA {aoa_deg:.2f}\n"
            "DUMP bl_dump.txt\n\n\n"
            "QUIT\n"
        )

        with open(script_file, 'w') as f:
            f.write(xfoil_script)

        try:
            import psutil
            with open(script_file, 'r') as script_in:
                proc = subprocess.Popen(
                    str(config.XFOIL_BIN),
                    stdin         = script_in,
                    stdout        = subprocess.PIPE,
                    stderr        = subprocess.PIPE,
                    text          = True,
                    shell         = config.XFOIL_SHELL,
                    cwd           = tmpdir,
                    creationflags = subprocess.CREATE_NEW_PROCESS_GROUP,
                )

            try:
                stdout, stderr = proc.communicate(timeout=12)
            except subprocess.TimeoutExpired:
                try:
                    parent = psutil.Process(proc.pid)
                    for child in parent.children(recursive=True):
                        child.kill()
                    parent.kill()
                except Exception:
                    proc.kill()
                proc.wait()
                return None

        except FileNotFoundError:
            raise FileNotFoundError(f"XFOIL not found at {config.XFOIL_BIN}")

        if "VISCAL:  Convergence failed" in stdout:
            return None

        if not os.path.exists(dump_file) or os.path.getsize(dump_file) < 50:
            return None

        return _parse_dump_file(dump_file, chord)


def blasius_fallback(
    Re     : float = None,
    aoa_deg: float = None,
    chord  : float = None,
) -> tuple:
    """Blasius flat-plate estimate of delta* when XFOIL fails."""
    if Re     is None: Re     = config.RE
    if aoa_deg is None: aoa_deg = config.AOA_DEG
    if chord  is None: chord  = config.CHORD

    dstar_base = 1.72 * chord / np.sqrt(Re)
    aoa_rad    = np.radians(aoa_deg)

    dstar_s = float(np.clip(dstar_base * (1.0 + 2.5 * aoa_rad), config.DSTAR_CLIP_MIN, config.DSTAR_CLIP_MAX))
    dstar_p = float(np.clip(dstar_base * (1.0 - 0.5 * aoa_rad), config.DSTAR_CLIP_MIN, config.DSTAR_CLIP_MAX))

    return dstar_p, dstar_s
