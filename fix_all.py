# fix_all.py
"""
Step 1: Clear ALL __pycache__ in the project (not just venv)
Step 2: Rewrite _parse_dump_file with a y-coordinate based approach
        that doesn't depend on ordering assumptions
Step 3: Verify the file was written correctly
Step 4: Run a quick inline test
"""

import shutil, subprocess, sys
from pathlib import Path

ROOT = Path(__file__).parent

# ── Step 1: Clear ALL cache ────────────────────────────────────────────────
print("Step 1: Clearing all __pycache__ folders...")
cleared = 0
for p in ROOT.rglob('__pycache__'):
    if 'venv' not in str(p):   # skip venv internals
        shutil.rmtree(p)
        print(f"  Deleted: {p}")
        cleared += 1
print(f"  Cleared {cleared} project cache folders.\n")

# ── Step 2: Write the new _parse_dump_file ────────────────────────────────
print("Step 2: Rewriting _parse_dump_file...")

NEW_FUNCTION = '''\
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

'''

# Read file with UTF-8
xfoil_path = ROOT / 'boundary_layer' / 'xfoil_bl.py'
content = xfoil_path.read_text(encoding='utf-8', errors='replace')

start_marker = 'def _parse_dump_file('
end_marker   = 'def blasius_fallback('

start_idx = content.find(start_marker)
end_idx   = content.find(end_marker)

if start_idx == -1 or end_idx == -1:
    print("  ERROR: Could not find function markers. Check xfoil_bl.py manually.")
    sys.exit(1)

new_content = content[:start_idx] + NEW_FUNCTION + '\n\n' + content[end_idx:]
xfoil_path.write_text(new_content, encoding='utf-8')
print(f"  Written {len(NEW_FUNCTION)} chars to xfoil_bl.py\n")

# ── Step 3: Verify ─────────────────────────────────────────────────────────
print("Step 3: Verifying written content...")
verify = xfoil_path.read_text(encoding='utf-8')
if 'y1 >= y2' in verify:
    print("  ✅ New function confirmed in file (y-coordinate logic present)\n")
else:
    print("  ❌ Write may have failed — 'y1 >= y2' not found in file\n")

# ── Step 4: Quick test ─────────────────────────────────────────────────────
print("Step 4: Running quick test...")
result = subprocess.run(
    [sys.executable, 'tests/test_xfoil_bl.py'],
    capture_output=False
)