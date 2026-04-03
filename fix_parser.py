# fix_parser.py
"""Rewrites _parse_dump_file in xfoil_bl.py with correct surface labelling."""

new_function = '''
def _parse_dump_file(
    dump_file : str,
    chord     : float,
):
    """
    Parse XFOIL DUMP file to extract trailing-edge delta* on both surfaces.

    XFOIL dump walks: TE (top) -> upper surface -> LE -> lower surface -> TE
    First segment (upper, y>0 at TE) = SUCTION side at positive AoA
    Second segment (lower, y<0 at TE) = PRESSURE side at positive AoA
    At AoA=5 deg: suction delta* > pressure delta* (physically required)

    Returns (delta_star_pressure, delta_star_suction) in metres.
    """
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

        import numpy as np
        import config

        data   = np.array(rows)
        x_col  = data[:, 1]
        ds_col = data[:, 4]

        # Split at leading edge (minimum x)
        le_idx = int(np.argmin(x_col))

        # First segment: TE -> LE over the TOP = suction side
        suction_x  = x_col[:le_idx + 1]
        suction_ds = ds_col[:le_idx + 1]

        # Second segment: LE -> TE along the BOTTOM = pressure side
        pressure_x  = x_col[le_idx:]
        pressure_ds = ds_col[le_idx:]

        if len(suction_x) < 3 or len(pressure_x) < 3:
            return None

        def te_mean(x_arr, ds_arr):
            mask = x_arr > 0.95
            if not np.any(mask):
                return float(np.mean(ds_arr[-3:]))
            vals = ds_arr[mask]
            vals = vals[np.isfinite(vals) & (vals > 0)]
            if len(vals) == 0:
                return None
            return float(np.mean(vals))

        dstar_suction_norm  = te_mean(suction_x,  suction_ds)
        dstar_pressure_norm = te_mean(pressure_x, pressure_ds)

        if dstar_suction_norm is None or dstar_pressure_norm is None:
            return None

        # Dimensionalise and clip
        dstar_s = float(np.clip(
            dstar_suction_norm  * chord,
            config.DSTAR_CLIP_MIN, config.DSTAR_CLIP_MAX
        ))
        dstar_p = float(np.clip(
            dstar_pressure_norm * chord,
            config.DSTAR_CLIP_MIN, config.DSTAR_CLIP_MAX
        ))

        return dstar_p, dstar_s   # always (pressure, suction)

    except Exception:
        return None
'''

# Read the existing file
with open('boundary_layer/xfoil_bl.py', 'r', encoding='utf-8', errors='replace') as f:
    content = f.read()

# Find start and end of _parse_dump_file
start_marker = 'def _parse_dump_file('
end_marker   = 'def blasius_fallback('

start_idx = content.find(start_marker)
end_idx   = content.find(end_marker)

if start_idx == -1:
    print("ERROR: Could not find _parse_dump_file in xfoil_bl.py")
elif end_idx == -1:
    print("ERROR: Could not find blasius_fallback in xfoil_bl.py")
else:
    new_content = content[:start_idx] + new_function + '\n\n' + content[end_idx:]
    with open('boundary_layer/xfoil_bl.py', 'w', encoding='utf-8') as f:
        f.write(new_content)
    print("Successfully rewrote _parse_dump_file.")
    print(f"  Old function was at char {start_idx}–{end_idx}")
    print(f"  New function is {len(new_function)} chars")