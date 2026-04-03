# check_fix.py
from pathlib import Path

content = Path('boundary_layer/xfoil_bl.py').read_text(encoding='utf-8', errors='replace')
lines = content.split('\n')
print("=== Lines containing dstar_suction / dstar_pressure / return dstar ===")
for i, line in enumerate(lines):
    if any(x in line for x in ['dstar_suction', 'dstar_pressure', 'return dstar']):
        print(f'Line {i+1}: {line}')