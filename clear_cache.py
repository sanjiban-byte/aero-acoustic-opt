# clear_cache.py
import shutil
from pathlib import Path

root = Path('.')
count = 0
for p in root.rglob('__pycache__'):
    shutil.rmtree(p)
    print(f'Deleted: {p}')
    count += 1
print(f'Done. Cleared {count} cache folders.')