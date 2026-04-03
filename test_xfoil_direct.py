# test_xfoil_direct.py
# Run this to test if XFOIL works on your machine
# Usage: python test_xfoil_direct.py

import subprocess
import os
from pathlib import Path

# Write the XFOIL command script to a file
script_content = """NACA 0012
OPER
VISC 500000
MACH 0.044
ITER 200
ALFA 5
DUMP xfoil_test_dump.txt


QUIT
"""

with open("xfoil_script_test.txt", "w") as f:
    f.write(script_content)

print("Script written. Running XFOIL...")
print("-" * 40)

# Run XFOIL with the script file as stdin
xfoil_path = str(Path(__file__).parent / "xfoil.exe")

try:
    with open("xfoil_script_test.txt", "r") as script_in:
        result = subprocess.run(
            xfoil_path,
            stdin          = script_in,
            capture_output = True,
            text           = True,
            timeout        = 20,
            shell          = True,
        )

    print("XFOIL stdout:")
    print(result.stdout[-2000:])   # last 2000 chars — XFOIL output is long

    if result.returncode != 0:
        print(f"Return code: {result.returncode}")
    if result.stderr:
        print("XFOIL stderr:", result.stderr[:500])

    # Check if dump file was created
    if os.path.exists("xfoil_test_dump.txt"):
        size = os.path.getsize("xfoil_test_dump.txt")
        print(f"\nDump file created: {size} bytes")
        if size > 50:
            print("First 300 chars of dump file:")
            with open("xfoil_test_dump.txt") as f:
                print(f.read(300))
            print("\n✅ XFOIL IS WORKING")
        else:
            print("⚠ Dump file exists but is empty — XFOIL ran but did not converge")
    else:
        print("\n❌ No dump file created — XFOIL did not reach the DUMP command")
        print("Possible causes:")
        print("  1. XFOIL.exe is a 32-bit binary — check if it runs manually")
        print("  2. Missing Visual C++ redistributable")
        print("  3. Antivirus blocking execution")

except subprocess.TimeoutExpired:
    print("❌ XFOIL timed out after 20 seconds")
except FileNotFoundError:
    print(f"❌ xfoil.exe not found at: {xfoil_path}")
    print("   Check that xfoil.exe is in the project root folder")
except Exception as e:
    print(f"❌ Unexpected error: {e}")

# Cleanup
for f in ["xfoil_script_test.txt"]:
    if os.path.exists(f):
        os.remove(f)