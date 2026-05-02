"""
evaluation/results_summary.py — Print final success-criteria table
===================================================================

Reads pareto_results.npy and prints a clean report-ready table showing
whether each λ run meets the project success criteria.

Usage
-----
    python evaluation/results_summary.py
"""

from __future__ import annotations

import sys
from pathlib import Path
import numpy as np

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import config


def main():
    path = config.RESULTS_DIR / "pareto_results.npy"
    if not path.exists():
        print(f"Results not found: {path}")
        print("Run  python evaluation/evaluate_all.py  first.")
        return

    data         = np.load(str(path), allow_pickle=True).item()
    results      = data["results"]
    baseline_spl = data["baseline_spl"]
    baseline_eff = data["baseline_eff"]

    spl_thresh = baseline_spl - config.SPL_REDUCTION_TARGET_DB
    ld_thresh  = baseline_eff * config.LD_RETENTION_FRACTION

    print("\n" + "=" * 80)
    print("SILENT MORPHING — Final Results Summary")
    print("ME 228 Project | Aeroacoustic Airfoil Optimisation via DRL")
    print("=" * 80)
    print(f"\nOperating condition: AoA={config.AOA_DEG}°, Re={config.RE:,}, "
          f"U∞={config.U_INF:.1f} m/s, chord={config.CHORD} m")
    print(f"\nBaseline (NACA 0012): L/D = {baseline_eff:.2f}, "
          f"SPL = {baseline_spl:.2f} dB")
    print(f"Success criteria:     L/D ≥ {ld_thresh:.2f}  |  "
          f"SPL ≤ {spl_thresh:.2f} dB  (ΔSPL ≥ −{config.SPL_REDUCTION_TARGET_DB:.1f} dB)")

    print("\n" + "-" * 80)
    hdr = (f"{'λ':>5}  {'L/D':>7}  {'σ(L/D)':>7}  "
           f"{'SPL':>7}  {'σ(SPL)':>7}  "
           f"{'ΔSPL':>7}  {'ΔL/D%':>7}  "
           f"{'XFOIL%':>7}  {'SPL✓':>5}  {'L/D✓':>5}  {'PASS':>5}")
    print(hdr)
    print("-" * 80)

    for r in results:
        spl_ok  = r["delta_spl_best"] <= -config.SPL_REDUCTION_TARGET_DB
        ld_ok   = r["ld_best"]        >= ld_thresh
        xfoil_ok= r["xfoil_rate"]     >= config.XFOIL_CONVERGENCE_MIN
        passed  = spl_ok and ld_ok

        print(
            f"{r['lam']:>5.2f}  "
            f"{r['ld_mean']:>7.2f}  "
            f"{r['ld_std']:>7.2f}  "
            f"{r['spl_mean']:>7.2f}  "
            f"{r['spl_std']:>7.2f}  "
            f"{r['delta_spl_mean']:>+7.2f}  "
            f"{r['delta_ld_pct_mean']:>+7.1f}%  "
            f"{r['xfoil_rate']*100:>6.0f}%  "
            f"{'✅' if spl_ok  else '❌':>5}  "
            f"{'✅' if ld_ok   else '❌':>5}  "
            f"{'✅' if passed  else '❌':>5}"
        )

    print("-" * 80)

    # Best single model across all λ
    best = max(results, key=lambda r: (
        r["delta_spl_best"] <= -config.SPL_REDUCTION_TARGET_DB
    ) * r["ld_best"])
    print(f"\nBest model: λ={best['lam']:.2f}  "
          f"L/D={best['ld_best']:.2f}  "
          f"SPL={best['spl_best']:.2f} dB  "
          f"ΔSPL={best['delta_spl_best']:+.2f} dB  "
          f"ΔL/D={best['delta_ld_pct_best']:+.1f}%")

    print("\nFigures: python evaluation/plot_results.py")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()
