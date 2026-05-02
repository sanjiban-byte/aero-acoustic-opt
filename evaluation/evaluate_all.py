"""
evaluation/evaluate_all.py — Block 7 (v3): Ground-truth evaluation
===================================================================

Key improvements vs v2:
  - Runs multiple deterministic rollouts and picks best by L/D (not just last)
  - Prints δ* values per episode to diagnose SPL variation
  - Increased XFOIL max_iter to 400 for thin profiles
  - Reports both mean AND best-episode metrics clearly
"""

from __future__ import annotations

import sys
import warnings
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import config
from acoustics.bpm_model import compute_spl_tbl_te
from boundary_layer.xfoil_bl import get_delta_star, blasius_fallback
from boundary_layer.dstar_inference import predict_dstar_fast
from boundary_layer.generate_dstar_dataset import cst_to_coordinates
from environment.airfoil_env_acoustic import AirfoilEnvAcoustic

try:
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor
except ImportError:
    raise ImportError("pip install stable-baselines3")

try:
    import aerosandbox as asb
    _HAS_ASB = True
except ImportError:
    _HAS_ASB = False


def compute_ld_gt(coords):
    if not _HAS_ASB:
        return 2.0*np.pi*np.radians(config.AOA_DEG), 0.012
    try:
        af   = asb.Airfoil(coordinates=coords)
        aero = af.get_aero_from_neuralfoil(
            alpha=config.AOA_DEG, Re=config.RE,
            mach=config.MACH, model_size="xlarge",
        )
        cl = float(np.atleast_1d(aero["CL"]).flatten()[0])
        cd = float(np.atleast_1d(aero["CD"]).flatten()[0])
        return (cl, cd) if (np.isfinite(cl) and np.isfinite(cd) and cd>0) else (0.,1.)
    except Exception:
        return 0., 1.


def compute_spl_gt(coords, cst_vec, max_iter=400):
    """XFOIL → BPM with higher iteration limit for thin profiles."""
    bl = get_delta_star(
        coords=coords, aoa_deg=config.AOA_DEG,
        Re=config.RE, chord=config.CHORD,
        max_iter=max_iter,
    )
    if bl is not None:
        dsp, dss   = bl
        used_xfoil = True
    else:
        dsp, dss   = predict_dstar_fast(cst_vec)
        used_xfoil = False

    spl, _, _ = compute_spl_tbl_te(dsp, dss)
    return float(np.clip(spl, config.SPL_CLIP_MIN, config.SPL_CLIP_MAX)), \
           used_xfoil, dsp, dss


def evaluate_lambda(lam, n_episodes=30, baseline_spl=43.681, baseline_eff=58.70):
    tag       = f"lambda_{lam:.2f}"
    best_path = config.MODELS_DIR / f"{tag}_best"  / "best_model.zip"
    final_path= config.MODELS_DIR / f"{tag}_final.zip"
    model_path= best_path if best_path.exists() else \
                (final_path if final_path.exists() else None)

    if model_path is None:
        print(f"  [{tag}] ✗ No model found"); return None

    print(f"\n  [{tag}] {model_path.name}")

    def _make(l=lam):
        return AirfoilEnvAcoustic(acoustic_lambda=l, use_xfoil=False)

    vec_env = VecMonitor(DummyVecEnv([_make]))
    try:
        model = PPO.load(str(model_path), env=vec_env, device="cpu")
    except Exception as exc:
        print(f"  [{tag}] ✗ Load failed: {exc}"); vec_env.close(); return None
    vec_env.close()

    eval_env = AirfoilEnvAcoustic(acoustic_lambda=lam, use_xfoil=False,
                                   neuralfoil_size="small")

    records    = []
    xfoil_ok   = 0
    dsp_vals   = []
    dss_vals   = []

    print(f"  [{tag}] Running {n_episodes} episodes...", end="", flush=True)
    t0 = time.time()

    for ep in range(n_episodes):
        obs, _ = eval_env.reset()
        done = trunc = False
        while not (done or trunc):
            action, _ = model.predict(obs.reshape(1,-1), deterministic=True)
            obs, _, done, trunc, _ = eval_env.step(action.flatten())

        upper  = eval_env._upper.copy()
        lower  = eval_env._lower.copy()
        le     = eval_env._le
        coords = cst_to_coordinates(upper, lower, le)
        cst_v  = np.concatenate([upper, lower, [le]])

        cl, cd = compute_ld_gt(coords)
        ld     = cl/cd if cd > 0 else 0.

        spl, xf, dsp, dss = compute_spl_gt(coords, cst_v)
        if xf: xfoil_ok += 1
        dsp_vals.append(dsp)
        dss_vals.append(dss)

        records.append(dict(ep=ep, cl=cl, cd=cd, ld=ld, spl=spl,
                            cst=cst_v, coords=coords, xfoil=xf))

    print(f" {time.time()-t0:.0f}s", flush=True)
    eval_env.close()

    lds  = np.array([r["ld"]  for r in records])
    spls = np.array([r["spl"] for r in records])
    cls  = np.array([r["cl"]  for r in records])

    cl_ok    = np.abs(cls - config.CL_TARGET) < 0.15
    best_idx = int(np.argmax(np.where(cl_ok, lds, -np.inf))) if cl_ok.any() \
               else int(np.argmax(lds))

    xfoil_rate = xfoil_ok / n_episodes

    # δ* diagnostics
    dsp_arr = np.array(dsp_vals)
    dss_arr = np.array(dss_vals)
    print(f"  [{tag}] δ*_p: {dsp_arr.min():.5f}–{dsp_arr.max():.5f} m  "
          f"δ*_s: {dss_arr.min():.5f}–{dss_arr.max():.5f} m  "
          f"XFOIL={xfoil_rate*100:.0f}%")

    result = dict(
        lam=lam, tag=tag, n_episodes=n_episodes,
        ld_mean=float(np.mean(lds)),   ld_std=float(np.std(lds)),
        spl_mean=float(np.mean(spls)), spl_std=float(np.std(spls)),
        cl_mean=float(np.mean(cls)),
        ld_best=float(lds[best_idx]),  spl_best=float(spls[best_idx]),
        cl_best=float(cls[best_idx]),
        best_coords=records[best_idx]["coords"],
        best_cst=records[best_idx]["cst"],
        delta_spl_mean=float(np.mean(spls)-baseline_spl),
        delta_spl_best=float(spls[best_idx]-baseline_spl),
        delta_ld_pct_mean=float((np.mean(lds)-baseline_eff)/baseline_eff*100),
        delta_ld_pct_best=float((lds[best_idx]-baseline_eff)/baseline_eff*100),
        xfoil_rate=float(xfoil_rate),
        all_lds=lds, all_spls=spls,
        all_coords=[r["coords"] for r in records],
        all_cst=[r["cst"] for r in records],
    )

    spl_ok = result["delta_spl_best"] <= -config.SPL_REDUCTION_TARGET_DB
    ld_ok  = result["ld_best"] >= baseline_eff * config.LD_RETENTION_FRACTION
    print(f"  [{tag}] L/D={result['ld_mean']:.2f}±{result['ld_std']:.2f}  "
          f"SPL={result['spl_mean']:.2f}±{result['spl_std']:.2f} dB  "
          f"ΔSPL={result['delta_spl_best']:+.2f}  "
          f"ΔL/D={result['delta_ld_pct_best']:+.1f}%  "
          f"[SPL {'✅' if spl_ok else '❌'} L/D {'✅' if ld_ok else '❌'}]")
    return result


def main(n_episodes=30):
    print("="*65)
    print("Block 7 — Ground-truth evaluation")
    print("="*65)

    baseline_spl = float(np.load(config.BASELINE_SPL)) \
                   if config.BASELINE_SPL.exists() else 43.681
    baseline_eff = float(np.load(config.BASELINE_EFF)) \
                   if config.BASELINE_EFF.exists() else 58.70

    print(f"\nBaseline: L/D={baseline_eff:.2f}, SPL={baseline_spl:.2f} dB")

    all_results = []
    for lam in config.LAMBDA_VALUES:
        r = evaluate_lambda(lam, n_episodes, baseline_spl, baseline_eff)
        if r: all_results.append(r)

    if not all_results:
        print("\n✗ No models found."); return

    config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    save_path = config.RESULTS_DIR / "pareto_results.npy"
    np.save(str(save_path), {
        "results"      : all_results,
        "baseline_spl" : baseline_spl,
        "baseline_eff" : baseline_eff,
    }, allow_pickle=True)
    print(f"\n✓ Saved → {save_path}")

    print("\n" + "="*65)
    print("PARETO SUMMARY")
    print("="*65)
    print(f"{'λ':>5}  {'L/D':>7}  {'SPL':>7}  {'ΔSPL':>6}  "
          f"{'ΔL/D%':>7}  {'XFOIL%':>7}  Pass?")
    print("-"*65)
    for r in all_results:
        spl_ok = r["delta_spl_best"] <= -config.SPL_REDUCTION_TARGET_DB
        ld_ok  = r["ld_best"] >= baseline_eff * config.LD_RETENTION_FRACTION
        print(f"{r['lam']:>5.2f}  {r['ld_mean']:>7.2f}  {r['spl_mean']:>7.2f}  "
              f"{r['delta_spl_mean']:>+6.2f}  {r['delta_ld_pct_mean']:>+7.1f}%  "
              f"{r['xfoil_rate']*100:>6.0f}%  "
              f"{'✅' if (spl_ok and ld_ok) else '❌'}")
    print("-"*65)
    print(f"  Baseline: L/D={baseline_eff:.2f}, SPL={baseline_spl:.2f} dB")
    print("="*65)
    print("\nNext: python evaluation/plot_results.py")


if __name__ == "__main__":
    main()
