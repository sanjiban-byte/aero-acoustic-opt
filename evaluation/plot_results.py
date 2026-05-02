"""
evaluation/plot_results.py — Block 8 figures (v2, fixed spectra)
"""
from __future__ import annotations
import sys, warnings
from pathlib import Path
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import config
from acoustics.bpm_model import compute_spl_tbl_te
from boundary_layer.xfoil_bl import get_delta_star, blasius_fallback
from boundary_layer.dstar_inference import predict_dstar_fast
from boundary_layer.generate_dstar_dataset import cst_to_coordinates

plt.rcParams.update({
    "font.family":"serif","font.size":10,"axes.labelsize":11,
    "axes.titlesize":12,"legend.fontsize":9,"figure.dpi":150,
    "savefig.dpi":300,"savefig.bbox":"tight","axes.grid":True,"grid.alpha":0.3,
})

_C  = {0.0:"#1f77b4",0.5:"#2ca02c",1.5:"#ff7f0e",3.0:"#d62728",6.0:"#9467bd"}
_NC = "#333333"

def _load():
    p = config.RESULTS_DIR/"pareto_results.npy"
    if not p.exists():
        raise FileNotFoundError(f"Run evaluate_all.py first. Missing: {p}")
    d = np.load(str(p), allow_pickle=True).item()
    return d["results"], d["baseline_spl"], d["baseline_eff"]

def _curve(lam):
    npz = config.MODELS_DIR/f"lambda_{lam:.2f}_logs"/"evaluations.npz"
    if not npz.exists(): return None, None
    try:
        d = np.load(str(npz))
        return d["timesteps"], d["results"].mean(axis=1)
    except: return None, None

def _naca():
    try:
        import aerosandbox as asb
        return asb.Airfoil("naca0012").coordinates
    except:
        x=0.5*(1-np.cos(np.linspace(0,np.pi,65))); t=0.12
        yt=5*t*(0.2969*x**0.5-0.1260*x-0.3516*x**2+0.2843*x**3-0.1015*x**4)
        return np.vstack([np.c_[x,yt],np.c_[x[::-1],-yt[::-1]][1:]])

def _spectrum(coords, cst_vec=None):
    bl = get_delta_star(coords, aoa_deg=config.AOA_DEG,
                        Re=config.RE, chord=config.CHORD, max_iter=400)
    if bl:
        dsp,dss=bl; src="XFOIL"
    elif cst_vec is not None:
        dsp,dss=predict_dstar_fast(cst_vec); src="surrogate"
    else:
        dsp,dss=blasius_fallback(Re=config.RE); src="Blasius"
    _,freqs,(sp,ss)=compute_spl_tbl_te(dsp,dss)
    spl=10*np.log10(10**(sp/10)+10**(ss/10))
    return freqs,spl,src


# ── Figure 1: Pareto ─────────────────────────────────────────────────────────
def fig1(results, bspl, beff):
    fig,ax=plt.subplots(figsize=(6.5,5))
    st=config.SPL_REDUCTION_TARGET_DB; lt=config.LD_RETENTION_FRACTION*100
    ax.axvline(lt,color="gray",ls="--",lw=0.9,alpha=0.6,label=f"L/D ≥ {lt:.0f}%")
    ax.axhline(st,color="gray",ls=":",lw=0.9,alpha=0.6,label=f"ΔSPL ≥ {st:.1f} dB")
    ax.fill_between([lt,130],st,15,alpha=0.06,color="green",label="Both targets met")
    pts=[]
    for r in results:
        lam=r["lam"]; xp=r["ld_mean"]/beff*100; yp=-r["delta_spl_mean"]
        c=_C.get(lam,"k")
        ax.scatter(xp,yp,color=c,s=120,zorder=5,
                   label=f"λ={lam:.1f}  L/D={r['ld_mean']:.1f}  ΔSPL={r['delta_spl_mean']:+.2f}dB")
        ax.annotate(f"λ={lam:.1f}",xy=(xp,yp),xytext=(5,4),
                    textcoords="offset points",fontsize=8,color=c)
        pts.append((xp,yp))
    ps=sorted(pts); xs,ys=zip(*ps)
    ax.plot(xs,ys,"k--",lw=1.0,alpha=0.35)
    ax.scatter(100,0,marker="D",color=_NC,s=100,zorder=6,label="NACA 0012")
    ax.annotate("NACA 0012",xy=(100,0),xytext=(4,-7),
                textcoords="offset points",fontsize=8,color=_NC)
    allx=[r["ld_mean"]/beff*100 for r in results]
    ally=[-r["delta_spl_mean"] for r in results]
    mx=max((max(allx)-min(allx))*0.5,3); my=max((max(ally)-min(ally))*0.6,0.5)
    ax.set_xlim(min(min(allx)-mx,97),max(allx)+mx+2)
    ax.set_ylim(min(min(ally)-my,-0.5),max(ally)+my+0.5)
    ax.set_xlabel("L/D retention  (% of NACA 0012 baseline)")
    ax.set_ylabel("SPL reduction  (dB,  positive = quieter)")
    ax.set_title("Pareto Frontier: Aeroacoustic Trade-off")
    ax.legend(loc="lower left",framealpha=0.85,fontsize=8)
    out=config.RESULTS_DIR/"fig_pareto_frontier.pdf"
    fig.savefig(str(out)); plt.close(fig); print(f"  ✓ {out.name}")


# ── Figure 2: Training curves ─────────────────────────────────────────────────
def fig2():
    fig,ax=plt.subplots(figsize=(7,4.5))
    any_data=False
    for lam in config.LAMBDA_VALUES:
        ts,rw=_curve(lam)
        if ts is None: continue
        any_data=True; c=_C.get(lam,"k")
        ax.plot(ts/1e6,rw,color=c,lw=1.5,label=f"λ={lam:.1f}")
        ax.scatter(ts[np.argmax(rw)]/1e6,rw.max(),color=c,s=45,zorder=5)
    if not any_data:
        ax.text(0.5,0.5,"evaluations.npz not found",ha="center",va="center",
                transform=ax.transAxes,fontsize=11,color="gray")
    else:
        starts=[np.load(str(config.MODELS_DIR/f"lambda_{l:.2f}_logs/evaluations.npz"))
                ["timesteps"].min()
                for l in config.LAMBDA_VALUES
                if (config.MODELS_DIR/f"lambda_{l:.2f}_logs/evaluations.npz").exists()]
        if starts and min(starts)>500_000:
            ax.text(0.02,0.05,
                    f"Note: data from {min(starts)/1e6:.1f}M steps onward\n"
                    "(early portion lost to system restart)",
                    transform=ax.transAxes,fontsize=7.5,color="gray",va="bottom")
    ax.set_xlabel("Training timesteps (millions)")
    ax.set_ylabel("Mean episodic reward")
    ax.set_title("PPO Training Convergence — All λ Values")
    ax.legend(loc="lower right",framealpha=0.85)
    out=config.RESULTS_DIR/"fig_training_curves.pdf"
    fig.savefig(str(out)); plt.close(fig); print(f"  ✓ {out.name}")


# ── Figure 3: Airfoil shapes ──────────────────────────────────────────────────
def fig3(results):
    fig,axes=plt.subplots(1,2,figsize=(12,4))
    af,at=axes
    nc=_naca(); um=nc[:,1]>=-1e-9
    for ax in axes:
        ax.plot(nc[um,0],nc[um,1],color=_NC,lw=2.5,label="NACA 0012")
        ax.plot(nc[~um,0],nc[~um,1],color=_NC,lw=2.5)
    for r in results:
        lam=r["lam"]; co=np.array(r["best_coords"])
        if co is None or len(co)<10: continue
        c=_C.get(lam,"k"); um=co[:,1]>=-1e-9
        lbl=f"λ={lam:.1f}  (ΔL/D={r['delta_ld_pct_mean']:+.1f}%, ΔSPL={r['delta_spl_mean']:+.1f}dB)"
        af.plot(co[um,0],co[um,1],color=c,lw=1.3,label=lbl)
        af.plot(co[~um,0],co[~um,1],color=c,lw=1.3)
        at.plot(co[um,0],co[um,1],color=c,lw=1.3)
        at.plot(co[~um,0],co[~um,1],color=c,lw=1.3)
    af.set_xlim(-0.02,1.02); af.set_ylim(-0.22,0.24); af.set_aspect("equal")
    af.set_xlabel("x/c"); af.set_ylabel("y/c")
    af.set_title("Optimised Airfoils vs NACA 0012")
    af.legend(loc="upper right",fontsize=7.5,framealpha=0.9)
    at.set_xlim(0.75,1.02); at.set_ylim(-0.06,0.09); at.set_aspect("equal")
    at.set_xlabel("x/c")
    at.set_title("Trailing-Edge Region (zoom)\nThin TE → low δ* → low SPL")
    fig.tight_layout()
    out=config.RESULTS_DIR/"fig_airfoil_comparison.pdf"
    fig.savefig(str(out)); plt.close(fig); print(f"  ✓ {out.name}")


# ── Figure 4: SPL spectra ─────────────────────────────────────────────────────
def fig4(results, bspl):
    fig,ax=plt.subplots(figsize=(7,4.5))
    # NACA 0012
    try:
        f,s,src=_spectrum(_naca())
        ax.semilogx(f,s,color=_NC,lw=2.5,label=f"NACA 0012 ({src})")
    except Exception as e:
        warnings.warn(f"NACA spectrum: {e}")

    for r in results:
        lam=r["lam"]
        if lam==0.0: continue   # skip λ=0 — nearly identical to NACA
        cst=r.get("best_cst")
        if cst is None: continue
        cst=np.array(cst)
        upper=cst[:config.N_CST_PARAMS]; lower=cst[config.N_CST_PARAMS:2*config.N_CST_PARAMS]
        coords=cst_to_coordinates(upper,lower,float(cst[-1]))
        try:
            f,s,src=_spectrum(coords,cst)
            c=_C.get(lam,"k")
            ax.semilogx(f,s,color=c,lw=1.5,
                        label=f"λ={lam:.1f}  (ΔSPL={r['delta_spl_mean']:+.2f}dB, {src})")
        except Exception as e:
            warnings.warn(f"λ={lam} spectrum: {e}")

    ax.set_xlabel("Frequency  (Hz)")
    ax.set_ylabel("SPL  (dB, BPM model)")
    ax.set_title("Trailing-Edge Noise Spectrum — NACA 0012 vs Optimised")
    ax.legend(loc="upper right",framealpha=0.85,fontsize=8)
    ax.set_xlim(100,10_000)
    out=config.RESULTS_DIR/"fig_spl_spectra.pdf"
    fig.savefig(str(out)); plt.close(fig); print(f"  ✓ {out.name}")


def main():
    print("="*55); print("Block 8 — Figures"); print("="*55)
    results,bspl,beff=_load()
    config.RESULTS_DIR.mkdir(parents=True,exist_ok=True)
    print(f"  {len(results)} results loaded\n")
    print("Figure 1: Pareto"); fig1(results,bspl,beff)
    print("Figure 2: Training"); fig2()
    print("Figure 3: Airfoils"); fig3(results)
    print("Figure 4: Spectra"); fig4(results,bspl)
    print(f"\n✓ All figures → {config.RESULTS_DIR}/")

if __name__ == "__main__":
    main()
