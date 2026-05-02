"""
training/debug_neuralfoil.py — diagnose small vs xlarge discrepancy
Run: python training/debug_neuralfoil.py
"""
import sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config
import aerosandbox as asb
from boundary_layer.generate_dstar_dataset import cst_to_coordinates

def test_airfoil(name, coords, sizes=["small", "medium", "large", "xlarge"]):
    print(f"\n  {name}")
    print(f"  {'Size':<10} {'CL':>8} {'CD':>10} {'L/D':>8}")
    print(f"  {'-'*40}")
    af = asb.Airfoil(coordinates=coords)
    for size in sizes:
        try:
            aero = af.get_aero_from_neuralfoil(
                alpha=config.AOA_DEG, Re=config.RE,
                mach=config.MACH, model_size=size,
            )
            cl = float(np.atleast_1d(aero["CL"]).flatten()[0])
            cd = float(np.atleast_1d(aero["CD"]).flatten()[0])
            ld = cl/cd if cd > 0 else 0
            print(f"  {size:<10} {cl:>8.4f} {cd:>10.6f} {ld:>8.2f}")
        except Exception as e:
            print(f"  {size:<10} ERROR: {e}")

print("="*55)
print("NeuralFoil model size comparison")
print(f"AoA={config.AOA_DEG}°  Re={config.RE:,}  Mach={config.MACH}")
print("="*55)

# 1. NACA 0012
try:
    naca = asb.Airfoil("naca0012").coordinates
    test_airfoil("NACA 0012", naca)
except:
    # analytic fallback
    x  = 0.5*(1-np.cos(np.linspace(0,np.pi,65)))
    t  = 0.12
    yt = 5*t*(0.2969*x**0.5-0.1260*x-0.3516*x**2+0.2843*x**3-0.1015*x**4)
    naca = np.vstack([np.c_[x,yt], np.c_[x[::-1],-yt[::-1]][1:]])
    test_airfoil("NACA 0012 (analytic)", naca)

# 2. Old seed (broken)  upper=[0.3]*8, lower=[0.3]*8
upper_old = np.array([0.3]*8)
lower_old = np.array([0.3]*8)
coords_old = cst_to_coordinates(upper_old, lower_old, 0.0)
test_airfoil("Old seed  upper=[0.3], lower=[0.3]", coords_old)

# 3. New seed  upper=[0.17]*8, lower=[-0.17]*8
upper_new = np.array([0.17]*8)
lower_new = np.array([-0.17]*8)
coords_new = cst_to_coordinates(upper_new, lower_new, 0.0)
test_airfoil("New seed  upper=[0.17], lower=[-0.17]", coords_new)

# 4. Load best model shape for lambda=0 and test it
best_path = config.MODELS_DIR / "lambda_0.00_best" / "best_model.zip"
if best_path.exists():
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor
    from environment.airfoil_env_acoustic import AirfoilEnvAcoustic

    def _make(): return AirfoilEnvAcoustic(acoustic_lambda=0.0, use_xfoil=False)
    vec_env = VecMonitor(DummyVecEnv([_make]))
    model   = PPO.load(str(best_path), env=vec_env, device="cpu")
    vec_env.close()

    env = AirfoilEnvAcoustic(acoustic_lambda=0.0, use_xfoil=False)
    obs, _ = env.reset()
    for _ in range(10):
        action, _ = model.predict(obs.reshape(1,-1), deterministic=True)
        obs, _, _, truncated, _ = env.step(action.flatten())
        if truncated: break

    coords_agent = cst_to_coordinates(env._upper, env._lower, env._le)
    test_airfoil("λ=0 best model (final shape)", coords_agent)

    # Also show what the env's internal NeuralFoil small says
    print(f"\n  Internal env CL={env.current_cl:.4f}  "
          f"CD={env.current_cd:.6f}  "
          f"L/D={env.current_ld:.2f}  (NeuralFoil small)")
    env.close()

print("\n" + "="*55)
print("KEY: if 'small' L/D << 'xlarge' L/D for NACA 0012,")
print("the agent was optimising the wrong objective.")
print("="*55)
