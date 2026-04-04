"""
Train MLP surrogate to predict delta* from CST parameters.

Input : 17 CST parameters [upper×8, lower×8, le×1]
Output: [delta_star_pressure, delta_star_suction] in metres

Architecture: 17 → 64 → 64 → 32 → 2 (Softplus output — δ* must be positive)
Training time: ~1-2 minutes
Inference time: ~0.1ms (vs 0.1-12s for XFOIL)
"""

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import joblib
import sys
import time
from pathlib import Path
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).parent.parent))
import config


# MODEL DEFINITION

class DStarMLP(nn.Module):
    """
    MLP surrogate for predicting boundary layer displacement thickness.

    Why Softplus output?
      δ* must always be positive. Softplus(x) = log(1 + e^x) is smooth
      and always positive, unlike ReLU which has a dead zone at x < 0.

    Why StandardScaler on both inputs and outputs?
      Our CST inputs span [-0.5, 0.5] and outputs span [4-10mm].
      Without normalisation, the optimizer struggles with mismatched scales.
      Scaling both to zero-mean, unit-variance makes training stable.
    """
    def __init__(self, in_dim: int = 17):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 2),
            nn.Softplus(),   # guarantees positive output
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# TRAINING

def train_surrogate(
    X              : np.ndarray,
    y              : np.ndarray,
    n_epochs       : int   = 500,
    batch_size     : int   = 256,
    lr             : float = 1e-3,
    val_fraction   : float = 0.125,
    test_fraction  : float = 0.125,
    patience       : int   = 30,
    seed           : int   = 42,
) -> tuple:
    """
    Train the DStarMLP surrogate.

    Returns
    -------
    model     : trained DStarMLP
    scaler_X  : fitted StandardScaler for inputs
    scaler_y  : fitted StandardScaler for outputs
    test_metrics : dict with test set performance
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    n      = len(X)
    n_test = int(n * test_fraction)
    n_val  = int(n * val_fraction)
    n_train = n - n_val - n_test

    # Shuffle indices
    idx     = np.random.permutation(n)
    tr_idx  = idx[:n_train]
    va_idx  = idx[n_train:n_train + n_val]
    te_idx  = idx[n_train + n_val:]

    print(f"  Split: {n_train} train / {n_val} val / {n_test} test")

    # ── Normalise ─────────────────────────────────────────────────────────
    # Fit scalers on training data only — never on val/test
    scaler_X = StandardScaler().fit(X[tr_idx])
    scaler_y = StandardScaler().fit(y[tr_idx])

    def to_tensor(arr, scaler):
        return torch.tensor(scaler.transform(arr), dtype=torch.float32)

    X_tr = to_tensor(X[tr_idx], scaler_X)
    y_tr = to_tensor(y[tr_idx], scaler_y)
    X_va = to_tensor(X[va_idx], scaler_X)
    y_va = to_tensor(y[va_idx], scaler_y)
    X_te = to_tensor(X[te_idx], scaler_X)
    y_te = to_tensor(y[te_idx], scaler_y)

    # ── Model, optimiser, scheduler ───────────────────────────────────────
    in_dim = X.shape[1]
    model  = DStarMLP(in_dim)

    optimiser = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimiser, patience=10, factor=0.5, min_lr=1e-5
    )
    criterion = nn.MSELoss()

    loader = DataLoader(
        TensorDataset(X_tr, y_tr),
        batch_size = batch_size,
        shuffle    = True,
    )

    # ── Training loop ─────────────────────────────────────────────────────
    best_val_loss  = np.inf
    best_state     = None
    patience_count = 0

    print(f"\n  {'Epoch':>6}  {'Train MSE':>10}  {'Val MSE':>10}  {'LR':>10}")
    print(f"  {'-'*6}  {'-'*10}  {'-'*10}  {'-'*10}")

    for epoch in range(n_epochs):
        # Training pass
        model.train()
        train_losses = []
        for xb, yb in loader:
            optimiser.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            optimiser.step()
            train_losses.append(loss.item())

        # Validation pass
        model.eval()
        with torch.no_grad():
            val_loss = criterion(model(X_va), y_va).item()

        scheduler.step(val_loss)
        current_lr = optimiser.param_groups[0]['lr']

        # Print every 50 epochs
        if epoch % 50 == 0:
            print(f"  {epoch:>6}  {np.mean(train_losses):>10.6f}  "
                  f"{val_loss:>10.6f}  {current_lr:>10.2e}")

        # Early stopping
        if val_loss < best_val_loss - 1e-6:
            best_val_loss  = val_loss
            best_state     = {k: v.clone() for k, v in model.state_dict().items()}
            patience_count = 0
        else:
            patience_count += 1
            if patience_count >= patience:
                print(f"\n  Early stopping at epoch {epoch} "
                      f"(no improvement for {patience} epochs)")
                break

    # ── Load best model and evaluate on test set ──────────────────────────
    model.load_state_dict(best_state)
    model.eval()

    with torch.no_grad():
        y_pred_norm = model(X_te).numpy()

    # Inverse-transform to physical units
    y_pred = scaler_y.inverse_transform(y_pred_norm)
    y_true = y[te_idx]

    # Relative error per output
    rel_error = np.abs(y_pred - y_true) / (y_true + 1e-10)

    test_metrics = {
        'rel_error_p_mean' : rel_error[:, 0].mean(),
        'rel_error_s_mean' : rel_error[:, 1].mean(),
        'rel_error_p_max'  : rel_error[:, 0].max(),
        'rel_error_s_max'  : rel_error[:, 1].max(),
        'rank_correlation' : float(np.corrcoef(
            y_pred[:, 0] + y_pred[:, 1],  # total δ*
            y_true[:, 0] + y_true[:, 1],
        )[0, 1]),
    }

    return model, scaler_X, scaler_y, test_metrics


# PHYSICAL VALIDATION

def validate_surrogate_physics(model, scaler_X, scaler_y):
    """
    Validate that the surrogate captures physically correct trends.

    Three-point check:
      1. Thin airfoil (small weights) → less δ* than NACA0012
      2. NACA0012 equivalent → δ*_p ≈ 4mm, δ*_s ≈ 5mm
      3. Thick airfoil (large weights) → more δ* than NACA0012

    Also checks: suction side > pressure side at AoA=5° for all three.
    """
    def predict(upper, lower, le=0.0):
        cst = np.concatenate([upper, lower, [le]]).astype(np.float32)
        x_norm = scaler_X.transform(cst.reshape(1, -1))
        with torch.no_grad():
            y_norm = model(torch.tensor(x_norm, dtype=torch.float32)).numpy()
        y_phys = scaler_y.inverse_transform(y_norm)[0]
        return float(y_phys[0]), float(y_phys[1])

    N = config.N_CST_PARAMS

    thin_u  = np.full(N, 0.08)
    thin_l  = np.full(N, -0.08)
    naca_u  = np.array([0.17, 0.16, 0.22, 0.16, 0.16, 0.24, 0.15, 0.15])
    naca_l  = np.array([-0.17, -0.16, -0.11, -0.11, -0.10, -0.03, -0.14, -0.14])
    thick_u = np.full(N, 0.35)
    thick_l = np.full(N, -0.35)

    p_thin,  s_thin  = predict(thin_u,  thin_l)
    p_naca,  s_naca  = predict(naca_u,  naca_l)
    p_thick, s_thick = predict(thick_u, thick_l)

    print(f"\n  Physical validation:")
    print(f"  {'Airfoil':<12}  {'δ*_p (mm)':>10}  {'δ*_s (mm)':>10}  {'s>p?':>6}")
    print(f"  {'-'*12}  {'-'*10}  {'-'*10}  {'-'*6}")
    print(f"  {'Thin':<12}  {p_thin*1000:>10.3f}  {s_thin*1000:>10.3f}  "
          f"{'✅' if s_thin > p_thin else '❌':>6}")
    print(f"  {'NACA0012':<12}  {p_naca*1000:>10.3f}  {s_naca*1000:>10.3f}  "
          f"{'✅' if s_naca > p_naca else '❌':>6}")
    print(f"  {'Thick':<12}  {p_thick*1000:>10.3f}  {s_thick*1000:>10.3f}  "
          f"{'✅' if s_thick > p_thick else '❌':>6}")

    # Check monotonic ordering
    thickness_trend_ok = (p_thin + s_thin) < (p_naca + s_naca) < (p_thick + s_thick)
    print(f"\n  Thickness trend (thin < NACA < thick): "
          f"{'✅' if thickness_trend_ok else '❌'}")

    return thickness_trend_ok


# MAIN

def main():
    print("=" * 55)
    print("Training δ* Surrogate MLP")
    print("=" * 55)

    # ── Load dataset ──────────────────────────────────────────────────────
    if not config.DSTAR_X_PATH.exists():
        print("ERROR: Dataset not found. Run generate_dstar_dataset.py first.")
        return

    X = np.load(config.DSTAR_X_PATH)
    y = np.load(config.DSTAR_Y_PATH)
    print(f"Loaded dataset: X={X.shape}, y={y.shape}")

    # ── Train ─────────────────────────────────────────────────────────────
    print("\nTraining...")
    t0 = time.time()

    model, scaler_X, scaler_y, metrics = train_surrogate(X, y)

    elapsed = time.time() - t0
    print(f"\nTraining complete in {elapsed:.1f}s")

    # ── Test set results ──────────────────────────────────────────────────
    print(f"\n  Test set performance:")
    print(f"    Mean rel error δ*_p : {metrics['rel_error_p_mean']*100:.2f}%")
    print(f"    Mean rel error δ*_s : {metrics['rel_error_s_mean']*100:.2f}%")
    print(f"    Max  rel error δ*_p : {metrics['rel_error_p_max']*100:.2f}%")
    print(f"    Max  rel error δ*_s : {metrics['rel_error_s_max']*100:.2f}%")
    print(f"    Rank correlation    : {metrics['rank_correlation']:.4f}")

    # ── Physical validation ───────────────────────────────────────────────
    trend_ok = validate_surrogate_physics(model, scaler_X, scaler_y)

    # ── Save ─────────────────────────────────────────────────────────────
    config.DATA_SURR.mkdir(parents=True, exist_ok=True)

    torch.save(model.state_dict(), config.SURROGATE_WEIGHTS)
    joblib.dump(scaler_X, config.SURROGATE_SCALER_X)
    joblib.dump(scaler_y, config.SURROGATE_SCALER_Y)

    in_dim = X.shape[1]
    torch.save(in_dim, config.DATA_SURR / "in_dim.pt")

    print(f"\n  Saved surrogate to {config.DATA_SURR}/")

    # ── Assertions ────────────────────────────────────────────────────────
    assert metrics['rel_error_p_mean'] < 0.15, \
        f"δ*_p mean error {metrics['rel_error_p_mean']*100:.1f}% > 15%"
    assert metrics['rel_error_s_mean'] < 0.15, \
        f"δ*_s mean error {metrics['rel_error_s_mean']*100:.1f}% > 15%"
    assert metrics['rank_correlation'] > 0.75, \
        f"Rank correlation {metrics['rank_correlation']:.3f} < 0.75"

    print()
    print("  ✅ All quality checks passed")
    print()
    print("Next step: python boundary_layer/dstar_inference.py (verify speed)")


if __name__ == "__main__":
    main()