"""
Brooks-Pope-Marcolini (1989) Trailing-Edge Noise Model
NASA Reference Publication 1218

This module implements the TBL-TE (turbulent boundary layer trailing edge)
noise mechanism, which is dominant for attached-flow conditions at
Re ≈ 500,000 and moderate angles of attack.

The model structure:
  1. For each frequency, compute a Strouhal number St = f * δ* / U
  2. Look up the spectral shape A(St / St_peak) — an empirical curve fit
  3. Add amplitude correction K1 (function of Reynolds number)
  4. Add directivity for the observer position
  5. Compute pressure-side and suction-side contributions separately
  6. Power-sum them to get total SPL at each frequency
  7. Power-sum across all frequencies to get overall SPL

Reference: Brooks, T.F., Pope, D.S., Marcolini, M.A. (1989)
           Airfoil Self-Noise and Prediction. NASA RP-1218.
           https://ntrs.nasa.gov/citations/19890016302
"""

import numpy as np
from typing import Union
import sys
from pathlib import Path

# Allow importing config from project root
sys.path.insert(0, str(Path(__file__).parent.parent))
import config


# SECTION 1 — SPECTRAL SHAPE FUNCTION A
# The A function describes the shape of the TBL-TE noise spectrum.
# It is a piece-wise empirical fit (BPM 1989, eq. A4–A6 and Fig. A2).
#
# The input `a` is log10(St / St_peak): how far the current frequency's
# Strouhal number is from the spectral peak.
#   a = 0  → you are AT the spectral peak → A = 0 (maximum contribution)
#   |a| large → you are far from peak → A is large negative → little contribution
#
# BPM provides two versions: A_min (minimum turbulence) and A_max (maximum).
# For our purposes, we use A_min as it represents attached flow, consistent
# with the operating conditions we target (AoA = 5°, Re = 500k).

def _A_min(a: float) -> float:
    """
    Spectral shape function A — minimum (attached flow) branch.
    BPM 1989, Appendix A, equations A4–A6.
    Input: a = log10(St / St_peak), scalar
    Output: spectral amplitude correction in dB (always ≤ 0)
    """
    a = float(np.abs(a))  # function is symmetric around peak

    if a <= 0.204:
        # Near-peak region: smooth parabolic shape
        # The sqrt gives a bell-curve top when added to -8.219
        return np.sqrt(67.552 - 886.788 * a**2) - 8.219

    elif a <= 0.244:
        # Transition region: linear bridge between parabolic and tail
        return -32.665 * a + 3.981

    else:
        # Tail region: cubic decay for frequencies far from peak
        return -142.795 * a**3 + 103.656 * a**2 - 57.757 * a + 6.006


def _A_max(a: float) -> float:
    """
    Spectral shape function A — maximum (higher turbulence) branch.
    BPM 1989, Appendix A, equations A7–A9.
    Used for suction side at non-zero AoA.
    """
    a = float(np.abs(a))

    if a <= 0.13:
        return np.sqrt(16.888 - 886.788 * a**2) - 4.109
    elif a <= 0.321:
        return -83.607 * a + 8.138
    else:
        return -817.810 * a**3 + 355.210 * a**2 - 135.024 * a + 10.619


def _A_weighted(a: float, aoa_deg: float) -> float:
    """
    Blend between A_min and A_max based on angle of attack.
    At AoA = 0, pure A_min. At high AoA, blend toward A_max.
    BPM 1989, eq. A10–A12.
    """
    # a0 is the AoA at which the blend switches from min to max
    # For NACA 0012, BPM gives a0 ≈ 12.5° (stall onset)
    # Below a0/2, use A_min; above a0, use A_max; blend in between
    a0 = 12.5  # degrees — NACA 0012 specific, acceptable for our use
    if aoa_deg <= a0 / 2:
        return _A_min(a)
    elif aoa_deg >= a0:
        return _A_max(a)
    else:
        # Linear interpolation
        t = (aoa_deg - a0 / 2) / (a0 / 2)
        return (1 - t) * _A_min(a) + t * _A_max(a)


# SECTION 2 — PEAK STROUHAL NUMBER
# The Strouhal number at peak emission. This is where the spectrum is loudest.
# St = f * δ* / U, so St_peak tells you the frequency of peak emission.
#
# BPM found that St_peak depends on AoA (eq. A14):
#   At zero AoA: St_peak = St1 = 0.02 (a reference value)
#   At non-zero AoA: St_peak increases (peak shifts to lower frequencies)
#
# Physically: at higher AoA, the boundary layer is thicker and less organised,
# so the spectral peak shifts to lower frequencies (longer wavelengths).

def _St1_peak() -> float:
    """Reference peak Strouhal number (zero AoA, pressure side). BPM eq. A14."""
    return 0.02


def _St_peak_pressure(aoa_deg: float) -> float:
    """
    Peak Strouhal number for the PRESSURE side contribution.
    Pressure side is less sensitive to AoA than suction side.
    BPM 1989, eq. A14.
    """
    St1 = _St1_peak()
    if aoa_deg < 1.333:
        return St1
    else:
        # AoA shifts peak to higher St (per BPM calibration)
        return St1 * 10.0 ** (0.0054 * (aoa_deg - 1.333) ** 2)


def _St_peak_suction(aoa_deg: float) -> float:
    """
    Peak Strouhal number for the SUCTION side contribution.
    Suction side boundary layer grows faster with AoA than pressure side,
    so its peak shifts more aggressively. BPM 1989, eq. A14.
    """
    St1 = _St1_peak()
    # Same functional form but applied to the suction-side BL growth
    return St1 * 10.0 ** (0.0054 * (aoa_deg - 1.333) ** 2)


# SECTION 3 — AMPLITUDE FUNCTIONS K1 AND ΔK1
# K1 is the overall amplitude of the TBL-TE noise spectrum as a function
# of chord Reynolds number. It captures how the turbulent boundary layer
# intensity varies with Re.
#
# BPM 1989, eqs. A40–A42, calibrated against NACA 0012 measurements.
# Three regimes with different slopes reflect the BL transition behaviour.

def _K1(Re_c: float) -> float:
    """
    Amplitude function K1 as a function of chord Reynolds number.
    BPM 1989, Appendix A, eqs. A40–A42.

    Re_c < 2.47e5  : laminar-dominant regime
    Re_c ≤ 8.0e5   : transitional regime
    Re_c > 8.0e5   : fully turbulent regime

    Returns K1 in dB.
    """
    log_Re = np.log10(Re_c)

    if Re_c < 2.47e5:
        return -4.31 * log_Re + 156.3
    elif Re_c <= 8.0e5:
        return -9.0 * log_Re + 181.6
    else:
        return 128.5


def _delta_K1(aoa_deg: float, Re_c: float) -> float:
    """
    Angle-of-attack correction to K1 for the suction side.
    At non-zero AoA the suction-side BL is amplified relative to pressure.
    BPM 1989, eq. A46.

    The formula uses AoA in radians and log10(Re_c) to correct amplitude.
    """
    aoa_rad = np.radians(aoa_deg)
    # BPM empirical fit for ΔK1 (dB)
    return aoa_rad * (1.43 * np.log10(Re_c) - 5.29)

# SECTION 4 — DIRECTIVITY
# Trailing-edge noise does not radiate equally in all directions — it has
# a characteristic cardioid-like directivity pattern.
#
# For our case (overhead observer, low Mach number) the directivity factor
# is nearly 1.0, so it doesn't strongly influence relative SPL comparisons
# between airfoils. But we include it for physical correctness.
#
# Dh = high-frequency directivity (used for most TBL-TE noise)
# BPM 1989, eq. A1.

def _Dh(theta_deg: float, phi_deg: float, Mach: float) -> float:
    """
    High-frequency directivity function for trailing-edge noise.
    BPM 1989, eq. A1.

    theta : polar angle from downstream axis (90° = overhead)
    phi   : azimuthal angle (90° = overhead in 2D)
    Mach  : free-stream Mach number

    At our conditions (M ≈ 0.044, theta=90°, phi=90°):
    The convection factor (1 - M*cos(theta)) ≈ 1 since M is small.
    """
    theta  = np.radians(theta_deg)
    phi    = np.radians(phi_deg)

    # Convection Mach factor — accounts for the fact that turbulent eddies
    # travel downstream at roughly 0.8 * U, creating a Doppler-like effect.
    # At our low Mach, this factor ≈ 1 and barely matters, but we keep it
    # for correctness.
    Mc = 0.8 * Mach  # convection Mach number

    numerator   = 2.0 * np.sin(theta / 2.0)**2 * np.sin(phi)**2
    denominator = (1.0 + Mc * np.cos(theta)) * (1.0 + (Mach - Mc) * np.cos(theta))**2

    # Guard against division by zero (shouldn't happen at our conditions)
    if np.abs(denominator) < 1e-12:
        return 1.0

    return numerator / denominator

# SECTION 5 — MAIN SPL COMPUTATION

def compute_spl_tbl_te(
    delta_star_p : float,
    delta_star_s : float,
    U            : float  = None,
    chord        : float  = None,
    span         : float  = None,
    aoa_deg      : float  = None,
    r            : float  = None,
    theta_deg    : float  = None,
    phi_deg      : float  = None,
    Mach         : float  = None,
    n_freqs      : int    = 40,
) -> tuple[float, np.ndarray, np.ndarray]:
    """
    Compute TBL-TE trailing-edge noise SPL via the BPM model.

    Uses config.py defaults for any parameter not explicitly provided.
    This makes it easy to call with just δ* values in the main pipeline:
        spl, freqs, spectrum = compute_spl_tbl_te(dsp, dss)

    Parameters
    ----------
    delta_star_p : displacement thickness, pressure side  [m]
    delta_star_s : displacement thickness, suction side   [m]
    U            : free-stream velocity                   [m/s]
    chord        : airfoil chord                          [m]
    span         : spanwise extent                        [m]
    aoa_deg      : angle of attack                        [degrees]
    r            : observer distance from trailing edge   [m]
    theta_deg    : polar observer angle (90° = overhead)  [degrees]
    phi_deg      : azimuthal observer angle               [degrees]
    Mach         : free-stream Mach number                [dimensionless]
    n_freqs      : number of frequency bins               [integer]

    Returns
    -------
    SPL_total    : overall (power-summed) SPL              [dB]
    freqs        : frequency array                         [Hz]
    SPL_spectrum : per-frequency SPL array                 [dB]
    """
    # ── Apply defaults from config ────────────────────────────────────────
    # This pattern means you can call compute_spl_tbl_te(dsp, dss) with
    # just two arguments and get sensible results for our fixed condition.
    if U         is None: U         = config.U_INF
    if chord     is None: chord     = config.CHORD
    if span      is None: span      = config.SPAN
    if aoa_deg   is None: aoa_deg   = config.AOA_DEG
    if r         is None: r         = config.R_OBS
    if theta_deg is None: theta_deg = config.THETA_OBS
    if phi_deg   is None: phi_deg   = config.PHI_OBS
    if Mach      is None: Mach = U / config.C_SOUND

    # ── Input validation and clipping ────────────────────────────────────
    # BPM will produce nonsense (NaN, -inf, +inf) for extreme δ* values.
    # These come from either XFOIL non-convergence or geometrically
    # degenerate CST shapes. Clipping here prevents cascading NaN errors.
    delta_star_p = float(np.clip(delta_star_p,
                                  config.DSTAR_CLIP_MIN,
                                  config.DSTAR_CLIP_MAX))
    delta_star_s = float(np.clip(delta_star_s,
                                  config.DSTAR_CLIP_MIN,
                                  config.DSTAR_CLIP_MAX))

    # ── Derived quantities ────────────────────────────────────────────────
    Re_c = U * chord / config.NU_AIR  # chord Reynolds number
    K1   = _K1(Re_c)                  # amplitude, dB
    dK1  = _delta_K1(aoa_deg, Re_c)   # AoA correction for suction side, dB
    Dh   = _Dh(theta_deg, phi_deg, Mach)  # directivity factor

    # Guard against non-positive directivity (can occur at grazing angles)
    if Dh <= 0:
        Dh = 1e-10

    # ── Frequency array ───────────────────────────────────────────────────
    # Log-spaced from 100 Hz to 10 kHz — covers the range where TBL-TE
    # noise is dominant and where the BPM empirical fits are valid.
    # The NASA dataset covers this range, so it matches our validation data.
    freqs = np.logspace(np.log10(100.0), np.log10(10_000.0), n_freqs)

    # ── Per-frequency SPL computation ─────────────────────────────────────
    SPL_p = np.full(n_freqs, -np.inf)  # pressure-side contribution
    SPL_s = np.full(n_freqs, -np.inf)  # suction-side contribution

    St_peak_p = _St_peak_pressure(aoa_deg)
    St_peak_s = _St_peak_suction(aoa_deg)

    for i, f in enumerate(freqs):

        # Strouhal number for this frequency and each surface's δ*
        # St = f * δ* / U: dimensionless frequency scaled by BL thickness.
        # This is what makes the spectrum "self-similar" across conditions.
        St_p = f * delta_star_p / U
        St_s = f * delta_star_s / U

        # How far is this St from the spectral peak? (in log space)
        # a = 0 means we're at the peak → maximum A → loudest at this freq
        # a large means we're far from peak → A is large negative → quiet
        a_p = np.log10(St_p / St_peak_p) if St_p > 1e-12 else -10.0
        a_s = np.log10(St_s / St_peak_s) if St_s > 1e-12 else -10.0

        # Spectral shape values
        A_p = _A_weighted(a_p, aoa_deg)
        A_s = _A_weighted(a_s, aoa_deg)

        # ── Pressure side SPL at this frequency ──────────────────────────
        # BPM eq. A13. The 10*log10(...) converts the physical scaling to dB.
        # Inside the log: δ*_p * M^5 * L * Dh / r²
        #   δ*_p   → how thick/energetic the pressure-side BL is
        #   M^5    → acoustic power scales as U^5 (Ffowcs-Williams scaling)
        #   L      → span (more span → more noise, linearly)
        #   Dh/r²  → how much reaches the observer
        # Then we add K1 (Reynolds correction), -3 (1/3-octave bandwidth),
        # and A_p (spectral shape).
        argument_p = (delta_star_p * (Mach**5) * span * Dh) / (r**2)
        if argument_p > 0:
            SPL_p[i] = 10.0 * np.log10(argument_p) + K1 - 3.0 + A_p

        # ── Suction side SPL at this frequency ───────────────────────────
        # Same structure but with suction-side δ* and the AoA correction ΔK1.
        # The suction side is louder at non-zero AoA because the boundary
        # layer there is thicker and more turbulent.
        argument_s = (delta_star_s * (Mach**5) * span * Dh) / (r**2)
        if argument_s > 0:
            SPL_s[i] = 10.0 * np.log10(argument_s) + K1 - 3.0 + A_s + dK1

    # ── Combine pressure and suction sides at each frequency ──────────────
    # Acoustic power addition: you can't just add dB values arithmetically.
    # 10 dB + 10 dB = 13 dB, not 20 dB.
    # The correct way: convert to linear power (10^(SPL/10)), sum, convert back.
    valid = (SPL_p > -900) & (SPL_s > -900)
    SPL_combined = np.full(n_freqs, -np.inf)

    for i in range(n_freqs):
        if valid[i]:
            SPL_combined[i] = 10.0 * np.log10(
                10.0 ** (SPL_p[i] / 10.0) + 10.0 ** (SPL_s[i] / 10.0)
            )
        elif SPL_p[i] > -900:
            SPL_combined[i] = SPL_p[i]
        elif SPL_s[i] > -900:
            SPL_combined[i] = SPL_s[i]

    # ── Overall SPL: power-sum across all frequencies ─────────────────────
    # Same principle: convert all frequency bins to linear, sum, convert back.
    # This gives the "broadband" SPL — what a sound level meter would read.
    valid_combined = SPL_combined > -900
    if not np.any(valid_combined):
        # Nothing computed — return the clipping floor
        return config.SPL_CLIP_MIN, freqs, SPL_combined

    linear_sum = np.sum(10.0 ** (SPL_combined[valid_combined] / 10.0))
    SPL_total  = 10.0 * np.log10(linear_sum)

    # Apply physical sanity bounds
    SPL_total = float(np.clip(SPL_total, config.SPL_CLIP_MIN, config.SPL_CLIP_MAX))

    return SPL_total, freqs, SPL_combined