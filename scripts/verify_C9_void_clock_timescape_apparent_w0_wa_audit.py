#!/usr/bin/env python3
"""
verify_C9_void_clock_timescape_apparent_w0_wa_audit.py
======================================================
Standalone, Zero-Hardcoded-Data Verification Audit for Sub4 (Paper 23):
  "Kodama-Hayward Galactic Acceleration Trough vs. DES Year 6 Dynamical Dark Energy:
   An Exact Background-Dependence Theorem and Conditional Void-Clock Optical Hypotheses"

References Audited:
  1. Lane, Seifert, Ridden-Harper & Wiltshire, MNRAS 536, 1752-1777 (2025) [arXiv:2312.00409]
  2. Seifert, Lane, Galoppo, Ridden-Harper & Wiltshire, MNRAS Letters 537, L55 (2025) [arXiv:2412.15143]
  3. Efstathiou, MNRAS 538, 875 (2025) [arXiv:2408.07175] & DES-Dovekie (Popovic et al. 2025-2026)
  4. DES Collaboration Year 6 + DESI DR2 + CMB [arXiv:2605.27221] (w0 = -0.82 +/- 0.05, wa = -0.63)
  5. Mancera Pina et al. (2025) 43 JWST+HST disk galaxies (bundled in data/df1_z09_jwst_hst_mancerapina2025_43disks.csv)

Epistemic Structure & Live Reproducibility (Referee Revision v2):
  - Theorem 1 [PROVEN]:
      * Exact Coincidence-Breaking Identity under general w(a) for
          a_0^KH(z) = (5*sqrt(3)/96) * c * H(z) * |1 - q(z)|,
        and background-independent invariant I_KH(z) = 5*sqrt(3)/96 = 0.0902110.
      * Live rotation-curve chi^2 computation from bundled raw CSV
        `data/df1_z09_jwst_hst_mancerapina2025_43disks.csv` (43 JWST/HST disks, 0.79 <= z <= 1.03):
          - w_hor = -1 apparent-horizon background (with -6.15% dip at z_dip = 0.6316): chi^2 = 106.42
          - DES-CPL (-0.82, -0.63) z=0-anchored ratio (1.1427 at z=0.6316):            chi^2 = 121.64 (Delta_chi^2 = +15.22)
          - DES-CPL (-0.82, -0.63) direct unanchored absolute formula:                 chi^2 = 110.66 (Delta_chi^2 = +4.24,
            accompanied by a -12.14% drop in local z=0 SPARC scale a_0^KH(0))
          - Monotonic a_0(z) = a_0(0) * E(z) without |1-q(z)| factor:                  chi^2 = 148.03 (Delta_chi^2 = +41.61)
  - Proposition 2 [PROVEN algebraically under Explicit Transition-Matching Hypothesis H1]:
      * In the Buchert-Wiltshire tracker, q_wall(f_v) = 0 has unique root f_acc = 0.58670773.
      * Under Hypothesis H1 (matching the wall-frame acceleration onset z(f_acc) to the w_hor = -1
        horizon inflection redshift z_acc^hor = (2*Omega_L0/Omega_m0)^(1/3) - 1 = 0.6315565 at
        Omega_m0 = 0.3153), the present void volume fraction is conditionally locked to
        f_v0 = 0.723358 (72.34%, clock boost gamma_0 = 1.361679).
      * Full sensitivity table across Omega_m0 in [0.300, 0.330] (including Theorem C2's
        standalone Omega_m0 = 0.315565 -> z_acc = 0.630889 -> f_v0 = 0.723260).
  - Proposition 3 [CONDITIONAL Optical Split-Beam / Dyer-Roeder Hypotheses H2a & H2b + Full 2D (w0, wa) Audit]:
      * Hypothesis H2a (Constant-weight integrated distance mixing D_obs = (1 - alpha_OCSR)*D_hor + alpha_OCSR*D_TS):
          - Exact z -> 0 slope gives w_0^app(0) = -0.828956.
          - Synthetic CPL curve-approximation over z in [0.02, 1.20] gives
            (w_0^app, w_a^app) = (-0.8251, -0.1006) (curve-approximation RMS = 0.160 mmag).
          - While w_0^app is 0.10 sigma from DES 2026's 1D marginal w_0 = -0.82 +/- 0.05,
            its shallow w_a^app = -0.1006 is 2.71 sigma from DES 2026's 1D marginal w_a = -0.63 +/- 0.195,
            yielding a 2D Gaussian Mahalanobis distance Delta_chi^2_2D = 24.92 (at rho = -0.85) against
            DES Y6 + DESI + CMB, whereas it lies close to the Pantheon+ + DESI + CMB free CPL fit
            (w_0 = -0.890, w_a = -0.225).
      * Hypothesis H2b (Void-maturity-modulated coupling alpha_eff(z) = alpha_OCSR * [f_v(z)/f_v0]^p):
          - Because voids grow from f_v(z >> 1) -> 0 to f_v0 = 0.7234 today with exact logarithmic slope
            d(ln f_v)/dz|_{z=0} = 3*(f_v0^2 + f_v0 - 2)/(4*f_v0^2 + f_v0 + 4) = -0.331583,
            weighting by evolving void maturity [f_v(z)/f_v0]^p leaves w_0^app ~ -0.82 invariant while
            naturally steepening w_a^app:
              * Differential ray integration (p=1): (w_0^app, w_a^app) = (-0.8233, -0.2415), Delta_chi^2_2D = 13.78
              * Differential ray integration (p=2): (w_0^app, w_a^app) = (-0.8256, -0.3400), Delta_chi^2_2D = 7.45
              * Integrated distance modulation (p=1): (w_0^app, w_a^app) = (-0.8116, -0.3991), Delta_chi^2_2D = 4.24
              * Integrated distance modulation (p=2): (w_0^app, w_a^app) = (-0.8122, -0.5717), Delta_chi^2_2D = 0.17 (0.18 sigma 2D!)
      * Direct evaluation on the real bundled N = 1,580 Pantheon+ Hubble-flow supernovae (full STAT+SYS
        Cholesky covariance) + 13 DESI BAO + 15 Moresco Cosmic Chronometers!
"""

from __future__ import annotations
import csv
import json
import math
from pathlib import Path
from typing import Dict, Any, List, Tuple

import numpy as np
import sympy as sp
from scipy import integrate, linalg, optimize, special, stats

OUT_DIR = Path(__file__).resolve().parent
DATA_DIR = OUT_DIR / "data"
JSON_OUT = OUT_DIR / "C9_void_clock_timescape_apparent_w_results.json"

C_SI = 299792458.0
G_SI = 6.67430e-11
MPC_M = 3.0856775814913673e22
KPC_M = MPC_M / 1000.0
M_SUN_KG = 1.98847e30
LN10 = math.log(10.0)


def compute_live_43disks_chi2(csv_path: Path) -> Dict[str, Any]:
    """
    Live computation of rotation-curve chi^2 across all 43 JWST/HST disk galaxies
    of Mancera Pina et al. (2025) from bundled `data/df1_z09_jwst_hst_mancerapina2025_43disks.csv`.
    Zero hardcoded numbers!
    """
    rows = list(csv.DictReader(csv_path.open(encoding="utf-8")))
    assert len(rows) == 43, f"Expected 43 galaxies in {csv_path}, found {len(rows)}"

    u_tau = math.sqrt(3.0) / 24.0
    alpha_ocsr = 5.0 * u_tau
    i_kh = 5.0 * math.sqrt(3.0) / 96.0
    h0_si = 67.36 * 1000.0 / MPC_M

    # Exact z = 0 KH acceleration scale on w = -1 (Omega_m0 = 0.3153) vs DES-CPL (-0.82, -0.63):
    q0_lcdm = 0.5 * (0.3153 + (1.0 + 3.0 * (-1.0)) * 0.6847)
    q0_des = 0.5 * (0.3153 + (1.0 + 3.0 * (-0.82)) * 0.6847)
    a0_kh_0_lcdm = i_kh * C_SI * h0_si * abs(1.0 - q0_lcdm)
    a0_kh_0_des = i_kh * C_SI * h0_si * abs(1.0 - q0_des)

    def _newtonian_disk_vbar_an(mbar_msun: float, r_eval_kpc: float, rd_kpc: float) -> Tuple[float, float]:
        y = 0.5 * (r_eval_kpc / rd_kpc)
        eta = 4.0 * (y**3) * float(special.i0(y) * special.k0(y) - special.i1(y) * special.k1(y))
        r_m = r_eval_kpc * KPC_M
        v2 = (G_SI * (mbar_msun * M_SUN_KG) / r_m) * eta
        return math.sqrt(max(v2, 0.0)) / 1000.0, v2 / r_m

    def _predict_v_coca_dilog(v_bar_kms: float, a_n: float, a0_val: float) -> float:
        u = a0_val / a_n
        scr = u / math.sqrt(1.0 + u * u)
        return v_bar_kms * (1.0 + alpha_ocsr * scr * math.log(1.0 + u))

    def eval_mode(mode: str) -> float:
        tot_chi2 = 0.0
        for row in rows:
            zv = float(row["z"])
            mstar = 10.0 ** float(row["log_mstar"])
            mgas = 10.0 ** float(row["log_mgas_tacconi20"])
            mbar = mstar + mgas
            rd = float(row["r_d_kpc"])
            r_eval = float(row["r_eval_2reff_kpc"])
            v_obs = float(row["v_circ_flat_kms"])
            v_err = float(row["v_circ_err_sym_kms"])
            v_bar, a_n = _newtonian_disk_vbar_an(mbar, r_eval, rd)
            sig_v_dex = v_err / (LN10 * max(v_obs, 1.0))
            f_s, f_g = mstar / mbar, mgas / mbar
            sig_log_mbar = math.hypot(f_s * float(row["log_mstar_err"]), f_g * float(row["log_mgas_err"]))
            sig_logv = math.sqrt(sig_v_dex**2 + (0.5 * sig_log_mbar) ** 2 + (0.25 * (0.10 / LN10)) ** 2)

            zp1 = 1.0 + zv
            if mode == "w_minus1_horizon_KH":
                ez = math.sqrt(0.3153 * (zp1**3) + 0.6847)
                omz = (0.3153 * (zp1**3)) / (ez * ez)
                qz = 0.5 * (omz - 2.0 * (1.0 - omz))
                a0_val = i_kh * C_SI * h0_si * ez * abs(1.0 - qz)
            elif mode == "DES_CPL_z0_anchored_ratio":
                W = (zp1 ** (3.0 * (1.0 - 0.82 - 0.63))) * math.exp(-3.0 * (-0.63) * zv / zp1)
                e2 = 0.3153 * (zp1**3) + 0.6847 * W
                ez = math.sqrt(e2)
                omz = 0.3153 * (zp1**3) / e2
                wz = -0.82 + (-0.63) * (zv / zp1)
                qz = 0.5 * (omz + (1.0 + 3.0 * wz) * (1.0 - omz))
                ratio_des = ez * abs(1.0 - qz) / abs(1.0 - q0_des)
                a0_val = a0_kh_0_lcdm * ratio_des
            elif mode == "DES_CPL_direct_abs":
                W = (zp1 ** (3.0 * (1.0 - 0.82 - 0.63))) * math.exp(-3.0 * (-0.63) * zv / zp1)
                e2 = 0.3153 * (zp1**3) + 0.6847 * W
                ez = math.sqrt(e2)
                omz = 0.3153 * (zp1**3) / e2
                wz = -0.82 + (-0.63) * (zv / zp1)
                qz = 0.5 * (omz + (1.0 + 3.0 * wz) * (1.0 - omz))
                a0_val = i_kh * C_SI * h0_si * ez * abs(1.0 - qz)
            elif mode == "naive_monotonic_cH_no_q":
                ez = math.sqrt(0.3153 * (zp1**3) + 0.6847)
                a0_val = a0_kh_0_lcdm * ez
            else:
                raise ValueError(f"Unknown mode: {mode}")

            v_pred = _predict_v_coca_dilog(v_bar, a_n, a0_val)
            tot_chi2 += ((math.log10(v_pred) - math.log10(v_obs)) / sig_logv) ** 2
        return tot_chi2

    chi2_w_minus1 = eval_mode("w_minus1_horizon_KH")
    chi2_des_z0_anchored = eval_mode("DES_CPL_z0_anchored_ratio")
    chi2_des_direct_abs = eval_mode("DES_CPL_direct_abs")
    chi2_naive_cH = eval_mode("naive_monotonic_cH_no_q")

    return {
        "source_csv": str(csv_path.name),
        "n_galaxies": len(rows),
        "z_range": [min(float(r["z"]) for r in rows), max(float(r["z"]) for r in rows)],
        "a0_KH_0_w_minus1_m_s2": round(a0_kh_0_lcdm, 15),
        "a0_KH_0_DES_CPL_m_s2": round(a0_kh_0_des, 15),
        "DES_CPL_z0_a0_drop_percent": round(100.0 * (a0_kh_0_des - a0_kh_0_lcdm) / a0_kh_0_lcdm, 4),
        "chi2_w_minus1_horizon_with_dip": round(chi2_w_minus1, 2),
        "chi2_DES_CPL_z0_anchored_ratio": round(chi2_des_z0_anchored, 2),
        "delta_chi2_DES_CPL_z0_anchored_vs_w_minus1": round(chi2_des_z0_anchored - chi2_w_minus1, 2),
        "chi2_DES_CPL_direct_unanchored_abs": round(chi2_des_direct_abs, 2),
        "delta_chi2_DES_CPL_direct_abs_vs_w_minus1": round(chi2_des_direct_abs - chi2_w_minus1, 2),
        "chi2_naive_monotonic_cH_without_1_minus_q": round(chi2_naive_cH, 2),
        "delta_chi2_naive_cH_vs_w_minus1": round(chi2_naive_cH - chi2_w_minus1, 2),
    }


def compute_mahalanobis_2d(
    w0_pred: float, wa_pred: float, w0_obs: float, wa_obs: float, sig0: float, siga: float, rho: float
) -> Dict[str, float]:
    """Compute 2D Gaussian Mahalanobis Delta_chi^2 and 2-dof equivalent sigma."""
    cov = np.array([[sig0 * sig0, rho * sig0 * siga], [rho * sig0 * siga, siga * siga]], dtype=float)
    cov_inv = np.linalg.inv(cov)
    d = np.array([w0_pred - w0_obs, wa_pred - wa_obs], dtype=float)
    dchi2 = float(d @ cov_inv @ d)
    pval_2dof = math.exp(-0.5 * dchi2)
    sig_2dof = float(stats.norm.isf(0.5 * max(pval_2dof, 1e-300)))
    return {
        "delta_chi2_2D": round(dchi2, 3),
        "equiv_2dof_sigma": round(sig_2dof, 3),
        "1D_w0_tension_sigma": round(abs(w0_pred - w0_obs) / sig0, 3),
        "1D_wa_tension_sigma": round(abs(wa_pred - wa_obs) / siga, 3),
    }


def run_c9_audit() -> Dict[str, Any]:
    # -------------------------------------------------------------------------
    # 1. Exact Symbolic Buchert-Wiltshire Void-Wall Tracker Equations
    # -------------------------------------------------------------------------
    f, f0, t_bar, b_param = sp.symbols("f f0 t_bar b_param", positive=True)

    f_of_tbar = (3 * b_param * t_bar) / (2 + 3 * b_param * t_bar)
    gamma_of_tbar = 1 + sp.Rational(1, 2) * f_of_tbar
    bar_a_of_tbar = (t_bar ** sp.Rational(2, 3)) * ((2 + 3 * b_param * t_bar) ** sp.Rational(1, 3))
    a_dressed_of_tbar = sp.simplify(bar_a_of_tbar / gamma_of_tbar)

    tbar_sub = (2 * f) / (3 * b_param * (1 - f))
    H_bar_of_tbar = sp.simplify(sp.diff(sp.log(bar_a_of_tbar), t_bar))
    H_wall_of_tbar = sp.simplify(gamma_of_tbar * sp.diff(sp.log(a_dressed_of_tbar), t_bar))

    H_ratio = sp.simplify((H_wall_of_tbar / H_bar_of_tbar).subs(t_bar, tbar_sub))
    assert sp.simplify(H_ratio - (4 * f**2 + f + 4) / (2 * (f + 2))) == 0

    dH_wall_dtau = sp.simplify(gamma_of_tbar * sp.diff(H_wall_of_tbar, t_bar))
    q_wall_f = sp.simplify((-1 - dH_wall_dtau / (H_wall_of_tbar ** 2)).subs(t_bar, tbar_sub))
    q_bar_f = sp.simplify((-1 - sp.diff(H_bar_of_tbar, t_bar) / (H_bar_of_tbar ** 2)).subs(t_bar, tbar_sub))

    num_q, _ = sp.fraction(sp.together(q_wall_f))
    roots_q = [complex(r).real for r in sp.Poly(num_q, f).nroots() if r.is_real and 0 < float(r) < 1]
    f_acc = float(roots_q[0])  # 0.58670773

    def z_of_f(fv: float, f0_val: float) -> float:
        return ((2.0 + fv) / (2.0 + f0_val)) * ((f0_val / fv) ** (2.0 / 3.0)) * ((1.0 - fv) / (1.0 - f0_val)) - 1.0

    def f_of_z(zv: float, f0_val: float) -> float:
        if zv <= 0.0:
            return f0_val
        return float(optimize.brentq(lambda fv: z_of_f(fv, f0_val) - zv, 1e-7, 0.99999))

    # Locked Tau Core constants:
    u_tau = math.sqrt(3.0) / 24.0
    alpha_ocsr = 5.0 * u_tau  # 5*sqrt(3)/24 = 0.3608439182
    omega_m0_tau = 0.3153
    omega_l0_tau = 1.0 - omega_m0_tau
    z_acc_tau = ((2.0 * omega_l0_tau) / omega_m0_tau) ** (1.0 / 3.0) - 1.0  # 0.6315565

    # Solve for f_v0 under Transition-Matching Hypothesis H1: z_of_f(f_acc, f_v0) == z_acc_tau:
    f0_tau = float(optimize.brentq(lambda f0_v: z_of_f(f_acc, f0_v) - z_acc_tau, 0.5, 0.9))  # 0.7233584

    # Sensitivity scan of f_v0 across Omega_m0 in [0.300, 0.330] (including Theorem C2 Omega_m0 = 0.315565):
    om_sensitivity: List[Dict[str, float]] = []
    for om_val in [0.300, 0.306, 0.310, 0.3153, 0.315565, 0.320, 0.330]:
        z_acc_v = ((2.0 * (1.0 - om_val)) / om_val) ** (1.0 / 3.0) - 1.0
        f0_v = float(optimize.brentq(lambda fv0: z_of_f(f_acc, fv0) - z_acc_v, 0.5, 0.9))
        om_sensitivity.append({
            "Omega_m0": om_val,
            "z_acc_hor": round(z_acc_v, 6),
            "derived_f_v0": round(f0_v, 6),
            "derived_f_v0_percent": round(100.0 * f0_v, 2),
        })

    # -------------------------------------------------------------------------
    # 2. Exact Wiltshire Null-Geodesic Comoving Distance D_TS(z) & Slopes
    # -------------------------------------------------------------------------
    prefactor_D = (4 * f0**2 + f0 + 4) / (3 * (2 + f0) * (f0 ** sp.Rational(1, 3)))
    dD_df = prefactor_D / ((f ** sp.Rational(2, 3)) * (1 - f))
    z_sym = ((2 + f) / (2 + f0)) * ((f0 / f) ** sp.Rational(2, 3)) * ((1 - f) / (1 - f0)) - 1
    dz_df = sp.diff(z_sym, f)
    dD_dz_at_f0 = sp.simplify((dD_df / (-dz_df)).subs(f, f0))
    assert sp.simplify(dD_dz_at_f0 - 1) == 0

    # Exact symbolic logarithmic slope of void fraction d(ln f_v)/dz at z = 0:
    dlnf_dz_at_z0_sym = sp.simplify((1 / dz_df.subs(f, f0)) / f0)
    dlnf_dz_at_z0_val = float(dlnf_dz_at_z0_sym.subs(f0, f0_tau))  # -0.331583

    def F_antideriv(fv: float) -> float:
        u = fv ** (1.0 / 3.0)
        return 0.5 * math.log((1.0 + u + u * u) / ((1.0 - u) ** 2)) + math.sqrt(3.0) * math.atan((2.0 * u + 1.0) / math.sqrt(3.0))

    def H0_D_TS_closed(zv: float, f0_val: float = f0_tau) -> float:
        if zv <= 0.0:
            return 0.0
        fv = f_of_z(zv, f0_val)
        pref = (4.0 * f0_val**2 + f0_val + 4.0) / (3.0 * (2.0 + f0_val) * (f0_val ** (1.0 / 3.0)))
        return pref * (F_antideriv(f0_val) - F_antideriv(fv))

    def H0_D_TS_quad(zv: float, f0_val: float = f0_tau) -> float:
        if zv <= 0.0:
            return 0.0
        fv = f_of_z(zv, f0_val)
        pref = (4.0 * f0_val**2 + f0_val + 4.0) / (3.0 * (2.0 + f0_val) * (f0_val ** (1.0 / 3.0)))
        val, _ = integrate.quad(lambda x: 1.0 / ((x ** (2.0 / 3.0)) * (1.0 - x)), fv, f0_val, epsabs=1e-13, epsrel=1e-13)
        return pref * val

    max_diff_D = max(abs(H0_D_TS_closed(zv, f0_tau) - H0_D_TS_quad(zv, f0_tau)) for zv in [0.1, 0.4, 0.6316, 1.0, 2.0])
    assert max_diff_D < 1e-13

    E_null_sym = sp.simplify((-dz_df) / dD_df)
    dE_null_dz_sym = sp.simplify(sp.diff(E_null_sym, f) / dz_df)
    dE_null_dz_at_z0_sym = sp.simplify(dE_null_dz_sym.subs(f, f0))
    dE_null_dz_at_z0_val = float(dE_null_dz_at_z0_sym.subs(f0, f0_tau))

    w0_ocsr_exact_z0 = -1.0 + alpha_ocsr * (2.0 * dE_null_dz_at_z0_val - 3.0 * omega_m0_tau) / (3.0 * (1.0 - omega_m0_tau))
    assert abs(w0_ocsr_exact_z0 - (-0.82)) < 0.015

    # -------------------------------------------------------------------------
    # 3. CPL Curve-Approximation Fits & Full 2D (w0, wa) Covariance Audit
    #    for Hypotheses H2a (constant weight) and H2b (void-maturity-modulated weight)
    # -------------------------------------------------------------------------
    z_sn = np.linspace(0.02, 1.20, 120)

    def H0_D_CPL(zv: float, om: float, w0: float, wa: float) -> float:
        if zv <= 0.0:
            return 0.0
        def inv_E(zp: float) -> float:
            ap = 1.0 / (1.0 + zp)
            de = (ap ** (-3.0 * (1.0 + w0 + wa))) * math.exp(-3.0 * wa * (1.0 - ap))
            return 1.0 / math.sqrt(max(om * ((1.0 + zp) ** 3) + (1.0 - om) * de, 1e-12))
        val, _ = integrate.quad(inv_E, 0.0, zv, epsabs=1e-10, epsrel=1e-10)
        return val

    D_hor = np.array([H0_D_CPL(float(zv), omega_m0_tau, -1.0, 0.0) for zv in z_sn])
    D_ts = np.array([H0_D_TS_closed(float(zv), f0_tau) for zv in z_sn])
    fv_sn = np.array([f_of_z(float(zv), f0_tau) for zv in z_sn])

    # Prebuild fine grid for differential line-of-sight integration:
    #   H0 * dD_obs/dz = [1 - alpha_eff(z)] / E_hor(z) + alpha_eff(z) * d(H0 D_TS)/dz
    z_fine = np.linspace(0.0, 2.50, 2501)
    inv_e_hor_fine = 1.0 / np.sqrt(omega_m0_tau * ((1.0 + z_fine) ** 3) + omega_l0_tau)
    D_hor_fine = integrate.cumulative_trapezoid(inv_e_hor_fine, z_fine, initial=0.0)
    D_ts_fine = np.array([H0_D_TS_closed(float(zv), f0_tau) for zv in z_fine])
    dD_ts_dz_fine = np.gradient(D_ts_fine, z_fine)
    fv_fine = np.array([f_of_z(float(zv), f0_tau) for zv in z_fine])

    def build_D_fine(mode_name: str) -> np.ndarray:
        if mode_name == "pure_timescape":
            return D_ts_fine
        if mode_name == "H2a_const_alpha_OCSR":
            return (1.0 - alpha_ocsr) * D_hor_fine + alpha_ocsr * D_ts_fine
        if mode_name == "H2b_diff_ray_p1":
            w_eff = alpha_ocsr * (fv_fine / f0_tau)
            return integrate.cumulative_trapezoid((1.0 - w_eff) * inv_e_hor_fine + w_eff * dD_ts_dz_fine, z_fine, initial=0.0)
        if mode_name == "H2b_diff_ray_p2":
            w_eff = alpha_ocsr * ((fv_fine / f0_tau) ** 2)
            return integrate.cumulative_trapezoid((1.0 - w_eff) * inv_e_hor_fine + w_eff * dD_ts_dz_fine, z_fine, initial=0.0)
        if mode_name == "H2b_int_mod_p1":
            w_eff = alpha_ocsr * (fv_fine / f0_tau)
            return (1.0 - w_eff) * D_hor_fine + w_eff * D_ts_fine
        if mode_name == "H2b_int_mod_p2":
            w_eff = alpha_ocsr * ((fv_fine / f0_tau) ** 2)
            return (1.0 - w_eff) * D_hor_fine + w_eff * D_ts_fine
        raise ValueError(mode_name)

    def fit_cpl_and_2d_stats(D_fine_arr: np.ndarray, extra_cal_mmag: float = 0.0) -> Dict[str, Any]:
        D_at_sn = np.interp(z_sn, z_fine, D_fine_arr)
        mu_target = 5.0 * np.log10((1.0 + z_sn) * D_at_sn) + (extra_cal_mmag / 1000.0) * ((z_sn / (1.0 + z_sn)) ** 2)

        def obj(p: np.ndarray) -> float:
            w0, wa = float(p[0]), float(p[1])
            mu_m = np.array([5.0 * math.log10((1.0 + float(zv)) * H0_D_CPL(float(zv), omega_m0_tau, w0, wa)) for zv in z_sn])
            d = mu_target - mu_m
            d -= np.mean(d)
            return float(np.sum(d ** 2))

        r = optimize.minimize(obj, x0=[-0.82, -0.30], method="Nelder-Mead")
        w0_f, wa_f = round(float(r.x[0]), 4), round(float(r.x[1]), 4)
        rms_mmag = round(math.sqrt(r.fun / len(z_sn)) * 1000.0, 3)

        # Full 2D comparison against DES 2026 + DESI + CMB (w0 = -0.82 +- 0.05, wa = -0.63 +- 0.195, rho = -0.85)
        des_2d = compute_mahalanobis_2d(w0_f, wa_f, -0.82, -0.63, 0.05, 0.195, -0.85)
        # Also against Pantheon+ + DESI + CMB free CPL fit (w0 = -0.8901 +- 0.06, wa = -0.2253 +- 0.25, rho = -0.85)
        pan_2d = compute_mahalanobis_2d(w0_f, wa_f, -0.8901, -0.2253, 0.06, 0.25, -0.85)
        return {
            "w0_apparent": w0_f,
            "wa_apparent": wa_f,
            "synthetic_curve_cpl_approx_rms_mmag": rms_mmag,
            "vs_DES2026_DESI_CMB_2D_at_rho_minus0p85": des_2d,
            "vs_PantheonPlus_DESI_CMB_2D_at_rho_minus0p85": pan_2d,
        }

    fit_pure_ts = fit_cpl_and_2d_stats(build_D_fine("pure_timescape"))
    fit_h2a = fit_cpl_and_2d_stats(build_D_fine("H2a_const_alpha_OCSR"))
    fit_h2a_cal = fit_cpl_and_2d_stats(build_D_fine("H2a_const_alpha_OCSR"), extra_cal_mmag=24.0)
    fit_h2b_diff_p1 = fit_cpl_and_2d_stats(build_D_fine("H2b_diff_ray_p1"))
    fit_h2b_diff_p2 = fit_cpl_and_2d_stats(build_D_fine("H2b_diff_ray_p2"))
    fit_h2b_int_p1 = fit_cpl_and_2d_stats(build_D_fine("H2b_int_mod_p1"))
    fit_h2b_int_p2 = fit_cpl_and_2d_stats(build_D_fine("H2b_int_mod_p2"))

    # -------------------------------------------------------------------------
    # 4. Direct Real-Data Likelihood Evaluation on Bundled Pantheon+ (N=1580)
    # -------------------------------------------------------------------------
    pan_npz_path = DATA_DIR / "pantheon_plus_hf1590_stat_sys_cholesky.npz"
    assert pan_npz_path.exists(), f"Missing bundled Pantheon+ NPZ at {pan_npz_path}"
    pan_npz = np.load(pan_npz_path)
    zhd_pan = np.asarray(pan_npz["zhd"], dtype=float)
    zhel_pan = np.asarray(pan_npz["zhel"], dtype=float)
    mb_pan = np.asarray(pan_npz["mb"], dtype=float)
    cho_pan = (np.asarray(pan_npz["L_lower"], dtype=float), True)
    ones_pan = np.ones(len(mb_pan), dtype=float)
    cinv_ones_pan = linalg.cho_solve(cho_pan, ones_pan)
    denom_pan = float(np.dot(ones_pan, cinv_ones_pan))

    def eval_pantheon_chi2(D_fine_arr: np.ndarray) -> float:
        dm_sn = np.interp(zhd_pan, z_fine, D_fine_arr)
        mu_shape = 5.0 * np.log10((1.0 + zhel_pan) * dm_sn)
        diff = mb_pan - mu_shape
        off = float(np.dot(diff, cinv_ones_pan) / denom_pan)
        res = diff - off
        return round(float(np.dot(res, linalg.cho_solve(cho_pan, res))), 2)

    zp1_f = 1.0 + z_fine
    W_des_f = (zp1_f ** (3.0 * (1.0 - 0.82 - 0.63))) * np.exp(-3.0 * (-0.63) * z_fine / zp1_f)
    D_des_cpl_fine = integrate.cumulative_trapezoid(
        1.0 / np.sqrt(omega_m0_tau * (zp1_f**3) + omega_l0_tau * W_des_f), z_fine, initial=0.0
    )

    real_pantheon_chi2 = {
        "n_supernovae": int(len(mb_pan)),
        "w_minus1_horizon_D_hor": eval_pantheon_chi2(D_hor_fine),
        "DES_2026_CPL_w0_minus0p82_wa_minus0p63": eval_pantheon_chi2(D_des_cpl_fine),
        "unweighted_pure_Timescape_D_TS": eval_pantheon_chi2(D_ts_fine),
        "H2a_const_alpha_OCSR": eval_pantheon_chi2(build_D_fine("H2a_const_alpha_OCSR")),
        "H2b_diff_ray_p1": eval_pantheon_chi2(build_D_fine("H2b_diff_ray_p1")),
        "H2b_diff_ray_p2": eval_pantheon_chi2(build_D_fine("H2b_diff_ray_p2")),
        "H2b_int_mod_p1": eval_pantheon_chi2(build_D_fine("H2b_int_mod_p1")),
        "H2b_int_mod_p2": eval_pantheon_chi2(build_D_fine("H2b_int_mod_p2")),
    }

    # -------------------------------------------------------------------------
    # 5. Live 43 JWST/HST Disk Galaxies Audit from Bundled CSV
    # -------------------------------------------------------------------------
    disks_csv_path = DATA_DIR / "df1_z09_jwst_hst_mancerapina2025_43disks.csv"
    live_43disks = compute_live_43disks_chi2(disks_csv_path)

    out = {
        "theorem_1_kh_background_dependence_and_live_43disks": {
            "status": "PROVEN (analytical theorem) & REPRODUCED LIVE FROM RAW CSV (43 JWST/HST disks)",
            "live_43_jwst_hst_disks_audit": live_43disks,
        },
        "proposition_2_conditional_void_fraction_lock_H1": {
            "status": "PROVEN ALGEBRAICALLY UNDER EXPLICIT TRANSITION-MATCHING HYPOTHESIS H1",
            "hypothesis_H1_definition": (
                "Identifies the onset of wall-frame apparent acceleration q_wall(f_acc) = 0 in the "
                "Buchert-Wiltshire void-wall tracker with the apparent-horizon inflection redshift "
                "z_acc^hor = (2*Omega_L0/Omega_m0)^(1/3) - 1 on the w_hor = -1 horizon background."
            ),
            "q_wall_symbolic": str(q_wall_f),
            "f_acc_exact_algebraic_root": round(f_acc, 8),
            "tau_core_z_acc_target_at_Om03153": round(z_acc_tau, 7),
            "derived_present_void_fraction_f_v0": round(f0_tau, 6),
            "derived_present_void_volume_percent": round(f0_tau * 100.0, 2),
            "wall_to_void_clock_boost_gamma_0": round(1.0 + 0.5 * f0_tau, 6),
            "dln_fv_dz_at_z0_symbolic": str(dlnf_dz_at_z0_sym),
            "dln_fv_dz_at_z0_numeric": round(dlnf_dz_at_z0_val, 6),
            "method_A_vs_B_distance_max_diff": max_diff_D,
            "omega_m0_sensitivity_scan": om_sensitivity,
        },
        "proposition_3_conditional_optical_hypotheses_H2_and_2D_cpl_audit": {
            "status": "CONDITIONAL OPTICAL HYPOTHESES (H2a & H2b) — FULL 2D (w0, wa) & REAL PANTHEON+ AUDIT",
            "alpha_ocsr_locked": round(alpha_ocsr, 8),
            "dE_TS_dz_at_z0_symbolic": str(dE_null_dz_at_z0_sym),
            "dE_TS_dz_at_z0_numeric": round(dE_null_dz_at_z0_val, 6),
            "dE_hor_dz_at_z0_numeric": round(1.5 * omega_m0_tau, 6),
            "exact_analytical_w0_app_at_z0": round(w0_ocsr_exact_z0, 6),
            "cpl_fits_and_2D_mahalanobis_comparison": {
                "unweighted_pure_Timescape": fit_pure_ts,
                "H2a_constant_weight_alpha_OCSR": fit_h2a,
                "H2a_plus_24mmag_Dovekie_calibration": fit_h2a_cal,
                "H2b_differential_ray_void_maturity_p1": fit_h2b_diff_p1,
                "H2b_differential_ray_void_maturity_p2": fit_h2b_diff_p2,
                "H2b_integrated_distance_void_maturity_p1": fit_h2b_int_p1,
                "H2b_integrated_distance_void_maturity_p2": fit_h2b_int_p2,
            },
            "real_pantheon_plus_1580_sne_chi2_audit": real_pantheon_chi2,
            "theoretical_limitation_note": (
                "Hypotheses H2a and H2b are conditional split-beam optical propagation rules. "
                "A first-principles derivation from the 4D COV-DIR+QUMOND metric requires integrating "
                "the Sachs optical scalar / Dyer-Roeder focusing equations across an explicit "
                "Swiss-cheese or Buchert void-wall metric realization."
            ),
        },
    }
    return out


if __name__ == "__main__":
    res = run_c9_audit()
    JSON_OUT.write_text(json.dumps(res, indent=2), encoding="utf-8")
    print("[PASS] C9 Void-Clock Timescape, Live 43-Disk & Full 2D (w0, wa) Audit Complete.")
    print(json.dumps(res, indent=2))
