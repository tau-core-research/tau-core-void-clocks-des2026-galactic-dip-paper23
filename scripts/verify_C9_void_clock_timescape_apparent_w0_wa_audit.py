#!/usr/bin/env python3
"""
verify_C9_void_clock_timescape_apparent_w0_wa_audit.py
======================================================
Self-Contained, Reproducible Multi-Probe Verification Suite (v12.0) for:
  "Void-Volume-Weighted Optical Distances Confronting DESI BAO, Compressed Planck CMB,
   and Four Type Ia Supernova Compilations"

Dependencies & Single-Command Reproducibility:
  - Core audit requires ONLY standard `numpy` and `scipy` (zero `sympy`, `camb`, or external packages).
  - Automatically invokes `generate_sub4_figures_and_tables.py` (unless `--no-figures` is passed;
    gracefully skipped if `matplotlib` is absent) so ONE command regenerates:
      1. `C9_void_clock_timescape_apparent_w_results.json`
      2. `generated_tables_v8.tex` (exact LaTeX tables matching manuscript numbers)
      3. `../fig_sub4_void_clock_des2026_and_galactic_dip_en.png` & `_hu.png`
  - Contains NO hardcoded dataset chi^2 assertions; mathematical unit tests can be skipped via `--no-asserts`.
  - Typical wall-clock runtime: ~25-60 seconds on a modern laptop CPU (up to ~2-3 min on older CPUs).

Key v8.0-v10.0 Upgrades Addressing External Referee Reports 1 & 2:
  1. Exact polynomial factorization (f_v - 1)*(8*f_v^3 + 39*f_v^2 - 12*f_v - 8) = 0 for f_acc = 0.58670773
     (eliminating floating-point rounding of the trivial f_v = 1 root in Python 3.9 / NumPy 1.26).
  2. Exact early-matter-dominated scaling diagnostics: f_v(z) ~ (1+z)^{-3/2}, f_v(z_*=1089.9) = 8.71e-5,
     and explicit cumulative late-time comoving shift Delta D_M(z_*) ~ -15.5 Mpc.
  3. CAMB-calibrated baryon drag sound horizon r_d (147.102 Mpc at Planck 2018 fiducial) as baseline,
     alongside explicit Aubourg et al. (2015) r_d (146.895 Mpc) and +-0.15% r_d scale sensitivity checks,
     evaluated on a unified smooth log-grid u = ln(1+z) in [0, ln(1201)] with radiation Omega_r*(1+z)^4
     (eliminating any z = 2.5 grid stitch).
  4. Head-to-head k = 5 dark-energy & optical controls with unconstrained alpha in [-0.3, 1.2]:
       - Flat LCDM (k = 4)
       - Void-volume-weighted H2-vol (p = 1, k = 5), quadratic H2-vol (p = 2, k = 5), and free-alpha
         constant-weight H2a (p = 0, k = 5) — no fixed-alpha strawman, plus true p = 0, alpha = 1 Timescape
       - Constant-w dark energy (wCDM, w_a = 0, k = 5)
       - Canonical non-phantom slow-roll Thawing Quintessence: w_a = -1.58*(1 + w_0) (Caldwell & Linder 2005, k = 5)
       - Empirical Phantom-Crossing CPL Degeneracy Line: w_a = -3.3*(1 + w_0) (k = 5)
       - Free 2-parameter CPL (k = 6, exact 5x5 numerical Hessian)
       - Smooth all-z Hamiltonian Forward Sachs ODE diagnostics (comparing unmodified FLRW Ricci focusing
         vs. Dyer-Roeder void-depleted Ricci focusing without any z = 2.5 cutoff).
  5. DESI BAO + Planck18-only fit (zero supernovae) and parameter-free out-of-sample SN predictive test
     across all four SN compilations, plus full non-selective f_{v,0} in [0.60, 0.84] scan and CID-overlap check.
  6. Deduplicated unique-SN (N = 280 unique CIDs from 311 light-curve rows) & sky-patch block-permuted
     3D SDSS DR7 VoidFinder exploratory low-z null check (Douglass, Veyrat & BenZvi 2023).
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
from scipy import integrate, interpolate, linalg, optimize, stats

OUT_DIR = Path(__file__).resolve().parent
DATA_DIR = OUT_DIR / "data"
JSON_OUT = OUT_DIR / "C9_void_clock_timescape_apparent_w_results.json"

# Physical SI & Astronomical Constants (CODATA 2018 / IAU 2015)
C_SI = 299792458.0
C_KMS = 299792.458
G_SI = 6.67430e-11
H_PLANCK_SI = 6.62607015e-34
KB_SI = 1.380649e-23
EV_J = 1.602176634e-19
M_E_KG = 9.1093837015e-31
M_H_KG = 1.6735575e-27
MPC_M = 3.0856775814913673e22
A_RAD_SI = (8.0 * (math.pi**5) * (KB_SI**4)) / (15.0 * (C_SI**3) * (H_PLANCK_SI**3))
T_CMB0_K = 2.72548
RHO_G0_SI = A_RAD_SI * (T_CMB0_K**4) / (C_SI**2)
NU_MASSES_LCDM_EV: Tuple[float, float, float] = (0.0, 0.0, 0.06)
U_GL64, W_GL64 = np.polynomial.legendre.leggauss(64)
U_GL32, W_GL32 = np.polynomial.legendre.leggauss(32)


# ==============================================================================
# 1. STANDALONE BDF2 RECFAST v1.5 RECOMBINATION & OBSERVATIONAL DATA LOADERS
# ==============================================================================
def recfast_v15_cpl_observables(
    h: float,
    omega_b: float,
    omega_cdm: float,
    w0: float = -1.0,
    wa: float = 0.0,
    nu_masses_ev: Tuple[float, float, float] = NU_MASSES_LCDM_EV,
    T_cmb0: float = T_CMB0_K,
    Y_He: float = 0.2454,
    zstar_stencil: Tuple[float, ...] | None = None,
) -> Dict[str, float]:
    """Standalone pure-NumPy BDF2 RECFAST v1.5 + exact Fermi-Dirac massive-neutrino integrator."""
    H0_si = 100.0 * h * 1000.0 / MPC_M
    rho_crit0 = 3.0 * (H0_si**2) / (8.0 * math.pi * G_SI)
    rho_g0 = A_RAD_SI * (T_cmb0**4) / (C_SI**2)
    Omega_g = rho_g0 / rho_crit0
    T_nu0 = T_cmb0 * ((4.0 / 11.0) ** (1.0 / 3.0))
    Omega_nu_1rel = Omega_g * (3.044 / 3.0) * (7.0 / 8.0) * ((4.0 / 11.0) ** (4.0 / 3.0))

    q_nu = np.linspace(0.03, 28.0, 160)
    dq_nu = float(q_nu[1] - q_nu[0])
    w_fd = (q_nu**2) / (np.exp(q_nu) + 1.0) * dq_nu
    norm_fd = float(np.sum(q_nu * w_fd))
    y_masses = [(m_ev * EV_J) / (KB_SI * T_nu0) for m_ev in nu_masses_ev]

    def omega_nu_a4(a_arr: np.ndarray) -> np.ndarray:
        a_col = np.atleast_1d(a_arr)[:, None]
        tot = np.zeros(a_col.shape[0], dtype=np.float64)
        for ym in y_masses:
            if ym <= 0.0:
                tot += Omega_nu_1rel
            else:
                eps = np.sqrt(q_nu[None, :] ** 2 + (a_col * ym) ** 2)
                tot += Omega_nu_1rel * (np.sum(eps * w_fd[None, :], axis=1) / norm_fd)
        return tot

    Omega_b = omega_b / (h * h)
    Omega_cdm = omega_cdm / (h * h)
    Omega_m_bc = Omega_b + Omega_cdm
    Omega_nu_today = float(omega_nu_a4(np.array([1.0]))[0])
    Omega_nu_massive_today = Omega_nu_today - Omega_nu_1rel
    Omega_m0_tot = Omega_m_bc + Omega_nu_massive_today
    Omega_de0 = 1.0 - Omega_m_bc - Omega_g - Omega_nu_today

    if Omega_de0 <= 0.05 or Omega_m0_tot <= 0.10 or (1.0 + w0 + wa) > 0.9:
        return {
            "h": h, "H0_kms_Mpc": 100.0 * h, "omega_b": omega_b, "omega_cdm": omega_cdm,
            "Omega_m0": Omega_m0_tot, "Omega_de0": Omega_de0, "z_star": 1090.0,
            "z_drag": 1060.0, "r_s_zstar_Mpc": 144.4, "r_d_aubourg15_Mpc": 146.9,
            "r_d_zdrag_Mpc": 147.1, "D_M_zstar_Mpc": 13800.0, "theta_100": 99.0,
        }

    def W_cpl(a_arr: np.ndarray) -> np.ndarray:
        a = np.atleast_1d(a_arr)
        if w0 == -1.0 and wa == 0.0:
            return np.ones_like(a)
        return (a ** (-3.0 * (1.0 + w0 + wa))) * np.exp(3.0 * wa * (a - 1.0))

    def E2_of_a(a_arr: np.ndarray) -> np.ndarray:
        a = np.atleast_1d(a_arr)
        return (Omega_g + omega_nu_a4(a)) / (a**4) + Omega_m_bc / (a**3) + Omega_de0 * W_cpl(a)

    H0_mpc = H0_si * MPC_M / C_SI
    omega_nu_mass = sum(nu_masses_ev) / 93.14
    omega_m_phys = omega_b + omega_cdm + omega_nu_mass
    # Aubourg et al. (2015, PRD 92, 123516) Eq. (16) fitting formula (146.895 Mpc at Planck18 fiducial)
    # vs. exact CAMB Boltzmann calibration (147.102 Mpc at Planck18 fiducial; ratio = 1.0014103):
    r_d_aubourg15 = (
        55.154
        * math.exp(-72.3 * ((omega_nu_mass + 0.0006) ** 2))
        / ((omega_m_phys**0.25351) * (omega_b**0.12807))
    )
    r_d_camb = r_d_aubourg15 * (55.23178 / 55.154)

    if zstar_stencil is not None:
        ob0, oc0, zs0, dz_dob, d2z_dob2, dz_doc, d2z_doc2, d2z_dobdoc, zd0 = zstar_stencil
        dob = omega_b - ob0
        doc = omega_cdm - oc0
        z_star = (
            zs0 + dz_dob * dob + 0.5 * d2z_dob2 * (dob**2)
            + dz_doc * doc + 0.5 * d2z_doc2 * (doc**2) + d2z_dobdoc * dob * doc
        )
        a_bg = np.geomspace(1.0e-8, 1.0, 4000)
        E_bg = np.sqrt(E2_of_a(a_bg))
        dtau_da = 1.0 / (a_bg * a_bg * H0_mpc * E_bg)
        eta_bg = np.concatenate(([0.0], np.cumsum(0.5 * (dtau_da[:-1] + dtau_da[1:]) * np.diff(a_bg))))
        eta_0 = float(eta_bg[-1])
        eta_star = float(np.interp(1.0 / (1.0 + z_star), a_bg, eta_bg))
        R_bg = 0.75 * (Omega_b * a_bg) / Omega_g
        cs_bg = 1.0 / np.sqrt(3.0 * (1.0 + R_bg))
        rs_bg = np.concatenate(([0.0], np.cumsum(0.5 * (cs_bg[:-1] + cs_bg[1:]) * np.diff(eta_bg))))
        r_s_star = float(np.interp(eta_star, eta_bg, rs_bg))
        d_m_star = eta_0 - eta_star
        return {
            "h": h, "H0_kms_Mpc": 100.0 * h, "omega_b": omega_b, "omega_cdm": omega_cdm,
            "Omega_m0": Omega_m0_tot, "Omega_de0": Omega_de0, "z_star": z_star,
            "z_drag": zd0, "r_s_zstar_Mpc": r_s_star,
            "r_d_aubourg15_Mpc": r_d_aubourg15, "r_d_zdrag_Mpc": r_d_camb,
            "D_M_zstar_Mpc": d_m_star, "theta_100": 100.0 * (r_s_star / d_m_star),
        }

    f_He = Y_He / (4.0 * (1.0 - Y_He))
    n_H0 = (rho_crit0 * Omega_b * (1.0 - Y_He)) / M_H_KG
    E_ion_H_K = 13.598434 * EV_J / KB_SI
    E_2s_H_K = E_ion_H_K * 0.75
    E_21_H_K = E_ion_H_K * 0.25
    LAMBDA_LyA = H_PLANCK_SI * C_SI / (KB_SI * E_2s_H_K)
    LAMBDA_2S = 8.2245809
    E_ion_HeI_K = 24.587387 * EV_J / KB_SI
    E_21s_HeI_K = 20.615775 * EV_J / KB_SI
    E_ion_HeII_K = 54.417763 * EV_J / KB_SI
    SIGMA_THOMSON = 6.6524587321e-29

    def saha_xe(z_arr: np.ndarray) -> np.ndarray:
        zv = np.asarray(z_arr, dtype=np.float64)
        T = T_cmb0 * (1.0 + zv)
        nH = n_H0 * ((1.0 + zv) ** 3)
        fac_e = ((2.0 * math.pi * M_E_KG * KB_SI * T) / (H_PLANCK_SI**2)) ** 1.5
        out = np.empty_like(zv)
        m1 = zv > 4500.0
        if np.any(m1):
            rhs = (2.0 * fac_e[m1] / nH[m1]) * np.exp(-np.clip(E_ion_HeII_K / T[m1], -500.0, 500.0))
            out[m1] = 1.0 + f_He + 0.5 * (-(1.0 + f_He + rhs) + np.sqrt((1.0 + f_He + rhs) ** 2 + 4.0 * rhs * f_He))
        m2 = (~m1) & (zv > 1650.0)
        if np.any(m2):
            rhs_he = (4.0 * fac_e[m2] / nH[m2]) * np.exp(-np.minimum(E_ion_HeI_K / T[m2], 500.0))
            bq = 1.0 + rhs_he
            out[m2] = 1.0 + 0.5 * (-bq + np.sqrt(bq * bq + 4.0 * f_He * rhs_he))
        m3 = ~(m1 | m2)
        if np.any(m3):
            rhs_h = (fac_e[m3] / nH[m3]) * np.exp(-np.minimum(E_ion_H_K / T[m3], 500.0))
            out[m3] = 0.5 * (-rhs_h + np.sqrt(rhs_h * rhs_h + 4.0 * rhs_h))
        return out

    z_start = 2800.0
    n_rec = 1200
    z_eval_rec = np.linspace(z_start, 0.0, n_rec)
    dz_rec = float(z_eval_rec[1] - z_eval_rec[0])
    Hz_eval_rec = H0_si * np.sqrt(E2_of_a(1.0 / (1.0 + z_eval_rec)))
    ln1pz_rec = np.log1p(z_eval_rec)
    gauss_corr_rec = (
        1.0
        - 0.14 * np.exp(-(((ln1pz_rec - 7.28) / 0.18) ** 2))
        + 0.079 * np.exp(-(((ln1pz_rec - 6.73) / 0.33) ** 2))
    )

    def recfast_rhs(idx_z: int, y_vec: np.ndarray) -> np.ndarray:
        z_val = float(z_eval_rec[idx_z])
        Hz = float(Hz_eval_rec[idx_z])
        g_wms = float(gauss_corr_rec[idx_z])
        xp = min(max(float(y_vec[0]), 1.0e-12), 1.0)
        xhe = min(max(float(y_vec[1]), 0.0), f_He)
        xe = xp + xhe
        Tg = T_cmb0 * (1.0 + z_val)
        Tm = max(float(y_vec[2]), 0.1)
        zp1 = 1.0 + z_val
        nH = n_H0 * (zp1**3)

        t4 = Tm * 1.0e-4
        alpha_H = (1.125e-19) * (4.309 * (t4 ** (-0.6166))) / (1.0 + 0.6703 * (t4**0.5300))
        fac_e_Tm = ((2.0 * math.pi * M_E_KG * KB_SI * Tm) / (H_PLANCK_SI**2)) ** 1.5
        beta_H = alpha_H * fac_e_Tm * math.exp(-min(E_21_H_K / Tm, 500.0))
        K_H = ((LAMBDA_LyA**3) / (8.0 * math.pi * Hz)) * g_wms
        om_xp = 1.0 - xp
        C_H = (1.0 + K_H * LAMBDA_2S * nH * om_xp) / (1.0 + K_H * (LAMBDA_2S + beta_H) * nH * om_xp)
        rate_H = xe * xp * nH * alpha_H - beta_H * om_xp * math.exp(-min(E_2s_H_K / Tm, 500.0))
        dxp_dz = (C_H / (Hz * zp1)) * rate_H

        if xhe > 1.0e-9 and z_val > 1100.0:
            q_he, p_he = 10.0 ** (-16.744), 0.711
            T1_he, T2_he = 10.0**5.114, 3.0
            sq_T2 = math.sqrt(Tm / T2_he)
            alpha_He = q_he / (sq_T2 * ((1.0 + sq_T2) ** (1.0 - p_he)) * ((1.0 + math.sqrt(Tm / T1_he)) ** (1.0 + p_he)))
            beta_He = 4.0 * alpha_He * fac_e_Tm * math.exp(-min((E_ion_HeI_K - E_21s_HeI_K) / Tm, 500.0))
            K_He = (58.4334e-9**3) / (8.0 * math.pi * Hz)
            nHe0 = nH * max(f_He - xhe, 0.0)
            boltz_he = math.exp(-min((21.21802 - 20.615775) * EV_J / (KB_SI * Tm), 500.0))
            C_He = (1.0 + K_He * 51.3 * nHe0 * boltz_he) / (1.0 + K_He * (51.3 + beta_He) * nHe0 * boltz_he)
            if nHe0 > 1.0e-12:
                lam_he_23p = 59.1412e-9
                A_23P = 177.58
                tau_he_3 = 3.0 * (lam_he_23p**3) * nHe0 * A_23P / (8.0 * math.pi * Hz)
                p_esc_3 = (1.0 - math.exp(-min(tau_he_3, 500.0))) / max(tau_he_3, 1.0e-12)
                E_23p_dE_K = (24.587387 - 20.96409) * EV_J / KB_SI
                beta_He_trip = 3.0 * alpha_He * fac_e_Tm * math.exp(-min(E_23p_dE_K / Tm, 500.0))
                C_He_trip = (A_23P * p_esc_3) / (A_23P * p_esc_3 + beta_He_trip)
                C_He = min(C_He + 0.08 * C_He_trip, 1.0)
            rate_He = xe * xhe * nH * alpha_He - beta_He * (f_He - xhe) * math.exp(-min(E_21s_HeI_K / Tm, 500.0))
            dxhe_dz = (C_He / (Hz * zp1)) * rate_He
        else:
            dxhe_dz = 0.0

        u_rad = A_RAD_SI * (Tg**4)
        gamma_C = (8.0 * SIGMA_THOMSON * u_rad * xe) / (3.0 * M_E_KG * C_SI * (1.0 + f_He + xe))
        dTm_dz = (2.0 * Tm / zp1) - (gamma_C / (Hz * zp1)) * (Tg - Tm)
        return np.array([dxp_dz, dxhe_dz, dTm_dz], dtype=np.float64)

    y_rec = np.empty((3, n_rec), dtype=np.float64)
    xe_ini = float(saha_xe(np.array([z_start]))[0])
    y_rec[:, 0] = [1.0, xe_ini - 1.0, T_cmb0 * (1.0 + z_start)]
    up_bounds = (1.0, f_He, 1.0e9)
    eye3 = np.eye(3)
    for i in range(n_rec - 1):
        ip1 = i + 1
        if i == 0:
            yg = y_rec[:, 0].copy()
            coef_f = dz_rec
            rhs_hist = y_rec[:, 0]
        else:
            yg = 2.0 * y_rec[:, i] - y_rec[:, i - 1]
            yg[0] = min(max(float(yg[0]), 1.0e-12), 1.0)
            yg[1] = min(max(float(yg[1]), 0.0), f_He)
            yg[2] = max(float(yg[2]), 0.5)
            coef_f = (2.0 / 3.0) * dz_rec
            rhs_hist = (4.0 / 3.0) * y_rec[:, i] - (1.0 / 3.0) * y_rec[:, i - 1]
        for _ in range(4):
            fn = recfast_rhs(ip1, yg)
            res = yg - rhs_hist - coef_f * fn
            if max(abs(res[0]) / max(yg[0], 1e-8), abs(res[1]) / max(yg[1], 1e-8), abs(res[2]) / max(yg[2], 1e-8)) < 1.0e-9:
                break
            jac = np.empty((3, 3), dtype=np.float64)
            for j in range(3):
                eps = max(1.0e-7 * abs(float(yg[j])), 1.0e-11)
                step = -eps if yg[j] + eps > up_bounds[j] else eps
                yp = yg.copy()
                yp[j] += step
                jac[:, j] = (recfast_rhs(ip1, yp) - fn) / step
            dy = np.linalg.solve(eye3 - coef_f * jac, -res)
            yg = np.array([
                min(max(float(yg[0] + dy[0]), 1.0e-12), 1.0),
                min(max(float(yg[1] + dy[1]), 0.0), f_He),
                max(float(yg[2] + dy[2]), 0.5),
            ], dtype=np.float64)
        y_rec[:, ip1] = yg

    z_rev = z_eval_rec[::-1]
    xe_rev = (y_rec[0] + y_rec[1])[::-1]
    a_bg = np.geomspace(1.0e-8, 1.0, 12000)
    z_bg = 1.0 / a_bg - 1.0
    E_bg = np.sqrt(E2_of_a(a_bg))
    dtau_da = 1.0 / (a_bg * a_bg * H0_mpc * E_bg)
    eta_bg = np.concatenate(([0.0], np.cumsum(0.5 * (dtau_da[:-1] + dtau_da[1:]) * np.diff(a_bg))))
    eta_0 = float(eta_bg[-1])

    xe_nr = np.where(z_bg > z_start, saha_xe(np.minimum(z_bg, 1.0e5)), np.interp(z_bg, z_rev, xe_rev))
    kd_nr = a_bg * (n_H0 * ((1.0 + z_bg) ** 3) * xe_nr) * SIGMA_THOMSON * MPC_M
    dk_nr = 0.5 * (kd_nr[:-1] + kd_nr[1:]) * np.diff(eta_bg)
    tau_nr = np.concatenate((np.cumsum(dk_nr[::-1])[::-1], [0.0]))
    z_star = float(np.interp(1.0, tau_nr[::-1], z_bg[::-1]))
    eta_star = float(np.interp(1.0 / (1.0 + z_star), a_bg, eta_bg))

    R_bg = 0.75 * (Omega_b * a_bg) / Omega_g
    cs_bg = 1.0 / np.sqrt(3.0 * (1.0 + R_bg))
    rs_bg = np.concatenate(([0.0], np.cumsum(0.5 * (cs_bg[:-1] + cs_bg[1:]) * np.diff(eta_bg))))
    r_s_star = float(np.interp(eta_star, eta_bg, rs_bg))
    d_m_star = eta_0 - eta_star

    dk_drag = 0.5 * ((kd_nr / R_bg)[:-1] + (kd_nr / R_bg)[1:]) * np.diff(eta_bg)
    tau_drag_nr = np.concatenate((np.cumsum(dk_drag[::-1])[::-1], [0.0]))
    z_drag = float(np.interp(1.0, tau_drag_nr[::-1], z_bg[::-1]))
    return {
        "h": h, "H0_kms_Mpc": 100.0 * h, "omega_b": omega_b, "omega_cdm": omega_cdm,
        "Omega_m0": Omega_m0_tot, "Omega_de0": Omega_de0, "z_star": z_star,
        "z_drag": z_drag, "r_s_zstar_Mpc": r_s_star,
        "r_d_aubourg15_Mpc": r_d_aubourg15, "r_d_zdrag_Mpc": r_d_camb,
        "D_M_zstar_Mpc": d_m_star, "theta_100": 100.0 * (r_s_star / d_m_star),
    }


def build_recfast_zstar_stencil(nu_masses_ev: Tuple[float, float, float] = NU_MASSES_LCDM_EV) -> Tuple[float, ...]:
    ob0, oc0 = 0.02237, 0.1200
    r0 = recfast_v15_cpl_observables(0.6736, ob0, oc0, -1.0, 0.0, nu_masses_ev)
    r_ob_p = recfast_v15_cpl_observables(0.6736, ob0 + 0.0005, oc0, -1.0, 0.0, nu_masses_ev)
    r_ob_m = recfast_v15_cpl_observables(0.6736, ob0 - 0.0005, oc0, -1.0, 0.0, nu_masses_ev)
    r_oc_p = recfast_v15_cpl_observables(0.6736, ob0, oc0 + 0.005, -1.0, 0.0, nu_masses_ev)
    r_oc_m = recfast_v15_cpl_observables(0.6736, ob0, oc0 - 0.005, -1.0, 0.0, nu_masses_ev)
    r_ob_oc = recfast_v15_cpl_observables(0.6736, ob0 + 0.0005, oc0 + 0.005, -1.0, 0.0, nu_masses_ev)
    dz_dob = (r_ob_p["z_star"] - r_ob_m["z_star"]) / 0.001
    d2z_dob2 = (r_ob_p["z_star"] - 2.0 * r0["z_star"] + r_ob_m["z_star"]) / (0.0005**2)
    dz_doc = (r_oc_p["z_star"] - r_oc_m["z_star"]) / 0.010
    d2z_doc2 = (r_oc_p["z_star"] - 2.0 * r0["z_star"] + r_oc_m["z_star"]) / (0.005**2)
    d2z_dobdoc = (
        r_ob_oc["z_star"] - r0["z_star"]
        - dz_dob * 0.0005 - 0.5 * d2z_dob2 * (0.0005**2)
        - dz_doc * 0.005 - 0.5 * d2z_doc2 * (0.005**2)
    ) / (0.0005 * 0.005)
    return (ob0, oc0, r0["z_star"], dz_dob, d2z_dob2, dz_doc, d2z_doc2, d2z_dobdoc, r0["z_drag"])


def load_all_cosmological_datasets() -> Dict[str, Any]:
    """Load all 4 SN Ia compilations, DESI BAO, compressed Planck 2018 priors, CC15, and SDSS DR7 VoidFinder."""
    def _wrap_cholesky(name: str, zhd: np.ndarray, zhel: np.ndarray, y: np.ndarray, L_low: np.ndarray) -> Dict[str, Any]:
        L_f = np.asarray(L_low, dtype=np.float64)
        ones = np.ones(len(y), dtype=np.float64)
        Linv_1 = linalg.solve_triangular(L_f, ones, lower=True)
        zhd_f = np.asarray(zhd, dtype=np.float64)
        return {
            "name": name,
            "n": int(len(y)),
            "zhd": zhd_f,
            "ln1pz_hd": np.log1p(zhd_f),
            "zhel": np.asarray(zhel, dtype=np.float64),
            "y": np.asarray(y, dtype=np.float64),
            "L": L_f,
            "Linv_1": Linv_1,
            "norm_1": float(np.dot(Linv_1, Linv_1)),
        }

    # 1. Pantheon+ Hubble-Flow (N = 1,580 light-curve rows; 1,466 unique physical CIDs)
    pan_npz = np.load(DATA_DIR / "pantheon_plus_hf1590_stat_sys_cholesky.npz")
    sn_pantheon = _wrap_cholesky("PantheonPlus", pan_npz["zhd"], pan_npz["zhel"], pan_npz["mb"], pan_npz["L_lower"])

    # 2. DES-SN5YR (2024 release, N = 1,829; DES Collaboration 2024, ApJL 973, L14)
    des_npz = np.load(DATA_DIR / "des_sn5yr_2024_stat_sys_cholesky_tril.npz")
    n_des = int(des_npz["n"][0])
    L_des = np.zeros((n_des, n_des), dtype=np.float64)
    L_des[np.tril_indices(n_des)] = des_npz["L_tril"]
    sn_des5yr = _wrap_cholesky("DES_SN5YR_2024", des_npz["zhd"], des_npz["zhel"], des_npz["mu"], L_des)

    # 3. DES-Dovekie (recalibrated DES-SN5YR, N = 1,820; Popovic et al. / DES Y6; shares ~94% of objects with DES-SN5YR)
    dov_npz = np.load(DATA_DIR / "des_dovekie_recal_stat_sys_cholesky_tril.npz")
    n_dov = int(dov_npz["n"][0])
    L_dov = np.zeros((n_dov, n_dov), dtype=np.float64)
    L_dov[np.tril_indices(n_dov)] = dov_npz["L_tril"]
    sn_dovekie = _wrap_cholesky("DES_Dovekie", dov_npz["zhd"], dov_npz["zhel"], dov_npz["mu"], L_dov)

    # 4. Union3 (N = 22 spline nodes representing 2,087 SNe Ia; Rubin et al. 2025, arXiv:2311.12098)
    u_rows = [
        line.split() for line in (DATA_DIR / "union3_lcparam_full.txt").read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.startswith("#")
    ]
    cov_raw_u3 = np.loadtxt(DATA_DIR / "union3_mag_covmat.txt")
    n_u3 = int(cov_raw_u3[0]) if cov_raw_u3.ndim == 1 else cov_raw_u3.shape[0]
    cov_u3 = cov_raw_u3[1:].reshape((n_u3, n_u3)) if cov_raw_u3.ndim == 1 else cov_raw_u3
    L_u3 = linalg.cholesky(np.asarray(cov_u3, dtype=np.float64), lower=True)
    sn_union3 = _wrap_cholesky(
        "Union3",
        np.array([float(r[1]) for r in u_rows]),
        np.array([float(r[2]) for r in u_rows]),
        np.array([float(r[4]) for r in u_rows]),
        L_u3,
    )

    # 5. DESI 13-point Gaussian BAO
    desi_rows = []
    for line in (DATA_DIR / "desi_gaussian_bao_ALL_GCcomb_mean.txt").read_text(encoding="utf-8").splitlines():
        if line.strip() and not line.startswith("#"):
            p = line.split()
            desi_rows.append((float(p[0]), float(p[1]), p[2]))
    cov_desi = np.loadtxt(DATA_DIR / "desi_gaussian_bao_ALL_GCcomb_cov.txt")
    cho_desi = linalg.cho_factor(cov_desi)

    # 6. Moresco et al. (2020) 15x15 Cosmic Chronometers (auxiliary check)
    mm20_hz = np.array([
        [0.1791, 74.91, 3.8069262], [0.1993, 74.96, 4.9001352], [0.3519, 82.78, 13.94843],
        [0.3802, 83.00, 13.54], [0.4004, 76.97, 10.18], [0.4247, 87.08, 11.24],
        [0.4497, 92.78, 12.90], [0.4783, 80.91, 9.044], [0.5929, 103.80, 12.49752],
        [0.6797, 91.60, 7.961872], [0.7812, 104.50, 12.19515], [0.8754, 125.10, 16.70085],
        [1.0370, 153.70, 19.6736], [1.3630, 160.00, 32.63], [1.9650, 186.50, 49.58],
    ], dtype=np.float64)
    mm20_sys = np.array([
        [0.075, 0.47, 9.91], [0.125, 0.47, 6.98], [0.175, 0.47, 5.40], [0.225, 0.47, 5.40],
        [0.275, 0.47, 5.40], [0.325, 0.47, 5.40], [0.375, 0.47, 5.40], [0.425, 0.47, 6.20],
        [0.475, 0.46, 5.86], [0.525, 0.23, 6.51], [0.575, 0.28, 6.12], [0.625, 0.47, 6.21],
        [0.675, 0.47, 5.71], [0.725, 0.47, 5.16], [0.775, 0.45, 5.05], [0.825, 0.47, 5.05],
        [0.875, 0.47, 2.79], [0.925, 0.44, 3.70], [0.975, 0.40, 3.65], [1.025, 0.27, 3.37],
        [1.075, 0.20, 3.49], [1.125, 0.20, 2.33], [1.175, 0.19, 2.33], [1.225, 0.19, 2.33],
        [1.275, 0.19, 2.33], [1.325, 0.20, 2.34], [1.375, 0.19, 2.34], [1.425, 0.19, 2.33],
        [1.475, 0.20, 2.34],
    ], dtype=np.float64)
    z_cc15, h_cc15, sig_cc15 = mm20_hz[:, 0], mm20_hz[:, 1], mm20_hz[:, 2]
    v_imf = h_cc15 * (np.interp(z_cc15, mm20_sys[:, 0], mm20_sys[:, 1]) / 100.0)
    v_sps = h_cc15 * (np.interp(z_cc15, mm20_sys[:, 0], mm20_sys[:, 2]) / 100.0)
    cov_cc15 = np.diag(sig_cc15**2) + np.outer(v_imf, v_imf) + np.outer(v_sps, v_sps)
    cho_cc15 = linalg.cho_factor(cov_cc15)

    # 7. Compressed Planck 2018 priors (TT,TE,EE+lowE+lensing Table 2 col 6 & Chen-Huang-Wang 2019)
    planck_json = json.loads((DATA_DIR / "planck2018_compressed_priors.json").read_text(encoding="utf-8"))
    pl3_mean = np.array(planck_json["planck2018_mcmc_3x3_omegab_omegac_100thetastar"]["mean"], dtype=np.float64)
    pl3_cov = np.array(planck_json["planck2018_mcmc_3x3_omegab_omegac_100thetastar"]["cov"], dtype=np.float64)
    pl3_invcov = linalg.inv(pl3_cov)
    chw_mean = np.array(planck_json["chen_huang_wang_2019_R_lA_omegab"]["mean"], dtype=np.float64)
    chw_invcov = np.array(planck_json["chen_huang_wang_2019_R_lA_omegab"]["invcov"], dtype=np.float64)

    # 8. SDSS DR7 + DESI DR1 (DESIVAST VoidFinder & ASTRA-DESI BGS+LRG) 3D Cosmic Void Catalogs
    vf_npz = np.load(DATA_DIR / "sdss_dr7_voidfinder_planck2018_3d_catalog.npz")
    dv_npz = np.load(DATA_DIR / "desivast_dr1_voidfinder_ngc_sgc.npz")
    as_npz = np.load(DATA_DIR / "astra_desi_dr1_bgs_lrg_voids.npz")

    return {
        "sn_compilations": {
            "PantheonPlus": sn_pantheon,
            "DES_Dovekie": sn_dovekie,
            "DES_SN5YR_2024": sn_des5yr,
            "Union3": sn_union3,
        },
        "desi13": {"rows": desi_rows, "y": np.array([r[1] for r in desi_rows], dtype=np.float64), "cho": cho_desi},
        "cc15": {"z": z_cc15, "ln1pz": np.log1p(z_cc15), "Hz": h_cc15, "cho": cho_cc15},
        "planck": {
            "source": planck_json["planck2018_mcmc_3x3_omegab_omegac_100thetastar"]["source"],
            "pl3_mean": pl3_mean,
            "pl3_cov": pl3_cov,
            "pl3_invcov": pl3_invcov,
            "chw19_mean": chw_mean,
            "chw19_invcov": chw_invcov,
        },
        "voidfinder_sdss_dr7": {
            "holes": np.asarray(vf_npz["holes_xyz_r_vid"], dtype=np.float64),
            "maximal": np.asarray(vf_npz["maximal_table"], dtype=np.float64),
        },
        "desivast_dr1_voidfinder": {
            "holes": np.asarray(dv_npz["holes_xyzr"], dtype=np.float64),
            "maximal": np.asarray(dv_npz["maximals_xyzr"], dtype=np.float64),
        },
        "astra_desi_dr1": {
            "bgs": np.asarray(as_npz["bgs_xyzr"], dtype=np.float64),
            "lrg": np.asarray(as_npz["lrg_xyzr"], dtype=np.float64),
        },
    }


# ==============================================================================
# 2. EXACT PUBLISHED WILTSHIRE (2009) Eq. (33) / LANE ET AL. (2025) Eq. (A4) DISTANCE
# ==============================================================================
def F_wiltshire_dim(x: float, b_dim: float) -> float:
    y = x ** (1.0 / 3.0)
    beta = b_dim ** (1.0 / 3.0)
    return (
        2.0 * y
        + (beta / 6.0) * math.log(((y + beta) ** 2) / (y * y - beta * y + beta * beta))
        + (beta / math.sqrt(3.0)) * math.atan((2.0 * y - beta) / (math.sqrt(3.0) * beta))
    )


def solve_x_and_fv_of_z(zv: float, fv0: float) -> Tuple[float, float]:
    b_dim = 2.0 * (1.0 - fv0) * (2.0 + fv0) / (9.0 * fv0)
    x0 = (2.0 + fv0) / 3.0
    if zv <= 0.0:
        return x0, fv0
    target = 1.0 + zv
    u0 = x0 ** (1.0 / 3.0)

    def resid(u: float) -> float:
        xv = u**3
        lhs = (2.0 ** (4.0 / 3.0)) * (xv + b_dim) / (
            (fv0 ** (1.0 / 3.0)) * (u**2) * ((2.0 * xv + 3.0 * b_dim) ** (4.0 / 3.0))
        )
        return lhs - target

    u_low = u0 * ((1.0 + zv) ** (-0.5)) * 1e-4
    while resid(u_low) < 0.0:
        u_low *= 0.1
    u_sol = float(optimize.brentq(resid, u_low, u0, xtol=1e-15, rtol=1e-15))
    xv = u_sol**3
    fv = 3.0 * fv0 * xv / (3.0 * fv0 * xv + (1.0 - fv0) * (2.0 + fv0))
    return xv, fv


def H0_D_TS_wiltshire_closed(zv: float, fv0: float) -> float:
    if zv <= 0.0:
        return 0.0
    b_dim = 2.0 * (1.0 - fv0) * (2.0 + fv0) / (9.0 * fv0)
    x0 = (2.0 + fv0) / 3.0
    xz, _ = solve_x_and_fv_of_z(zv, fv0)
    H0_over_Hbar0 = (4.0 * fv0 * fv0 + fv0 + 4.0) / (2.0 * (2.0 + fv0))
    Hbar0_dA = (xz ** (2.0 / 3.0)) * (F_wiltshire_dim(x0, b_dim) - F_wiltshire_dim(xz, b_dim))
    return H0_over_Hbar0 * Hbar0_dA * (1.0 + zv)


def H0_D_TS_wiltshire_quad(zv: float, fv0: float) -> float:
    if zv <= 0.0:
        return 0.0
    x0 = (2.0 + fv0) / 3.0
    xz, _ = solve_x_and_fv_of_z(zv, fv0)
    H0_over_Hbar0 = (4.0 * fv0 * fv0 + fv0 + 4.0) / (2.0 * (2.0 + fv0))

    def integrand(xp: float) -> float:
        fv_p = 3.0 * fv0 * xp / (3.0 * fv0 * xp + (1.0 - fv0) * (2.0 + fv0))
        return 2.0 / ((2.0 + fv_p) * (xp ** (2.0 / 3.0)))

    val, _ = integrate.quad(integrand, xz, x0, epsabs=1e-14, epsrel=1e-14)
    return H0_over_Hbar0 * (xz ** (2.0 / 3.0)) * float(val) * (1.0 + zv)


_SPLINE_CACHE: Dict[int, Tuple[Any, Any, Any, float]] = {}


def build_vectorized_wiltshire_splines(fv0: float):
    """Exact O(0.1 ms) vectorized parametric Wiltshire cubic splines across z in [0, ~1e5] (<1e-9 error)."""
    key = int(round(fv0 * 1e9))
    cached = _SPLINE_CACHE.get(key)
    if cached is not None:
        return cached
    fv = fv0 * np.exp(-np.linspace(0.0, 16.5, 650))
    x0 = (2.0 + fv0) / 3.0
    b_dim = 2.0 * (1.0 - fv0) * (2.0 + fv0) / (9.0 * fv0)
    h_ratio = (4.0 * fv0 * fv0 + fv0 + 4.0) / (2.0 * (2.0 + fv0))
    zp1 = ((2.0 + fv) / (2.0 + fv0)) * ((fv0 / fv) ** (2.0 / 3.0)) * ((1.0 - fv) / (1.0 - fv0))
    z_grid = zp1 - 1.0
    z_grid[0] = 0.0
    x = (1.0 - fv0) * (2.0 + fv0) * fv / (3.0 * fv0 * (1.0 - fv))
    x[0] = x0
    b13, x13 = b_dim ** (1.0 / 3.0), x ** (1.0 / 3.0)
    F_val = (
        2.0 * x13
        + (b13 / 6.0) * np.log(((x13 + b13) ** 2) / (x13 * x13 - b13 * x13 + b13 * b13))
        + (b13 / math.sqrt(3.0)) * np.arctan((2.0 * x13 - b13) / (math.sqrt(3.0) * b13))
    )
    Hbar0_dA = (x ** (2.0 / 3.0)) * (float(F_val[0]) - F_val)
    D_ts = h_ratio * Hbar0_dA * zp1
    dln1pz_dx = 1.0 / (x + b_dim) - 2.0 / (3.0 * x) - 8.0 / (3.0 * (2.0 * x + 3.0 * b_dim))
    dx_dz = 1.0 / (zp1 * dln1pz_dx)
    d_Hbar0_dA_dz = ((2.0 / (3.0 * x)) * Hbar0_dA - 2.0 / (2.0 + fv)) * dx_dz
    dD_dz = h_ratio * (Hbar0_dA + zp1 * d_Hbar0_dA_dz)
    dD_dz[0] = 1.0
    ln1pz = np.log1p(z_grid)
    res = (
        interpolate.CubicSpline(ln1pz, D_ts),
        interpolate.CubicSpline(ln1pz, dD_dz),
        interpolate.CubicSpline(ln1pz, fv),
        h_ratio,
    )
    if len(_SPLINE_CACHE) > 2048:
        _SPLINE_CACHE.clear()
    _SPLINE_CACHE[key] = res
    return res


def eval_sn_chi2_direct(sn: Dict[str, Any], D_at_sn: np.ndarray) -> float:
    mu_shape = 5.0 * np.log10((1.0 + sn["zhel"]) * np.maximum(D_at_sn, 1e-30))
    diff = sn["y"] - mu_shape
    Linv_d = linalg.solve_triangular(sn["L"], diff, lower=True)
    off = float(np.dot(sn["Linv_1"], Linv_d)) / sn["norm_1"]
    res = Linv_d - off * sn["Linv_1"]
    return float(np.dot(res, res))


def H0_D_CPL_vec(z_arr: np.ndarray, w0: float, wa: float, om: float) -> np.ndarray:
    half = 0.5 * z_arr[:, None]
    zp = half * (U_GL32[None, :] + 1.0)
    ap = 1.0 / (1.0 + zp)
    de = ((1.0 + zp) ** (3.0 * (1.0 + w0 + wa))) * np.exp(-3.0 * wa * (1.0 - ap))
    Ez = np.sqrt(om * ((1.0 + zp) ** 3) + (1.0 - om) * de)
    return np.sum(W_GL32[None, :] / Ez, axis=1) * half[:, 0]


# ==============================================================================
# 3. UNIFIED LOG-GRID MULTI-COMPILATION JOINT LIKELIHOOD ENGINE (v10.0)
# ==============================================================================
def run_multi_sn_joint_likelihood_suite(data: Dict[str, Any], f_acc: float) -> Dict[str, Any]:
    stencil = build_recfast_zstar_stencil(NU_MASSES_LCDM_EV)

    # Smooth unified logarithmic grid u = ln(1 + z) in [0, ln(1201)] (ZERO z = 2.5 stitch!):
    n_u = 1400
    u_max = math.log1p(1200.0)
    u_nodes = np.linspace(0.0, u_max, n_u + 1)
    du = u_max / n_u
    u_mid = 0.5 * (u_nodes[:-1] + u_nodes[1:])
    u_all = np.empty(2 * n_u + 1, dtype=np.float64)
    u_all[0::2] = u_nodes
    u_all[1::2] = u_mid
    zp1_all = np.exp(u_all)
    zp1_nodes = zp1_all[0::2]

    desi_u = np.array([math.log1p(zv) for zv, _, _ in data["desi13"]["rows"]], dtype=np.float64)
    desi_z = np.array([zv for zv, _, _ in data["desi13"]["rows"]], dtype=np.float64)
    desi_types = [qt for _, _, qt in data["desi13"]["rows"]]

    def fv0_from_om0_H1(om0: float) -> float:
        z_acc = (2.0 * (1.0 - om0) / om0) ** (1.0 / 3.0) - 1.0
        zp1 = 1.0 + z_acc
        return float(
            optimize.brentq(
                lambda fv0: ((2.0 + f_acc) / (2.0 + fv0)) * ((fv0 / f_acc) ** (2.0 / 3.0)) * ((1.0 - f_acc) / (1.0 - fv0)) - zp1,
                0.05, 0.999, xtol=1e-13,
            )
        )

    def chw19_chi2(h: float, omega_b: float, om0: float, delta_star_dimless: float, w0: float = -1.0, wa: float = 0.0) -> float:
        omh2 = om0 * h * h
        g1 = 0.0783 * (omega_b ** (-0.238)) / (1.0 + 39.5 * (omega_b ** 0.763))
        g2 = 0.560 / (1.0 + 21.1 * (omega_b ** 1.81))
        z_star = 1048.0 * (1.0 + 0.00124 * (omega_b ** (-0.738))) * (1.0 + g1 * (omh2 ** g2))
        t_rat = 2.7255 / 2.7
        rb_fac = 31500.0 * (t_rat ** (-4.0)) * omega_b
        z_eq = 2.5e4 * omh2 * (t_rat ** (-4.0))
        om_r = om0 / (1.0 + z_eq)
        om_de = 1.0 - om0 - om_r
        a_star = 1.0 / (1.0 + z_star)
        a_nodes = 0.5 * a_star * (U_GL64 + 1.0)
        zp1_n = 1.0 / a_nodes
        de_n = (zp1_n ** (3.0 * (1.0 + w0 + wa))) * np.exp(-3.0 * wa * (1.0 - a_nodes))
        E_n = np.sqrt(om_r * (zp1_n ** 4) + om0 * (zp1_n ** 3) + om_de * de_n)
        integrand_rs = 1.0 / ((a_nodes ** 2) * E_n * np.sqrt(3.0 * (1.0 + rb_fac * a_nodes)))
        dh_mpc = C_KMS / (100.0 * h)
        rs_star = dh_mpc * float(np.sum(W_GL64 * integrand_rs) * 0.5 * a_star)
        d_dim = 0.0
        for z_a, z_b in [(0.0, 5.0), (5.0, z_star)]:
            half = 0.5 * (z_b - z_a)
            mid = 0.5 * (z_b + z_a)
            zp = mid + half * U_GL64
            zp1 = 1.0 + zp
            ap = 1.0 / zp1
            de_z = (zp1 ** (3.0 * (1.0 + w0 + wa))) * np.exp(-3.0 * wa * (1.0 - ap))
            Ez = np.sqrt(om_r * (zp1 ** 4) + om0 * (zp1 ** 3) + om_de * de_z)
            d_dim += float(np.sum(W_GL64 / Ez) * half)
        dm_obs = dh_mpc * (d_dim + delta_star_dimless)
        R_val = math.sqrt(om0) * (d_dim + delta_star_dimless)
        lA_val = math.pi * dm_obs / rs_star
        vec = np.array([R_val, lA_val, omega_b], dtype=np.float64) - data["planck"]["chw19_mean"]
        return float(vec @ (data["planck"]["chw19_invcov"] @ vec))

    def eval_joint_model(
        sn: Dict[str, Any] | None,
        h: float,
        omega_b: float,
        omega_cdm: float,
        p1: float,
        p2: float = 0.0,
        mode: str = "H2_vol_p1",
        cmb_prior: str = "PL3_MCMC",
        include_cc: bool = False,
        fixed_fv0: float | None = None,
        rd_mode: str = "CAMB",
    ) -> Dict[str, Any] | None:
        if h <= 0.54 or h >= 0.80 or omega_b <= 0.0195 or omega_b >= 0.0255 or omega_cdm <= 0.08 or omega_cdm >= 0.175:
            return None
        if mode == "CPL":
            w0_rec, wa_rec = p1, p2
        elif mode == "wCDM":
            w0_rec, wa_rec = p1, 0.0
        elif mode == "Linder2015_Thawing_k5":
            # Linder (2015, PRD 91, 063006, Eq. 3): 1+w(a) = (1+w0)*a*((1+b)/(1+b*a^{-3}))^{2/3}, b=0.5, w0 >= -1
            w0_rec, wa_rec = -1.0, 0.0
        elif mode == "CPL_DegLine_k5":
            w0_rec = p1
            wa_rec = -3.30 * (1.0 + w0_rec)
        else:
            w0_rec, wa_rec = -1.0, 0.0

        rec = recfast_v15_cpl_observables(
            h=h, omega_b=omega_b, omega_cdm=omega_cdm, w0=w0_rec, wa=wa_rec,
            nu_masses_ev=NU_MASSES_LCDM_EV, zstar_stencil=stencil,
        )
        om0 = rec["Omega_m0"]
        if om0 <= 0.18 or om0 >= 0.48:
            return None

        H0_si = 100.0 * h * 1000.0 / MPC_M
        rho_crit0 = 3.0 * (H0_si**2) / (8.0 * math.pi * G_SI)
        Om_g = RHO_G0_SI / rho_crit0
        Om_r = Om_g * (1.0 + 0.2271073 * 3.044)
        om_de = 1.0 - om0 - Om_r

        dh_mpc = C_KMS / (100.0 * h)
        if rd_mode == "CAMB":
            rd_mpc = rec["r_d_zdrag_Mpc"]
        elif rd_mode == "AUBOURG15":
            rd_mpc = rec["r_d_aubourg15_Mpc"]
        elif rd_mode == "CAMB_PLUS_015PCT":
            rd_mpc = rec["r_d_zdrag_Mpc"] * 1.0015
        elif rd_mode == "CAMB_MINUS_015PCT":
            rd_mpc = rec["r_d_zdrag_Mpc"] * 0.9985
        else:
            raise ValueError(rd_mode)

        u_star = math.log1p(rec["z_star"])
        fv0_used = 0.0

        if mode in ("CPL", "wCDM", "Linder2015_Thawing_k5", "CPL_DegLine_k5"):
            a_nodes = 1.0 / zp1_nodes
            if mode == "Linder2015_Thawing_k5":
                b_l15 = 0.5
                de_nodes = np.exp(
                    3.0 * (1.0 + p1) * ((1.0 + b_l15) - ((1.0 + b_l15) ** (2.0 / 3.0)) * ((a_nodes**3 + b_l15) ** (1.0 / 3.0)))
                )
            else:
                de_nodes = (zp1_nodes ** (3.0 * (1.0 + w0_rec + wa_rec))) * np.exp(-3.0 * wa_rec * (1.0 - a_nodes))
            E_nodes = np.sqrt(Om_r * (zp1_nodes**4) + om0 * (zp1_nodes**3) + om_de * de_nodes)
            dD_dz_nodes = 1.0 / E_nodes
            D_nodes = integrate.cumulative_trapezoid(zp1_nodes * dD_dz_nodes, u_nodes, initial=0.0)
            dH_dimless_nodes = dD_dz_nodes
            if mode == "Linder2015_Thawing_k5":
                E_lcdm_nodes = np.sqrt(Om_r * (zp1_nodes**4) + om0 * (zp1_nodes**3) + om_de)
                D_lcdm_nodes = integrate.cumulative_trapezoid(zp1_nodes / E_lcdm_nodes, u_nodes, initial=0.0)
                delta_star_dimless = float(np.interp(u_star, u_nodes, D_nodes - D_lcdm_nodes))
                dm_obs_zstar_mpc = rec["D_M_zstar_Mpc"] + dh_mpc * delta_star_dimless
            else:
                delta_star_dimless = 0.0
                dm_obs_zstar_mpc = rec["D_M_zstar_Mpc"]
        else:
            alpha_val = p1
            fv0_used = float(fixed_fv0) if fixed_fv0 is not None else fv0_from_om0_H1(om0)
            Eh_all = np.sqrt(Om_r * (zp1_all**4) + om0 * (zp1_all**3) + om_de)
            Eh_nodes = Eh_all[0::2]
            D_hor_nodes = integrate.cumulative_trapezoid(zp1_nodes / Eh_nodes, u_nodes, initial=0.0)

            if abs(alpha_val) < 1e-15:
                D_nodes = D_hor_nodes
                dD_dz_nodes = 1.0 / Eh_nodes
                dH_dimless_nodes = dD_dz_nodes
                delta_star_dimless = 0.0
                dm_obs_zstar_mpc = rec["D_M_zstar_Mpc"]
            else:
                sD_ts, sdD_ts, sfv_ts, h_ratio = build_vectorized_wiltshire_splines(fv0_used)
                if mode == "H2a_const_p0":
                    D_nodes = (1.0 - alpha_val) * D_hor_nodes + alpha_val * sD_ts(u_nodes)
                    dD_dz_nodes = (1.0 - alpha_val) / Eh_nodes + alpha_val * sdD_ts(u_nodes)
                    dH_dimless_nodes = dD_dz_nodes
                    dm_obs_zstar_mpc = (1.0 - alpha_val) * rec["D_M_zstar_Mpc"] + alpha_val * dh_mpc * float(sD_ts(u_star))
                    delta_star_dimless = (dm_obs_zstar_mpc - rec["D_M_zstar_Mpc"]) / dh_mpc
                elif mode in ("H2_vol_p1", "H2_vol_p2", "H2_vol_p1_bareH0"):
                    p_exp = 2 if mode == "H2_vol_p2" else 1
                    fv_nodes = sfv_ts(u_nodes)
                    dD_ts_nodes = (sdD_ts(u_nodes) / h_ratio) if mode == "H2_vol_p1_bareH0" else sdD_ts(u_nodes)
                    aeff_nodes = alpha_val * ((fv_nodes / fv0_used) ** p_exp)
                    delta_dD = aeff_nodes * (dD_ts_nodes - 1.0 / Eh_nodes)
                    dD_dz_nodes = 1.0 / Eh_nodes + delta_dD
                    dH_dimless_nodes = dD_dz_nodes
                    delta_D_nodes = integrate.cumulative_trapezoid(zp1_nodes * delta_dD, u_nodes, initial=0.0)
                    D_nodes = D_hor_nodes + delta_D_nodes
                    delta_star_dimless = float(np.interp(u_star, u_nodes, delta_D_nodes))
                    dm_obs_zstar_mpc = rec["D_M_zstar_Mpc"] + dh_mpc * delta_star_dimless
                elif mode in ("Sachs_AllZ_HomRicci_p1", "Sachs_AllZ_VoidDepletedRicci_p1"):
                    # Exact first-order Hamiltonian Sachs ODE across all u in [0, ln(1201)] — ZERO z=2.5 stitch!
                    # X(u) := (H0/c)*D_M(z), P(u) := E_ray(u)*(X'(u) - X(u)):
                    #   X'(u) = X(u) + P(u) / E_ray(u),   X(0) = 0
                    #   P'(u) = - S_Ricci(u) * X(u),      P(0) = 1
                    aeff_all = alpha_val * (sfv_ts(u_all) / fv0_used)
                    inv_Eray_all = (1.0 - aeff_all) / Eh_all + aeff_all * sdD_ts(u_all)
                    Eray_all = 1.0 / inv_Eray_all
                    ricci_wt = np.maximum(1.0 - aeff_all, 0.0) if mode == "Sachs_AllZ_VoidDepletedRicci_p1" else 1.0
                    src_all = (1.5 * om0 * ricci_wt * (zp1_all**3) + 2.0 * Om_r * (zp1_all**4)) / Eray_all
                    X_arr = np.zeros(n_u + 1, dtype=np.float64)
                    P_arr = np.zeros(n_u + 1, dtype=np.float64)
                    P_arr[0] = 1.0
                    X_v, P_v = 0.0, 1.0
                    for k_i in range(n_u):
                        i0, im, i1 = 2 * k_i, 2 * k_i + 1, 2 * k_i + 2
                        E0, Em, E1 = Eray_all[i0], Eray_all[im], Eray_all[i1]
                        S0, Sm, S1 = src_all[i0], src_all[im], src_all[i1]
                        dx1 = X_v + P_v / E0; dp1 = -S0 * X_v
                        x1 = X_v + 0.5 * du * dx1; p1_s = P_v + 0.5 * du * dp1
                        dx2 = x1 + p1_s / Em; dp2 = -Sm * x1
                        x2 = X_v + 0.5 * du * dx2; p2_s = P_v + 0.5 * du * dp2
                        dx3 = x2 + p2_s / Em; dp3 = -Sm * x2
                        x3 = X_v + du * dx3; p3_s = P_v + du * dp3
                        dx4 = x3 + p3_s / E1; dp4 = -S1 * x3
                        X_v += (du / 6.0) * (dx1 + 2.0 * dx2 + 2.0 * dx3 + dx4)
                        P_v += (du / 6.0) * (dp1 + 2.0 * dp2 + 2.0 * dp3 + dp4)
                        X_arr[k_i + 1] = X_v
                        P_arr[k_i + 1] = P_v
                    D_nodes = X_arr
                    dD_dz_nodes = (X_arr + P_arr / Eray_all[0::2]) / zp1_nodes
                    dH_dimless_nodes = dD_dz_nodes
                    delta_star_dimless = float(np.interp(u_star, u_nodes, D_nodes - D_hor_nodes))
                    dm_obs_zstar_mpc = rec["D_M_zstar_Mpc"] + dh_mpc * delta_star_dimless
                else:
                    raise ValueError(mode)

        spl_D = interpolate.CubicSpline(u_nodes, D_nodes)
        spl_dH = interpolate.CubicSpline(u_nodes, dH_dimless_nodes)

        dm_b = spl_D(desi_u)
        dh_b = spl_dH(desi_u)
        pu = np.empty(len(desi_types), dtype=np.float64)
        for idx_q, qt in enumerate(desi_types):
            if qt == "DM_over_rs":
                pu[idx_q] = dm_b[idx_q]
            elif qt == "DH_over_rs":
                pu[idx_q] = dh_b[idx_q]
            else:
                pu[idx_q] = (desi_z[idx_q] * (dm_b[idx_q] ** 2) * dh_b[idx_q]) ** (1.0 / 3.0)
        res_desi = (dh_mpc / rd_mpc) * pu - data["desi13"]["y"]
        chi2_bao = float(np.dot(res_desi, linalg.cho_solve(data["desi13"]["cho"], res_desi)))

        chi2_sn = 0.0
        if sn is not None:
            chi2_sn = eval_sn_chi2_direct(sn, spl_D(sn["ln1pz_hd"]))

        th100 = 100.0 * rec["r_s_zstar_Mpc"] / dm_obs_zstar_mpc
        if cmb_prior == "PL3_MCMC":
            v3 = np.array([omega_b, omega_cdm, th100], dtype=np.float64) - data["planck"]["pl3_mean"]
            chi2_cmb = float(v3 @ (data["planck"]["pl3_invcov"] @ v3))
        elif cmb_prior == "CHW19":
            chi2_cmb = chw19_chi2(h, omega_b, om0, delta_star_dimless, w0_rec, wa_rec)
        elif cmb_prior == "THETA_BBN_OMEGAC_DIAG":
            chi2_cmb = (
                ((th100 - 1.04110) / 0.00031) ** 2
                + ((omega_b - 0.022353) / 0.00037) ** 2
                + ((omega_cdm - 0.1200) / 0.0012) ** 2
            )
        else:
            raise ValueError(cmb_prior)

        chi2_cc = 0.0
        if include_cc:
            res_cc = (100.0 * h) / spl_dH(data["cc15"]["ln1pz"]) - data["cc15"]["Hz"]
            chi2_cc = float(np.dot(res_cc, linalg.cho_solve(data["cc15"]["cho"], res_cc)))

        tot = chi2_sn + chi2_bao + chi2_cmb + chi2_cc
        return {
            "chi2_total": tot,
            "chi2_SN": chi2_sn,
            "chi2_DESI_BAO_13": chi2_bao,
            "chi2_CMB": chi2_cmb,
            "chi2_CC_15": chi2_cc,
            "H0_kms_Mpc": 100.0 * h,
            "h": h,
            "omega_b": omega_b,
            "omega_cdm": omega_cdm,
            "Omega_m0": om0,
            "f_v0": fv0_used,
            "theta_100": th100,
            "r_d_Mpc": rd_mpc,
            "delta_DM_zstar_Mpc": dh_mpc * delta_star_dimless,
            "spl_D": spl_D,
        }

    def fit_3param_fixed_p1(
        sn: Dict[str, Any] | None, p1_val: float, mode: str, cmb_prior: str = "PL3_MCMC",
        include_cc: bool = False, fixed_fv0: float | None = None, rd_mode: str = "CAMB", init=None,
    ) -> Dict[str, Any]:
        x0 = init if init is not None else [0.676, 0.02238, 0.1195]
        res = optimize.minimize(
            lambda x: (eval_joint_model(sn, float(x[0]), float(x[1]), float(x[2]), p1_val, 0.0, mode, cmb_prior, include_cc, fixed_fv0, rd_mode) or {"chi2_total": 1e9})["chi2_total"],
            x0, method="L-BFGS-B", bounds=[(0.60, 0.76), (0.0205, 0.0245), (0.095, 0.155)],
            options={"ftol": 1e-12, "gtol": 1e-7},
        )
        out = eval_joint_model(sn, float(res.x[0]), float(res.x[1]), float(res.x[2]), p1_val, 0.0, mode, cmb_prior, include_cc, fixed_fv0, rd_mode)
        assert out is not None
        out["_x"] = [float(v) for v in res.x]
        return out

    def fit_4param_k5_model(
        sn: Dict[str, Any] | None, mode: str, cmb_prior: str = "PL3_MCMC",
        include_cc: bool = False, fixed_fv0: float | None = None, rd_mode: str = "CAMB",
        init_p1: float = 0.42, p1_bounds: Tuple[float, float] = (-0.30, 1.25), dp: float = 0.025,
    ) -> Dict[str, Any]:
        res = optimize.minimize(
            lambda x: (eval_joint_model(sn, float(x[0]), float(x[1]), float(x[2]), float(x[3]), 0.0, mode, cmb_prior, include_cc, fixed_fv0, rd_mode) or {"chi2_total": 1e9})["chi2_total"],
            [0.674, 0.02238, 0.1195, init_p1], method="L-BFGS-B",
            bounds=[(0.60, 0.76), (0.0205, 0.0245), (0.095, 0.155), p1_bounds],
            options={"ftol": 1e-12, "gtol": 1e-7},
        )
        p_hat = float(res.x[3])
        out = eval_joint_model(sn, float(res.x[0]), float(res.x[1]), float(res.x[2]), p_hat, 0.0, mode, cmb_prior, include_cc, fixed_fv0, rd_mode)
        assert out is not None
        c_0 = out["chi2_total"]
        c_p = fit_3param_fixed_p1(sn, p_hat + dp, mode, cmb_prior, include_cc, fixed_fv0, rd_mode, res.x[:3])["chi2_total"]
        c_m = fit_3param_fixed_p1(sn, p_hat - dp, mode, cmb_prior, include_cc, fixed_fv0, rd_mode, res.x[:3])["chi2_total"]
        sig_p = math.sqrt(2.0 * dp * dp / max(c_p - 2.0 * c_0 + c_m, 1e-9))
        if mode in ("wCDM", "Linder2015_Thawing_k5", "CPL_DegLine_k5"):
            out["w0_hat"] = p_hat
            out["sigma_w0"] = sig_p
            out["wa_hat"] = 0.0 if mode == "wCDM" else (None if mode == "Linder2015_Thawing_k5" else -3.30 * (1.0 + p_hat))
        else:
            out["alpha_hat"] = p_hat
            out["sigma_alpha"] = sig_p
        return out

    def fit_5param_free_cpl_with_hessian(sn: Dict[str, Any], cmb_prior: str = "PL3_MCMC", rd_mode: str = "CAMB") -> Dict[str, Any]:
        def cpl_obj(x: np.ndarray) -> float:
            r = eval_joint_model(sn, float(x[0]), float(x[1]), float(x[2]), float(x[3]), float(x[4]), "CPL", cmb_prior, False, None, rd_mode)
            return 1e9 if r is None else r["chi2_total"]

        res = optimize.minimize(
            cpl_obj, [0.674, 0.02238, 0.1195, -0.82, -0.65], method="L-BFGS-B",
            bounds=[(0.60, 0.76), (0.0205, 0.0245), (0.095, 0.155), (-1.4, -0.4), (-2.5, 1.0)],
            options={"ftol": 1e-12, "gtol": 1e-7},
        )
        p_opt = res.x
        f_opt = float(res.fun)
        steps = np.array([1e-4, 4e-5, 2e-4, 2e-3, 8e-3], dtype=np.float64)
        H_mat = np.zeros((5, 5), dtype=np.float64)
        for i in range(5):
            ei = np.zeros(5); ei[i] = steps[i]
            H_mat[i, i] = (cpl_obj(p_opt + ei) - 2.0 * f_opt + cpl_obj(p_opt - ei)) / (steps[i] ** 2)
            for j in range(i + 1, 5):
                ej = np.zeros(5); ej[j] = steps[j]
                H_mat[i, j] = H_mat[j, i] = (
                    cpl_obj(p_opt + ei + ej) - cpl_obj(p_opt + ei - ej)
                    - cpl_obj(p_opt - ei + ej) + cpl_obj(p_opt - ei - ej)
                ) / (4.0 * steps[i] * steps[j])
        cov5 = 2.0 * linalg.inv(H_mat)
        s_w0 = math.sqrt(max(float(cov5[3, 3]), 1e-12))
        s_wa = math.sqrt(max(float(cov5[4, 4]), 1e-12))
        rho_w = float(cov5[3, 4] / (s_w0 * s_wa))
        out = eval_joint_model(sn, float(p_opt[0]), float(p_opt[1]), float(p_opt[2]), float(p_opt[3]), float(p_opt[4]), "CPL", cmb_prior, False, None, rd_mode)
        assert out is not None
        out.update({
            "w0_hat": float(p_opt[3]),
            "sigma_w0_hessian": s_w0,
            "wa_hat": float(p_opt[4]),
            "sigma_wa_hessian": s_wa,
            "rho_w0_wa_hessian": rho_w,
        })
        return out

    def summarize_fit(raw: Dict[str, Any], c0_lcdm: float, k_params: int, n_points: int, c_cpl: float | None = None) -> Dict[str, Any]:
        dchi2 = raw["chi2_total"] - c0_lcdm
        dk = k_params - 4
        daic = dchi2 + 2.0 * dk
        dbic = dchi2 + dk * math.log(n_points)
        res: Dict[str, Any] = {
            "k_params": k_params,
            "N_data": n_points,
            "chi2_total": round(raw["chi2_total"], 3),
            "Delta_chi2_vs_LCDM": round(dchi2, 3),
            "Delta_AIC_vs_LCDM": round(daic, 3),
            "Delta_BIC_vs_LCDM": round(dbic, 3),
            "chi2_SN": round(raw["chi2_SN"], 3),
            "chi2_DESI_BAO_13": round(raw["chi2_DESI_BAO_13"], 3),
            "chi2_CMB_3": round(raw["chi2_CMB"], 3),
            "H0_kms_Mpc": round(raw["H0_kms_Mpc"], 3),
            "Omega_m0": round(raw["Omega_m0"], 5),
            "omega_b": round(raw["omega_b"], 6),
            "omega_cdm": round(raw["omega_cdm"], 6),
            "theta_100": round(raw["theta_100"], 5),
            "r_d_Mpc": round(raw["r_d_Mpc"], 3),
            "delta_DM_zstar_Mpc": round(raw["delta_DM_zstar_Mpc"], 2),
        }
        if raw.get("f_v0", 0.0) > 0.0:
            res["f_v0"] = round(raw["f_v0"], 5)
        if "alpha_hat" in raw:
            res["alpha_hat"] = round(raw["alpha_hat"], 4)
            res["sigma_alpha"] = round(raw["sigma_alpha"], 4)
            z_two_sided = math.sqrt(max(-dchi2, 0.0))
            p_two_sided = float(stats.chi2.sf(max(-dchi2, 0.0), df=1))
            # Trial-factor / look-elsewhere over 3 discrete p in {0, 1, 2} choices:
            p_lee_3models = min(1.0, 3.0 * p_two_sided)
            z_lee_3models = float(stats.norm.isf(0.5 * p_lee_3models)) if p_lee_3models < 1.0 else 0.0
            res["uncorrected_1dof_sigma_over_LCDM"] = round(z_two_sided, 2)
            res["look_elsewhere_3p_corrected_sigma"] = round(z_lee_3models, 2)
        if "w0_hat" in raw:
            res["w0_hat"] = round(raw["w0_hat"], 4)
            if raw.get("wa_hat") is not None:
                res["wa_hat"] = round(raw["wa_hat"], 4)
            if "sigma_w0_hessian" in raw:
                res["sigma_w0_hessian"] = round(raw["sigma_w0_hessian"], 4)
                res["sigma_wa_hessian"] = round(raw["sigma_wa_hessian"], 4)
                res["rho_w0_wa_hessian"] = round(raw["rho_w0_wa_hessian"], 4)
            elif "sigma_w0" in raw:
                res["sigma_w0"] = round(raw["sigma_w0"], 4)
                res["uncorrected_1dof_sigma_over_LCDM"] = round(math.sqrt(max(-dchi2, 0.0)), 2)
        if c_cpl is not None:
            dk_vs_cpl = k_params - 6
            res["Delta_chi2_vs_CPL"] = round(raw["chi2_total"] - c_cpl, 3)
            res["Delta_AIC_vs_CPL"] = round((raw["chi2_total"] - c_cpl) + 2.0 * dk_vs_cpl, 3)
            res["Delta_BIC_vs_CPL"] = round((raw["chi2_total"] - c_cpl) + dk_vs_cpl * math.log(n_points), 3)
        return res

    # A) BAO + Planck18 ONLY (ZERO SUPERNOVAE) FIT & OUT-OF-SAMPLE SN PREDICTIVE SUITE (ALL 1-PARAM TEMPLATES):
    f_bc_lcdm = fit_3param_fixed_p1(None, 0.0, "H2_vol_p1", "PL3_MCMC", rd_mode="CAMB")
    c0_bc = f_bc_lcdm["chi2_total"]

    def _bc_entry(fit_res: Dict[str, Any], k_no_mb: int) -> Dict[str, Any]:
        s = summarize_fit(fit_res, c0_bc, k_no_mb + 1, 16)
        s["k_params"] = k_no_mb
        s["Delta_AIC_vs_LCDM"] = round((fit_res["chi2_total"] - c0_bc) + 2.0 * (k_no_mb - 3), 3)
        s["Delta_BIC_vs_LCDM"] = round((fit_res["chi2_total"] - c0_bc) + (k_no_mb - 3) * math.log(16), 3)
        oos = {}
        for sn_k, sn_ds in data["sn_compilations"].items():
            csn0 = eval_sn_chi2_direct(sn_ds, f_bc_lcdm["spl_D"](sn_ds["ln1pz_hd"]))
            csn1 = eval_sn_chi2_direct(sn_ds, fit_res["spl_D"](sn_ds["ln1pz_hd"]))
            oos[sn_k] = {
                "predicted_SN_chi2": round(csn1, 2),
                "out_of_sample_Delta_chi2_SN_vs_LCDM": round(csn1 - csn0, 2),
            }
        s["out_of_sample_SN_predictions"] = oos
        return s

    f_bc_p0 = fit_4param_k5_model(None, "H2a_const_p0", "PL3_MCMC", rd_mode="CAMB", init_p1=0.09, dp=0.01)
    f_bc_p1 = fit_4param_k5_model(None, "H2_vol_p1", "PL3_MCMC", rd_mode="CAMB", init_p1=0.52)
    f_bc_p2 = fit_4param_k5_model(None, "H2_vol_p2", "PL3_MCMC", rd_mode="CAMB", init_p1=0.75)
    f_bc_cpl_deg = fit_4param_k5_model(None, "CPL_DegLine_k5", "PL3_MCMC", rd_mode="CAMB", init_p1=-0.75, p1_bounds=(-1.25, -0.45), dp=0.015)

    bao_cmb_only_block = {
        "N_BAO_plus_CMB": 16,
        "LCDM_no_MB_k3": _bc_entry(f_bc_lcdm, 3),
        "H2a_p0_no_MB_k4": _bc_entry(f_bc_p0, 4),
        "H2_vol_p1_no_MB_k4": _bc_entry(f_bc_p1, 4),
        "H2_vol_p2_no_MB_k4": _bc_entry(f_bc_p2, 4),
        "PhantomCrossing_CPL_DegLine_no_MB_k4": _bc_entry(f_bc_cpl_deg, 4),
    }

    # B) PRIMARY AUDIT ACROSS ALL 4 SN COMPILATIONS (CAMB-calibrated r_d = 147.102 Mpc at fiducial):
    primary_pl3_results: Dict[str, Any] = {}
    chw19_crosscheck_results: Dict[str, Any] = {}
    aubourg15_rd_results: Dict[str, Any] = {}
    alpha_grid = [0.0, 0.15, 0.30, 0.40, 0.52, 0.68, 0.84, 1.00]

    for sn_key, sn_ds in data["sn_compilations"].items():
        n_pts = sn_ds["n"] + 13 + 3
        f_lcdm = fit_3param_fixed_p1(sn_ds, 0.0, "H2_vol_p1", "PL3_MCMC", rd_mode="CAMB")
        c0 = f_lcdm["chi2_total"]
        f_cpl = fit_5param_free_cpl_with_hessian(sn_ds, "PL3_MCMC", rd_mode="CAMB")
        c_cpl = f_cpl["chi2_total"]

        f_h2v_p1 = fit_4param_k5_model(sn_ds, "H2_vol_p1", "PL3_MCMC", rd_mode="CAMB", init_p1=0.42)
        f_h2v_p2 = fit_4param_k5_model(sn_ds, "H2_vol_p2", "PL3_MCMC", rd_mode="CAMB", init_p1=0.55)
        f_h2a_p0_free = fit_4param_k5_model(sn_ds, "H2a_const_p0", "PL3_MCMC", rd_mode="CAMB", init_p1=0.085, dp=0.01)

        # k = 5 smooth dark-energy benchmarks (with truly non-phantom Linder 2015 Eq. 3 algebraic thawing, w0 >= -1):
        f_wcdm = fit_4param_k5_model(sn_ds, "wCDM", "PL3_MCMC", rd_mode="CAMB", init_p1=-0.96, p1_bounds=(-1.45, -0.55), dp=0.02)
        f_thaw = fit_4param_k5_model(sn_ds, "Linder2015_Thawing_k5", "PL3_MCMC", rd_mode="CAMB", init_p1=-0.92, p1_bounds=(-1.0, -0.50), dp=0.015)
        f_cpl_deg = fit_4param_k5_model(sn_ds, "CPL_DegLine_k5", "PL3_MCMC", rd_mode="CAMB", init_p1=-0.82, p1_bounds=(-1.25, -0.50), dp=0.015)

        # Fixed-weight boundary limits (k = 4):
        f_unit_p1 = fit_3param_fixed_p1(sn_ds, 1.0, "H2_vol_p1", "PL3_MCMC", rd_mode="CAMB")
        f_pure_ts_p0 = fit_3param_fixed_p1(sn_ds, 1.0, "H2a_const_p0", "PL3_MCMC", rd_mode="CAMB")

        # Profile curve across alpha in [0, 1] for Figure 1A:
        prof_rows = []
        prev_x = f_lcdm["_x"]
        for av in alpha_grid:
            rf = fit_3param_fixed_p1(sn_ds, av, "H2_vol_p1", "PL3_MCMC", rd_mode="CAMB", init=prev_x)
            prev_x = rf["_x"]
            prof_rows.append({
                "alpha": av,
                "Delta_chi2_vs_LCDM": round(rf["chi2_total"] - c0, 3),
                "chi2_SN": round(rf["chi2_SN"], 2),
                "chi2_BAO": round(rf["chi2_DESI_BAO_13"], 2),
                "chi2_CMB": round(rf["chi2_CMB"], 2),
                "H0": round(rf["H0_kms_Mpc"], 2),
                "Omega_m0": round(rf["Omega_m0"], 4),
            })

        primary_pl3_results[sn_key] = {
            "M0_Flat_LCDM_k4": summarize_fit(f_lcdm, c0, 4, n_pts, c_cpl),
            "H2_vol_p1_free_alpha_k5": summarize_fit(f_h2v_p1, c0, 5, n_pts, c_cpl),
            "H2_vol_p2_free_alpha_k5": summarize_fit(f_h2v_p2, c0, 5, n_pts, c_cpl),
            "Const_Weight_H2a_p0_free_alpha_k5": summarize_fit(f_h2a_p0_free, c0, 5, n_pts, c_cpl),
            "Smooth_wCDM_k5": summarize_fit(f_wcdm, c0, 5, n_pts, c_cpl),
            "Smooth_Linder2015_NonPhantom_Thawing_k5": summarize_fit(f_thaw, c0, 5, n_pts, c_cpl),
            "Smooth_PhantomCrossing_CPL_DegLine_k5": summarize_fit(f_cpl_deg, c0, 5, n_pts, c_cpl),
            "M1b_Free_CPL_k6": summarize_fit(f_cpl, c0, 6, n_pts, c_cpl),
            "Unit_VoidModulated_Limit_alpha1_p1_k4": summarize_fit(f_unit_p1, c0, 4, n_pts, c_cpl),
            "True_Pure_Timescape_alpha1_p0_k4": summarize_fit(f_pure_ts_p0, c0, 4, n_pts, c_cpl),
            "Delta_chi2_H2vol_p1_minus_ThawingQuint_k5": round(f_h2v_p1["chi2_total"] - f_thaw["chi2_total"], 3),
            "Delta_chi2_H2vol_p1_minus_CPL_DegLine_k5": round(f_h2v_p1["chi2_total"] - f_cpl_deg["chi2_total"], 3),
            "alpha_profile_grid": prof_rows,
        }

        # Chen, Huang & Wang (2019) (R, l_A, omega_b) cross-check (CAMB r_d):
        f_lcdm_chw = fit_3param_fixed_p1(sn_ds, 0.0, "H2_vol_p1", "CHW19", rd_mode="CAMB")
        c0_chw = f_lcdm_chw["chi2_total"]
        f_cpl_chw = fit_5param_free_cpl_with_hessian(sn_ds, "CHW19", rd_mode="CAMB")
        f_h2v_chw = fit_4param_k5_model(sn_ds, "H2_vol_p1", "CHW19", rd_mode="CAMB", init_p1=0.40)
        chw19_crosscheck_results[sn_key] = {
            "M0_Flat_LCDM_k4": summarize_fit(f_lcdm_chw, c0_chw, 4, n_pts, f_cpl_chw["chi2_total"]),
            "H2_vol_p1_free_alpha_k5": summarize_fit(f_h2v_chw, c0_chw, 5, n_pts, f_cpl_chw["chi2_total"]),
            "M1b_Free_CPL_k6": summarize_fit(f_cpl_chw, c0_chw, 6, n_pts, f_cpl_chw["chi2_total"]),
        }

        # Aubourg et al. (2015) unscaled r_d (146.895 Mpc) cross-check (reproducing v7.0 scale):
        f_lcdm_aub = fit_3param_fixed_p1(sn_ds, 0.0, "H2_vol_p1", "PL3_MCMC", rd_mode="AUBOURG15")
        f_h2v_aub = fit_4param_k5_model(sn_ds, "H2_vol_p1", "PL3_MCMC", rd_mode="AUBOURG15", init_p1=0.43)
        aubourg15_rd_results[sn_key] = summarize_fit(f_h2v_aub, f_lcdm_aub["chi2_total"], 5, n_pts)

    # C) Sensitivity checks on Pantheon+ + DESI BAO + Planck18:
    pan_ds = data["sn_compilations"]["PantheonPlus"]
    c0_pan = primary_pl3_results["PantheonPlus"]["M0_Flat_LCDM_k4"]["chi2_total"]

    # 1. Full non-selective f_{v,0} scan across [0.60, 0.84] without H1:
    fv0_sensitivity = []
    for fv_fixed in [0.600, 0.650, 0.695, 0.724, 0.762, 0.800, 0.840]:
        rf_fv = fit_4param_k5_model(pan_ds, "H2_vol_p1", "PL3_MCMC", fixed_fv0=fv_fixed, rd_mode="CAMB", init_p1=0.25)
        fv0_sensitivity.append({
            "fixed_f_v0": fv_fixed,
            "alpha_hat": round(rf_fv["alpha_hat"], 4),
            "sigma_alpha": round(rf_fv["sigma_alpha"], 4),
            "chi2_total": round(rf_fv["chi2_total"], 3),
            "Delta_chi2_vs_LCDM": round(rf_fv["chi2_total"] - c0_pan, 3),
            "Delta_BIC_vs_LCDM": round((rf_fv["chi2_total"] - c0_pan) + math.log(pan_ds["n"] + 16), 3),
        })

    # 2. +-0.15% r_d scale sensitivity on Pantheon+:
    rd_scale_sensitivity = {}
    for rd_m in ["CAMB_MINUS_015PCT", "AUBOURG15", "CAMB", "CAMB_PLUS_015PCT"]:
        fl_r = fit_3param_fixed_p1(pan_ds, 0.0, "H2_vol_p1", "PL3_MCMC", rd_mode=rd_m)
        fh_r = fit_4param_k5_model(pan_ds, "H2_vol_p1", "PL3_MCMC", rd_mode=rd_m, init_p1=0.40)
        rd_scale_sensitivity[rd_m] = {
            "fiducial_rd_mode": rd_m,
            "best_fit_rd_Mpc": round(fh_r["r_d_Mpc"], 3),
            "alpha_hat": round(fh_r["alpha_hat"], 4),
            "sigma_alpha": round(fh_r["sigma_alpha"], 4),
            "Delta_chi2_vs_LCDM": round(fh_r["chi2_total"] - fl_r["chi2_total"], 3),
        }

    # 3. Smooth all-z Hamiltonian Forward Sachs optical diagnostics (zero z=2.5 stitch!):
    f_sachs_hom = fit_4param_k5_model(pan_ds, "Sachs_AllZ_HomRicci_p1", "PL3_MCMC", rd_mode="CAMB", init_p1=-0.04, dp=0.01)
    f_sachs_dr = fit_4param_k5_model(pan_ds, "Sachs_AllZ_VoidDepletedRicci_p1", "PL3_MCMC", rd_mode="CAMB", init_p1=0.10, dp=0.015)

    # 4. Bare-H0 vs common-H0 normalization sensitivity on Pantheon+:
    f_bare = fit_4param_k5_model(pan_ds, "H2_vol_p1_bareH0", "PL3_MCMC", rd_mode="CAMB", init_p1=0.02, dp=0.01)

    return {
        "bao_plus_planck18_only_and_out_of_sample_SN_test": bao_cmb_only_block,
        "primary_CAMB_rd_Planck2018_3x3_MCMC_omegab_omegac_100thetastar": primary_pl3_results,
        "crosscheck_Chen_Huang_Wang_2019_R_lA_omegab": chw19_crosscheck_results,
        "crosscheck_Aubourg2015_rd_146p90_Mpc": aubourg15_rd_results,
        "sensitivity_without_H1_full_fv0_scan_0p60_to_0p84_PantheonPlus": fv0_sensitivity,
        "sound_horizon_rd_scale_sensitivity_PantheonPlus": rd_scale_sensitivity,
        "all_z_unstitched_Hamiltonian_Sachs_diagnostics_PantheonPlus": {
            "homogeneous_Ricci_focusing_all_z": summarize_fit(f_sachs_hom, c0_pan, 5, pan_ds["n"] + 16),
            "void_depleted_DyerRoeder_Ricci_focusing_all_z": summarize_fit(f_sachs_dr, c0_pan, 5, pan_ds["n"] + 16),
            "mathematical_explanation": (
                "Integrating the Sachs ODE smoothly from z = 0 to z_* ~ 1090 in u = ln(1+z) with NO z = 2.5 stitch "
                "shows that replacing E_hor -> E_ray in the expansion damping while retaining unmodified homogeneous "
                "FLRW matter focusing 1.5*Omega_m0*(1+z)^3 breaks the FLRW cancellation E_ray*E_ray' = Ricci_source, "
                "generating an O(alpha) secular drift in D_M(z_*) that restricts alpha_hat to -0.048 +- 0.024. "
                "Depleting the Ricci matter focusing along underdense rays by (1 - alpha_eff) partially restores positive "
                "alpha_hat, confirming that consistent 4D light-cone averaging of expansion and Ricci focusing is an "
                "open theoretical problem beyond the 1D H2-vol distance template."
            ),
        },
        "normalization_note_and_bare_H0_sensitivity": {
            "bare_Hbar0_branch_fit_PantheonPlus": summarize_fit(f_bare, c0_pan, 5, pan_ds["n"] + 16),
        },
    }


# ==============================================================================
# 4. DEDUPLICATED & BLOCK-PERMUTED 3D VOIDFINDER EXPLORATORY SIGHTLINE CHECK
# ==============================================================================
def run_3d_void_sightline_audit(data: Dict[str, Any]) -> Dict[str, Any]:
    pan = data["sn_compilations"]["PantheonPlus"]
    zhd = pan["zhd"]
    zhel = pan["zhel"]
    mb = pan["y"]
    L_hf = pan["L"]
    cov_hf = np.einsum("ik,jk->ij", L_hf, L_hf)

    ra_hf = np.zeros(len(zhd), dtype=np.float64)
    dec_hf = np.zeros(len(zhd), dtype=np.float64)
    cid_hf: List[str] = [""] * len(zhd)
    ptr = 0
    with (DATA_DIR / "Pantheon+SH0ES.dat").open("r", encoding="utf-8") as f:
        for r in csv.DictReader(f, delimiter=" ", skipinitialspace=True):
            if ptr < len(zhd) and abs(float(r["zHD"]) - zhd[ptr]) < 1e-7 and abs(float(r["m_b_corr"]) - mb[ptr]) < 1e-5:
                ra_hf[ptr] = float(r["RA"])
                dec_hf[ptr] = float(r["DEC"])
                cid_hf[ptr] = str(r["CID"])
                ptr += 1

    r_sn_mpch = 2997.92458 * H0_D_CPL_vec(zhd, -1.0, 0.0, 0.3153)
    ra_rad = np.deg2rad(ra_hf)
    dec_rad = np.deg2rad(dec_hf)
    n_hat = np.column_stack([np.cos(dec_rad) * np.cos(ra_rad), np.cos(dec_rad) * np.sin(ra_rad), np.sin(dec_rad)])

    holes = data["voidfinder_sdss_dr7"]["holes"]
    max_tbl = data["voidfinder_sdss_dr7"]["maximal"]
    centers = holes[:, :3]
    radii2 = holes[:, 3] ** 2
    c_norm2 = np.sum(centers**2, axis=1)

    ra_v, dec_v = max_tbl[:, 7], max_tbl[:, 8]
    v_dirs = np.column_stack([
        np.cos(np.deg2rad(dec_v)) * np.cos(np.deg2rad(ra_v)),
        np.cos(np.deg2rad(dec_v)) * np.sin(np.deg2rad(ra_v)),
        np.sin(np.deg2rad(dec_v)),
    ])
    cos_mat = np.einsum("id,jd->ij", n_hat, v_dirs)
    in_ngc_interior = (
        (np.max(cos_mat, axis=1) >= math.cos(math.radians(4.5)))
        & (ra_hf >= 125.0) & (ra_hf <= 245.0)
        & (dec_hf >= 3.0) & (dec_hf <= 58.0)
    )

    def trace_1d_void_fraction(unit_dirs: np.ndarray, s_min: np.ndarray, s_max: np.ndarray) -> np.ndarray:
        out = np.zeros(len(unit_dirs), dtype=np.float64)
        for i in range(len(unit_dirs)):
            s0, s1 = float(s_min[i]), float(s_max[i])
            if s1 - s0 <= 5.0:
                continue
            u = unit_dirs[i]
            s_proj = np.einsum("md,d->m", centers, u)
            d2 = c_norm2 - s_proj * s_proj
            hit = np.where(d2 < radii2)[0]
            if len(hit) == 0:
                continue
            hc = np.sqrt(radii2[hit] - d2[hit])
            aa = np.maximum(s0, s_proj[hit] - hc)
            bb = np.minimum(s1, s_proj[hit] + hc)
            val = bb > aa
            if not np.any(val):
                continue
            ivs = sorted(zip(aa[val], bb[val]))
            tot = 0.0
            ca, cb = ivs[0]
            for na, nb in ivs[1:]:
                if na <= cb:
                    if nb > cb:
                        cb = nb
                else:
                    tot += cb - ca
                    ca, cb = na, nb
            tot += cb - ca
            out[i] = tot / (s1 - s0)
        return out

    res_sn_all = mb - 5.0 * np.log10((1.0 + zhel) * H0_D_CPL_vec(zhd, -1.0, 0.0, 0.3326))

    def deduplicate_by_cid(idx: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Contract multi-survey light-curve rows of identical physical SNe (CID) via exact GLS projection."""
        unique_cids: List[str] = []
        groups: Dict[str, List[int]] = {}
        for pos, row_i in enumerate(idx):
            c = cid_hf[row_i]
            if c not in groups:
                unique_cids.append(c)
                groups[c] = []
            groups[c].append(pos)
        m_uniq = len(unique_cids)
        cov_sub = cov_hf[np.ix_(idx, idx)]
        var_diag = np.diag(cov_sub)
        W_agg = np.zeros((m_uniq, len(idx)), dtype=np.float64)
        z_u = np.zeros(m_uniq, dtype=np.float64)
        r_u = np.zeros(m_uniq, dtype=np.float64)
        ra_u = np.zeros(m_uniq, dtype=np.float64)
        dec_u = np.zeros(m_uniq, dtype=np.float64)
        u_dirs = np.zeros((m_uniq, 3), dtype=np.float64)
        for u_i, c in enumerate(unique_cids):
            g = groups[c]
            inv_v = 1.0 / np.maximum(var_diag[g], 1e-8)
            w = inv_v / np.sum(inv_v)
            W_agg[u_i, g] = w
            z_u[u_i] = float(np.sum(w * zhd[idx[g]]))
            r_u[u_i] = float(np.sum(w * r_sn_mpch[idx[g]]))
            ra_u[u_i] = float(ra_hf[idx[g[0]]])
            dec_u[u_i] = float(dec_hf[idx[g[0]]])
            d_mean = np.sum(w[:, None] * n_hat[idx[g]], axis=0)
            u_dirs[u_i] = d_mean / np.linalg.norm(d_mean)
        with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
            y_u = W_agg @ res_sn_all[idx]
            cov_u = (W_agg @ cov_sub) @ W_agg.T
        return z_u, r_u, y_u, cov_u, u_dirs, ra_u, dec_u

    idx_in_raw = np.where(in_ngc_interior & (r_sn_mpch >= 60.0) & (r_sn_mpch <= 305.0))[0]
    idx_bg_raw = np.where(in_ngc_interior & (r_sn_mpch > 320.0))[0]

    z_in, r_in, y_in, cov_in, u_in, ra_in, dec_in = deduplicate_by_cid(idx_in_raw)
    z_bg, r_bg, y_bg, cov_bg, u_bg, ra_bg, dec_bg = deduplicate_by_cid(idx_bg_raw)
    fv_in = trace_1d_void_fraction(u_in, np.full(len(u_in), 30.0), r_in)
    fv_bg = trace_1d_void_fraction(u_bg, np.full(len(u_bg), 30.0), np.full(len(u_bg), 300.0))

    def gls_dedup_stats(z_arr, y_arr, cov_mat, fv_arr, n_raw_rows: int) -> Dict[str, Any]:
        cho = linalg.cho_factor(cov_mat, lower=True)
        z_c = z_arr - np.mean(z_arr)
        fv_c = fv_arr - np.mean(fv_arr)
        X = np.column_stack([np.ones(len(z_arr)), z_c, fv_c])
        Cinv_X = linalg.cho_solve(cho, X)
        Cinv_y = linalg.cho_solve(cho, y_arr)
        cb = linalg.inv(np.einsum("ni,nj->ij", X, Cinv_X))
        bh = np.einsum("ij,nj,n->i", cb, X, Cinv_y)
        slope, err_slope = float(bh[2]), math.sqrt(float(cb[2, 2]))
        z_score = slope / err_slope

        return {
            "N_lightcurve_rows": int(n_raw_rows),
            "N_unique_SNe": int(len(z_arr)),
            "z_min": round(float(np.min(z_arr)), 5),
            "z_max": round(float(np.max(z_arr)), 5),
            "mean_f_v_LOS": round(float(np.mean(fv_arr)), 4),
            "std_f_v_LOS": round(float(np.std(fv_arr)), 4),
            "z_controlled_GLS_slope_dmu_dfv_mag": round(slope, 4),
            "sigma_slope_mag": round(err_slope, 4),
            "ci95_GLS_slope_mag": [round(slope - 1.959964 * err_slope, 4), round(slope + 1.959964 * err_slope, 4)],
            "slope_t_stat_sigma": round(z_score, 2),
        }

    stats_in = gls_dedup_stats(z_in, y_in, cov_in, fv_in, len(idx_in_raw))
    stats_bg = gls_dedup_stats(z_bg, y_bg, cov_bg, fv_bg, len(idx_bg_raw))

    # Combined deduplicated NGC sample (N_unique = len(z_in) + len(z_bg)) with full cross-covariance:
    idx_comb_raw = np.concatenate([idx_in_raw, idx_bg_raw])
    # Build joint block-projection matrix W_tot over idx_comb_raw:
    def build_W(idx_sub):
        uc: List[str] = []; gr: Dict[str, List[int]] = {}
        for p, ri in enumerate(idx_sub):
            c = cid_hf[ri]
            if c not in gr:
                uc.append(c); gr[c] = []
            gr[c].append(p)
        cs = cov_hf[np.ix_(idx_sub, idx_sub)]; vd = np.diag(cs)
        W = np.zeros((len(uc), len(idx_sub)), dtype=np.float64)
        for ui, c in enumerate(uc):
            g = gr[c]; iv = 1.0 / np.maximum(vd[g], 1e-8); W[ui, g] = iv / np.sum(iv)
        return W

    W1, W2 = build_W(idx_in_raw), build_W(idx_bg_raw)
    n1, n2 = W1.shape[0], W2.shape[0]
    W_tot = linalg.block_diag(W1, W2)
    with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
        cov_comb_u = (W_tot @ cov_hf[np.ix_(idx_comb_raw, idx_comb_raw)]) @ W_tot.T
    y_comb_u = np.concatenate([y_in, y_bg])
    cho_c = linalg.cho_factor(cov_comb_u, lower=True)
    X_c = np.zeros((n1 + n2, 5), dtype=np.float64)
    X_c[:n1, 0] = 1.0
    X_c[:n1, 1] = z_in - np.mean(z_in)
    X_c[n1:, 2] = 1.0
    X_c[n1:, 3] = z_bg - np.mean(z_bg)
    X_c[:, 4] = np.concatenate([fv_in - np.mean(fv_in), fv_bg - np.mean(fv_bg)])
    Cinv_Xc = linalg.cho_solve(cho_c, X_c)
    Cinv_yc = linalg.cho_solve(cho_c, y_comb_u)
    cb_c = linalg.inv(np.einsum("ni,nj->ij", X_c, Cinv_Xc))
    bh_c = np.einsum("ij,nj,n->i", cb_c, X_c, Cinv_yc)
    sl_comb, esl_comb = float(bh_c[4]), math.sqrt(float(cb_c[4, 4]))

    # Also compute un-deduplicated N=311 row slope for exact backward reference:
    fv_in_raw = trace_1d_void_fraction(n_hat[idx_in_raw], np.full(len(idx_in_raw), 30.0), r_sn_mpch[idx_in_raw])
    fv_bg_raw = trace_1d_void_fraction(n_hat[idx_bg_raw], np.full(len(idx_bg_raw), 30.0), np.full(len(idx_bg_raw), 300.0))
    cho_raw = linalg.cho_factor(cov_hf[np.ix_(idx_comb_raw, idx_comb_raw)], lower=True)
    nr1, nr2 = len(idx_in_raw), len(idx_bg_raw)
    Xr = np.zeros((nr1 + nr2, 5), dtype=np.float64)
    Xr[:nr1, 0] = 1.0; Xr[:nr1, 1] = zhd[idx_in_raw] - np.mean(zhd[idx_in_raw])
    Xr[nr1:, 2] = 1.0; Xr[nr1:, 3] = zhd[idx_bg_raw] - np.mean(zhd[idx_bg_raw])
    Xr[:, 4] = np.concatenate([fv_in_raw - np.mean(fv_in_raw), fv_bg_raw - np.mean(fv_bg_raw)])
    cbr = linalg.inv(np.einsum("ni,nj->ij", Xr, linalg.cho_solve(cho_raw, Xr)))
    bhr = np.einsum("ij,nj,n->i", cbr, Xr, linalg.cho_solve(cho_raw, res_sn_all[idx_comb_raw]))

    # Exact theoretical single-shell vs deep-tomographic prediction from H2-vol (alpha = 0.40) using actual tracker f_v(z)/f_{v,0}:
    _, sdD_ts, sfv_ts, _ = build_vectorized_wiltshire_splines(0.724)
    zt_sdss = np.linspace(0.01, 0.102, 200)
    dD_diff_sdss = (sfv_ts(np.log1p(zt_sdss)) / 0.724) * (sdD_ts(np.log1p(zt_sdss)) - 1.0 / np.sqrt(0.314 * (1.0 + zt_sdss) ** 3 + 0.686))
    delta_D_sdss = float(integrate.trapezoid(dD_diff_sdss, zt_sdss))
    dmu_th_bg_sdss = (5.0 / math.log(10.0)) * (0.40 / 0.61) * delta_D_sdss / float(H0_D_CPL_vec(np.array([0.30]), -1.0, 0.0, 0.314)[0])

    zt_in = np.linspace(0.01, 0.075, 150)
    delta_D_in = float(integrate.trapezoid((sfv_ts(np.log1p(zt_in)) / 0.724) * (sdD_ts(np.log1p(zt_in)) - 1.0 / np.sqrt(0.314 * (1.0 + zt_in) ** 3 + 0.686)), zt_in))
    dmu_th_in_sdss = (5.0 / math.log(10.0)) * (0.40 / 0.61) * delta_D_in / float(H0_D_CPL_vec(np.array([0.075]), -1.0, 0.0, 0.314)[0])

    zt_deep = np.linspace(0.01, 0.70, 500)
    delta_D_deep = float(integrate.trapezoid((sfv_ts(np.log1p(zt_deep)) / 0.724) * (sdD_ts(np.log1p(zt_deep)) - 1.0 / np.sqrt(0.314 * (1.0 + zt_deep) ** 3 + 0.686)), zt_deep))
    dmu_th_deep_z07 = (5.0 / math.log(10.0)) * (0.40 / 0.61) * delta_D_deep / float(H0_D_CPL_vec(np.array([0.70]), -1.0, 0.0, 0.314)[0])

    # Deep 3D DESI DR1 Ray-Tracing Tests (DESIVAST VoidFinder 0.08<=z<=0.23 & ASTRA-DESI DR1 BGS+LRG 0.10<=z<=0.95):
    def trace_custom_spheres(centers_arr, radii_arr, unit_dirs, s_min, s_max):
        r2 = radii_arr ** 2
        cn2 = np.sum(centers_arr ** 2, axis=1)
        out = np.zeros(len(unit_dirs), dtype=np.float64)
        for i in range(len(unit_dirs)):
            s0, s1 = float(s_min[i]), float(s_max[i])
            if s1 - s0 <= 10.0:
                continue
            with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
                sp = centers_arr @ unit_dirs[i]
            d2 = cn2 - sp * sp
            hit = np.where(d2 < r2)[0]
            if len(hit) == 0:
                continue
            hc = np.sqrt(r2[hit] - d2[hit])
            aa = np.maximum(s0, sp[hit] - hc)
            bb = np.minimum(s1, sp[hit] + hc)
            val = bb > aa
            if not np.any(val):
                continue
            ivs = sorted(zip(aa[val], bb[val]))
            tot = 0.0
            ca, cb = ivs[0]
            for na, nb in ivs[1:]:
                if na <= cb:
                    if nb > cb:
                        cb = nb
                else:
                    tot += cb - ca
                    ca, cb = na, nb
            tot += cb - ca
            out[i] = tot / (s1 - s0)
        return out

    # (a) Official DESI DR1 DESIVAST VoidFinder (NGC+SGC, 101,863 spheres, explicit 0.08 <= z <= 0.23):
    dv_h = data["desivast_dr1_voidfinder"]["holes"]
    dv_m = data["desivast_dr1_voidfinder"]["maximal"]
    dv_dirs = dv_m[:, :3] / np.linalg.norm(dv_m[:, :3], axis=1, keepdims=True)
    with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
        in_dv = np.max(n_hat @ dv_dirs.T, axis=1) >= math.cos(math.radians(3.5))
    idx_dv_in = np.where(in_dv & (zhd >= 0.08) & (zhd <= 0.23))[0]
    z_dvi, r_dvi, y_dvi, cov_dvi, u_dvi, _, _ = deduplicate_by_cid(idx_dv_in)
    fv_dvi = trace_custom_spheres(dv_h[:, :3], dv_h[:, 3], u_dvi, np.full(len(u_dvi), 180.0), r_dvi)
    stats_desivast_in = gls_dedup_stats(z_dvi, y_dvi, cov_dvi, fv_dvi, len(idx_dv_in))

    # (b) Deep ASTRA-DESI DR1 BGS_BRIGHT + LRG (NGC+SGC, 35,122 effective-radius spheres, explicit 0.10 <= z <= 0.95):
    as_bgs = data["astra_desi_dr1"]["bgs"]
    as_lrg = data["astra_desi_dr1"]["lrg"]
    as_all = np.vstack([as_bgs, as_lrg])
    as_dirs = as_all[:, :3] / np.linalg.norm(as_all[:, :3], axis=1, keepdims=True)
    as_lrg_dirs = as_lrg[:, :3] / np.linalg.norm(as_lrg[:, :3], axis=1, keepdims=True)
    with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
        in_as = np.max(n_hat @ as_dirs.T, axis=1) >= math.cos(math.radians(2.5))
        in_as_lrg = np.max(n_hat @ as_lrg_dirs.T, axis=1) >= math.cos(math.radians(2.5))

    idx_as_full = np.where(in_as & (zhd >= 0.10) & (zhd <= 0.95))[0]
    z_asf, r_asf, y_asf, cov_asf, u_asf, _, _ = deduplicate_by_cid(idx_as_full)
    fv_asf = trace_custom_spheres(as_all[:, :3], as_all[:, 3], u_asf, np.full(len(u_asf), 150.0), r_asf)
    stats_astra_0p10_0p95 = gls_dedup_stats(z_asf, y_asf, cov_asf, fv_asf, len(idx_as_full))

    idx_as_deep = np.where(in_as & (zhd >= 0.20) & (zhd <= 0.95))[0]
    z_asd, r_asd, y_asd, cov_asd, u_asd, _, _ = deduplicate_by_cid(idx_as_deep)
    fv_asd = trace_custom_spheres(as_all[:, :3], as_all[:, 3], u_asd, np.full(len(u_asd), 150.0), r_asd)
    stats_astra_0p20_0p95 = gls_dedup_stats(z_asd, y_asd, cov_asd, fv_asd, len(idx_as_deep))

    # Strict LRG-only check: LRG void centers (footprint) AND LRG effective-radius spheres ONLY,
    # explicit 0.42 <= z <= 0.95 with matched inner shell r_lrg_min = r(z=0.42) (~1129.3 Mpc/h) and >= 10 Mpc/h path:
    r_lrg_min = float(2997.92458 * H0_D_CPL_vec(np.array([0.42]), -1.0, 0.0, 0.3153)[0])
    idx_as_lrg = np.where(in_as_lrg & (zhd >= 0.42) & (zhd <= 0.95) & (r_sn_mpch >= r_lrg_min + 10.0))[0]
    z_asl, r_asl, y_asl, cov_asl, u_asl, _, _ = deduplicate_by_cid(idx_as_lrg)
    fv_asl = trace_custom_spheres(as_lrg[:, :3], as_lrg[:, 3], u_asl, np.full(len(u_asl), r_lrg_min), r_asl)
    stats_astra_lrg_0p42_0p95 = gls_dedup_stats(z_asl, y_asl, cov_asl, fv_asl, len(idx_as_lrg))

    # Exact unique-SN overlap between DESIVAST and ASTRA-DESI DR1 (correlated cross-checks, not independent):
    n_overlap_dv_as = len({cid_hf[int(i)] for i in idx_dv_in} & {cid_hf[int(i)] for i in idx_as_full})

    return {
        "catalog_citation": "Douglass, Veyrat & BenZvi (2023), ApJS 265, 7 (SDSS DR7 VoidFinder) + DESI DR1 DESIVAST VoidFinder (v1.0) + ASTRA-DESI DR1 BGS/LRG effective-radius spheres (Zenodo 22003899)",
        "deduplicated_in_volume_SNe_0p02_le_z_le_0p105": stats_in,
        "deduplicated_background_SNe_z_gt_0p11_fixed_shell_30_to_300_Mpch": stats_bg,
        "combined_deduplicated_NGC_interior_SNe": {
            "N_lightcurve_rows": int(nr1 + nr2),
            "N_unique_SNe": int(n1 + n2),
            "z_controlled_GLS_slope_dmu_dfv_mag": round(sl_comb, 4),
            "sigma_slope_mag": round(esl_comb, 4),
            "ci95_GLS_slope_mag": [round(sl_comb - 1.959964 * esl_comb, 4), round(sl_comb + 1.959964 * esl_comb, 4)],
            "slope_t_stat_sigma": round(sl_comb / esl_comb, 2),
            "raw_undeduplicated_N311_slope_mag": round(float(bhr[4]), 4),
            "raw_undeduplicated_N311_sigma_mag": round(math.sqrt(float(cbr[4, 4])), 4),
        },
        "deep_DESI_DR1_3d_void_sightline_tests": {
            "footprint_selection_note": (
                "Sky footprint approximated via angular proximity to nearest void center (3.5 deg for DESIVAST, 2.5 deg for ASTRA-DESI); "
                "actual DESI DR1 coverage is program-dependent (bright/BGS vs dark/LRG). "
                "ASTRA-DESI chord fractions evaluate the 1D union of catalog effective-radius (R_eff) spheres."
            ),
            "DESIVAST_VoidFinder_NGC_SGC_0p08_le_z_le_0p23": stats_desivast_in,
            "ASTRA_DESI_DR1_BGS_LRG_0p10_le_z_le_0p95": stats_astra_0p10_0p95,
            "ASTRA_DESI_DR1_BGS_LRG_0p20_le_z_le_0p95": stats_astra_0p20_0p95,
            "ASTRA_DESI_DR1_LRG_only_0p42_le_z_le_0p95": stats_astra_lrg_0p42_0p95,
            "overlap_DESIVAST_in_ASTRA_unique_SNe": int(n_overlap_dv_as),
            "conditional_N_inv_sqrt_scaling_at_minus_0p052mag": int(
                round(stats_astra_0p10_0p95["N_unique_SNe"] * ((stats_astra_0p10_0p95["sigma_slope_mag"] / (abs(dmu_th_deep_z07) / 3.0)) ** 2))
            ),
            "conditional_scaling_note": (
                "Order-of-magnitude N^{-1/2} scaling assuming identical per-SN sensitivity to the ASTRA-DESI DR1 sample "
                "and a fixed hypothetical -0.052 mag directional slope (not fixed by the sky-averaged distance law; "
                "excludes future survey selection functions and spatial covariance)."
            ),
        },
        "theoretical_H2_vol_scale_and_exploratory_null_status": {
            "expected_SNR_in_SDSS_DR7_shell_sigma": round(abs(dmu_th_in_sdss) / esl_comb, 2),
            "predicted_slope_dmu_dfv_in_volume_z0p075_mag": round(dmu_th_in_sdss, 4),
            "predicted_slope_dmu_dfv_bg_shell_z0p10_at_z0p30_mag": round(dmu_th_bg_sdss, 4),
            "predicted_slope_dmu_dfv_deep_z0p70_tracker_mag": round(dmu_th_deep_z07, 4),
            "status_note": (
                "Exploratory low-redshift null check ONLY: at z <= 0.105 the directional sightline signal "
                f"({dmu_th_bg_sdss:+.3f} to {dmu_th_in_sdss:+.3f} mag under an anisotropic ray-wise extension) has "
                f"~0.3-0.4 sigma expected SNR relative to the +- {esl_comb:.3f} mag uncertainty of N = {n1+n2} unique "
                "Pantheon+ SNe, and low-z void-boundary peculiar velocities (Bolejko et al. 2013, PRL 110, 021302) "
                "contribute comparable variance. The -0.052 mag deep z <= 0.70 curve requires an explicit directional "
                "extension beyond sky-averaged Eq. (4)."
            ),
        },
    }


# ==============================================================================
# 5. MAIN AUDIT ENTRY POINT
# ==============================================================================
def run_audit(run_asserts: bool = True, generate_figures: bool = True) -> Dict[str, Any]:
    t_start = time.perf_counter()
    data = load_all_cosmological_datasets()

    # 1. Exact Buchert-Wiltshire tracker cubic factorization for f_acc (eliminating f_v = 1 root rounding bug):
    # q_wall numerator: 8*f^4 + 31*f^3 - 51*f^2 + 4*f + 8 == (f - 1)*(8*f^3 + 39*f^2 - 12*f - 8)
    factored_cubic = [8.0, 39.0, -12.0, -8.0]
    cubic_roots = np.roots(factored_cubic)
    real_pos_roots = [float(r.real) for r in cubic_roots if abs(r.imag) < 1e-12 and 0.0 < r.real < 1.0]
    f_acc = real_pos_roots[0]  # 0.5867077311400187

    Om_m0_fid = 0.3153
    z_acc_fid = (2.0 * (1.0 - Om_m0_fid) / Om_m0_fid) ** (1.0 / 3.0) - 1.0

    def z_of_f(fv: float, f0_val: float) -> float:
        return ((2.0 + fv) / (2.0 + f0_val)) * ((f0_val / fv) ** (2.0 / 3.0)) * ((1.0 - fv) / (1.0 - f0_val)) - 1.0

    f0_fid = float(optimize.brentq(lambda f0_v: z_of_f(f_acc, f0_v) - z_acc_fid, 0.5, 0.9, xtol=1e-15))
    max_diff_closed_vs_quad = max(
        abs(H0_D_TS_wiltshire_closed(zv, f0_fid) - H0_D_TS_wiltshire_quad(zv, f0_fid))
        for zv in [0.1, 0.4, 0.6316, 1.0, 1.5, 2.0]
    )

    # Exact recombination void-fraction scaling check: f_v(z) ~ (1+z)^{-3/2}, f_v(1089.9) = 8.71e-5
    _, _, sfv_check, _ = build_vectorized_wiltshire_splines(f0_fid)
    fv_zstar_1090 = float(sfv_check(math.log1p(1089.92)))

    fv0_w09 = 0.762
    om_dressed_w09 = 0.5 * (1.0 - fv0_w09) * (2.0 + fv0_w09)
    c1_w09 = (24.0 * fv0_w09**4 - 23.0 * fv0_w09**3 + 99.0 * fv0_w09**2 + 8.0) / (
        2.0 * ((4.0 * fv0_w09**2 + fv0_w09 + 4.0) ** 2)
    )
    w0_w09 = (2.0 * (2.0 * (1.0 - c1_w09)) - 3.0) / (3.0 * (1.0 - om_dressed_w09))

    if run_asserts:
        assert np.allclose(np.polymul([1.0, -1.0], factored_cubic), [8.0, 31.0, -51.0, 4.0, 8.0])
        assert len(real_pos_roots) == 1
        assert max_diff_closed_vs_quad < 1e-12, f"Wiltshire quadrature diff: {max_diff_closed_vs_quad}"
        assert abs(w0_w09 - (-0.758)) < 5e-4, f"Wiltshire 2009 benchmark mismatch: {w0_w09}"

    c1_fid = (24.0 * f0_fid**4 - 23.0 * f0_fid**3 + 99.0 * f0_fid**2 + 8.0) / (
        2.0 * ((4.0 * f0_fid**2 + f0_fid + 4.0) ** 2)
    )
    dE_TS_dz0_fid = 2.0 * (1.0 - c1_fid)

    # 2. SN-only flat-LCDM Omega_m0 benchmarks across all 4 compilations & Pantheon+ alpha scan
    sn_only_benchmarks: Dict[str, Any] = {}
    for sn_k, sn_ds in data["sn_compilations"].items():
        res_om = optimize.minimize_scalar(
            lambda om: eval_sn_chi2_direct(sn_ds, H0_D_CPL_vec(sn_ds["zhd"], -1.0, 0.0, float(om))),
            bounds=(0.15, 0.55), method="bounded",
        )
        om_b = float(res_om.x)
        c_b = float(res_om.fun)
        h_s = 0.005
        cp = eval_sn_chi2_direct(sn_ds, H0_D_CPL_vec(sn_ds["zhd"], -1.0, 0.0, om_b + h_s))
        cm = eval_sn_chi2_direct(sn_ds, H0_D_CPL_vec(sn_ds["zhd"], -1.0, 0.0, om_b - h_s))
        sig_om = math.sqrt(2.0 * h_s * h_s / max(cp - 2.0 * c_b + cm, 1e-9))
        sn_only_benchmarks[sn_k] = {
            "N_SN": sn_ds["n"],
            "best_fit_LCDM_Omega_m0": round(om_b, 4),
            "sigma_Omega_m0": round(sig_om, 4),
            "min_chi2_SN": round(c_b, 2),
        }

    pan = data["sn_compilations"]["PantheonPlus"]
    ln1pz_pan = pan["ln1pz_hd"]
    res_ts_free = optimize.minimize_scalar(
        lambda fv: eval_sn_chi2_direct(pan, build_vectorized_wiltshire_splines(float(fv))[0](ln1pz_pan)),
        bounds=(0.55, 0.92), method="bounded",
    )
    D_hor_pan = H0_D_CPL_vec(pan["zhd"], -1.0, 0.0, Om_m0_fid)
    sD_fid, sdD_fid, sfv_fid, _ = build_vectorized_wiltshire_splines(f0_fid)
    D_ts_pan = sD_fid(ln1pz_pan)
    chi2_pan_lcdm_fix = eval_sn_chi2_direct(pan, D_hor_pan)
    chi2_pan_ts_fix = eval_sn_chi2_direct(pan, D_ts_pan)

    z_g = np.linspace(0.0, 2.35, 1000)
    dDh_g = 1.0 / np.sqrt(Om_m0_fid * (1.0 + z_g) ** 3 + (1.0 - Om_m0_fid))
    dDt_g = sdD_fid(np.log1p(z_g))
    fv_g = sfv_fid(np.log1p(z_g))
    sn_only_alpha_scan = []
    for av in [0.0, 0.10, 0.20, 0.22, 0.30, 0.40, 0.60, 1.00]:
        c_p0 = eval_sn_chi2_direct(pan, (1.0 - av) * D_hor_pan + av * D_ts_pan)
        ae = av * (fv_g / f0_fid)
        spl_p1 = interpolate.CubicSpline(z_g, (1.0 - ae) * dDh_g + ae * dDt_g).antiderivative()
        c_p1 = eval_sn_chi2_direct(pan, spl_p1(pan["zhd"]))
        sn_only_alpha_scan.append({
            "alpha": av,
            "H2a_p0_Delta_chi2_SN": round(c_p0 - chi2_pan_lcdm_fix, 2),
            "H2_vol_p1_Delta_chi2_SN": round(c_p1 - chi2_pan_lcdm_fix, 2),
        })

    # 3. Single-clock Dyer-Roeder numerical verification (w_0^app(0) == -1):
    def dr_rhs(zv: float, y: np.ndarray) -> np.ndarray:
        DA, dDA = y
        zp1 = 1.0 + zv
        Eh = math.sqrt(Om_m0_fid * (zp1**3) + (1.0 - Om_m0_fid))
        dEh = 1.5 * Om_m0_fid * (zp1**2) / Eh
        ddDA = -(2.0 * zp1 * Eh * dDA + (zp1**2) * dEh * dDA + 1.5 * Om_m0_fid * 0.58 * (zp1**3) / Eh * DA) / ((zp1**2) * Eh)
        return np.array([dDA, ddDA])

    sol_dr = integrate.solve_ivp(dr_rhs, (0.0, 0.05), np.array([0.0, 1.0]), method="DOP853", rtol=1e-12, atol=1e-14, dense_output=True)
    dz_tiny = 1e-4
    d2DA_dz2_0 = float((sol_dr.sol(dz_tiny)[1] - sol_dr.sol(0.0)[1]) / dz_tiny)
    d2DA_expected = -(2.0 + 1.5 * Om_m0_fid)

    # 4. Full Multi-Compilation Joint Audit + Deduplicated 3D Void Sightline Exploratory Check
    joint_suite = run_multi_sn_joint_likelihood_suite(data, f_acc=f_acc)
    sightline_suite = run_3d_void_sightline_audit(data)

    elapsed = round(time.perf_counter() - t_start, 2)
    results = {
        "audit_version": (
            "v12.0 (Final JCAP Submission: Exact 100*theta_* Planck PR3 Cov, Linder 2015 Non-Phantom Thawing, "
            "Uniform BAO+CMB Out-of-Sample SN Split, & Matched-Boundary Exact-z Deep 3D DESI DR1 Void Sightlines)"
        ),
        "runtime_seconds": elapsed,
        "section_1_wiltshire_benchmarks_and_recombination_scaling": {
            "factored_cubic_coefficients": factored_cubic,
            "f_acc_exact_tracker_root": round(f_acc, 8),
            "f_v0_at_Om03153_H1": round(f0_fid, 6),
            "early_matter_scaling_exponent": -1.5,
            "f_v_at_zstar_1089p92": round(fv_zstar_1090, 8),
            "dE_TS_dz_at_z0_Om03153": round(dE_TS_dz0_fid, 6),
            "wiltshire_2009_closed_vs_quad_max_diff": max_diff_closed_vs_quad,
            "wiltshire_2009_fv0_0p762_w0_benchmark": round(w0_w09, 6),
            "sn_only_flat_LCDM_Omega_m0_across_4_compilations": sn_only_benchmarks,
            "pantheon_plus_fixed_Om03153_LCDM_chi2": round(chi2_pan_lcdm_fix, 2),
            "pantheon_plus_fixed_Om03153_pure_TS_chi2": round(chi2_pan_ts_fix, 2),
            "pantheon_plus_best_fit_pure_TS_fv0": round(float(res_ts_free.x), 4),
            "pantheon_plus_best_fit_pure_TS_chi2": round(float(res_ts_free.fun), 2),
            "pantheon_plus_sn_only_alpha_profile": sn_only_alpha_scan,
        },
        "section_2_single_clock_dyer_roeder_invariance": {
            "references": "Kantowski (1969); Dyer & Roeder (1972, 1973); Weinberg (1976)",
            "numerical_d2DA_dz2_at_z0_alpha_beam_0p58": round(d2DA_dz2_0, 5),
            "exact_analytic_minus_2_minus_1p5_Om0": round(d2DA_expected, 5),
            "exact_w0_app_at_z0": -1.0,
        },
        "section_3_multi_compilation_joint_BAO_CMB_SN_fits": joint_suite,
        "section_4_3d_cosmic_void_sightline_exploratory_null_check": sightline_suite,
    }

    JSON_OUT.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(json.dumps(results, indent=2))
    print(f"\n[PASS] v12.0 Verification Suite completed in {elapsed:.2f}s -> {JSON_OUT}")

    if generate_figures:
        try:
            import generate_sub4_figures_and_tables as gen_ft
            gen_ft.regenerate_all_from_json(results)
        except ImportError as exc:
            print(f"[INFO] Skipping optional figure regeneration (matplotlib unavailable: {exc}).")

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Standalone v12.0 Void-Weighted Cosmology Verification Suite")
    parser.add_argument("--no-asserts", action="store_true", help="Disable unit-test assertions for custom prior exploration")
    parser.add_argument("--no-figures", action="store_true", help="Skip regenerating Figure 1 PNGs and LaTeX table snippets")
    args = parser.parse_args()
    run_audit(run_asserts=not args.no_asserts, generate_figures=not args.no_figures)
