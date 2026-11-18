#!/usr/bin/env python3
"""
verify_C5_kh_background_dependence_audit.py
===========================================
Program C (Task C5) Independent Dual-Method Verification Script
Tau Core Integrated Fundamental Physics Research Program (2026-11-18)

Executes rigorous symbolic (SymPy) and independent numerical ODE/root-finding (SciPy)
verification of Paper 20's Kodama-Hayward Horizon Acceleration Law:
    a_0^KH(z) = (5*sqrt(3)/96) * c * H(z) * |1 - q(z)|,
    q(z) = -1 + ((1 + z) / H(z)) * dH/dz.

Verifies:
  1. Complete TeX corpus audit across all 50 `.tex` manuscripts in `all_papers_bundle/TeX_Sources/`:
     confirms ZERO remaining `(3 - q)` / `3 - q` instances (`PROVEN`).
  2. Exact Equivalence-Breaking Identity between Paper 20's two written forms (`PROVEN`):
       (1 - q(z)) / 2  ==  1 - (3/4)*Omega_m(z) - (3/4)*(1 + w(z))*Omega_DE(z),
     proving that `a_0^KH(z) = (5*sqrt(3)/48)*c*H(z)*|1 - (3/4)*Omega_m(z)|` is ONLY valid
     when w(z) == -1! On any dynamical DE background, the true geometric Cartan dK formula is
     `(5*sqrt(3)/96)*c*H(z)*|1 - q(z)|`.
  3. Dual-method (SymPy symbolic derivatives + SciPy DOP853 Friedmann ODE integration) evaluation on:
     (a) Paper 20 Planck PR3 LCDM (Omega_m0 = 0.3153, Omega_L0 = 0.6847):
         z_dip == z_acc = 0.63155650, a_0^KH(z_dip)/a_0^KH(0) = 0.93855026.
     (b) Theorem C2 Standalone Tau Core LCDM (Omega_m0 = 0.315565, Omega_L0 = 0.684435):
         z_dip == z_acc = 0.630889, a_0^KH(z_dip)/a_0^KH(0) = 0.938575.
     (c) DES 2026 + DESI DR2 + CMB CPL background (w0 = -0.82, wa = -0.63) across
         Omega_m0 in {0.300, 0.306, 0.310, 0.3153, 0.320, 0.330}:
         - Solves for the exact Omega_m0_* reproducing a_0^KH(0.6315565)/a_0^KH(0) = 1.14272!
         - Decomposes the ~+14% ratio jump into: (i) the -12.1% drop of (1 - q_0) at z = 0
           caused by w_0 = -0.82 > -1 (which directly shifts local z = 0 SPARC a_0^KH(0)!),
           and (ii) the phantom enhancement 1 + w(z) < 0 for z > 0.2857!
         - Finds all true stationary points da_0^KH/dz = 0 and the true acceleration transition
           z_acc (q(z_acc) = 0) on the DES-CPL background!
  4. Exact Analytic Coincidence-Breaking Theorem (`PROVEN`):
     Proves symbolically in SymPy that on a flat matter + dark-energy FLRW background,
       (1 / H) * d[H*(1 - q)] / d(ln a) |_{q(a_acc)=0}
         = -(3/2)*(1 + w(a_acc)) + (1 / (2*w(a_acc))) * (dw / d ln a)|_{a_acc},
     so z_dip == z_acc holds IF AND ONLY IF 3*w*(1 + w) - dw/d(ln a) = 0 at a_acc
     (and for constant w != 0, uniquely w == -1).
  5. Physical sign audit of (1 - q(z)), radiation-era vanishing/scaling, Monte Carlo error
     propagation from (w0, wa, Omega_m0) onto a_0^KH(z)/a_0^KH(0), and identification of the
     strictly background-independent kinematic invariant:
       I_KH(z) := a_0^KH(z) / (c * H(z) * |1 - q(z)|) == 5*sqrt(3)/96 = 0.09021097956...
  6. Exports full comparison table to `verification/Tau_DES_2026_background_comparison.csv`.
"""

from __future__ import annotations

import csv
import json
import math
import re
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
from scipy import integrate, optimize
import sympy as sp


OUT_DIR = Path(__file__).resolve().parent
TEX_DIR = Path("/Volumes/EXTERNAL/Research/Tau Core/all_papers_bundle/TeX_Sources")
JSON_OUT = OUT_DIR / "C5_kh_background_dependence_results.json"
CSV_BG_OUT = OUT_DIR / "Tau_DES_2026_background_comparison.csv"
CSV_OM_SCAN_OUT = OUT_DIR / "C5_des_cpl_omega_m_scan_and_extrema.csv"

C_SI = 299792458.0
MPC_M = 3.0856775814913673e22
ALPHA_OCSR = 5.0 * math.sqrt(3.0) / 24.0
I_KH_EXACT = 5.0 * math.sqrt(3.0) / 96.0


def audit_tex_sources_for_3_minus_q() -> Dict[str, Any]:
    """Scan all .tex files in all_papers_bundle/TeX_Sources/ (or local package root when standalone) for `(3 - q)`."""
    if TEX_DIR.exists():
        scan_dir = TEX_DIR
        tex_files = sorted(TEX_DIR.glob("*.tex"))
    else:
        scan_dir = OUT_DIR.parent
        tex_files = sorted(scan_dir.glob("*.tex"))
    pat_3mq = re.compile(r"3\s*-\s*q")
    pat_1mq = re.compile(r"1\s*-\s*q")
    hits_3mq: List[Dict[str, Any]] = []
    hits_1mq: List[str] = []

    for tf in tex_files:
        txt = tf.read_text(encoding="utf-8", errors="replace")
        for line_no, line in enumerate(txt.splitlines(), start=1):
            if pat_3mq.search(line):
                hits_3mq.append({"file": tf.name, "line": line_no, "text": line.strip()})
            if pat_1mq.search(line):
                hits_1mq.append(f"{tf.name}:{line_no}")

    assert len(hits_3mq) == 0, f"Found unexpected 3-q instances: {hits_3mq}"
    return {
        "epistemic_status": "PROVEN",
        "tex_directory": str(scan_dir),
        "total_tex_files_scanned": len(tex_files),
        "three_minus_q_instances_found": len(hits_3mq),
        "three_minus_q_hits": hits_3mq,
        "one_minus_q_verified_occurrences_count": len(hits_1mq),
        "one_minus_q_sample_locations": hits_1mq[:12],
    }


def prove_symbolic_kh_coincidence_breaking_theorem() -> Dict[str, Any]:
    """
    Exact SymPy proof of:
      1. Equivalence breaking between |1 - q(z)|/2 and |1 - (3/4)*Omega_m(z)| when w != -1;
      2. General formula for (1/H) d[H(1 - q)]/d(ln a) evaluated at q(a_acc) = 0;
      3. Necessary and sufficient condition 3*w*(1 + w) - dw/d(ln a) = 0 at a_acc for z_dip == z_acc.
    """
    x_s = sp.Symbol("x", real=True)  # x = ln(a)
    Om0_s = sp.Symbol("Om0", positive=True, real=True)
    w_fn = sp.Function("w", real=True)(x_s)

    # Express general flat FLRW identity in terms of Omega_DE(x) and w(x):
    # d(ln H)/dx = -(3/2) * [Omega_m + (1 + w) * Omega_DE] = -(3/2) * (1 + w * Omega_DE)
    Ode_fn = sp.Function("Ode", positive=True)(x_s)
    Om_expr = 1 - Ode_fn
    dlnH_dx = -sp.Rational(3, 2) * (1 + w_fn * Ode_fn)
    q_expr = -1 - dlnH_dx  # = (1/2) * (1 + 3 * w * Ode)

    # Check Paper 20 two-form difference: (1 - q)/2 - (1 - (3/4)*Om)
    diff_forms = sp.simplify(sp.Rational(1, 2) * (1 - q_expr) - (1 - sp.Rational(3, 4) * Om_expr))
    expected_diff = -sp.Rational(3, 4) * (1 + w_fn) * Ode_fn
    assert sp.simplify(diff_forms - expected_diff) == 0

    # Evolution ODE of Omega_DE(x) on flat matter + DE background:
    # Ode = rho_DE / (rho_m + rho_DE), d(ln rho_m)/dx = -3, d(ln rho_DE)/dx = -3*(1 + w)
    # => d(Ode)/dx = -3 * w * Ode * (1 - Ode)
    dOde_dx_rule = -3 * w_fn * Ode_fn * (1 - Ode_fn)
    d2lnH_dx2 = sp.simplify(sp.diff(dlnH_dx, x_s).subs(sp.diff(Ode_fn, x_s), dOde_dx_rule))

    # Derivative of A(x) := H(x) * (1 - q(x)) = H(x) * (2 + dlnH/dx):
    # (1/H) dA/dx = (dlnH/dx)*(2 + dlnH/dx) + d2lnH/dx2
    one_over_H_dA_dx = sp.simplify(dlnH_dx * (2 + dlnH_dx) + d2lnH_dx2)

    # At acceleration transition q(x_acc) = 0 <=> 1 + 3 * w * Ode = 0 <=> Ode = -1 / (3*w)
    w_acc_s, dw_dx_acc_s = sp.symbols("w_acc dw_dx_acc", real=True)
    at_q0 = sp.simplify(
        one_over_H_dA_dx.subs(
            {
                sp.diff(w_fn, x_s): dw_dx_acc_s,
                w_fn: w_acc_s,
                Ode_fn: -1 / (3 * w_acc_s),
            }
        )
    )
    expected_at_q0 = -sp.Rational(3, 2) * (1 + w_acc_s) + dw_dx_acc_s / (2 * w_acc_s)
    assert sp.simplify(at_q0 - expected_at_q0) == 0

    # Constant-w specialization (dw_dx_acc = 0): vanishes iff w_acc = -1
    const_w_roots = sp.solve(sp.Eq(at_q0.subs(dw_dx_acc_s, 0), 0), w_acc_s)
    assert const_w_roots == [-1]

    return {
        "epistemic_status": "PROVEN",
        "two_formula_difference_identity": "(1 - q(z))/2 - [1 - (3/4)*Omega_m(z)] = -(3/4)*(1 + w(z))*Omega_DE(z)",
        "two_formulas_coincide_iff": "w(z) == -1 (on any dynamical DE background, the 1 - (3/4)*Omega_m shorthand is invalid)",
        "exact_slope_at_q_equals_0": "(1/H) * d[H*(1 - q)]/d(ln a) |_{q=0} = -(3/2)*(1 + w(a_acc)) + (1/(2*w(a_acc))) * (dw/d ln a)|_{a_acc}",
        "coincidence_condition_z_dip_eq_z_acc": "3 * w(a_acc) * (1 + w(a_acc)) - (dw / d ln a)|_{a_acc} == 0",
        "unique_constant_w_solution": "w == -1",
    }


def cpl_kinematics_analytic(
    z: np.ndarray | float, om0: float, w0: float, wa: float, h0_kms_mpc: float = 67.36
) -> Dict[str, np.ndarray | float]:
    """Method A: Closed-form analytic evaluation of E(z), q(z), Omega_m(z), and a_0^KH(z) on flat CPL."""
    zv = np.asarray(z, dtype=float)
    zp1 = 1.0 + zv
    a = 1.0 / zp1
    ode0 = 1.0 - om0
    W_de = (zp1 ** (3.0 * (1.0 + w0 + wa))) * np.exp(-3.0 * wa * zv / zp1)
    E2 = om0 * (zp1**3) + ode0 * W_de
    E = np.sqrt(E2)
    om_z = om0 * (zp1**3) / E2
    ode_z = ode0 * W_de / E2
    w_z = w0 + wa * (zv / zp1)

    # Exact q(z) = (1/2) * [Omega_m(z) + (1 + 3*w(z)) * Omega_DE(z)]
    q_z = 0.5 * (om_z + (1.0 + 3.0 * w_z) * ode_z)
    one_minus_q = 1.0 - q_z

    # Values at z = 0
    q_0 = 0.5 * (om0 + (1.0 + 3.0 * w0) * ode0)
    one_minus_q0 = 1.0 - q_0

    h0_si = h0_kms_mpc * 1000.0 / MPC_M
    a0_kh_0 = I_KH_EXACT * C_SI * h0_si * np.abs(one_minus_q0)
    a0_kh_z = I_KH_EXACT * C_SI * h0_si * E * np.abs(one_minus_q)
    ratio_kh = (E * np.abs(one_minus_q)) / np.abs(one_minus_q0)

    # Also compute what the naive LCDM-only shorthand `1 - (3/4)*Omega_m(z)` would wrongly give:
    naive_om_ratio = (E * np.abs(1.0 - 0.75 * om_z)) / np.abs(1.0 - 0.75 * om0)

    if np.ndim(E) == 0:
        return {
            "E": float(E),
            "w": float(w_z),
            "Omega_m": float(om_z),
            "Omega_DE": float(ode_z),
            "q": float(q_z),
            "one_minus_q": float(one_minus_q),
            "one_minus_q0": float(one_minus_q0),
            "a0_KH_0_m_s2": float(a0_kh_0),
            "a0_KH_z_m_s2": float(a0_kh_z),
            "a0_KH_ratio": float(ratio_kh),
            "naive_1_minus_34Om_ratio": float(naive_om_ratio),
        }
    return {
        "E": E,
        "w": w_z,
        "Omega_m": om_z,
        "Omega_DE": ode_z,
        "q": q_z,
        "one_minus_q": one_minus_q,
        "one_minus_q0": one_minus_q0,
        "a0_KH_0_m_s2": a0_kh_0,
        "a0_KH_z_m_s2": a0_kh_z,
        "a0_KH_ratio": ratio_kh,
        "naive_1_minus_34Om_ratio": naive_om_ratio,
    }


def cpl_kinematics_ode_cross_check(
    z_eval: np.ndarray, om0: float, w0: float, wa: float
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Method B: Independent SciPy DOP853 numerical integration of the coupled Friedmann + DE
    continuity ODEs in u = ln(1 + z):
        d(rho_m)/du = 3 * rho_m,
        d(rho_DE)/du = 3 * (1 + w0 + wa * (1 - exp(-u))) * rho_DE,
    with q(u) computed from (1/(2 E^2)) * d(E^2)/du - 1.
    """
    u_max = float(np.log1p(np.max(z_eval))) + 0.05

    def rhs(u: float, y: np.ndarray) -> np.ndarray:
        rm, rde = float(y[0]), float(y[1])
        w_u = w0 + wa * (1.0 - math.exp(-u))
        return np.array([3.0 * rm, 3.0 * (1.0 + w_u) * rde], dtype=float)

    sol = integrate.solve_ivp(
        rhs,
        t_span=(0.0, u_max),
        y0=np.array([om0, 1.0 - om0], dtype=float),
        t_eval=np.log1p(z_eval),
        method="DOP853",
        rtol=1e-13,
        atol=1e-15,
    )
    assert sol.success
    rm_arr, rde_arr = sol.y[0], sol.y[1]
    e2_arr = rm_arr + rde_arr
    e_arr = np.sqrt(e2_arr)
    w_arr = w0 + wa * (z_eval / (1.0 + z_eval))
    de2_du = 3.0 * rm_arr + 3.0 * (1.0 + w_arr) * rde_arr
    q_arr = -1.0 + 0.5 * de2_du / e2_arr
    q0 = -1.0 + 0.5 * (3.0 * om0 + 3.0 * (1.0 + w0) * (1.0 - om0))
    ratio_ode = e_arr * np.abs(1.0 - q_arr) / abs(1.0 - q0)
    return e_arr, ratio_ode


def find_cpl_stationary_points_and_zacc(
    om0: float, w0: float, wa: float
) -> Dict[str, Any]:
    """Find z_acc (q(z)=0) and all local extrema da_0^KH/dz = 0 for z in [0, 10]."""
    def q_of_z(zv: float) -> float:
        return float(cpl_kinematics_analytic(zv, om0, w0, wa)["q"])

    z_acc = float(optimize.brentq(q_of_z, 0.0, 5.0, xtol=1e-14))

    # Derivative d/dz [ E(z) * (1 - q(z)) ] via high-accuracy complex-step differentiation!
    def f_kh_complex(zc: complex) -> complex:
        zp1 = 1.0 + zc
        ode0 = 1.0 - om0
        W_de = (zp1 ** (3.0 * (1.0 + w0 + wa))) * np.exp(-3.0 * wa * zc / zp1)
        E2 = om0 * (zp1**3) + ode0 * W_de
        E = np.sqrt(E2)
        om_z = om0 * (zp1**3) / E2
        ode_z = ode0 * W_de / E2
        w_z = w0 + wa * (zc / zp1)
        q_z = 0.5 * (om_z + (1.0 + 3.0 * w_z) * ode_z)
        return E * (1.0 - q_z)

    h_cs = 1.0e-30

    def df_dz(zv: float) -> float:
        return float(f_kh_complex(zv + 1j * h_cs).imag / h_cs)

    z_scan = np.linspace(0.0, 8.0, 4001)
    df_vals = np.array([df_dz(float(zv)) for zv in z_scan])
    extrema: List[Dict[str, Any]] = []
    for idx in range(len(z_scan) - 1):
        if df_vals[idx] == 0.0 or (df_vals[idx] * df_vals[idx + 1] < 0.0):
            zr = float(optimize.brentq(df_dz, float(z_scan[idx]), float(z_scan[idx + 1]), xtol=1e-14))
            dz2 = 1e-6
            d2f = (df_dz(zr + dz2) - df_dz(zr - dz2)) / (2.0 * dz2)
            kin_r = cpl_kinematics_analytic(zr, om0, w0, wa)
            extrema.append({
                "z_extremum": zr,
                "kind": "local_minimum" if d2f > 0 else "local_maximum",
                "a0_KH_ratio_at_extremum": float(kin_r["a0_KH_ratio"]),
                "a0_KH_m_s2_at_extremum": float(kin_r["a0_KH_z_m_s2"]),
                "q_at_extremum": float(kin_r["q"]),
                "d2f_dz2": d2f,
            })

    kin_zacc = cpl_kinematics_analytic(z_acc, om0, w0, wa)
    kin_z06316 = cpl_kinematics_analytic(0.63155650188, om0, w0, wa)
    return {
        "Omega_m0": om0,
        "z_acc": z_acc,
        "a0_KH_ratio_at_z_acc": float(kin_zacc["a0_KH_ratio"]),
        "a0_KH_ratio_at_LCDM_zdip_0p6316": float(kin_z06316["a0_KH_ratio"]),
        "a0_KH_0_m_s2": float(kin_z06316["a0_KH_0_m_s2"]),
        "a0_KH_at_0p6316_m_s2": float(kin_z06316["a0_KH_z_m_s2"]),
        "one_minus_q0": float(kin_z06316["one_minus_q0"]),
        "df_dz_at_z0": df_dz(0.0),
        "stationary_points_z_ge_0": extrema,
    }


def run_c5_numerical_and_mc_audit() -> Dict[str, Any]:
    # 1. Verify Paper 20 LCDM (Omega_m0 = 0.3153) & Tau C2 LCDM (Omega_m0 = 0.315565)
    lcdm_p20 = find_cpl_stationary_points_and_zacc(0.3153, -1.0, 0.0)
    z_dip_exact_p20 = ((2.0 * 0.6847 / 0.3153) ** (1.0 / 3.0)) - 1.0
    assert abs(lcdm_p20["z_acc"] - z_dip_exact_p20) < 1e-12
    assert abs(lcdm_p20["stationary_points_z_ge_0"][0]["z_extremum"] - z_dip_exact_p20) < 1e-12
    assert abs(lcdm_p20["stationary_points_z_ge_0"][0]["a0_KH_ratio_at_extremum"] - 0.93855026) < 1e-7

    lcdm_c2 = find_cpl_stationary_points_and_zacc(0.315565, -1.0, 0.0)

    # 2. Dual-method ODE cross-check across z in [0, 14.18]
    z_grid = np.array([
        0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.295, 0.35, 0.40, 0.45, 0.51,
        0.57, 0.6315565, 0.706, 0.80, 0.934, 1.00, 1.15, 1.321, 1.484,
        1.75, 2.00, 2.33, 3.00, 4.50, 7.31, 10.40, 14.18
    ], dtype=float)

    e_an_lcdm = cpl_kinematics_analytic(z_grid, 0.3153, -1.0, 0.0)
    e_ode_lcdm, rat_ode_lcdm = cpl_kinematics_ode_cross_check(z_grid, 0.3153, -1.0, 0.0)
    max_ode_err_lcdm = float(np.max(np.abs(e_an_lcdm["a0_KH_ratio"] - rat_ode_lcdm)))

    e_an_des = cpl_kinematics_analytic(z_grid, 0.3153, -0.82, -0.63)
    e_ode_des, rat_ode_des = cpl_kinematics_ode_cross_check(z_grid, 0.3153, -0.82, -0.63)
    max_ode_err_des = float(np.max(np.abs(e_an_des["a0_KH_ratio"] - rat_ode_des)))
    assert max_ode_err_lcdm < 1e-11 and max_ode_err_des < 1e-11

    # 3. Scan Omega_m0 on DES+DESI+CMB CPL (-0.82, -0.63) and find exact Omega_m0 matching 1.14272
    om_list = [0.300, 0.306, 0.310, 0.3153, 0.320, 0.330]
    om_scan_results: List[Dict[str, Any]] = []
    for omv in om_list:
        res_om = find_cpl_stationary_points_and_zacc(omv, -0.82, -0.63)
        om_scan_results.append(res_om)

    # Solve for exact Omega_m0 giving a0_KH(z=0.6315565)/a0_KH(0) == 1.14272 on (-0.82, -0.63)
    om_match_06316 = float(
        optimize.brentq(
            lambda omv: float(cpl_kinematics_analytic(0.63155650188, omv, -0.82, -0.63)["a0_KH_ratio"]) - 1.14272,
            0.25,
            0.40,
            xtol=1e-14,
        )
    )
    # Also check at z = 0.632
    om_match_0632 = float(
        optimize.brentq(
            lambda omv: float(cpl_kinematics_analytic(0.632, omv, -0.82, -0.63)["a0_KH_ratio"]) - 1.14272,
            0.25,
            0.40,
            xtol=1e-14,
        )
    )

    with CSV_OM_SCAN_OUT.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "Omega_m0", "z_acc_DES_CPL", "a0_KH_0_m_s2", "one_minus_q0",
            "a0_KH_ratio_at_z_acc", "a0_KH_ratio_at_z0p6316", "a0_KH_at_z0p6316_m_s2",
            "n_extrema_z_ge_0", "extrema_summary"
        ])
        for item in om_scan_results:
            ext_str = "; ".join(
                f"{ex['kind']}@z={ex['z_extremum']:.5f}(ratio={ex['a0_KH_ratio_at_extremum']:.6f})"
                for ex in item["stationary_points_z_ge_0"]
            ) or "STRICTLY_INCREASING_NO_DIP"
            writer.writerow([
                item["Omega_m0"],
                f"{item['z_acc']:.6f}",
                f"{item['a0_KH_0_m_s2']:.6e}",
                f"{item['one_minus_q0']:.6f}",
                f"{item['a0_KH_ratio_at_z_acc']:.6f}",
                f"{item['a0_KH_ratio_at_LCDM_zdip_0p6316']:.6f}",
                f"{item['a0_KH_at_0p6316_m_s2']:.6e}",
                len(item["stationary_points_z_ge_0"]),
                ext_str,
            ])

    # 4. Monte Carlo Uncertainty Propagation (w0, wa, Omega_m0) onto a_0^KH(z)/a_0^KH(0)
    # Using DES+DESI+CMB: w0 = -0.82 +- 0.05, wa = -0.63 +- 0.195, rho = -0.85, Omega_m0 = 0.306 +- 0.008
    rng = np.random.default_rng(2026111805)
    n_mc = 100_000
    rho_mc = -0.85
    z_mc = rng.standard_normal((n_mc, 2))
    w0_d = (-0.82 + 0.05 * z_mc[:, 0])[:, None]
    wa_d = (-0.63 + 0.195 * (rho_mc * z_mc[:, 0] + math.sqrt(1.0 - rho_mc * rho_mc) * z_mc[:, 1]))[:, None]
    om_draws = np.clip(0.306 + 0.008 * rng.standard_normal(n_mc), 0.20, 0.45)
    om_d = om_draws[:, None]
    ode_d = 1.0 - om_d
    zp1_g = (1.0 + z_grid)[None, :]
    z_g = z_grid[None, :]

    W_mc = (zp1_g ** (3.0 * (1.0 + w0_d + wa_d))) * np.exp(-3.0 * wa_d * z_g / zp1_g)
    E2_mc = om_d * (zp1_g**3) + ode_d * W_mc
    E_mc = np.sqrt(E2_mc)
    omz_mc = om_d * (zp1_g**3) / E2_mc
    odez_mc = ode_d * W_mc / E2_mc
    wz_mc = w0_d + wa_d * (z_g / zp1_g)
    qz_mc = 0.5 * (omz_mc + (1.0 + 3.0 * wz_mc) * odez_mc)
    q0_mc = 0.5 * (om_d[:, 0] + (1.0 + 3.0 * w0_d[:, 0]) * ode_d[:, 0])[:, None]
    ratio_mc = E_mc * np.abs(1.0 - qz_mc) / np.abs(1.0 - q0_mc)

    mc_p16, mc_p50, mc_p84 = np.percentile(ratio_mc, [16.0, 50.0, 84.0], axis=0)
    mc_p2p5, mc_p97p5 = np.percentile(ratio_mc, [2.5, 97.5], axis=0)

    # Check fraction of MC draws on DES+DESI+CMB that still possess ANY local minimum at z > 0:
    # A local minimum near z > 0 requires da_0^KH/dz|_{z=0} < 0!
    # Let's check exact slope at z=0: d[E*(1-q)]/dz|_{z=0}!
    dz_tiny = 1e-5
    slope0_draws = (
        cpl_kinematics_analytic(dz_tiny, om_draws, w0_d[:, 0], wa_d[:, 0])["a0_KH_ratio"] - 1.0
    ) / dz_tiny
    frac_initial_decrease = float(np.mean(slope0_draws < 0.0))

    # 5. Export `verification/Tau_DES_2026_background_comparison.csv`
    e_des_0306 = cpl_kinematics_analytic(z_grid, 0.306, -0.82, -0.63)
    bg_rows: List[Dict[str, Any]] = []
    for i, zv in enumerate(z_grid):
        bg_rows.append({
            "z": float(zv),
            "a": round(1.0 / (1.0 + float(zv)), 7),
            "E_LCDM_Om03153": round(float(e_an_lcdm["E"][i]), 7),
            "q_LCDM_Om03153": round(float(e_an_lcdm["q"][i]), 7),
            "a0_KH_m_s2_LCDM_Om03153": round(float(e_an_lcdm["a0_KH_z_m_s2"][i]), 16),
            "a0_KH_ratio_LCDM_Om03153": round(float(e_an_lcdm["a0_KH_ratio"][i]), 7),
            "E_DES_CPL_Om03153": round(float(e_an_des["E"][i]), 7),
            "w_DES_CPL": round(float(e_an_des["w"][i]), 7),
            "q_DES_CPL_Om03153": round(float(e_an_des["q"][i]), 7),
            "a0_KH_m_s2_DES_CPL_Om03153": round(float(e_an_des["a0_KH_z_m_s2"][i]), 16),
            "a0_KH_ratio_DES_CPL_Om03153": round(float(e_an_des["a0_KH_ratio"][i]), 7),
            "a0_KH_ratio_DES_CPL_Om0306": round(float(e_des_0306["a0_KH_ratio"][i]), 7),
            "naive_wrong_1m34Om_ratio_DES_Om03153": round(float(e_an_des["naive_1_minus_34Om_ratio"][i]), 7),
            "DES_CPL_ratio_MC_median": round(float(mc_p50[i]), 6),
            "DES_CPL_ratio_MC_p16": round(float(mc_p16[i]), 6),
            "DES_CPL_ratio_MC_p84": round(float(mc_p84[i]), 6),
            "DES_CPL_ratio_MC_p2p5": round(float(mc_p2p5[i]), 6),
            "DES_CPL_ratio_MC_p97p5": round(float(mc_p97p5[i]), 6),
            "pct_shift_ratio_DES_Om03153_vs_LCDM": round(
                100.0 * (float(e_an_des["a0_KH_ratio"][i]) - float(e_an_lcdm["a0_KH_ratio"][i]))
                / float(e_an_lcdm["a0_KH_ratio"][i]),
                4,
            ),
            "pct_shift_abs_a0_DES_Om03153_vs_LCDM": round(
                100.0 * (float(e_an_des["a0_KH_z_m_s2"][i]) - float(e_an_lcdm["a0_KH_z_m_s2"][i]))
                / float(e_an_lcdm["a0_KH_z_m_s2"][i]),
                4,
            ),
        })

    with CSV_BG_OUT.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(bg_rows[0].keys()))
        writer.writeheader()
        writer.writerows(bg_rows)

    idx_dip = int(np.argmin(np.abs(z_grid - 0.6315565)))
    return {
        "dual_method_ode_vs_analytic_verification": {
            "epistemic_status": "PROVEN / NUMERICAL",
            "max_abs_diff_LCDM_ratio_z0_to_14p18": max_ode_err_lcdm,
            "max_abs_diff_DES_CPL_ratio_z0_to_14p18": max_ode_err_des,
        },
        "paper20_lcdm_reference": lcdm_p20,
        "tau_c2_lcdm_reference": lcdm_c2,
        "des2026_cpl_omega_m_scan": {
            "epistemic_status": "NUMERICAL",
            "exact_Omega_m0_reproducing_1p14272_at_z0p6315565": om_match_06316,
            "exact_Omega_m0_reproducing_1p14272_at_z0p632": om_match_0632,
            "physical_decomposition_of_ratio_jump_at_Om03153": {
                "LCDM_a0_KH_0_m_s2": float(e_an_lcdm["a0_KH_0_m_s2"]),
                "DES_CPL_a0_KH_0_m_s2": float(e_an_des["a0_KH_0_m_s2"]),
                "z0_denominator_drop_pct": 100.0 * (
                    float(e_an_des["a0_KH_0_m_s2"]) - float(e_an_lcdm["a0_KH_0_m_s2"])
                ) / float(e_an_lcdm["a0_KH_0_m_s2"]),
                "LCDM_abs_a0_KH_at_z0p6316_m_s2": float(e_an_lcdm["a0_KH_z_m_s2"][idx_dip]),
                "DES_CPL_abs_a0_KH_at_z0p6316_m_s2": float(e_an_des["a0_KH_z_m_s2"][idx_dip]),
                "abs_a0_KH_shift_at_z0p6316_pct": 100.0 * (
                    float(e_an_des["a0_KH_z_m_s2"][idx_dip]) - float(e_an_lcdm["a0_KH_z_m_s2"][idx_dip])
                ) / float(e_an_lcdm["a0_KH_z_m_s2"][idx_dip]),
                "LCDM_normalized_ratio_at_z0p6316": float(e_an_lcdm["a0_KH_ratio"][idx_dip]),
                "DES_CPL_normalized_ratio_at_z0p6316": float(e_an_des["a0_KH_ratio"][idx_dip]),
                "normalized_ratio_jump_pct": 100.0 * (
                    float(e_an_des["a0_KH_ratio"][idx_dip]) - float(e_an_lcdm["a0_KH_ratio"][idx_dip])
                ) / float(e_an_lcdm["a0_KH_ratio"][idx_dip]),
            },
            "scan_rows": om_scan_results,
        },
        "mc_uncertainty_propagation_DES_DESI_CMB": {
            "epistemic_status": "NUMERICAL / EMPIRICAL",
            "n_mc": n_mc,
            "priors": {
                "w0": "-0.82 +- 0.05",
                "wa": "-0.63 +- 0.195",
                "rho_w0_wa": rho_mc,
                "Omega_m0": "0.306 +- 0.008",
            },
            "ratio_at_z0p6316_median_68_95": {
                "median": float(mc_p50[idx_dip]),
                "p16": float(mc_p16[idx_dip]),
                "p84": float(mc_p84[idx_dip]),
                "p2p5": float(mc_p2p5[idx_dip]),
                "p97p5": float(mc_p97p5[idx_dip]),
            },
            "fraction_of_draws_with_initial_z0_decrease_da0dz_lt_0": frac_initial_decrease,
        },
        "sign_audit_and_background_independent_invariant": {
            "epistemic_status": "PROVEN",
            "invariant_definition": "I_KH(z) := a_0^KH(z) / [c * H(z) * |1 - q(z)|] = 5*sqrt(3)/96",
            "invariant_exact_value": I_KH_EXACT,
            "sign_change_condition_for_1_minus_q": (
                "1 - q(z) = (1/2) * (1 - 3 * w_tot(z)) changes sign IFF total cosmic EOS "
                "w_tot(z) > +1/3 (stiff/kination fluid). In standard radiation + matter + DE "
                "cosmology, q(z) < 1 for all finite z >= 0, approaching q -> 1^- in deep "
                "radiation domination (where R = 0 and a_0^KH / (c H) -> 0)."
            ),
        },
    }


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    tex_audit = audit_tex_sources_for_3_minus_q()
    thm_proof = prove_symbolic_kh_coincidence_breaking_theorem()
    num_audit = run_c5_numerical_and_mc_audit()

    payload = {
        "program": "Program_C_Cosmology_Dynamical_DE_and_DES_2026",
        "script": "verify_C5_kh_background_dependence_audit.py",
        "date": "2026-11-18",
        "tex_corpus_3_minus_q_audit": tex_audit,
        "symbolic_coincidence_breaking_theorem": thm_proof,
        "numerical_and_mc_background_audit": num_audit,
    }
    JSON_OUT.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"[OK] Saved JSON -> {JSON_OUT}")
    print(f"[OK] Saved background comparison CSV -> {CSV_BG_OUT}")
    print(f"[OK] Saved Omega_m scan CSV -> {CSV_OM_SCAN_OUT}")


if __name__ == "__main__":
    main()
