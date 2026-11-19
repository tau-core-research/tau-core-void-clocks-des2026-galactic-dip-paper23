#!/usr/bin/env python3
"""
generate_sub4_figures_and_tables.py
===================================
Bundled figure & LaTeX table generator for Sub4 v11.0.
Automatically invoked by `verify_C9_void_clock_timescape_apparent_w0_wa_audit.py`
(or callable standalone via `python3 generate_sub4_figures_and_tables.py`).
"""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from typing import Any, Dict, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import numpy as np
from scipy import integrate, interpolate, linalg, optimize

SUP_DIR = Path(__file__).resolve().parent
SUB4_DIR = SUP_DIR.parent
DATA_DIR = SUP_DIR / "data"
JSON_PATH = SUP_DIR / "C9_void_clock_timescape_apparent_w_results.json"
TABLES_TEX_OUT = SUP_DIR / "generated_tables_v8.tex"


def _add_cov_ellipse(ax, mean_x, mean_y, sx, sy, rho, n_chi2_2dof, facecolor, edgecolor, label=None, alpha=0.20, ls="-", lw=1.35):
    cov = np.array([[sx * sx, rho * sx * sy], [rho * sx * sy, sy * sy]])
    vals, vecs = np.linalg.eigh(cov)
    order = vals.argsort()[::-1]
    vals, vecs = vals[order], vecs[:, order]
    theta = np.degrees(np.arctan2(*vecs[:, 0][::-1]))
    width, height = 2.0 * np.sqrt(n_chi2_2dof * vals)
    ax.add_patch(Ellipse(
        xy=(mean_x, mean_y), width=width, height=height, angle=theta,
        facecolor=facecolor, edgecolor=edgecolor, lw=lw, ls=ls, alpha=alpha, label=label,
    ))
    ax.add_patch(Ellipse(
        xy=(mean_x, mean_y), width=width, height=height, angle=theta,
        facecolor="none", edgecolor=edgecolor, lw=lw, ls=ls,
    ))


def compute_apparent_cpl_loci() -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Compute exact joint (SN + DESI BAO + Planck18) apparent CPL (w_0^app, w_a^app) loci for BOTH
    H2-vol (p=1, alpha in [0, 0.75]) AND Linder (2015, Eq. 3) strictly non-phantom thawing (w_0 in [-1.0, -0.60])."""
    import verify_C9_void_clock_timescape_apparent_w0_wa_audit as c9
    data = c9.load_all_cosmological_datasets()
    st = c9.build_recfast_zstar_stencil()
    z_sn = data["sn_compilations"]["DES_Dovekie"]["zhd"]
    u_sn = np.log1p(z_sn)
    L_sn = data["sn_compilations"]["DES_Dovekie"]["L"]
    Linv_1 = data["sn_compilations"]["DES_Dovekie"]["Linv_1"]
    norm_1 = data["sn_compilations"]["DES_Dovekie"]["norm_1"]
    desi_u = np.array([math.log1p(zv) for zv, _, _ in data["desi13"]["rows"]])
    desi_z = np.array([zv for zv, _, _ in data["desi13"]["rows"]])
    desi_types = [qt for _, _, qt in data["desi13"]["rows"]]
    cho_desi = data["desi13"]["cho"]
    pl_invcov = data["planck"]["pl3_invcov"]

    u_nodes = np.linspace(0.0, math.log1p(1200.0), 1001)
    zp1_nodes = np.exp(u_nodes)
    a_nodes = 1.0 / zp1_nodes
    h_fid, ob_fid, oc_fid = 0.6736, 0.0223732, 0.1199938
    rec0 = c9.recfast_v15_cpl_observables(h_fid, ob_fid, oc_fid, -1.0, 0.0, zstar_stencil=st)
    om0_fid = rec0["Omega_m0"]
    fv0_fid = 0.7239
    H0_si0 = 100.0 * h_fid * 1000.0 / c9.MPC_M
    Om_g0 = c9.RHO_G0_SI / (3.0 * H0_si0**2 / (8.0 * math.pi * c9.G_SI))
    Om_r0 = Om_g0 * (1.0 + 0.2271073 * 3.044)
    om_de0 = 1.0 - om0_fid - Om_r0
    Eh0 = np.sqrt(Om_r0 * zp1_nodes**4 + om0_fid * zp1_nodes**3 + om_de0)
    D_hor0 = integrate.cumulative_trapezoid(zp1_nodes / Eh0, u_nodes, initial=0.0)
    _, sdD_ts, sfv_ts, _ = c9.build_vectorized_wiltshire_splines(fv0_fid)
    u_star0 = math.log1p(rec0["z_star"])
    dh0 = c9.C_KMS / (100.0 * h_fid)

    def _project_to_cpl(spl_D_syn, spl_dH_syn, dm_star_syn, prev_p):
        mu_syn = 5.0 * np.log10((1.0 + z_sn) * spl_D_syn(u_sn))
        dm_b0, dh_b0 = spl_D_syn(desi_u), spl_dH_syn(desi_u)
        bao_syn = (dh0 / rec0["r_d_zdrag_Mpc"]) * np.array([
            dm_b0[i] if qt == "DM_over_rs" else (dh_b0[i] if qt == "DH_over_rs" else (desi_z[i] * dm_b0[i]**2 * dh_b0[i])**(1.0/3.0))
            for i, qt in enumerate(desi_types)
        ])
        cmb_syn = np.array([ob_fid, oc_fid, 100.0 * rec0["r_s_zstar_Mpc"] / dm_star_syn])

        def cpl_syn_chi2(x):
            h, ob, oc, w0_v, wa_v = float(x[0]), float(x[1]), float(x[2]), float(x[3]), float(x[4])
            if h <= 0.56 or h >= 0.78 or ob <= 0.020 or ob >= 0.025 or oc <= 0.095 or oc >= 0.155 or (1.0 + w0_v + wa_v) > 0.8:
                return 1e9
            rc = c9.recfast_v15_cpl_observables(h, ob, oc, w0_v, wa_v, zstar_stencil=st)
            om = rc["Omega_m0"]
            H0s = 100.0 * h * 1000.0 / c9.MPC_M
            Or = (c9.RHO_G0_SI / (3.0 * H0s**2 / (8.0 * math.pi * c9.G_SI))) * (1.0 + 0.2271073 * 3.044)
            de_n = (zp1_nodes ** (3.0 * (1.0 + w0_v + wa_v))) * np.exp(-3.0 * wa_v * (1.0 - a_nodes))
            En = np.sqrt(Or * zp1_nodes**4 + om * zp1_nodes**3 + (1.0 - om - Or) * de_n)
            Dn = integrate.cumulative_trapezoid(zp1_nodes / En, u_nodes, initial=0.0)
            sD = interpolate.CubicSpline(u_nodes, Dn)
            sdH = interpolate.CubicSpline(u_nodes, 1.0 / En)
            diff_sn = mu_syn - 5.0 * np.log10((1.0 + z_sn) * np.maximum(sD(u_sn), 1e-30))
            Ld = linalg.solve_triangular(L_sn, diff_sn, lower=True)
            rsn = Ld - (float(np.dot(Linv_1, Ld)) / norm_1) * Linv_1
            c_sn = float(np.dot(rsn, rsn))
            dmb, dhb = sD(desi_u), sdH(desi_u)
            dh_m = c9.C_KMS / (100.0 * h)
            b_mod = (dh_m / rc["r_d_zdrag_Mpc"]) * np.array([
                dmb[i] if qt == "DM_over_rs" else (dhb[i] if qt == "DH_over_rs" else (desi_z[i] * dmb[i]**2 * dhb[i])**(1.0/3.0))
                for i, qt in enumerate(desi_types)
            ])
            rb = b_mod - bao_syn
            c_bao = float(np.dot(rb, linalg.cho_solve(cho_desi, rb)))
            vc = np.array([ob, oc, rc["theta_100"]]) - cmb_syn
            c_cmb = float(vc @ (pl_invcov @ vc))
            return c_sn + c_bao + c_cmb

        sol = optimize.minimize(cpl_syn_chi2, prev_p, method="L-BFGS-B",
                                bounds=[(0.60, 0.75), (0.021, 0.024), (0.105, 0.140), (-1.2, -0.4), (-2.0, 0.5)])
        return sol.x

    # 1. H2-vol (p=1) locus:
    alphas = np.linspace(0.0, 0.75, 16)
    w0s, was = [], []
    prev_p = np.array([h_fid, ob_fid, oc_fid, -1.0, 0.0])
    for av in alphas:
        if av == 0.0:
            w0s.append(-1.0); was.append(0.0); continue
        aeff = av * (sfv_ts(u_nodes) / fv0_fid)
        delta_dD = aeff * (sdD_ts(u_nodes) - 1.0 / Eh0)
        dD_syn = 1.0 / Eh0 + delta_dD
        delta_D = integrate.cumulative_trapezoid(zp1_nodes * delta_dD, u_nodes, initial=0.0)
        D_syn = D_hor0 + delta_D
        dm_star_syn = rec0["D_M_zstar_Mpc"] + dh0 * float(np.interp(u_star0, u_nodes, delta_D))
        prev_p = _project_to_cpl(interpolate.CubicSpline(u_nodes, D_syn), interpolate.CubicSpline(u_nodes, dD_syn), dm_star_syn, prev_p)
        w0s.append(float(prev_p[3])); was.append(float(prev_p[4]))

    # 2. Exact Linder (2015, Eq. 3, b=0.5, p=1) non-phantom algebraic thawing CPL projection:
    w0_l15_in = np.linspace(-1.0, -0.60, 9)
    w0_l15_proj, wa_l15_proj = [], []
    prev_l = np.array([h_fid, ob_fid, oc_fid, -1.0, 0.0])
    for w0_v in w0_l15_in:
        if w0_v == -1.0:
            w0_l15_proj.append(-1.0); wa_l15_proj.append(0.0); continue
        b_l = 0.5
        de_l = np.exp(3.0 * (1.0 + w0_v) * ((1.0 + b_l) - ((1.0 + b_l)**(2.0/3.0)) * ((a_nodes**3 + b_l)**(1.0/3.0))))
        E_l = np.sqrt(Om_r0 * zp1_nodes**4 + om0_fid * zp1_nodes**3 + om_de0 * de_l)
        D_l = integrate.cumulative_trapezoid(zp1_nodes / E_l, u_nodes, initial=0.0)
        dm_star_l = rec0["D_M_zstar_Mpc"] + dh0 * float(np.interp(u_star0, u_nodes, D_l - D_hor0))
        prev_l = _project_to_cpl(interpolate.CubicSpline(u_nodes, D_l), interpolate.CubicSpline(u_nodes, 1.0 / E_l), dm_star_l, prev_l)
        w0_l15_proj.append(float(prev_l[3])); wa_l15_proj.append(float(prev_l[4]))

    return alphas, np.array(w0s), np.array(was), np.array(w0_l15_proj), np.array(wa_l15_proj)


def compute_deduplicated_sightline_bins() -> Tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """Compute 5 equal-count bins of z-detrended Hubble residuals across N = 280 deduplicated unique SNe."""
    import verify_C9_void_clock_timescape_apparent_w0_wa_audit as c9
    data = c9.load_all_cosmological_datasets()
    pan = data["sn_compilations"]["PantheonPlus"]
    zhd, zhel, mb, L_hf = pan["zhd"], pan["zhel"], pan["y"], pan["L"]
    cov_hf = np.einsum("ik,jk->ij", L_hf, L_hf)

    ra_hf = np.zeros(len(zhd)); dec_hf = np.zeros(len(zhd)); cid_hf = [""] * len(zhd); ptr = 0
    with (DATA_DIR / "Pantheon+SH0ES.dat").open("r", encoding="utf-8") as f:
        for r in csv.DictReader(f, delimiter=" ", skipinitialspace=True):
            if ptr < len(zhd) and abs(float(r["zHD"]) - zhd[ptr]) < 1e-7 and abs(float(r["m_b_corr"]) - mb[ptr]) < 1e-5:
                ra_hf[ptr] = float(r["RA"]); dec_hf[ptr] = float(r["DEC"]); cid_hf[ptr] = str(r["CID"]); ptr += 1

    r_sn = 2997.92458 * c9.H0_D_CPL_vec(zhd, -1.0, 0.0, 0.3153)
    n_hat = np.column_stack([
        np.cos(np.deg2rad(dec_hf)) * np.cos(np.deg2rad(ra_hf)),
        np.cos(np.deg2rad(dec_hf)) * np.sin(np.deg2rad(ra_hf)),
        np.sin(np.deg2rad(dec_hf)),
    ])
    holes = data["voidfinder_sdss_dr7"]["holes"]
    max_tbl = data["voidfinder_sdss_dr7"]["maximal"]
    centers = holes[:, :3]; radii2 = holes[:, 3] ** 2; c_norm2 = np.sum(centers**2, axis=1)
    ra_v, dec_v = max_tbl[:, 7], max_tbl[:, 8]
    v_dirs = np.column_stack([
        np.cos(np.deg2rad(dec_v)) * np.cos(np.deg2rad(ra_v)),
        np.cos(np.deg2rad(dec_v)) * np.sin(np.deg2rad(ra_v)),
        np.sin(np.deg2rad(dec_v)),
    ])
    cos_mat = np.einsum("id,jd->ij", n_hat, v_dirs)
    in_ngc = (np.max(cos_mat, axis=1) >= math.cos(math.radians(4.5))) & (ra_hf >= 125.0) & (ra_hf <= 245.0) & (dec_hf >= 3.0) & (dec_hf <= 58.0)

    res_raw = mb - 5.0 * np.log10((1.0 + zhel) * c9.H0_D_CPL_vec(zhd, -1.0, 0.0, 0.3326))

    def dedup(idx):
        uc = []; gr = {}
        for p, ri in enumerate(idx):
            c = cid_hf[ri]
            if c not in gr:
                uc.append(c); gr[c] = []
            gr[c].append(p)
        cs = cov_hf[np.ix_(idx, idx)]; vd = np.diag(cs)
        W = np.zeros((len(uc), len(idx)))
        zu = np.zeros(len(uc)); ru = np.zeros(len(uc)); ud = np.zeros((len(uc), 3))
        for ui, c in enumerate(uc):
            g = gr[c]; iv = 1.0 / np.maximum(vd[g], 1e-8); w = iv / np.sum(iv)
            W[ui, g] = w
            zu[ui] = float(np.sum(w * zhd[idx[g]]))
            ru[ui] = float(np.sum(w * r_sn[idx[g]]))
            dm = np.sum(w[:, None] * n_hat[idx[g]], axis=0); ud[ui] = dm / np.linalg.norm(dm)
        return zu, ru, np.einsum("ui,i->u", W, res_raw[idx]), np.einsum("ui,ij,vj->uv", W, cs, W), ud

    def trace_1d(unit_dirs, s_min, s_max):
        out = np.zeros(len(unit_dirs))
        for i in range(len(unit_dirs)):
            s0, s1 = float(s_min[i]), float(s_max[i])
            sp = np.einsum("md,d->m", centers, unit_dirs[i])
            d2 = c_norm2 - sp * sp
            hit = np.where(d2 < radii2)[0]
            if len(hit) == 0:
                continue
            hc = np.sqrt(radii2[hit] - d2[hit])
            aa = np.maximum(s0, sp[hit] - hc); bb = np.minimum(s1, sp[hit] + hc)
            val = bb > aa
            if not np.any(val):
                continue
            ivs = sorted(zip(aa[val], bb[val]))
            tot = 0.0; ca, cb = ivs[0]
            for na, nb in ivs[1:]:
                if na <= cb:
                    cb = max(cb, nb)
                else:
                    tot += cb - ca; ca, cb = na, nb
            tot += cb - ca
            out[i] = tot / (s1 - s0)
        return out

    idx_in = np.where(in_ngc & (r_sn >= 60.0) & (r_sn <= 305.0))[0]
    idx_bg = np.where(in_ngc & (r_sn > 320.0))[0]
    z_in, r_in, y_in0, cov_in, u_in = dedup(idx_in)
    z_bg, _, y_bg0, cov_bg, u_bg = dedup(idx_bg)
    fv_in = trace_1d(u_in, np.full(len(u_in), 30.0), r_in)
    fv_bg = trace_1d(u_bg, np.full(len(u_bg), 30.0), np.full(len(u_bg), 300.0))

    def detrend(zu, yu, covu):
        cho = linalg.cho_factor(covu, lower=True)
        X = np.column_stack([np.ones(len(zu)), zu - np.mean(zu)])
        CX = linalg.cho_solve(cho, X); Cy = linalg.cho_solve(cho, yu)
        b = linalg.inv(np.einsum("ni,nj->ij", X, CX)) @ np.einsum("ni,n->i", X, Cy)
        return yu - X @ b, np.diag(covu)

    yd_in, v_in = detrend(z_in, y_in0, cov_in)
    yd_bg, v_bg = detrend(z_bg, y_bg0, cov_bg)
    fv_all = np.concatenate([fv_in, fv_bg])
    y_all = np.concatenate([yd_in, yd_bg])
    w_all = 1.0 / np.concatenate([v_in, v_bg])

    order = np.argsort(fv_all)
    splits = np.array_split(order, 5)
    bx, by, be = [], [], []
    for sp_idx in splits:
        ww = w_all[sp_idx]
        bx.append(float(np.mean(fv_all[sp_idx])))
        by.append(float(np.sum(ww * y_all[sp_idx]) / np.sum(ww)))
        be.append(float(1.0 / math.sqrt(np.sum(ww))))
    return np.array(bx), np.array(by), np.array(be), float(np.mean(fv_all))


def make_figure(lang: str, res_json: Dict[str, Any], locus_data, bin_data) -> Path:
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 8.4,
        "axes.titlesize": 9.1,
        "axes.labelsize": 8.6,
        "legend.fontsize": 6.85,
    })
    fig, axes = plt.subplots(1, 3, figsize=(14.5, 4.4), dpi=300)
    fig.subplots_adjust(left=0.052, right=0.988, bottom=0.155, top=0.885, wspace=0.255)

    pl3 = res_json["section_3_multi_compilation_joint_BAO_CMB_SN_fits"]["primary_CAMB_rd_Planck2018_3x3_MCMC_omegab_omegac_100thetastar"]
    sn_prof = res_json["section_1_wiltshire_benchmarks_and_recombination_scaling"]["pantheon_plus_sn_only_alpha_profile"]

    # --- PANEL A: Profile Delta_chi^2(alpha) across 4 SN compilations ---
    ax = axes[0]
    a_sn = [r["alpha"] for r in sn_prof]
    d_sn = [r["H2_vol_p1_Delta_chi2_SN"] for r in sn_prof]
    spl_sn = interpolate.CubicSpline(a_sn, d_sn)
    ag_fine = np.linspace(0.0, 1.0, 200)
    ax.plot(ag_fine, spl_sn(ag_fine), color="#7f8c8d", ls="--", lw=1.45,
            label=("Pantheon+ only (no BAO/CMB)" if lang == "en" else "Csak Pantheon+ (BAO/CMB nélkül)"))

    def _lbl(name_disp: str, key: str) -> str:
        fk = pl3[key]["H2_vol_p1_free_alpha_k5"]
        dc = fk["Delta_chi2_vs_LCDM"]
        ah = fk["alpha_hat"]
        sa = fk["sigma_alpha"]
        if lang == "en":
            return f"{name_disp} (Δχ²={dc:+.2f}, α̂={ah:.2f}±{sa:.2f})"
        return f"{name_disp} (Δχ²={dc:+.2f}; α̂={ah:.2f}±{sa:.2f})".replace(".", ",")

    styles = [
        ("PantheonPlus", "#1f77b4", "-", _lbl("Pantheon+", "PantheonPlus")),
        ("DES_Dovekie", "#009e73", "-.", _lbl("DES-Dovekie", "DES_Dovekie")),
        ("Union3", "#ff7f0e", ":", _lbl("Union3", "Union3")),
        ("DES_SN5YR_2024", "#d62728", "-", _lbl("DES-SN5YR", "DES_SN5YR_2024")),
    ]
    for key, col, ls, lbl in styles:
        rows = pl3[key]["alpha_profile_grid"]
        xa = [r["alpha"] for r in rows]
        yd = [r["Delta_chi2_vs_LCDM"] for r in rows]
        spl = interpolate.CubicSpline(xa, yd)
        ax.plot(ag_fine, spl(ag_fine), color=col, ls=ls, lw=2.0, label=lbl)
        fit_k = pl3[key]["H2_vol_p1_free_alpha_k5"]
        ax.scatter([fit_k["alpha_hat"]], [fit_k["Delta_chi2_vs_LCDM"]], color=col, s=28, zorder=5)

    ax.axhline(0.0, color="black", lw=0.8, alpha=0.5)
    a_min = pl3["PantheonPlus"]["H2_vol_p1_free_alpha_k5"]["alpha_hat"] - pl3["PantheonPlus"]["H2_vol_p1_free_alpha_k5"]["sigma_alpha"]
    a_max = pl3["Union3"]["H2_vol_p1_free_alpha_k5"]["alpha_hat"] + pl3["Union3"]["H2_vol_p1_free_alpha_k5"]["sigma_alpha"]
    ax.axvspan(a_min, a_max, color="#1f77b4", alpha=0.07)
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(-21.5, 13.5)
    ax.set_xlabel("Void-volume optical weight α" if lang == "en" else "Üreg-térfogati optikai súly α")
    ax.set_ylabel("Profile Δχ²(α) vs. flat ΛCDM (α = 0)" if lang == "en" else "Profil Δχ²(α) a sík ΛCDM-hez (α = 0) képest")
    ax.set_title(r"(A) Multi-Compilation Profile $\Delta\chi^2(\alpha)$ (CAMB $r_d$)" if lang == "en" else r"(A) Többkatalógusos $\Delta\chi^2(\alpha)$ profilok (CAMB $r_d$)", fontweight="bold")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="upper left", frameon=True, framealpha=0.92)

    # --- PANEL B: 2D CPL Hessian Contours vs H2-vol, Thawing Quintessence & CPL Degeneracy Line ---
    ax = axes[1]
    cpl_sets = [
        ("PantheonPlus", "#1f77b4", "Pantheon+ + DESI + Planck18"),
        ("DES_Dovekie", "#009e73", "DES-Dovekie + DESI + Planck18"),
        ("DES_SN5YR_2024", "#d62728", "DES-SN5YR + DESI + Planck18"),
    ]
    for key, col, lbl in cpl_sets:
        m = pl3[key]["M1b_Free_CPL_k6"]
        _add_cov_ellipse(ax, m["w0_hat"], m["wa_hat"], m["sigma_w0_hessian"], m["sigma_wa_hessian"], m["rho_w0_wa_hessian"], 2.2957, col, col, label=lbl + " (68%/95%)", alpha=0.17, lw=1.35)
        _add_cov_ellipse(ax, m["w0_hat"], m["wa_hat"], m["sigma_w0_hessian"], m["sigma_wa_hessian"], m["rho_w0_wa_hessian"], 6.1801, "none", col, alpha=0.0, lw=0.85, ls="--")

    # Exact joint CPL projection of Linder (2015, PRD 91, 063006, Eq. 3) non-phantom algebraic thawing:
    alphas_loc, w0_loc, wa_loc, w0_l15, wa_l15 = locus_data
    ax.plot(w0_l15, wa_l15, color="#8e44ad", ls=":", lw=1.85, zorder=5,
            label=("Linder (2015) non-phantom thawing (k=5)" if lang == "en" else "Linder (2015) nem-fantom thawing (k=5)"))
    w0_line = np.linspace(-1.0, -0.60, 100)
    ax.plot(w0_line, -3.30 * (1.0 + w0_line), color="#7f8c8d", ls="-.", lw=1.55, zorder=5,
            label=("Phantom-crossing line wₐ = −3.3(1+w₀) (k=5)" if lang == "en" else "Fantom-átlépő egyenes wₐ = −3,3(1+w₀) (k=5)"))

    ax.plot(w0_loc, wa_loc, color="#111111", lw=2.15, zorder=6,
            label=("H2-vol (p=1) locus α ∈ [0, 0.75] (k=5)" if lang == "en" else "H2-vol (p=1) görbe α ∈ [0; 0,75] (k=5)"))
    for a_mark, col_m in [(0.40, "#1f77b4"), (0.52, "#d62728")]:
        idx_m = int(np.argmin(np.abs(alphas_loc - a_mark)))
        ax.scatter([w0_loc[idx_m]], [wa_loc[idx_m]], color=col_m, edgecolor="black", s=44, marker="D", zorder=7)
    ax.scatter([-1.0], [0.0], color="black", s=52, marker="s", zorder=8, label="ΛCDM (−1, 0)")
    ax.axvline(-1.0, color="gray", lw=0.7, ls=":")
    ax.axhline(0.0, color="gray", lw=0.7, ls=":")
    ax.set_xlim(-1.06, -0.60)
    ax.set_ylim(-1.45, 0.25)
    ax.set_xlabel("Apparent / Effective w₀" if lang == "en" else "Látszólagos / effektív w₀")
    ax.set_ylabel("Apparent / Effective wₐ" if lang == "en" else "Látszólagos / effektív wₐ")
    ax.set_title("(B) CPL Contours vs. k = 5 DE & Void Curves" if lang == "en" else "(B) CPL kontúrok vs. k = 5 DE és üreg-görbék", fontweight="bold")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="lower left", frameon=True, framealpha=0.92)

    # --- PANEL C: Multi-Catalog 3D Void Sightline GLS Measurements (SDSS DR7 + Deep DESI DR1) ---
    ax = axes[2]
    bx, by, be, f_mean = bin_data
    sl_res = res_json["section_4_3d_cosmic_void_sightline_exploratory_null_check"]
    comb = sl_res["combined_deduplicated_NGC_interior_SNe"]
    sl_emp = comb["z_controlled_GLS_slope_dmu_dfv_mag"]
    esl_emp = comb["sigma_slope_mag"]
    n_uniq = comb["N_unique_SNe"]
    deep_d = sl_res["deep_DESI_DR1_3d_void_sightline_tests"]["ASTRA_DESI_DR1_BGS_LRG_0p10_le_z_le_0p95"]
    sl_desi = deep_d["z_controlled_GLS_slope_dmu_dfv_mag"]
    esl_desi = deep_d["sigma_slope_mag"]
    n_desi = deep_d["N_unique_SNe"]
    th = sl_res["theoretical_H2_vol_scale_and_exploratory_null_status"]
    sl_th_deep = th["predicted_slope_dmu_dfv_deep_z0p70_tracker_mag"]

    xf = np.linspace(0.12, 0.92, 100)
    dx = xf - f_mean
    ax.fill_between(xf, (sl_emp - esl_emp) * dx * 1000.0, (sl_emp + esl_emp) * dx * 1000.0, color="#1f77b4", alpha=0.14)
    ax.plot(xf, sl_emp * dx * 1000.0, color="#1f77b4", lw=1.9,
            label=(f"SDSS DR7 z≤0.105 GLS ({sl_emp*1000:+.0f}±{esl_emp*1000:.0f} mmag, N={n_uniq})"))
    ax.plot(xf, sl_desi * dx * 1000.0, color="#009e73", lw=1.85, ls="--",
            label=(f"DESI DR1 z≤0.95 GLS ({sl_desi*1000:+.0f}±{esl_desi*1000:.0f} mmag, N={n_desi})"))
    ax.plot(xf, sl_th_deep * dx * 1000.0, color="#d62728", lw=1.45, ls=":",
            label=(f"Conditional directional z≤0.70 ({sl_th_deep*1000:+.0f} mmag)" if lang == "en"
                   else f"Feltételes irányfüggő z≤0,70 ({sl_th_deep*1000:+.0f} mmag)"))
    ax.errorbar(bx, by * 1000.0, yerr=be * 1000.0, fmt="o", color="#222222", capsize=3, ms=5.0, zorder=6,
                label=(f"SDSS DR7 unique SNe (N = {n_uniq}, 5 bins)" if lang == "en" else f"SDSS DR7 egyedi SNe (N = {n_uniq}, 5 bin)"))
    ax.axhline(0.0, color="black", lw=0.8, alpha=0.5)
    ax.set_xlim(0.12, 0.92)
    ax.set_ylim(-68.0, 58.0)
    ax.set_xlabel(r"3D ray-traced void path fraction $f_{v,\mathrm{LOS}}$" if lang == "en" else r"3D sugárkövetett üreg-útarány $f_{v,\mathrm{LOS}}$")
    ax.set_ylabel("Detrended Hubble residual Δμ [mmag]" if lang == "en" else "Detrendelt Hubble-maradék Δμ [mmag]")
    ax.set_title("(C) 3D Void Sightline GLS Checks (SDSS + DESI DR1)" if lang == "en" else "(C) 3D üreg-látóirány GLS-tesztek (SDSS + DESI DR1)", fontweight="bold")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="lower left", frameon=True, framealpha=0.92)

    out_fn = f"fig_sub4_void_clock_des2026_and_galactic_dip_{lang}.png"
    out_p = SUB4_DIR / out_fn
    fig.savefig(out_p, dpi=300)
    plt.close(fig)
    return out_p


def write_latex_tables_snippet(res_json: Dict[str, Any]) -> Path:
    """Export machine-generated LaTeX table rows directly from JSON so zero manual transcription is needed."""
    pl3 = res_json["section_3_multi_compilation_joint_BAO_CMB_SN_fits"]["primary_CAMB_rd_Planck2018_3x3_MCMC_omegab_omegac_100thetastar"]
    lines = ["% Auto-generated by generate_sub4_figures_and_tables.py from C9_void_clock_timescape_apparent_w_results.json"]
    for sn_k, block in pl3.items():
        lines.append(f"% === {sn_k} ===")
        for m_k in [
            "M0_Flat_LCDM_k4",
            "H2_vol_p1_free_alpha_k5",
            "H2_vol_p2_free_alpha_k5",
            "Const_Weight_H2a_p0_free_alpha_k5",
            "Smooth_wCDM_k5",
            "Smooth_Linder2015_NonPhantom_Thawing_k5",
            "Smooth_PhantomCrossing_CPL_DegLine_k5",
            "M1b_Free_CPL_k6",
        ]:
            r = block[m_k]
            lines.append(
                f"% {m_k}: k={r['k_params']}, H0={r['H0_kms_Mpc']:.2f}, Om0={r['Omega_m0']:.4f}, "
                f"chi2={r['chi2_total']:.2f}, dchi2={r['Delta_chi2_vs_LCDM']:+.2f}, "
                f"dAIC={r['Delta_AIC_vs_LCDM']:+.2f}, dBIC={r['Delta_BIC_vs_LCDM']:+.2f}, dBIC_CPL={r['Delta_BIC_vs_CPL']:+.2f}"
            )
    TABLES_TEX_OUT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return TABLES_TEX_OUT


def regenerate_all_from_json(res_json: Dict[str, Any] | None = None) -> None:
    if res_json is None:
        res_json = json.loads(JSON_PATH.read_text(encoding="utf-8"))
    loc = compute_apparent_cpl_loci()
    bins = compute_deduplicated_sightline_bins()
    p_en = make_figure("en", res_json, loc, bins)
    p_hu = make_figure("hu", res_json, loc, bins)
    p_tb = write_latex_tables_snippet(res_json)
    print(f"[PASS] Regenerated figures & LaTeX table summary:\n  - {p_en}\n  - {p_hu}\n  - {p_tb}")


if __name__ == "__main__":
    regenerate_all_from_json()
