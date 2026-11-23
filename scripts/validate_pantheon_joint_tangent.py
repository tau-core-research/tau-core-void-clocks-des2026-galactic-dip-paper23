#!/usr/bin/env python3
"""One-catalog compressed-geometry refit and matched tangent-bank null test.

Python + numpy + scipy only. Pass the *working* Sub4 supplementary directory
as --source. Read-only with respect to upstream sources; output is exclusive.
This is a plug-in linear-Gaussian null, not nonlinear mock-survey inference.
"""

import argparse
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import sys

import numpy as np
import scipy
from scipy import integrate, interpolate, linalg, optimize, sparse, stats

MODEL = ("LCDM", "p0", "p1", "p2", "mirage")
OPTICAL = MODEL[1:4]
FILES = (
    "desi_gaussian_bao_ALL_GCcomb_mean.txt",
    "desi_gaussian_bao_ALL_GCcomb_cov.txt",
    "planck2018_compressed_priors.json",
    "pantheon_plus_hf1590_stat_sys_cholesky.npz",
    "camb205_3d_emulator_grid.npz",
)
HELPER = "verify_C9_void_clock_timescape_apparent_w0_wa_audit.py"
EXPECTED = (
    "9ac154ab583ce759c0f7eef3c978c7c70a6ead2d18774caceadf1a350a640585",
    "252a143274c8a07c78694c119617d36594f6d7965d00319ca611c6ffb886e509",
    "fd31b6910518c8a840d64b9b978b9ca81977c38912ba0a03a756f8e86e9a6385",
    "5ed92c8a836d63f646316da418baf00be24fedef02ff58082269a04beb87c761",
    "9eca7b504fc913f721b586d8358c380bdc388c5130aa8a1f0d678f60621e8524",
    "9a0615f0244553df7e0b4e84770a1ee9759dd622099c3493750a7f8ea1ee65a6",
)
BASE_LO = np.array([0.560001, 0.020001, 0.090001])
BASE_HI = np.array([0.794999, 0.024799, 0.164999])
INITIAL = np.array([0.682, 0.0224, 0.118])
DIFF = np.array([5e-5, 2e-6, 2e-5, 1e-3])


def digest(path):
    sha = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            sha.update(chunk)
    return sha.hexdigest()


class Experiment:
    def __init__(self, source):
        d = source / "data"
        paths = [d / name for name in FILES] + [source / HELPER]
        hashes = tuple(map(digest, paths))
        if hashes != EXPECTED:
            raise ValueError("A frozen input hash changed; do not compare with this receipt")
        self.hashes = {p.name: h for p, h in zip(paths, hashes)}
        spec = importlib.util.spec_from_file_location("frozen_c9", source / HELPER)
        self.c9 = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(self.c9)
        with np.load(d / FILES[3], allow_pickle=False) as sn:
            self.sn_u = np.log1p(sn["zhd"])
            self.sn_zhel = sn["zhel"]
            self.sn_y = sn["mb"]
            self.sn_L = sn["L_lower"]
        self.offset = linalg.solve_triangular(self.sn_L, np.ones(len(self.sn_y)), lower=True)
        self.offset /= linalg.norm(self.offset)
        rows = np.loadtxt(d / FILES[0], dtype=str, skiprows=1)
        self.bao_z = rows[:, 0].astype(float)
        self.bao_u = np.log1p(self.bao_z)
        self.bao_kind = rows[:, 2]
        self.bao_y = rows[:, 1].astype(float)
        self.bao_L = linalg.cholesky(np.loadtxt(d / FILES[1]), lower=True)
        c = json.loads((d / FILES[2]).read_text())["planck2018_mcmc_3x3_omegab_omegac_100thetastar"]
        self.cmb_L = linalg.cholesky(c["cov"], lower=True)
        self.y = np.r_[self.sn_y, self.bao_y, c["mean"]]
        assert (len(self.sn_y), len(self.bao_y), self.cmb_L.shape) == (1580, 13, (3, 3))
        with np.load(d / FILES[4], allow_pickle=False) as grid:
            axes = tuple(grid[key] for key in ("h_g", "ob_g", "oc_g"))
            values = np.stack([grid[key] for key in ("zs", "rs", "rd", "dm")], axis=-1)
            self.acoustics = interpolate.RegularGridInterpolator(
                axes, values, method="cubic", bounds_error=True, solver=sparse.linalg.spsolve)
        self.f_acc = optimize.brentq(lambda f: 8*f**3 + 39*f*f - 12*f - 8, .4, .8)
        self.units = (self.c9.C_KMS, self.c9.MPC_M, self.c9.G_SI, self.c9.RHO_G0_SI)

    def white(self, x):
        sn, bao, cmb = np.split(x, (len(self.sn_y), len(self.sn_y) + len(self.bao_y)))
        snw = linalg.solve_triangular(self.sn_L, sn, lower=True, check_finite=False)
        # Explicit reductions avoid spurious Accelerate floating-status flags.
        coeff = np.einsum("i,i...->...", self.offset, snw)
        norm_proj = self.offset * coeff if snw.ndim == 1 else self.offset[:, None] * coeff
        return np.concatenate([
            snw - norm_proj,
            linalg.solve_triangular(self.bao_L, bao, lower=True),
            linalg.solve_triangular(self.cmb_L, cmb, lower=True),
        ], axis=0)

    def forward(self, mode, theta, n):
        h, ob, oc = theta[:3]
        h0, c = 100*h, self.units[0]
        zstar, rs, rd, dmstar = self.acoustics(np.array([theta[:3]]))[0]
        matter = (ob + oc + .06/93.14) / h**2
        rho = 3*(1000*h0/self.units[1])**2/(8*np.pi*self.units[2])
        rad = self.units[3]/rho * (1 + .2271073*3.044)
        dark = 1 - matter - rad
        u = np.linspace(0, np.log1p(1200), n+1)
        e = np.exp(u)
        eh2 = rad*e**4 + matter*e**3 + dark
        null = eh2**(-.5)
        dh = null
        if mode in OPTICAL and theta[3] != 0:
            fa = self.f_acc
            zp = (2*(1-matter)/matter)**(1/3)
            fv0 = optimize.brentq(
                lambda f: (2+fa)/(2+f)*(f/fa)**(2/3)*(1-fa)/(1-f)-zp,
                .05, .999, xtol=1e-13)
            _, tracker_dh, tracker_f, _ = self.c9.build_vectorized_wiltshire_splines(fv0)
            dh = null + theta[3]*(tracker_f(u)/fv0)**int(mode[1])*(tracker_dh(u)-null)
        if mode == "mirage" and theta[3] != 0:
            dw = theta[3]
            wa = -3.3*dw
            dh = (rad*e**4 + matter*e**3
                  + dark*np.exp(3*(dw+wa)*u - 3*wa*(1-1/e)))**(-.5)
        if (dh <= 0).any() or not np.isfinite(dh).all():
            raise ValueError("outside monotone-distance domain")
        dist = integrate.cumulative_trapezoid(e*dh, u, initial=0)
        hs = interpolate.CubicHermiteSpline(u, dist, e*dh)
        dm = hs(self.bao_u)
        dr = np.interp(self.bao_u, u, dh)
        dv = np.cbrt(self.bao_z*dm**2*dr)
        bao = c/h0/rd*np.where(self.bao_kind == "DM_over_rs", dm,
                                np.where(self.bao_kind == "DH_over_rs", dr, dv))
        star_shift = integrate.cumulative_trapezoid(e*(dh-null), u, initial=0)
        th = 100*rs/(dmstar + c/h0*np.interp(np.log1p(zstar), u, star_shift))
        return np.r_[5*np.log10((1+self.sn_zhel)*hs(self.sn_u)), bao, ob, oc, th]

    def fit(self, mode, n, null_start=None):
        if mode == "LCDM":
            starts = [INITIAL, np.array([.71, .0222, .118])]
            lo, hi = BASE_LO, BASE_HI
        else:
            lo, hi = (np.r_[BASE_LO, -.30], np.r_[BASE_HI, .50]) if mode == "mirage" else (
                np.r_[BASE_LO, -.60], np.r_[BASE_HI, 2.20])
            starts = [np.r_[null_start, a] for a in (0., .07, -.05)]

        def err(x):
            try:
                return self.white(self.y-self.forward(mode, x, n))
            except ValueError:
                return np.full_like(self.y, 1e6)

        results = [optimize.least_squares(err, x, bounds=(lo, hi),
                                        x_scale="jac", diff_step=2e-5,
                                        max_nfev=160, ftol=1e-10,
                                        xtol=1e-10, gtol=1e-8)
                   for x in starts]
        fit = min((f for f in results if f.success and not f.active_mask.any()), key=lambda f: f.cost)
        chi = 2*fit.cost
        assert all(f.success and 2*f.cost-chi < 1e-3 for f in results), (mode, [2*f.cost for f in results])
        cut = len(self.sn_y)
        components = [float(np.sum(v*v)) for v in np.split(fit.fun, (cut, cut+len(self.bao_z)))]
        assert abs(sum(components)-chi) < 2e-10
        return fit.x, chi, components

    def design(self, theta, n, scale):
        anchor = self.forward("LCDM", theta, n)
        eye = np.eye(3)
        columns = []
        for i, step in enumerate(DIFF[:3]*scale):
            d = eye[i]*step
            columns.append((self.forward("LCDM", theta+d, n)-self.forward("LCDM", theta-d, n))/(2*step))
        jac = self.white(np.array(columns).T)
        q, r = linalg.qr(jac, mode="economic")
        assert (abs(np.diag(r)) > 1).all()
        step = float(DIFF[3]*scale)
        cols = []
        for mode in MODEL[1:]:
            assert np.array_equal(self.forward(mode, np.r_[theta, 0.], n), anchor)
            cols.append((self.forward(mode, np.r_[theta, step], n)
                         - self.forward(mode, np.r_[theta, -step], n))/(2*step))
        z = self.white(np.array(cols).T)
        z -= np.einsum("ij,jk->ik",q,np.einsum("ij,ik->jk",q,z))
        z /= linalg.norm(z, axis=0)
        np.testing.assert_allclose(np.einsum("ij,ik->jk",q,z), 0, atol=2e-13)
        white_res = self.white(self.y-anchor)
        base_projection = white_res-np.einsum("ij,j->i",q,np.einsum("ij,i->j",q,white_res))
        scores = np.einsum("ij,i->j",z,base_projection)
        # Independent local likelihood-ratio evaluation using 4-column solves.
        direct = []
        for col in z.T:
            coeff = linalg.lstsq(np.c_[q,col], white_res)[0]
            v = white_res - np.einsum("ij,j->i",np.c_[q,col],coeff)
            direct.append(np.sum(base_projection**2) - np.sum(v*v))
        np.testing.assert_allclose(scores**2, direct, atol=5e-11)
        return scores, np.einsum("ij,ik->jk",z,z)


def bank_tail(scores, matrix, draws, seed):
    t = float(np.max(scores**2))
    rng = np.random.default_rng(seed)
    L = linalg.cholesky(matrix, lower=True)
    k = n1 = 0
    for offset in range(0, draws, 100_000):
        z = np.einsum("ij,kj->ik",rng.standard_normal((min(100_000, draws-offset), len(scores))),L)
        k += int(np.count_nonzero(np.max(z*z, axis=1) >= t))
        n1 += int(np.count_nonzero(z[:,0]**2 >= t))
    single = float(stats.chi2.sf(t,1))
    mc, e1 = k/draws, n1/draws
    assert k >= n1 and abs(e1-single) < 6*math.sqrt(single*(1-single)/draws)
    assert mc <= len(scores)*single + 6*math.sqrt(max(mc,1/draws)/draws)
    ci = stats.binomtest(k, draws).proportion_ci(confidence_level=.95)
    return {"test_stat":t, "best_template_single_tail":single,
            "bonferroni_upper_bound":min(1.,len(scores)*single),
            "common_null_tail":mc, "mc_95_interval":[ci.low,ci.high],
            "exceedances":k, "draws":draws, "seed":seed,
            "duplicate_template_control": e1}


def execute(source, draws, seed):
    e = Experiment(source)
    x0, c0, _ = e.fit("LCDM", 2800)
    runs = {}
    for n in (1400,2800):
        fits = {}
        for model in MODEL:
            param, chi, blocks = e.fit(model,n,x0)
            fits[model] = {"parameters":param.tolist(),"chi2":float(chi),"blocks_SN_BAO_CMB":blocks}
        x = np.array(fits["LCDM"]["parameters"])
        s, mat = e.design(x,n,1.)
        half, mhalf = e.design(x,n,.5)
        np.testing.assert_allclose(s, half, atol=3e-4, rtol=0)
        np.testing.assert_allclose(mat, mhalf, atol=3e-4, rtol=0)
        for model in MODEL:
            fits[model]["chi2_gain"] = float(fits["LCDM"]["chi2"]-fits[model]["chi2"])
        runs[str(n)] = {"fits":fits, "linear_scores":s.tolist(),
                        "template_correlation":mat.tolist(),
                        "half_step_score_max_abs": float(np.max(abs(half-s)))}
    fine, coarse = runs["2800"],runs["1400"]
    max_grid_error = max(abs(fine["fits"][m]["chi2_gain"]-coarse["fits"][m]["chi2_gain"])
                         for m in MODEL)
    assert max_grid_error < .03, max_grid_error
    np.testing.assert_allclose(fine["linear_scores"],coarse["linear_scores"],atol=.015,rtol=0)
    scores, mat = np.array(fine["linear_scores"]),np.array(fine["template_correlation"])
    tail = bank_tail(scores[:3],mat[:3,:3],draws,seed)
    rho = abs(float(mat[1,3]))
    # Fixed 3-sigma mean shift along either line. In their two-dimensional
    # projected span the sum and difference of the signed scores are independent.
    a, b = stats.norm.cdf(3*np.sqrt((1-rho)/2)), stats.norm.cdf(3*np.sqrt((1+rho)/2))
    return {"scope":"same-data retrospective compressed Gaussian tangent bank; not a global nonlinear or blind test",
            "python":sys.version.split()[0],"numpy":np.__version__,"scipy":scipy.__version__,
            "input_sha256":e.hashes, "script_sha256":digest(Path(__file__)),
            "model_order":MODEL, "shared_nuisance":["h","omega_b","omega_c","SN magnitude shift"],
            "fit_shape_parameter":"signed alpha (optical), w0+1 (mirage); wa=-3.3*(w0+1)",
            "linear_score_order":MODEL[1:], "grids":runs, "max_delta_chi2_grid_shift":max_grid_error,
            "primary_optical_bank_null":tail,
            "p1_vs_mirage": {"abs_tangent_correlation":rho,
                             "wrong_line_win_probability_for_fixed_3sigma_mean":float(a*(1-b)+(1-a)*b)}}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--draws", type=int, default=2_000_000)
    p.add_argument("--seed", type=int, default=20261123)
    a = p.parse_args()
    if a.output.exists() or a.draws < 50_000:
        p.error("use a new output path and at least 50000 draws")
    out = execute(a.source.resolve(),a.draws,a.seed)
    with a.output.open("x", encoding="utf-8") as handle:
        json.dump(out,handle,indent=2,allow_nan=False)
        handle.write("\n")
    print("PAPER23_MATCHED_TANGENT_PASS",json.dumps(out["primary_optical_bank_null"]))


if __name__ == "__main__":
    main()
