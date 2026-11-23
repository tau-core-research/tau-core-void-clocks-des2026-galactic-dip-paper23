#!/usr/bin/env python3
"""Frozen Paper 23 distances vs the six public H0LiCOW XIII lens likelihoods."""
import argparse
import hashlib
import io
import json
import math
from pathlib import Path
from urllib.parse import quote
from urllib.request import urlopen

import numpy as np
import scipy
from scipy import integrate, optimize, stats

import verify_C9_void_clock_timescape_apparent_w0_wa_audit as tracker

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "derived"
PROTOCOL = DATA / "h0licow_holdout_protocol_v1.json"
TRAINING = DATA / "pantheon_joint_tangent_20261123_strict.json"
FREEZE = DATA / "h0licow_holdout_predictions_v1.json"
PROTOCOL_SHA = "516394d95046a4bc6b2286d459c1464c8fa14919427ace1e85a448e44a50ebdf"


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def read_json(path, expected):
    raw = path.read_bytes()
    if sha(raw) != expected:
        raise ValueError(f"Unexpected hash: {path}")
    return json.loads(raw)


def expansion(model, theta):
    h, ob, oc = theta[:3]
    om = (ob + oc + .06 / 93.14) / h**2
    rho = 3 * (1000*100*h/tracker.MPC_M)**2 / (8*np.pi*tracker.G_SI)
    rad = tracker.RHO_G0_SI / rho * (1 + .2271073*3.044)
    dark = 1 - om - rad
    if model == "p1" and theta[3] != 0:
        fa = optimize.brentq(lambda f: 8*f**3 + 39*f*f - 12*f - 8, .4, .8)
        zp = (2*(1-om)/om)**(1/3)
        fv0 = optimize.brentq(
            lambda f: (2+fa)/(2+f)*(f/fa)**(2/3)*(1-fa)/(1-f)-zp,
            .05, .999, xtol=1e-13)
        _, tr_d, tr_f, _ = tracker.build_vectorized_wiltshire_splines(fv0)

    def x(z):
        e = 1 + z
        base = rad*e**4 + om*e**3
        if model == "mirage":
            delta = theta[3]
            wa = -3.3*delta
            result = (base + dark*math.exp(
                3*(delta+wa)*math.log1p(z) - 3*wa*(1-1/e)))**(-.5)
        else:
            null = (base + dark)**(-.5)
            result = null
            if model == "p1" and theta[3] != 0:
                u = math.log1p(z)
                result += theta[3]*float(tr_f(u))/fv0*(float(tr_d(u))-null)
        if not math.isfinite(result) or result <= 0:
            raise ValueError("Nonmonotone or invalid optical derivative")
        return result
    return x


def distances(zd, zs, scale, x):
    # Integrate the lens-to-source leg directly to avoid subtractive loss.
    assert 0 < zd < zs
    cd, ed = integrate.quad(x, 0, zd, epsabs=5e-11, epsrel=5e-11)
    cds, es = integrate.quad(x, zd, zs, epsabs=5e-11, epsrel=5e-11)
    assert min(cd, cds) > 0 and max(ed, es) < 1e-9
    return [scale*cd*(cd+cds)/cds, scale*cd/(1+zd)]


def freeze(protocol):
    models = protocol["model_order"]
    training = read_json(TRAINING, protocol["training_receipt_sha256"])
    assert sha(Path(tracker.__file__).read_bytes()) == protocol["tracker_implementation_sha256"]
    zd, zs, scale = .5, 1., 6000.
    np.testing.assert_allclose(distances(zd, zs, scale, lambda z: 1),
                               [6000., 2000.], rtol=1e-15)
    predictions = {}
    for grid in (protocol["grid"], min(training["grids"], key=int)):
        fits = training["grids"][grid]["fits"]
        predictions[grid] = {}
        for model in models:
            theta = fits[model]["parameters"]
            x = expansion(model, theta)
            x0 = expansion("LCDM", theta[:3])
            for redshift in (0., .2, .8, 1.9):
                for alt in ("p1", "mirage"):
                    assert expansion(alt, [*theta[:3], 0.])(redshift) == x0(redshift)
            assert abs(x(0) - 1) < 1e-14
            predictions[grid][model] = {
                "parameters": theta,
                "lenses_Ddt_Dd_Mpc": {
                    name: distances(zd, zs, tracker.C_KMS/(100*theta[0]), x)
                    for name, zd, zs, _, _ in protocol["lenses"]
                },
            }
    return {
        "protocol_sha256": PROTOCOL_SHA, "units": "Mpc",
        "controls": ["analytic constant X distance", "alpha=0 and dw=0 recover LCDM",
                     "X(0)=1 and X positive on quadrature nodes"],
        "predictions": predictions,
    }


def lens_likelihood(raw, kind, factor):
    buf = io.BytesIO(raw)
    if kind == "analytic":
        coeff = np.loadtxt(buf, skiprows=9)
        # Upstream product of shifted lognormals, independent in this fit only.
        a, mean, sigma, b, expmean, tau = coeff[2:8]
        d1, d2 = stats.lognorm(sigma, loc=a, scale=np.exp(mean)), stats.lognorm(
            tau, loc=b, scale=expmean)

        def score(p):
            values = d1.logpdf(p[:, 0]) + d2.logpdf(p[:, 1])
            assert np.isfinite(values).all()
            return values
        return score, np.array([d1.std(), d2.std()]), 2, None
    if kind == "dt":
        samples, weights = np.loadtxt(buf)[None, :], None
    elif kind == "dt_dd_csv":
        samples, weights = np.loadtxt(buf, delimiter=",", skiprows=1).T, None
    elif kind == "weighted_dt_csv":
        rows = np.loadtxt(buf, delimiter=",", skiprows=1).T
        samples, weights = rows[:1], rows[1]
    elif kind == "dd_dt":
        samples, weights = np.loadtxt(buf, skiprows=1).T[::-1], None
    else:
        raise ValueError(kind)
    kde = stats.gaussian_kde(samples, weights=weights,
                             bw_method=lambda k: factor*k.scotts_factor())
    ndim, nrows = samples.shape
    sigma = np.sqrt(np.diag(kde.covariance)) / kde.factor

    def score(p):
        q = p[:, :ndim].T
        ll = kde.logpdf(q)
        assert np.isfinite(ll).all()
        ordinary = kde.pdf(q)
        assert (ordinary > 0).all()
        np.testing.assert_allclose(ll, np.log(ordinary), atol=1e-10, rtol=0)
        return ll
    return score, sigma, ndim, nrows


def evaluate(protocol, frozen, frozen_raw):
    root = (
        "https://raw.githubusercontent.com/shsuyu/H0LiCOW-public/"
        + protocol["upstream_commit"]
        + "/MontePython_cosmo_sampling/data/timedelay_6lenses/"
    )
    models = protocol["model_order"]
    primary = protocol["grid"]
    predictions = frozen["predictions"]
    results, inputs = {}, {}
    for name, _, _, filename, kind in protocol["lenses"]:
        url = root + quote(filename)
        with urlopen(url, timeout=20) as response:
            raw = response.read(100_000_001)
        assert 0 < len(raw) <= 100_000_000, (name, len(raw))
        inputs[name] = {"url": url, "sha256": sha(raw), "bytes": len(raw)}
        r = results[name] = {"scores": {}}
        for factor in protocol["bandwidth_factors"]:
            score, sigma, ndim, rows = lens_likelihood(raw, kind, factor)
            for key, fits in predictions.items():
                p = np.array([fits[m]["lenses_Ddt_Dd_Mpc"][name] for m in models])
                ll = score(p)
                r["scores"].setdefault(key, {})[str(factor)] = dict(zip(models, ll.tolist()))
                if key == primary and factor == 1:
                    exact = score(np.repeat(p[:1], 2, axis=0))
                    assert exact[0] == exact[1]
                    rel = np.abs(p[0, :ndim] / p[1, :ndim] - 1)
                    sep = np.abs(p[0, :ndim] - p[1, :ndim]) / sigma
                    r.update({
                        "active_coordinates": ["Ddt", "Dd"][:ndim],
                        "chain_rows": rows,
                        "marginal_distance_width_Mpc": sigma.tolist(),
                        "p1_mirage_relative_separation": rel.tolist(),
                        "p1_mirage_separation_over_marginal_width": sep.tolist(),
                        "primary_delta_loglike": float(ll[0]-ll[1]),
                    })
    sums = {}
    for key in predictions:
        sums[key] = {}
        for band in map(str, protocol["bandwidth_factors"]):
            totals = {m: math.fsum(r["scores"][key][band][m] for r in results.values())
                      for m in models}
            sums[key][band] = {
                "relative_to_LCDM": {m: totals[m]-totals["LCDM"] for m in models},
                "p1_minus_mirage": totals["p1"]-totals["mirage"],
            }
    delta = sums[primary]["1.0"]["p1_minus_mirage"]
    return {
        "scope": "Held-out fixed-distance comparison under a common flat distance-sum and the released H0LiCOW lens profile assumptions; no mechanism inference",
        "protocol_sha256": PROTOCOL_SHA, "prediction_sha256": sha(frozen_raw),
        "versions": {"numpy": np.__version__, "scipy": scipy.__version__},
        "inputs": inputs, "lenses": results, "totals": sums,
        "primary_fixed_likelihood_ratio_p1_over_mirage": math.exp(delta),
        "leave_one_out_p1_minus_mirage": {
            name: delta-r["primary_delta_loglike"] for name, r in results.items()},
        "max_relative_separation": max(max(r["p1_mirage_relative_separation"])
                                       for r in results.values()),
        "max_separation_over_marginal_width": max(
            max(r["p1_mirage_separation_over_marginal_width"]) for r in results.values()),
        "controls": ["same prediction => identical score", "KDE pdf vs logpdf",
                     "both predeclared bandwidth perturbations and lower training grid",
                     "leave-one-lens-out without refitting"],
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("step", choices=["predict", "score"])
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--predictions", type=Path, default=FREEZE)
    args = ap.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    protocol = read_json(PROTOCOL, PROTOCOL_SHA)
    frozen = freeze(protocol)
    if args.step == "predict":
        result = frozen   # No lens measurement data loaded here.
    else:
        raw = args.predictions.read_bytes()
        saved = json.loads(raw)
        assert saved["protocol_sha256"] == PROTOCOL_SHA and saved["units"] == "Mpc"
        for key, fits in frozen["predictions"].items():
            for model, pred in fits.items():
                old = saved["predictions"][key][model]
                assert pred["parameters"] == old["parameters"]
                for name, value in pred["lenses_Ddt_Dd_Mpc"].items():
                    np.testing.assert_allclose(value, old["lenses_Ddt_Dd_Mpc"][name],
                                               atol=1e-8, rtol=1e-12)
        result = evaluate(protocol, saved, raw)
    with args.output.open("x", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2, allow_nan=False)
        handle.write("\n")


if __name__ == "__main__":
    main()
