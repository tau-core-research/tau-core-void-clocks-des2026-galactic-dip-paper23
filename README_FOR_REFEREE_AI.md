# Referee & Reproducibility Guide (v7.0 — JCAP / PRD / OJA Cosmology Edition)

## 1. One-Command Standalone Verification (`NumPy` + `SciPy` Only)

Run the entire multi-compilation cosmological likelihood and 3D void-sightline ray-tracing suite from raw bundled data (~30–45 seconds on a single laptop CPU core; requires **only** `numpy` and `scipy`):

```bash
python3 supplementary_code_and_data/verify_C9_void_clock_timescape_apparent_w0_wa_audit.py
```

If you modify priors, parameters, or datasets in a copy of the script for custom cross-checks, no hardcoded dataset $\chi^2$ assertions exist, and mathematical unit-test assertions can also be disabled via:

```bash
python3 supplementary_code_and_data/verify_C9_void_clock_timescape_apparent_w0_wa_audit.py --no-asserts
```

The complete machine-readable output is written to `supplementary_code_and_data/C9_void_clock_timescape_apparent_w_results.json`.

---

## 2. Summary of v7.0 Upgrades Addressing External Review of v6.0

1. **Focused Scope & Restrained JCAP/PRD Style:**
   - Removed all galactic rotation-curve / 43-disk material (retained in companion galactic work) and removed all algebraic simplex priors ($\alpha_0 = 5\sqrt{3}/24$) and corpus self-citations.
   - Replaced emphatic labels and boldface rhetoric with standard academic prose.
2. **Honest $k = 5$ Parameter Accounting:**
   - `H2-vol` ($p = 1$) and `Forward H2-Sachs` ($p = 1$) are penalized throughout as $k = 5$ models ($h, \omega_b, \omega_c, M_B, \alpha$; $\Delta k = +1$ over $k = 4$ flat $\Lambda$CDM, $\Delta k = -1$ vs. $k = 6$ CPL), explicitly noting post-data selection of $p = 1$ after $p = 0$ failed CMB+BAO.
3. **Full Compressed Planck 2018 Priors & $H_0$ / $f_{v,0}$ Sensitivity:**
   - Primary CMB: marginalized $3\times 3$ `Planck 2018 TT,TE,EE+lowE` MCMC covariance on $(\omega_b, \omega_c, 100\theta_*)$ (`planck2018_compressed_priors.json`).
   - Cross-check CMB: `Chen, Huang & Wang (2019, JCAP 02, 028)` $(R, \ell_A, \omega_b)$ inverse covariance, plus reproduction of the referee's diagonal $\theta_* + \mathrm{BBN} + \omega_c + \mathrm{CC15}$ check ($\chi^2_{\Lambda\mathrm{CDM}} = 1413.85$, $\Delta\chi^2 = -11.31$, $\hat{\alpha} = 0.425 \pm 0.125$).
   - Explicitly tests unnormalized bare-$\bar{H}_0$ weighting ($\hat{\alpha} = 0.000 \pm 0.043$) and fixed $f_{v,0} \in [0.65, 0.80]$ without `H1` ($f_{v,0} = 0.762$ gives $\Delta\chi^2 = -11.27$).
4. **All Four Public Type Ia Supernova Compilations Bundled & Fitted:**
   - `Pantheon+` ($N = 1,580$), `DES-Dovekie` recalibrated ($N = 1,820$), `DES-SN5YR` 2024 ($N = 1,829$), and `Union3` ($N = 22$ spline nodes) yield mutually consistent $\hat{\alpha} \in [0.422, 0.594]$ with $\Delta\chi^2 = -10.96, -12.66, -20.30, -15.15$ ($\Delta\mathrm{BIC} = -3.59, -5.14, -12.78, -11.51$ vs. $\Lambda$CDM, and $-3.76$ to $-9.51$ vs. CPL).
5. **Direct 3D Cosmic-Void Sightline Ray-Tracing (`Pantheon+` $\times$ `SDSS DR7 VoidFinder`):**
   - Exact 1D chord-union ray-tracing through the $39,735$ 3D comoving spheres of Douglass, Veyrat & BenZvi (2023, *ApJS* 265, 7; `sdss_dr7_voidfinder_planck2018_3d_catalog.npz`) across $N = 311$ NGC interior sightlines gives $\mathrm{d}\Delta\mu/\mathrm{d}f_{v,\mathrm{LOS}} = -0.058 \pm 0.036\;\mathrm{mag}$ ($-1.59\sigma$), matching the predicted sign while remaining statistically consistent with both the small $z \le 0.105$ single-shell amplitude ($-0.007$ to $-0.014\;\mathrm{mag}$) and noise.
