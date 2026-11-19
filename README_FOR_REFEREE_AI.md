# Referee & Reproducibility Guide (v11.0 — Final JCAP Submission-Ready Edition)

## 1. Single-Command Verification & Planck PR3 Chain Extraction

Run the entire multi-compilation likelihood, Linder (2015) non-phantom thawing control, uniform `BAO+CMB` out-of-sample SN split, and exact-$z$ multi-catalog 3D void-sightline pipeline (`SDSS DR7` + `DESI DR1 DESIVAST` + deep `ASTRA-DESI DR1 BGS+LRG` & strict `LRG-only` out to $z \le 0.95$; ~1–3 min depending on CPU; core audit requires only `numpy` and `scipy`, and calls `generate_sub4_figures_and_tables.py` if `matplotlib` is installed):

```bash
python3 supplementary_code_and_data/verify_C9_void_clock_timescape_apparent_w0_wa_audit.py
```

To regenerate `data/planck2018_compressed_priors.json` directly from the official Planck PR3 `COM_CosmoParams_base-plikHM_R3.01.zip` archive (column `thetastar*` $= 100\theta_*$, $N_{\mathrm{samples}} = 25,225$):

```bash
python3 supplementary_code_and_data/build_planck2018_compressed_prior.py /path/to/COM_CosmoParams_base-plikHM_R3.01.zip
```

---

## 2. Targeted v11.0 Methodological Fixes Addressing the v10.0 JCAP Re-Review

1. **Exact Redshift Cuts & Strict `LRG-only` Void Filtering:** Replaced the $2350\,h^{-1}\mathrm{Mpc}$ ($z \simeq 1.035$) radial cut with explicit $z_{\mathrm{HD}}$ cuts matching every label, and restricted the `LRG-only` test strictly to `LRG` void centers and `LRG` effective-radius spheres:
   - `SDSS DR7 VoidFinder` ($z \le 0.105$, $N_{\mathrm{uniq}} = 280$): $-0.0552 \pm 0.0362\;\mathrm{mag}$ ($-1.52\sigma$, $95\%$ CI $[-0.126, +0.016]\;\mathrm{mag}$)
   - `DESI DR1 DESIVAST` ($0.08 \le z \le 0.23$, $N_{\mathrm{uniq}} = 68$): $-0.0471 \pm 0.0497\;\mathrm{mag}$ ($-0.95\sigma$, $95\%$ CI $[-0.145, +0.050]\;\mathrm{mag}$)
   - `ASTRA-DESI DR1 BGS+LRG` ($0.10 \le z \le 0.95$, $N_{\mathrm{uniq}} = 734$): $-0.0352 \pm 0.0340\;\mathrm{mag}$ ($-1.03\sigma$, $95\%$ CI $[-0.102, +0.032]\;\mathrm{mag}$); $0.20 \le z \le 0.95$ ($N_{\mathrm{uniq}} = 547$): $-0.0439 \pm 0.0439\;\mathrm{mag}$ ($-1.00\sigma$)
   - Strict `ASTRA-DESI DR1 LRG-only` ($0.42 \le z \le 0.95$, $N_{\mathrm{uniq}} = 128$): $+0.0159 \pm 0.0460\;\mathrm{mag}$ ($+0.35\sigma$, $95\%$ CI $[-0.074, +0.106]\;\mathrm{mag}$)
2. **Documented Footprint, $R_{\mathrm{eff}}$ Sphere Union, Overlap & Conditional Scaling:** Explicitly documented angular-proximity footprint approximation, `ASTRA` $R_{\mathrm{eff}}$ sphere-union estimator, exact overlap (`67/68` `DESIVAST` SNe in `ASTRA-DESI`, making them correlated cross-checks), and qualified $N_{\mathrm{SN}} \sim 2.8 \times 10^3$ as a conditional $N^{-1/2}$ order-of-magnitude scaling assuming identical per-SN sensitivity and fixed $-0.052\;\mathrm{mag}$.
3. **Standalone Data & Code Availability Section & 6× Speedup:** Added a dedicated JCAP `\section*{Data and Code Availability}` and replaced unoptimized `np.einsum` covariance/ray-tracing loops with BLAS matrix products, cutting full runtime to ~1–3 min.
