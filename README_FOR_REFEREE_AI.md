# Referee & Reproducibility Guide (v10.0 — Final JCAP Submission-Ready Edition + Deep 3D DESI DR1 Void Sightlines)

## 1. Single-Command Verification & Planck PR3 Chain Extraction

Run the entire multi-compilation likelihood, Linder (2015) non-phantom thawing control, uniform `BAO+CMB` out-of-sample SN split, and multi-catalog 3D void-sightline pipeline (`SDSS DR7` + `DESI DR1 DESIVAST` + deep `ASTRA-DESI DR1 BGS+LRG` out to $z \le 0.95$; ~2–10 min depending on CPU; core audit requires only `numpy` and `scipy`, and calls `generate_sub4_figures_and_tables.py` if `matplotlib` is installed):

```bash
python3 supplementary_code_and_data/verify_C9_void_clock_timescape_apparent_w0_wa_audit.py
```

To regenerate `data/planck2018_compressed_priors.json` directly from the official Planck PR3 `COM_CosmoParams_base-plikHM_R3.01.zip` archive (column `thetastar*` $= 100\theta_*$, $N_{\mathrm{samples}} = 25,225$):

```bash
python3 supplementary_code_and_data/build_planck2018_compressed_prior.py /path/to/COM_CosmoParams_base-plikHM_R3.01.zip
```

---

## 2. Targeted v10.0 Upgrades & Referee Polish (Addressing Both JCAP Re-Reviews of v9.0)

1. **Deep 3D `DESI DR1` Void-Sightline Extension ($z \le 0.95$):** Bundled compact `.npz` extracts of official `DESIVAST DR1 VoidFinder` (`101,863` spheres, $z \le 0.24$) and deep `ASTRA-DESI DR1 BGS+LRG` (`35,122` voids, $0.05 \le z \le 1.05$, Zenodo `22003899`). Deduplicated `Pantheon+` GLS slopes:
   - `SDSS DR7 VoidFinder` ($z \le 0.105$, $N_{\mathrm{uniq}} = 280$): $-0.0552 \pm 0.0362\;\mathrm{mag}$ ($-1.52\sigma$)
   - `DESI DR1 DESIVAST` ($0.08 \le z \le 0.23$, $N_{\mathrm{uniq}} = 68$): $-0.0471 \pm 0.0497\;\mathrm{mag}$ ($-0.95\sigma$)
   - `ASTRA-DESI DR1 BGS+LRG` ($0.10 \le z \le 0.95$, $N_{\mathrm{uniq}} = 735$): $-0.0344 \pm 0.0341\;\mathrm{mag}$ ($-1.01\sigma$); $0.20 \le z \le 0.95$ ($N_{\mathrm{uniq}} = 540$): $-0.0478 \pm 0.0443\;\mathrm{mag}$ ($-1.08\sigma$); `LRG`-only $0.42 \le z \le 0.95$ ($N_{\mathrm{uniq}} = 167$): $+0.0341 \pm 0.0365\;\mathrm{mag}$ ($+0.94\sigma$).
   - Empirical $3\sigma$ sample-size forecast for a $-0.052\;\mathrm{mag}$ directional signal: $N_{\mathrm{SN}} \gtrsim 2,860$ unique SNe Ia inside the `DESI` footprint.
2. **Exact Figure 1B Linder (2015) Projection & Figure 1C Labeling:** Replaced the proxy line in Panel B with the exact joint projected `(w_0^eff, w_a^eff)` trajectory (passing through `(-0.904, -0.146)` at $w_0 = -0.90$), plotted both `SDSS DR7` and deep `ASTRA-DESI DR1` in Panel C, and labeled the `-52 mmag` reference explicitly as a conditional directional illustration.
3. **Table 1 Framing, Table 3 Exact Rounding & Banner Sync:** Clarified fixed-parameter evaluation in Table 1 ($0.8\sigma$ vs. $1.65\sigma$ tension), explained Linder's CMB penalty ($\chi^2_{\mathrm{CMB}} = 5.07\text{--}6.05$), matched Table 3 digits exactly to the regenerated JSON, updated Abstract range to $2.2\text{--}4.2\sigma$, and synced the script banner to `v10.0`.
