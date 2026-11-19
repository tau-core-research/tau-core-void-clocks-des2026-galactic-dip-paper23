# Referee & Reproducibility Guide (v9.0 — Final JCAP Submission-Ready Edition)

## 1. Single-Command Verification & Planck PR3 Chain Extraction

Run the entire multi-compilation likelihood, Linder (2015) non-phantom thawing control, uniform `BAO+CMB` out-of-sample SN split, and deduplicated 3D void-sightline pipeline (~1–6 min depending on CPU; core audit requires only `numpy` and `scipy`, and calls `generate_sub4_figures_and_tables.py` if `matplotlib` is installed):

```bash
python3 supplementary_code_and_data/verify_C9_void_clock_timescape_apparent_w0_wa_audit.py
```

To regenerate `data/planck2018_compressed_priors.json` directly from the official Planck PR3 `COM_CosmoParams_base-plikHM_R3.01.zip` archive (column `thetastar*` $= 100\theta_*$, $N_{\mathrm{samples}} = 25,225$):

```bash
python3 supplementary_code_and_data/build_planck2018_compressed_prior.py /path/to/COM_CosmoParams_base-plikHM_R3.01.zip
```

---

## 2. Targeted v9.0 Fixes Addressing Both JCAP Re-Reviews of v8.0

1. **Exact `100*theta_*` Planck PR3 Covariance:** Replaced the `100*theta_MC` covariance with the exact weighted covariance of `thetastar*` (`std = 0.0003068`) from the four `base_plikHM_TTTEEE_lowl_lowE_lensing` chains and bundled `build_planck2018_compressed_prior.py` ($p=1$ $\Delta\chi^2 = -9.80, -11.34, -18.53, -13.62$).
2. **Strictly Non-Phantom Linder (2015, Eq. 3) Control:** Replaced $w_a = -1.58(1+w_0)$ with Linder's algebraic thawing model ($p = 1, b = 0.5, w_0 \ge -1$, $w(a) \ge -1$ at all $z$; $\Delta\chi^2 = -2.54, -3.75, -8.79, -3.54$).
3. **Uniform `BAO+CMB`-Only & Out-of-Sample SN Table:** Added Table 1 comparing $p = 0, 1, 2$ and the phantom-crossing CPL line ($k = 4$ without $M_B$; fixed `LCDM_no_MB_k3` $k=3$).
4. **Exploratory Sightline Check, Abstract Scope & JCAP Compliance:** Removed permutation $p$-values, qualified the $-0.052\;\mathrm{mag}$ deep scale as directional, stated bare-$\bar{H}_0$ and all-$z$ Sachs limitations in the Abstract, narrowed the $D_{\mathrm{TS}}$ + $\Lambda$CDM acoustic calibration phrasing, and trimmed JCAP keywords to 4.
