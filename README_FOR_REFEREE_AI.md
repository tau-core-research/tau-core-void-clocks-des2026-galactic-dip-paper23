# Referee Audit Package (v3 — Complete 4/4 Reproducible Suite & Model-Selection Audit): Sub4 Letter (`MNRAS Letters / JCAP`, 2026-11-18)
**Title:** *Kodama–Hayward Galactic Acceleration Trough vs. DES Year 6 Dynamical Dark Energy: An Exact Background-Dependence Theorem and Conditional Void-Clock Optical Hypotheses*  
**Author:** Jozsef Olcsak (`ORCID: 0009-0003-7311-6522`)

---

## 1. Summary of Revisions Addressing All External Referee Points
1. **`4/4` Standalone Reproducibility Confirmed (Typical Total Runtime `~10–30 seconds`, Hardware-Dependent):**
   - All four verification scripts (`verify_C9`, `verify_C5`, `verify_C3_C4`, and `verify_C6_C7`) and all 9 raw observational data files are bundled in `supplementary_code_and_data/` and execute 100% standalone without external paths.
   - In `verify_C6_C7_canonical_cosmological_closure_and_likelihoods.py`, the 6-point BDF2 `RECFAST v1.5` 2nd-order mixed-derivative stencil used during intermediate optimizer steps is now explicitly validated across 6 independent out-of-sample test points (`max_abs_err_z_star = 1.56e-3`, `max_abs_err_100_theta_star = 3.09e-6`, i.e. $< 0.01\sigma$ of the Planck $\theta_*$ uncertainty), and every final optimum is re-evaluated with the **full unapproximated 1200-step BDF2 `RECFAST v1.5` solver**.
2. **Physical Motivation & Separate Sensitivity Accounting ($\Delta k_{\mathrm{cont}} = 0$ vs. Heuristic $\Delta k_{\mathrm{hyp}} \in \{1, 2\}$) for `H2b`:**
   - We define the physical meaning of $p \in \{1, 2\}$ in $\alpha_{\mathrm{eff}}(z) = \alpha_{\mathrm{OCSR}}[f_v(z)/f_{v,0}]^p$ ($p=1$: 1-point local void volume fraction along the ray; $p=2$: 2-point emitter–observer / quadratic susceptibility void-clock dressing).
   - Because each pre-specified variant (`H2a`, `H2b` $p=1,2$) has zero continuous parameters fitted to supernova data ($\Delta k_{\mathrm{cont}} = 0$), standard continuous AIC/BIC counts equal $\Delta\chi^2_{\mathrm{SN}}$ (`+0.44` for `H2b` $p=2$ vs. $\Lambda$CDM on the $N = 1,580$ `Pantheon+` supernovae), while we also report a separate heuristic sensitivity table using $\Delta k_{\mathrm{hyp}} \in \{1, 2\}$ (`+4.44` in AIC, `+15.17` in BIC for $\Delta k_{\mathrm{hyp}} = 2$) to account transparently for discrete hypothesis selection prior to a full Sachs optical scalar derivation.
3. **Full Correlation Range $\rho \in [-0.60, -0.90]$ & Asymmetric Error Check for $\Delta\chi^2_{\mathrm{2D}}$:**
   - `verify_C9` and Proposition 3 report the full sensitivity range across $\rho \in [-0.60, -0.90]$ as well as the asymmetric upper $+0.21$ error bar check (for `H2b` $p=2$ integrated modulation: $\Delta\chi^2_{\mathrm{2D}} = 0.70$ at $\rho = -0.85$, $0.63$ with upper $+0.21$ error, and $[0.27, 1.04]$ across $\rho \in [-0.60, -0.90]$), explicitly labeled as an approximate 2D Gaussian compatibility metric.
4. **Catalog Summary-Statistic Qualification on the 43 `JWST/HST` Disks, Negative Control Retention & Cross-Domain Rigidity:**
   - Theorem 1 explicitly notes that the 43-galaxy test ($\chi^2 = 106.42$ vs. $121.64$ anchored and $110.66$ unanchored) is a catalog summary-statistic rotation-velocity evaluation ($v_{\mathrm{circ,flat}}$ at $2R_{\mathrm{eff}}$ with Tacconi+20 gas scaling and error propagation), Table 1 retains `M2a-Strict-Frozen` ($\Delta\chi^2 = +20.60, \Delta\mathrm{AIC} = +14.60$) as an explicit negative control, and Section 5.2 documents the cross-domain parameter rigidity of $\alpha_{\mathrm{OCSR}} = \frac{5\sqrt{3}}{24}$ and $f_{v,0} = 72.34\%$.

---

## 2. Standalone Verification Commands (Typical Total Runtime `~10–30 seconds`, Hardware-Dependent)
```bash
python3 supplementary_code_and_data/verify_C9_void_clock_timescape_apparent_w0_wa_audit.py
python3 supplementary_code_and_data/verify_C5_kh_background_dependence_audit.py
python3 supplementary_code_and_data/verify_C3_C4_paper10_nogo_and_des2026_cpl_audit.py
python3 supplementary_code_and_data/verify_C6_C7_canonical_cosmological_closure_and_likelihoods.py
```
