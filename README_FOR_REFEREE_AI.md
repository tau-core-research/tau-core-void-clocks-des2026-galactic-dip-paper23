# Referee Audit Package (v2 — Revised after External Referee Review): Sub4 Wedge Letter (`MNRAS Letters / JCAP`, 2026-11-18)
**Title:** *Kodama–Hayward Galactic Acceleration Trough vs. DES Year 6 Dynamical Dark Energy: An Exact Background-Dependence Theorem and Conditional Void-Clock Optical Hypotheses*  
**Author:** Jozsef Olcsak (`ORCID: 0009-0003-7311-6522`)

---

## 1. Summary of Revisions Addressing All External Referee Points (v2)
All P0 and P1 points raised in the external referee validation (`Tau Core Sub4 – szigorú tudományos validáció`) have been resolved in full:
1. **100% Standalone Reproducibility (`4/4` Scripts `PASS` with Bundled Raw Data, Zero External Paths):**
   - All raw observational datasets required by `verify_C9`, `verify_C5`, `verify_C3_C4`, and `verify_C6_C7` are now bundled directly in `supplementary_code_and_data/data/`:
     * `df1_z09_jwst_hst_mancerapina2025_43disks.csv` (`43` JWST/HST disk galaxies of Mancera Piña et al. 2025)
     * `desi_gaussian_bao_ALL_GCcomb_mean.txt` & `desi_gaussian_bao_ALL_GCcomb_cov.txt` (`13` DESI BAO measurements + $13\times 13$ covariance)
     * `pantheon_plus_hf1590_stat_sys_cholesky.npz` & `Pantheon+SH0ES.dat` (`1,580` Pantheon+ Hubble-flow Type Ia supernovae + exact $1580\times 1580$ STAT+SYS Cholesky factor)
     * `sdss_DR16_MultiTracerELGLRG_BAORSD_FS_PK.dat`, `_cov.dat`, & `growth_web_source_native_design_matrix_v1.csv`
   - All hardcoded `/Volumes/EXTERNAL/...` workspace paths have been removed; every script uses relative paths (`Path(__file__).resolve().parent / "data"`).
   - Fixed the platform-dependent `SLSQP` starting-point tolerance check in `verify_C3_C4_paper10_nogo_and_des2026_cpl_audit.py`.
   - In `verify_C9_void_clock_timescape_apparent_w0_wa_audit.py`, the 43 JWST/HST disk galaxy $\chi^2$ values (`106.42` vs. `121.64` vs. `110.66`) are no longer hardcoded literals — they are computed **live from the bundled raw CSV** on every run!
2. **Full 2D $(w_0, w_a)$ Mahalanobis Analysis & Void-Maturity Modulation (`H2a` vs. `H2b`):**
   - We no longer claim that the 1D $w_0^{\mathrm{app}} = -0.8251$ ($0.10\sigma$) match alone constitutes a full 2D match to DES 2026.
   - Instead, Proposition 3 and Figure 1B report the **full 2D $(w_0, w_a)$ Mahalanobis $\Delta\chi^2_{\mathrm{2D}}$** (at $\rho = -0.85$) against both `DES Y6 + DESI + CMB` $(-0.82, -0.63)$ and `Pantheon+ + DESI + CMB` $(-0.8901, -0.2253)$, as well as the direct $\chi^2_{\mathrm{SN}}$ on the real bundled $1,580$ `Pantheon+` supernovae:
     * **Hypothesis H2a (Constant-weight $\alpha_{\mathrm{OCSR}}$ split-beam):** Gives $(w_0^{\mathrm{app}}, w_a^{\mathrm{app}}) = (-0.8251, -0.1006)$ ($0.10\sigma$ in 1D $w_0$, $2.71\sigma$ in 1D $w_a$, $\Delta\chi^2_{\mathrm{2D}} = 24.90$ vs. DES Y6, $\Delta\chi^2_{\mathrm{2D}} = 8.44$ vs. Pantheon+, real Pantheon+ $\chi^2_{\mathrm{SN}} = 1392.40$ vs. $1470.99$ for unweighted Timescape).
     * **Hypothesis H2b (Void-maturity-modulated coupling $\alpha_{\mathrm{eff}}(z) = \alpha_{\mathrm{OCSR}}[f_v(z)/f_{v,0}]^p$):** Because voids grow from $f_v(z\gg 1)\to 0$ to $f_{v,0}=0.7234$ today with exact slope $\mathrm{d}\ln f_v/\mathrm{d}z|_0 = -0.331584$, weighting by $[f_v(z)/f_{v,0}]^p$ leaves $w_0^{\mathrm{app}} \approx -0.82$ invariant while steepening $w_a^{\mathrm{app}}$ into the 2D confidence region: $(-0.8233, -0.2414)$ and $(-0.8256, -0.3399)$ for differential ray integration ($p=1,2$), and $(-0.8116, -0.3990)$ and $(-0.8122, -0.5717)$ ($\Delta\chi^2_{\mathrm{2D}} = 0.70$, $0.38\sigma$ 2D vs. DES 2026; real Pantheon+ $\chi^2_{\mathrm{SN}} = 1387.77$) for integrated distance modulation ($p=1,2$).
3. **Strict Epistemic Separation (`Theorem 1 [PROVEN]`, `Proposition 2 [PROVEN under H1]`, `Proposition 3 [CONDITIONAL under H2a/H2b]`) & Unmerged Table 1:**
   - Replaced "Zero-Parameter Synthesis" in the title and text with explicit epistemic labels (`Hypothesis H1` for horizon–wall transition matching; `Hypotheses H2a/H2b` for split-beam optical propagation).
   - Table 1 reports `M2a-Profiled` ($k_{\mathrm{tot}}=3, \Delta\chi^2=+2.53, \Delta\mathrm{BIC}=-4.85$), `M2a-Strict-Frozen` ($k_{\mathrm{tot}}=1, \Delta\chi^2=+20.60, \Delta\mathrm{AIC}=+14.60$), and the conditional void-clock branches (`H1+H2a`, `H1+H2b`) on separate, unmerged rows.

---

## 2. Standalone Verification Commands (Run from Any Directory)
```bash
python3 supplementary_code_and_data/verify_C9_void_clock_timescape_apparent_w0_wa_audit.py
python3 supplementary_code_and_data/verify_C5_kh_background_dependence_audit.py
python3 supplementary_code_and_data/verify_C3_C4_paper10_nogo_and_des2026_cpl_audit.py
python3 supplementary_code_and_data/verify_C6_C7_canonical_cosmological_closure_and_likelihoods.py
```
