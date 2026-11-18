# Referee Audit Package (v3 — Fast 8-Second Complete Suite & Full Model-Selection Audit): Sub4 Letter (`MNRAS Letters / JCAP`, 2026-11-18)
**Title:** *Kodama–Hayward Galactic Acceleration Trough vs. DES Year 6 Dynamical Dark Energy: An Exact Background-Dependence Theorem and Conditional Void-Clock Optical Hypotheses*  
**Author:** Jozsef Olcsak (`ORCID: 0009-0003-7311-6522`)

---

## 1. Summary of v3 Revisions Addressing All Remaining Referee Points
1. **`C6/C7` Timeout Eliminated (`55 s` $\to$ `5.1 s`, Total 4-Script Suite Runtime `< 9 seconds`):**
   - Previously, `verify_C6_C7_canonical_cosmological_closure_and_likelihoods.py` called the full 1200-step implicit Newton-BDF2 `RECFAST v1.5` recombination ODE inside every Nelder-Mead step ($\sim 780$ calls $\times 70\;\mathrm{ms} \approx 55\;\mathrm{s}$).
   - Because dark energy is $< 10^{-9}$ of the cosmic energy density at recombination ($z_* \approx 1090$), $z_*(\omega_b, \omega_c)$ depends only on $(\omega_b, \omega_c)$. We now build an exact 6-point BDF2 `RECFAST v1.5` 2nd-order mixed-derivative stencil (accurate to $< 8\times 10^{-4}$ in $z_*$) for intermediate optimizer steps while retaining the exact conformal-time and sound-horizon integrals at every step, and then evaluate every final optimum with the **full unapproximated 1200-step BDF2 `RECFAST v1.5` solver**.
   - **Result:** All numbers in `C6_C7_canonical_closure_and_likelihood_results.json` and `C7_multiprobe_M0_M1_M2_comparison.csv` are **identical**, and `verify_C6_C7` now finishes in **5.1 seconds**!
2. **Physical Motivation & Explicit Hypothesis-Selection Penalty ($\Delta k_{\mathrm{hyp}} \in \{1, 2\}$) for `H2b`:**
   - We explicitly define the physical meaning of the exponent $p \in \{1, 2\}$ in $\alpha_{\mathrm{eff}}(z) = \alpha_{\mathrm{OCSR}}[f_v(z)/f_{v,0}]^p$ ($p=1$: 1-point local void volume fraction along the ray; $p=2$: 2-point emitter–observer / quadratic susceptibility void-clock dressing).
   - Because selecting $p \in \{1, 2\}$ and differential-ray vs. integrated-distance dressing is not yet derived from the 4D `COV-DIR+QUMOND` Sachs optical scalar equations, we explicitly assign a **hypothesis-selection degree-of-freedom penalty $\Delta k_{\mathrm{hyp}} = 1$ ($p=1$) and $\Delta k_{\mathrm{hyp}} = 2$ ($p=2$, $+4.0$ in AIC and $+14.73$ in BIC on $N = 1,580$ Pantheon+ supernovae)** in `verify_C9` and Table 1.
3. **Full Correlation Range $\rho \in [-0.60, -0.90]$ & Asymmetric Error Check for $\Delta\chi^2_{\mathrm{2D}}$:**
   - Rather than relying solely on $\rho = -0.85$, `verify_C9` and Proposition 3 report the full sensitivity range across $\rho \in [-0.60, -0.90]$ as well as the asymmetric upper $+0.21$ error bar check (for `H2b` $p=2$ integrated modulation: $\Delta\chi^2_{\mathrm{2D}} = 0.70$ at $\rho = -0.85$, $0.63$ with upper $+0.21$ error, and $[0.27, 1.04]$ across $\rho \in [-0.60, -0.90]$), explicitly labeled as an approximate 2D Gaussian compatibility metric.
4. **Catalog Summary-Statistic Qualification on the 43 `JWST/HST` Disks & Negative Control Retention:**
   - Theorem 1 explicitly notes that the 43-galaxy test ($\chi^2 = 106.42$ vs. $121.64$ anchored and $110.66$ unanchored) is a catalog summary-statistic rotation-velocity evaluation ($v_{\mathrm{circ,flat}}$ at $2R_{\mathrm{eff}}$ with Tacconi+20 gas scaling and error propagation), and Table 1 retains `M2a-Strict-Frozen` ($\Delta\chi^2 = +20.60, \Delta\mathrm{AIC} = +14.60$) as an explicit negative control.
5. **Cross-Domain Parameter Rigidity Audit (Why `H1` and `H2a`/`H2b` Are Not Post-Hoc Curve Fits):**
   - Even though `H1` and `H2a`/`H2b` are conditional hypotheses rather than completed covariant derivations, they contain **zero continuous parameters tuned to supernova data**:
     * **Independent Empirical Match of $f_{v,0} = 72.34\%$:** Equating the Buchert–Wiltshire quartic root $f_{\mathrm{acc}} = 0.586708$ and the horizon inflection redshift $z_{\mathrm{acc}}^{\mathrm{hor}} = 0.631556$ (`H1`) yields $f_{v,0} = 72.34\%$ ($[71.79\%, 72.91\%]$ across $\Omega_{m,0} \in [0.300, 0.330]$), landing squarely inside the empirical $69.5\%\text{--}76\%$ void-fraction interval independently fitted to `Pantheon+` by Lane, Seifert, Ridden-Harper & Wiltshire (*MNRAS* 536, 1752, 2025).
     * **Cross-Scale Transfer of the Galactic Constant $\alpha_{\mathrm{OCSR}} = \frac{5\sqrt{3}}{24} \approx 0.360844$:** Fixed prior to any supernova analysis from the discrete $K_4$ simplex and `SPARC` galactic rotation curves, applying $\alpha_{\mathrm{OCSR}}$ to the Wiltshire null distance simultaneously shifts unweighted Timescape's $w_0^{\mathrm{app}} = -0.4930$ ($6.5\sigma$ away from DES 2026) to $w_0^{\mathrm{app}}(0) = -0.8290$ ($0.18\sigma$ from DES 2026) and eliminates **$99.5\%$ of unweighted Timescape's $+83.66$ $\chi^2_{\mathrm{SN}}$ penalty** on the $1,580$ `Pantheon+` supernovae ($\chi^2_{\mathrm{SN}}: 1470.99 \to 1387.77$).
     * **Two-Sided Kinematic Vice on Homogeneous CPL (`Theorem 1`):** A homogeneous DES 2026 CPL fluid either boosts the $z=0.6316$ galactic scale by $+21.75\%$ ($\Delta\chi^2 = +15.22$ on the 43 `JWST/HST` disks when anchored at $z=0$) or compresses the local $z=0$ `SPARC` acceleration scale $a_0^{\mathrm{KH}}(0)$ by $-12.11\%$ ($\Delta\chi^2 = +4.24$ on the 43 disks when unanchored).

---

## 2. Fast Standalone Verification Commands (Total Runtime `< 9 seconds`)
```bash
python3 supplementary_code_and_data/verify_C9_void_clock_timescape_apparent_w0_wa_audit.py
python3 supplementary_code_and_data/verify_C5_kh_background_dependence_audit.py
python3 supplementary_code_and_data/verify_C3_C4_paper10_nogo_and_des2026_cpl_audit.py
python3 supplementary_code_and_data/verify_C6_C7_canonical_cosmological_closure_and_likelihoods.py
```
