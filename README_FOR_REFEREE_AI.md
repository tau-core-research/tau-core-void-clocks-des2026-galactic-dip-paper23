# Referee Audit Package (`v6.0` — Direct `Pantheon++DESI BAO+CMB` Joint Fits, Full `alpha in [0,1]` Profile Likelihood, Exact `5x5` Hessian Covariance, and Asymptotic-Flat MOND/RAR Kernels): Sub4 Letter (`MNRAS Letters / JCAP`, 2026-11-19)
**Title:** *Kodama–Hayward Galactic Acceleration Trough vs. DES Year 6 Dynamical Dark Energy: Background-Dependence Theorem, Direct Pantheon++DESI BAO+CMB Likelihood Tests, and Void-Volume-Weighted Optical Models*  
**Author:** Jozsef Olcsak (`ORCID: 0009-0003-7311-6522`)

---

## 1. Point-by-Point Summary of Revisions in `v6.0` Addressing All `v5` Referee Critiques

1. **[Referee Point 1] Retraction of SN-Only "Beats $\Lambda$CDM" Wording & Full $\alpha \in [0, 1]$ Profile Table (`verify_C9`, Table 1):**
   - On `Pantheon+` alone ($N=1,580$, fixed $\Omega_{m,0}=0.3153$), $\Delta\chi^2_{\mathrm{SN}} = -0.74$ (`H2a`, $p=0$), $-1.02$ (`H2-vol`, $p=1$), and $-1.21$ (`Forward H2-Sachs`, $p=1$) are now strictly described as **statistically equivalent to $\Lambda$CDM** ($< 1.1\sigma$), while eliminating the $+16.50$ penalty of pure unweighted Timescape ($\alpha=1$, $\chi^2_{\mathrm{SN}} = 1403.83$).
   - Table 1 explicitly reports $\Delta\chi^2_{\mathrm{SN}}(\alpha)$ across $\alpha \in \{0.00, 0.10, 0.20, 0.218, 0.30, 0.330, 0.361, 0.45, 0.60, 1.00\}$, reproducing the referee's exact plateau (`0.00, -0.93, -1.31, -1.32, -1.13, -0.95, -0.74, +0.21, +2.86, +16.50` for `H2a`) and noting its degeneracy with a small $\Omega_{m,0}$ shift on SNe Ia alone (free-$\Omega_{m,0}$ $\Lambda$CDM gives $\Delta\chi^2_{\mathrm{SN}} = -0.92$ at $\Omega_{m,0} = 0.3326$).

2. **[Referee Point 2] Direct Joint Fitting of $D_{\mathrm{obs}}(z)$ to `Pantheon+ + DESI BAO + Planck CMB 100*theta_* + BBN (+ Moresco CC)` & Exact `5x5` Hessian Covariance (`verify_C9`, Tables 1–2):**
   - **Constant-weight `H2a` ($p=0$) is ruled out by direct CMB $\theta_*$ + BAO fitting:** keeping $\alpha_{\mathrm{eff}}(z) \equiv \alpha_0$ constant up to recombination ($z_* \simeq 1090$, where nonlinear cosmic voids do not yet exist, $f_v(z_*) \approx 0.002$) yields $100\theta_* = 1.0230$ ($\chi^2_{\theta_*} = 3408.5$) at Planck parameters, and $\Delta\chi^2 = \mathbf{+5.56}$ worse than $\Lambda$CDM on $N=1,610$ ($\mathbf{+4.87}$ on $N=1,595$ without CC) even after profiling $(h, \omega_b, \omega_c, M_B)$.
   - **Physical void-volume-fraction weighting $\alpha_{\mathrm{eff}}(z) = \alpha\,f_v(z)/f_{v,0}$ (`H2-vol`, $p=1$) breaks the $\Omega_{m,0}$ degeneracy:** because $f_v(z)/f_{v,0} \to 0$ automatically as $z \to z_*$, profiling $(h, \omega_b, \omega_c, M_B)$ across $\alpha \in [0, 1]$ on $N=1,610$ yields a well-defined empirical maximum-likelihood estimate **$\hat{\alpha}_{\mathrm{MLE}} = \mathbf{0.330 \pm 0.139}$** ($\min\chi^2 = 1401.15$, $\Delta\chi^2 = -6.52$, a moderate $\sim 2.55\sigma$ preference over $\alpha=0$ $\Lambda$CDM while excluding pure Timescape $\alpha=1$ at $+11.02$). Thus no reader need assume any external algebraic origin for $\alpha$: the prior hypothesis $\alpha_0 = \frac{5\sqrt{3}}{24} = 0.3608$ sits within **$0.22\sigma$ ($\Delta\chi^2 = +0.05$)** of $\hat{\alpha}_{\mathrm{MLE}}$ and achieves $\chi^2_{N=1610} = \mathbf{1401.20}$ ($\Delta\chi^2 = \Delta\mathrm{AIC} = \Delta\mathrm{BIC} = \mathbf{-6.47}$ vs. $\Lambda$CDM `M0` $1407.67$ at identical $k=4$, and $\Delta\mathrm{BIC} = \mathbf{-14.71}$ vs. 6-parameter free CPL `M1b` $1401.14$; $\Delta\chi^2 = \mathbf{-6.57}$ on `SN+BAO+CMB+BBN` $N=1,595$; `Forward H2-Sachs` $p=1$: $\chi^2_{N=1610} = \mathbf{1401.30}$, $\Delta\chi^2 = \mathbf{-6.37}$).
   - **Zero hand-typed `M1b` covariance numbers:** `verify_C9` computes the exact $5\times 5$ numerical Hessian of the bundled $N=1,610$ likelihood around `M1b`'s optimum $(-0.8895, -0.2250)$, yielding $\sigma_{w_0} = 0.0597$, $\sigma_{w_a} = 0.2686$, $\rho = -0.8860$.

3. **[Referee Point 3] True Asymptotic-Flat MOND & McGaugh RAR Kernels on 43 `JWST/HST` Disks & Formal Retraction of Deep-MOND RAR Equivalence for the Companion Logarithmic Kernel (`verify_C9`):**
   - Explicitly tabulates $g_{\mathrm{obs}}/g_{\mathrm{bar}} = [1.43, 3.14, 6.57, 11.49]$ vs. McGaugh RAR $[1.58, 3.69, 10.51, 32.13]$ at $g_{\mathrm{bar}}/(1.2\times 10^{-10}) = [1, 0.1, 0.01, 0.001]$ and **formally retracts** any claim of deep-MOND RAR equivalence for $v_{\mathrm{log}} \propto r^{-1/2}\ln r \to 0$ (removing the auxiliary Cardano parameter $\mu$ from the Letter).
   - Evaluates the 43 Mancera Piña+ (2025) `JWST/HST` disks primarily using **true asymptotic-flat MOND and RAR laws** ($v_\infty^4 = GM_b a_0(z)$):
     * **Standard Milgrom (1983) MOND $\mu(y) = y/\sqrt{1+y^2}$ at canonical $a_0(0) = 1.20\times 10^{-10}\;\mathrm{m/s^2}$:** $\chi^2 = \mathbf{107.53}$ ($\chi^2/\mathrm{dof} = \mathbf{2.501}$) on $w_{\mathrm{hor}}=-1$ KH vs. $\mathbf{121.65}$ on anchored DES-CPL ($\Delta\chi^2 = \mathbf{+14.12}$ raw / $\mathbf{+5.65}$ rescaled) and $\mathbf{169.44}$ on monotonic $a_0 \propto H(z)$ ($\mathbf{+61.91}$; at $a_0^{\mathrm{KH}}(0) = 0.9015\times 10^{-10}\;\mathrm{m/s^2}$: $\mathbf{95.93}$, $\chi^2/\mathrm{dof} = \mathbf{2.231}$ vs. $102.49$ and $131.05$).
     * **McGaugh+ (2016) RAR $(1 - e^{-\sqrt{y}})^{-1}$ at $a_0(0) = 1.20\times 10^{-10}\;\mathrm{m/s^2}$:** $\chi^2 = \mathbf{186.55}$ vs. $\mathbf{217.44}$ anchored DES-CPL ($\Delta\chi^2 = \mathbf{+30.89}$ raw / $\mathbf{+7.12}$ rescaled) and $\mathbf{297.76}$ monotonic $H(z)$.

4. **[Referee Points 4–5] Classical Optical Citations & Primary-Source DES/DESI Precision:**
   - Cites Kantowski (1969), Dyer & Roeder (1972, 1973), and Weinberg (1976) in Theorem 2, and separates DES Y6 alone $(-0.84 \pm 0.10, -0.44^{+0.60}_{-0.55})$, DES Y6 + DESI BAO $(-0.84^{+0.06}_{-0.07}, -0.53^{+0.33}_{-0.28})$, and DES Y6 + CMB $(-0.82 \pm 0.05, -0.63^{+0.21}_{-0.18}$; `arXiv:2605.27221`) alongside DESI DR2 (`arXiv:2503.14738`) and DES-SN5YR (`arXiv:2401.02929`).

---

## 2. Standalone Verification Commands (`NumPy` + `SciPy` Only, No `SymPy` Required)
```bash
python3 supplementary_code_and_data/verify_C9_void_clock_timescape_apparent_w0_wa_audit.py
python3 supplementary_code_and_data/verify_C5_kh_background_dependence_audit.py
python3 supplementary_code_and_data/verify_C3_C4_paper10_nogo_and_des2026_cpl_audit.py
python3 supplementary_code_and_data/verify_C6_C7_canonical_cosmological_closure_and_likelihoods.py
```
