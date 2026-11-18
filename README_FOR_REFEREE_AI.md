# Referee Audit Package (v5 — Exact Published Wiltshire 2009 Distance, Zero-Choice p=0 Baseline & Complete Numerical/Bibliographic Cleanup): Sub4 Letter (`MNRAS Letters / JCAP`, 2026-11-18)
**Title:** *Kodama–Hayward Galactic Acceleration Trough vs. DES Year 6 Dynamical Dark Energy: Exact Background-Dependence and Forward Optical No-Go Theorems with Phenomenological Void-Clock Models*  
**Author:** Jozsef Olcsak (`ORCID: 0009-0003-7311-6522`)

---

## 1. Summary of Revisions in `v5` Addressing All External Referee Critiques
1. **Exact Published Wiltshire (2009) / Lane et al. (2025) Dressed Distance Formula & Primary-Literature Unit Tests (`verify_C9`):**
   - Replaced the simplified $\int \mathrm{d}z/H_{\mathrm{wall}}(z)$ integral with the exact published dressed angular diameter distance of Wiltshire (*Phys. Rev. D* **80**, 123512, 2009, `arXiv:0909.0749`, Eqs. 33–35) and Lane, Seifert, Ridden-Harper & Wiltshire (*MNRAS* **536**, 1752, 2025, `arXiv:2311.01438`, Appendix A, Eqs. A3–A9):
     $$d_A^{\mathrm{TS}}(z) = c\,t(z)^{2/3}\int_{t(z)}^{t_0}\frac{2\,\mathrm{d}\tilde{t}}{\bigl[2 + f_v(\tilde{t})\bigr]\tilde{t}^{2/3}} = c\,t(z)^{2/3}\bigl[\mathcal{F}(t_0) - \mathcal{F}\bigl(t(z)\bigr)\bigr], \qquad D_{\mathrm{TS}}(z) = (1+z)\,d_A^{\mathrm{TS}}(z).$$
   - Added 4 mandatory primary-literature unit tests in `verify_C9`:
     1. Closed-form $\mathcal{F}(t)$ vs. `SciPy` quadrature (`max_diff = 5.55e-16`);
     2. Exact match to Lane et al. (2025) Eq. (A9) $O(z)$ coefficient $c_1(f_{v,0}) \implies \mathrm{d}E_{\mathrm{TS}}/\mathrm{d}z|_0 = 2(1 - c_1(f_{v,0})) = 0.758852$ at $f_{v,0} = 0.723358$;
     3. Exact match to Wiltshire (2009, line 900) canonical benchmark $f_{v,0} = 0.762, \Omega_{M0} = 0.328916 \implies w(0) = -0.757856 \simeq -0.758$;
     4. Exact reproduction of the external referee's pure Timescape `Pantheon+` ($N=1,580$) benchmarks: $\chi^2_{\mathrm{SN}}(f_{v,0}=0.7234) = \mathbf{1403.83}$ (`1403.8`) and free-$f_{v,0}$ best fit $\chi^2_{\mathrm{SN}}(f_{v,0}=0.8526) = \mathbf{1391.55}$ (`1391.5`).
2. **Zero-Choice ($p = 0$, $\Delta k_{\mathrm{hyp}} = 0$) Phenomenological Baselines Outperform $\Lambda$CDM on `Pantheon+` (No $p = 2$ Selection Needed):**
   - With the true Wiltshire distance, no $p = 2$ exponent selection is needed:
     * **Baseline `H2a` ($p = 0$, constant $\alpha_{\mathrm{OCSR}} = \frac{5\sqrt{3}}{24}$, $\Delta k_{\mathrm{hyp}} = 0$):** exact $z \to 0$ limit $w_0^{\mathrm{app}}(0) = \mathbf{-0.8996}$; synthetic $[0.02, 1.20]$ CPL projection $(\mathbf{-0.8843}, \mathbf{-0.2042})$; `Pantheon+` covariance-weighted GLS CPL projection $(\mathbf{-0.8912}, \mathbf{-0.1709})$; **$\mathbf{0.07\sigma}$ ($\Delta\chi^2_{\mathrm{2D}} = 0.11$) from `Pantheon++DESI+CMB` `(-0.8901, -0.2253)`**, **$\mathbf{1.95\sigma}$ ($\Delta\chi^2_{\mathrm{2D}} = 5.94$, inside $95\%$) from `DES Y6+DESI+CMB` `(-0.82, -0.63)`**, and **real $1,580$-supernova `Pantheon+` $\chi^2_{\mathrm{SN}} = \mathbf{1386.59}$ ($\Delta\chi^2_{\mathrm{SN}} = -0.74$ better than baseline $\Lambda$CDM `1387.33`)**.
     * **Baseline `Forward H2-Sachs` ($p = 0$, constant $\alpha_{\mathrm{OCSR}}$, $\Delta k_{\mathrm{hyp}} = 0$):** synthetic CPL $(\mathbf{-0.8885}, \mathbf{-0.2782})$ (GLS: $(\mathbf{-0.8941}, \mathbf{-0.2479})$), **$\mathbf{0.08\sigma}$ ($\Delta\chi^2_{\mathrm{2D}} = 0.13$) from `Pantheon++DESI+CMB`**, **$\mathbf{1.32\sigma}$ ($\Delta\chi^2_{\mathrm{2D}} = 3.35$) from `DES Y6+DESI+CMB`**, and **real `Pantheon+` $\chi^2_{\mathrm{SN}} = \mathbf{1386.21}$ ($\Delta\chi^2_{\mathrm{SN}} = -1.12$ better than $\Lambda$CDM `1387.33`, and lower than free-$\Omega_{m,0}$ flat $\Lambda$CDM `1386.41`)**.
     * Secondary sensitivity grid $p \in \{1, 2\}$ (`H2b-diff`, `H2b-int`, `Forward H2-Sachs`): all variants beat $\Lambda$CDM with $\chi^2_{\mathrm{SN}} \in [1386.06, 1386.31]$ and span $\Delta\chi^2_{\mathrm{2D,DES}} \in [1.83, 4.10]$ ($0.84\sigma\text{--}1.52\sigma$).
3. **Phenomenological Scoping, Removal of Tautological Sturm–Liouville Check, and Complete Internal Numerical/Bibliographic Cleanup:**
   - Removed the tautological inverse Sturm–Liouville check (`E_eff = 1/D'`), framed `H2a`/`H2b`/`H2-Sachs` strictly as phenomenological effective split-beam models, documented the shallow Jacobian of `H1` ($\mathrm{d}f_{v,0}/\mathrm{d}z_{\mathrm{acc}} \approx 0.141$, mapping $z_{\mathrm{acc}} \in [0.456, 0.918]$ to $f_{v,0} \in [69.5\%, 76.0\%]$), qualified the 43 `JWST/HST` disks ($0.787 \le z \le 1.03$) as a post-trough $z \approx 0.9$ amplitude test with $\chi^2/\mathrm{dof} = 2.475$-rescaled $\Delta\chi^2_{\mathrm{rescaled}} = +6.15 / +1.71$ and explained the logarithmic `OCSR` kernel scale $a_0^{\mathrm{KH}}(0) = 0.9015\times 10^{-10}\;\mathrm{m\,s^{-2}}$ vs. McGaugh's $1.20\times 10^{-10}\;\mathrm{m\,s^{-2}}$ RAR scale.
   - Fixed Theorem 1 proof intermediate numbers ($E(z) = 1.450613$, $q(z) = -0.057302$, $q(0) = -0.342181 \implies 1.142720$), unified Table 1 with the exact `verify_C6_C7` joint $N=1,610$ profiled numbers (`M0` $\chi^2_{\mathrm{SN}}=1390.57$, `M1a` $\Delta\chi^2=-4.24$, `M1b` $(-0.8895,-0.2250)$, $\Delta\chi^2=-6.53$, `M2a-Profiled` $\chi^2_{\mathrm{SN}}=1389.62$) and `verify_C9` fixed-$\Omega_{m,0}=0.3153$ numbers ($\chi^2_{\mathrm{SN}}=1387.33$), corrected `arXiv:2311.01438` (Lane+ 2025) and `arXiv:2511.08685` (Mancera Piña+ 2025), and made `SymPy` optional across **all four** verification scripts (`verify_C3_C4`, `verify_C5`, `verify_C6_C7`, `verify_C9`).

---

## 2. Standalone Verification Commands
```bash
python3 supplementary_code_and_data/verify_C9_void_clock_timescape_apparent_w0_wa_audit.py
python3 supplementary_code_and_data/verify_C5_kh_background_dependence_audit.py
python3 supplementary_code_and_data/verify_C3_C4_paper10_nogo_and_des2026_cpl_audit.py
python3 supplementary_code_and_data/verify_C6_C7_canonical_cosmological_closure_and_likelihoods.py
```
