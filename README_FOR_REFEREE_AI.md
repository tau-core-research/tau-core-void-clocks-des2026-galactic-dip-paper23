# Referee Audit Package: Sub4 Wedge Letter (`MNRAS Letters / JCAP`, 2026-11-18)
**Title:** *Apparent Dynamical Dark Energy from Void Clock Rates and the $z = 0.6316$ Galactic Acceleration Trough: A Zero-Parameter Synthesis of DES Year 6 Supernovae, MNRAS Void-Clock Cosmology, and JWST/ALMA Disk Kinematics*  
**Author:** Jozsef Olcsak (`ORCID: 0009-0003-7311-6522`)

## 1. Contents of This Archive
- `main.tex` & `Sub4_MNRAS_JCAP_Letter_Void_Clocks_DES2026_and_Galactic_Dip.pdf` — English manuscript (6 pages, 3 theorems, 1 figure, 1 summary table).
- `main_HU.tex` & `Sub4_MNRAS_JCAP_Letter_Void_Clocks_DES2026_and_Galactic_Dip_HU.pdf` — Complete Hungarian edition.
- `fig_sub4_void_clock_des2026_and_galactic_dip_en.png` / `_hu.png` — 300 DPI 3-panel figures.
- `COVER_LETTER.txt` — Submission cover letter for *MNRAS Letters* / *JCAP*.
- `supplementary_code_and_data/` — Standalone executable Python (`SymPy` + `SciPy`) verification scripts and machine-readable JSON/CSV outputs:
  - `verify_C9_void_clock_timescape_apparent_w0_wa_audit.py` (`C9_void_clock_timescape_apparent_w_results.json`)
  - `verify_C5_kh_background_dependence_audit.py` (`C5_kh_background_dependence_results.json`)
  - `verify_C3_C4_paper10_nogo_and_des2026_cpl_audit.py` (`C3_C4_paper10_nogo_and_des2026_results.json`)
  - `verify_C6_C7_canonical_cosmological_closure_and_likelihoods.py` (`C6_C7_canonical_closure_and_likelihood_results.json`)
  - `Tau_DES_2026_background_comparison.csv`
  - `C7_multiprobe_M0_M1_M2_comparison.csv`
  - `C7_growth_fsigma8_and_lensing_slip_comparison.csv`

## 2. Quick Verification Command
```bash
python3 supplementary_code_and_data/verify_C9_void_clock_timescape_apparent_w0_wa_audit.py
python3 supplementary_code_and_data/verify_C5_kh_background_dependence_audit.py
python3 supplementary_code_and_data/verify_C3_C4_paper10_nogo_and_des2026_cpl_audit.py
```

## 3. Epistemic Scope and Mathematical Claims Audited
1. **Theorem 1 (`PROVEN` analytically & `NUMERICAL`):**
   - For any smooth $w(a)$, the Kodama-Hayward scale $a_0^{\mathrm{KH}}(z) = \frac{5\sqrt{3}}{96}cH(z)|1-q(z)|$ satisfies $\left.\frac{1}{H}\frac{\mathrm{d}[H(1-q)]}{\mathrm{d}\ln a}\right|_{q=0} = -\frac{3}{2}(1+w) + \frac{1}{2w}\frac{\mathrm{d}w}{\mathrm{d}\ln a}$.
   - On a homogeneous DES 2026 CPL background ($w_0 = -0.82, w_a = -0.63$, $\Omega_{m,0} = 0.3153$), $a_0^{\mathrm{KH}}(0.6316)/a_0^{\mathrm{KH}}(0) = 1.142720$ (erasing the $0.938550$ trough in $99.725\%$ of draws and degrading the 43 JWST/HST disk fit by $\Delta\chi^2 = +15.22$).
2. **Theorem 2 (`PROVEN`):**
   - In the Buchert-Wiltshire tracker (`MNRAS` 536, 1752, 2025), $q_{\mathrm{wall}}(f_v) = \frac{8f_v^4+31f_v^3-51f_v^2+4f_v+8}{16f_v^4+8f_v^3+33f_v^2+8f_v+16} = 0 \iff f_{\mathrm{acc}} = 0.58670773$.
   - Locking $z(f_{\mathrm{acc}}) = z_{\mathrm{acc}} = (2\Omega_{\Lambda,0}/\Omega_{m,0})^{1/3} - 1 = 0.6315565$ uniquely yields $f_{v,0} = 0.723358$ ($72.34\%$) and $\bar{\gamma}_0 = 1.361679$ with zero free parameters.
3. **Theorem 3 (`PROVEN` formula / `CONDITIONAL` on OCSR optical weight $\alpha_{\mathrm{OCSR}} = \frac{5\sqrt{3}}{24}$):**
   - Exact $z=0$ slope of the Wiltshire null distance: $\left.\frac{\mathrm{d}E_{\mathrm{TS}}}{\mathrm{d}z}\right|_{z=0} = \frac{3(8f_0^4+13f_0^3-6f_0^2+4f_0+8)}{16f_0^4+8f_0^3+33f_0^2+8f_0+16} = 0.959784$.
   - For $D_{\mathrm{obs}}(z) = (1 - \alpha_{\mathrm{OCSR}})D_{\mathrm{hor}}(z) + \alpha_{\mathrm{OCSR}}D_{\mathrm{TS}}(z)$, a flat-FLRW observer infers $w_0^{\mathrm{app}}(0) = -0.828956$ and full-range CPL $(w_0^{\mathrm{app}}, w_a^{\mathrm{app}}) = (-0.8251, -0.1006)$ ($0.10\sigma$ vs. DES 2026 $w_0 = -0.82 \pm 0.05$).
