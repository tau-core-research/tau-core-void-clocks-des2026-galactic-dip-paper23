# Referee & Reproducibility Guide (v8.0 — JCAP Major Revision)

## 1. True Single-Command Regeneration (JSON + LaTeX Tables + Figure 1)

Execute the complete multi-compilation likelihood, dark-energy control, out-of-sample SN split, and deduplicated 3D void-sightline pipeline from bundled public data (~1–3 minutes on a single CPU core; core audit requires only `numpy` and `scipy`, and automatically calls bundled `generate_sub4_figures_and_tables.py` if `matplotlib` is installed):

```bash
python3 supplementary_code_and_data/verify_C9_void_clock_timescape_apparent_w0_wa_audit.py
```

Custom flags for independent experimentation:
- `--no-asserts`: disables unit-test assertions when testing modified priors or formulas.
- `--no-figures`: runs only the pure `NumPy`/`SciPy` numerical audit without `matplotlib`.

Generated outputs:
1. `supplementary_code_and_data/C9_void_clock_timescape_apparent_w_results.json`
2. `supplementary_code_and_data/generated_tables_v8.tex`
3. `fig_sub4_void_clock_des2026_and_galactic_dip_en.png` & `fig_sub4_void_clock_des2026_and_galactic_dip_hu.png`

---

## 2. Point-by-Point Checklist of v8.0 Fixes for Referees 1 & 2

1. **Bug Fix for Python 3.9 / NumPy 1.26 (`f_acc` Root):** Factored $(f_v - 1)(8f_v^3 + 39f_v^2 - 12f_v - 8) = 0$ analytically so $f_v = 1$ can never round below unity (`f_acc = 0.5867077311`).
2. **Exact Recombination Scaling & Cumulative $\Delta D_M(z_*)$:** Fixed $f_v(z) \propto (1+z)^{-3/2}$ ($f_v(1089.92) = 8.68\times 10^{-5}$) and reported the non-zero cumulative shift $\Delta D_M(z_*) \simeq -10.6$ to $-14.0\;\mathrm{Mpc}$ explicitly.
3. **CAMB $r_d$ Baseline & $+lensing$ Provenance:** Calibrated baseline $r_d$ to exact CAMB ($147.102\;\mathrm{Mpc}$ at Planck fiducial) on a smooth all-$z$ log-grid $u = \ln(1+z) \in [0, \ln 1201]$ (zero $z = 2.5$ stitch), added Aubourg et al. (2015) and $\pm 0.15\%$ $r_d$ sensitivity checks, and updated `planck2018_compressed_priors.json` to `TT,TE,EE+lowE+lensing` (Table 2 col. 6).
4. **Head-to-Head $k = 5$ Dark-Energy Controls & No $p = 0$ Strawman:** Tabulated free-$\alpha$ $p = 0, 1, 2$ alongside $w$CDM ($k = 5$), canonical NEC-obeying thawing quintessence $w_a = -1.58(1+w_0)$ ($k = 5$; `H2-vol` improves by $\Delta\chi^2 = -6.99$ to $-9.64$), and the empirical phantom-crossing CPL degeneracy line $w_a = -3.3(1+w_0)$ ($k = 5$; $|\Delta\chi^2| \le 1.73$, demonstrating background geometric degeneracy).
5. **`BAO + Planck18`-Only Out-of-Sample SN Test & Full $f_{v,0} \in [0.60, 0.84]$ Scan:** Added the zero-SN `BAO + CMB` fit ($\Delta\chi^2 = -7.43$, $\hat{\alpha} = 0.521$) with parameter-free out-of-sample improvements on all four SN datasets ($\Delta\chi^2_{\mathrm{SN}} = -1.48$ to $-11.60$), documented compilation overlap (`DES-Dovekie` shares $\sim 94\%$ of CIDs with `DES-SN5YR`), and reported the full $f_{v,0} \in [0.60, 0.84]$ scan showing strong $(\alpha, f_{v,0})$ degeneracy.
6. **Deduplicated Exploratory 3D Sightline Null Check:** Contracted duplicate survey rows ($311 \to N_{\mathrm{unique}} = 280$ physical SNe), added 8-patch sky-block permutations, cited Bolejko et al. (2013, *PRL* 110, 021302), removed `">5\sigma"` claims, and reported the $z \le 0.105$ test strictly as a $\sim 0.37\sigma$ exploratory null check.
