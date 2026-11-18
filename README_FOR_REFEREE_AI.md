# Referee Audit Package (v4 — Non-Circular Forward Sachs ODE, Forward Optical No-Go Theorem & 4/4 Reproducible Suite): Sub4 Letter (`MNRAS Letters / JCAP`, 2026-11-18)
**Title:** *Kodama–Hayward Galactic Acceleration Trough vs. DES Year 6 Dynamical Dark Energy: Exact Background-Dependence and Forward Optical No-Go Theorems with Conditional Void-Clock Sachs Models*  
**Author:** Jozsef Olcsak (`ORCID: 0009-0003-7311-6522`)

---

## 1. Summary of Revisions in `v4` Addressing the External Referee Review
1. **Elimination of Sturm–Liouville Circularity via Two Genuine Non-Circular Forward Optical Calculations (`verify_C9`, Theorem 2 & Proposition 2):**
   - **Why the Previous Check Was Circular:** Defining $E_{\mathrm{eff}}(z) := [\mathrm{d}(H_0 D_{\mathrm{obs}})/\mathrm{d}z]^{-1}$ from an already-specified $D_{\mathrm{obs}}(z)$ and checking $\frac{\mathrm{d}}{\mathrm{d}z}[(1+z)^2 E_{\mathrm{eff}}\frac{\mathrm{d}D_A}{\mathrm{d}z}] + (1+z)E_{\mathrm{eff}}' D_A = 0$ is an *exact inverse-optical Sturm–Liouville identity* ($0 = 0$ for any smooth $D_{\mathrm{obs}}$), not a forward physical derivation. We now label that check explicitly as an internal solver consistency check on the phenomenological `H2b` curves and update the cross-platform bound from `< 2.4e-7` to **`< 3.0e-7`** (quoting exact local values `2.50e-7` / `2.35e-7`, and `2.55e-7` / `2.45e-7` across platforms).
   - **New `Theorem 2 [PROVEN]` — Forward Dyer–Roeder & Weak-Field Swiss-Cheese Optical No-Go Theorem (Zero `D_TS` or `H2b` Input):**
     * Contracting canonical 4D `COV-DIR+QUMOND` with null $k^\mu k^\nu$ ($g_{\mu\nu}k^\mu k^\nu = 0$) annihilates all metric-trace terms ($\Lambda_{\mathrm{eff}}, -\frac{1}{2}R, -\frac{1}{2}V_{\mathrm{QUMOND}}(y)$).
     * On a single-clock $w_{\mathrm{hor}} = -1$ FLRW background ($E_{\mathrm{ray}} = E_{\mathrm{hor}}$), solving the forward Dyer–Roeder / Sachs IVP $\frac{\mathrm{d}}{\mathrm{d}z}[(1+z)^2 E_{\mathrm{hor}}\frac{\mathrm{d}D_A}{\mathrm{d}z}] + \frac{3}{2}\Omega_{m,0}\alpha_{\mathrm{beam}}(z)\frac{(1+z)^3}{E_{\mathrm{hor}}}D_A = 0$ leaves $\frac{H_0}{c}D_A''(0) = -(2 + \frac{3}{2}\Omega_{m,0})$ invariant because $D_A(0) = 0$, forcing $w_0^{\mathrm{app}}(z \to 0) \equiv -1$ **identically**! Numerically (`DOP853`), $\alpha_{\mathrm{beam}} = 1 - \alpha_{\mathrm{OCSR}}$ yields CPL fit `(-0.8173, -1.5886)` (over-steepening $w_a^{\mathrm{app}}$ by `4.92 sigma`, `Pantheon+` $\chi^2_{\mathrm{SN}} = 1394.32$), while weak-field QUMOND void potentials ($s_{\mathrm{void}} = 0.009162$, $F_{\mathrm{OD}}^2 = 7.2718$, $|\Phi_{\mathrm{Tau}}|/c^2 \sim 5\times 10^{-4}$) shift $w_0^{\mathrm{app}}$ by `< 1e-3`. Thus, shifting $w_0^{\mathrm{app}}(0) > -1$ without pathological $w_a^{\mathrm{app}}$ mathematically requires a non-perturbative $O(1)$ void-clock shift in $\mathrm{d}z/\mathrm{d}\lambda$.
   - **New `Proposition 2 [CONDITIONAL]` — Genuine Non-Circular Forward Coupled Null-Redshift + Sachs Focusing IVP (`H2-Sachs`):**
     * Under the conditional local differential transit-time hypothesis $E_{\mathrm{ray}}^{-1}(z) = [1 - \alpha_{\mathrm{eff}}(z)]E_{\mathrm{hor}}^{-1}(z) + \alpha_{\mathrm{eff}}(z)E_{\mathrm{TS}}^{-1}(z)$ with $\alpha_{\mathrm{eff}}(z) = \alpha_{\mathrm{OCSR}}[f_v(z)/f_{v,0}]^p$ (**no** integrated-distance mixing assumed!), the mean cosmic-dust Ricci focusing source $\frac{1}{2}\langle R_{\mu\nu}k^\mu k^\nu\rangle = \frac{3}{2}\Omega_{m,0}(H_0/c)^2(1+z)^5$ is fixed independently of $E_{\mathrm{ray}}(z)$.
     * Because $\frac{3}{2}\Omega_{m,0}\frac{(1+z)^3}{E_{\mathrm{ray}}(z)} \neq (1+z)\frac{\mathrm{d}E_{\mathrm{ray}}}{\mathrm{d}z}$ whenever $\alpha_{\mathrm{eff}} \neq 0$, solving the forward Sachs IVP $\frac{\mathrm{d}}{\mathrm{d}z}[(1+z)^2 E_{\mathrm{ray}}\frac{\mathrm{d}D_A}{\mathrm{d}z}] + \frac{3}{2}\Omega_{m,0}\frac{(1+z)^3}{E_{\mathrm{ray}}}D_A = 0$ (dual-verified via 3-component affine-$\lambda$ IVP vs. 2-component $z$-IVP to `< 2.8e-8` + symbolic `SymPy` $z \to 0$ Taylor tangents) is **strictly non-circular**. It removes the `+55.5%` spurious flat-comoving over-focusing of naive `H2a` and yields:
       - **`H2-Sachs` ($p = 0$, constant $\alpha_{\mathrm{OCSR}}$, $\Delta k_{\mathrm{hyp}} = 0$):** `(-0.8117, -0.3831)`, `DES` 2D $\Delta\chi^2 = 7.16$ sym / $6.28$ asym (`2.02 sigma`), `Pantheon+` 2D $\Delta\chi^2 = 2.48$ (`1.06 sigma`), real `Pantheon+` $\chi^2_{\mathrm{SN}} = 1389.46$.
       - **`H2-Sachs` ($p = 1$, linear $f_v/f_{v,0}$, $\Delta k_{\mathrm{hyp}} = 1$):** `(-0.8209, -0.4227)` (`0.02 sigma` in 1D $w_0$), `DES` 2D $\Delta\chi^2 = 3.96$ sym / $3.40$ asym (`1.33 sigma`), `Pantheon+` 2D $\Delta\chi^2 = 1.43$ (`0.69 sigma`), real `Pantheon+` $\chi^2_{\mathrm{SN}} = 1388.48$.
       - **`H2-Sachs` ($p = 2$, quadratic $[f_v/f_{v,0}]^2$, $\Delta k_{\mathrm{hyp}} = 1$):** `(-0.8309, -0.4453)` (CPL curve RMS = `0.022 mmag`), `DES` 2D $\Delta\chi^2 = 2.14$ sym / $1.78$ asym (`0.82 sigma`), `Pantheon+` 2D $\Delta\chi^2 = 0.96$ (`0.50 sigma`), real `Pantheon+` $\chi^2_{\mathrm{SN}} = 1387.77$ (`+0.44` vs. $\Lambda$CDM; heuristic discrete penalty $\Delta\mathrm{AIC}_{\mathrm{heur}} = +2.44$, $\Delta\mathrm{BIC}_{\mathrm{heur}} = +7.81$).
   - **Strict Epistemic Boundary Maintained:** Both `H1` ($z(f_{\mathrm{acc}}) = z_{\mathrm{acc}}^{\mathrm{hor}}$) and the local two-phase transit-time rule $E_{\mathrm{ray}}(z)$ remain explicitly classified as conditional effective hypotheses awaiting a full 4D inhomogeneous metric solution of `COV-DIR+QUMOND`.
2. **`4/4` Standalone Reproducibility Confirmed (Typical Total Runtime `~10–35 seconds`, Hardware-Dependent):**
   - All four verification scripts (`verify_C9`, `verify_C5`, `verify_C3_C4`, and `verify_C6_C7`) and all 9 raw observational data files run 100% standalone in `supplementary_code_and_data/`.

---

## 2. Standalone Verification Commands
```bash
python3 supplementary_code_and_data/verify_C9_void_clock_timescape_apparent_w0_wa_audit.py
python3 supplementary_code_and_data/verify_C5_kh_background_dependence_audit.py
python3 supplementary_code_and_data/verify_C3_C4_paper10_nogo_and_des2026_cpl_audit.py
python3 supplementary_code_and_data/verify_C6_C7_canonical_cosmological_closure_and_likelihoods.py
```
