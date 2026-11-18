# Tau Core Paper 23 (Wedge Letter v2 — `MNRAS Letters / JCAP`)
## Kodama–Hayward Galactic Acceleration Trough vs. DES Year 6 Dynamical Dark Energy: An Exact Background-Dependence Theorem and Conditional Void-Clock Optical Hypotheses

**Author:** Jozsef Olcsak (`ORCID: 0009-0003-7311-6522`)  
**Date:** 18 November 2026 (Revised Referee Edition v2)  
**Repository Visibility:** Private pre-submission repository  
**Epistemic Closure Status:** `NATURE_LEVEL_CLOSURE = false`

---

## 1. Repository Structure (100% Standalone Referee Bundle)
- `Tau_Core_Paper_23_Void_Clocks_DES2026_and_Galactic_Dip_EN.pdf` — Compiled English Letter manuscript (`manuscript_en/main.tex`).
- `Tau_Core_Paper_23_Void_Clocks_DES2026_and_Galactic_Dip_HU.pdf` — Compiled Hungarian Letter manuscript (`manuscript_hu/main_HU.tex`).
- `Sub4_MNRAS_JCAP_Wedge_Letter_Referee_Package.zip` — Self-contained referee audit zip bundle (includes all raw observational datasets in `supplementary_code_and_data/data/`).
- `COVER_LETTER.txt` — Journal submission cover letter (*MNRAS Letters* / *JCAP*).
- `README_FOR_REFEREE_AI.md` — Structured guide for automated referee auditing (v2).
- `figures/` — 300 DPI bilingual 3-panel figures (`fig_sub4_void_clock_des2026_and_galactic_dip_en.png` and `_hu.png`).
- `scripts/` — Standalone `SymPy` + `SciPy` verification scripts (`verify_C9_void_clock_timescape_apparent_w0_wa_audit.py`, `verify_C5_kh_background_dependence_audit.py`, `verify_C3_C4_paper10_nogo_and_des2026_cpl_audit.py`, `verify_C6_C7_canonical_cosmological_closure_and_likelihoods.py`) and bundled raw data (`scripts/data/`).
- `data/derived/` — Machine-verified JSON and CSV outputs.

## 2. Quick Standalone Verification (4/4 Suites PASS)
```bash
python3 scripts/verify_C9_void_clock_timescape_apparent_w0_wa_audit.py
python3 scripts/verify_C5_kh_background_dependence_audit.py
python3 scripts/verify_C3_C4_paper10_nogo_and_des2026_cpl_audit.py
python3 scripts/verify_C6_C7_canonical_cosmological_closure_and_likelihoods.py
```
