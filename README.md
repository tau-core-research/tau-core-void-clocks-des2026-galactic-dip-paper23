# Tau Core Paper 23 (`MNRAS Letters / JCAP` Letter v5)
## Kodama–Hayward Galactic Acceleration Trough vs. DES Year 6 Dynamical Dark Energy: Exact Background-Dependence and Forward Optical No-Go Theorems with Phenomenological Void-Clock Models

**Author:** Jozsef Olcsak (`ORCID: 0009-0003-7311-6522`)  
**Date:** 18 November 2026 (Revised Referee Edition v5)  
**Repository Visibility:** Public reproducibility repository<br>
**Epistemic Closure Status:** `NATURE_LEVEL_CLOSURE = false`

## 2026-11-23: held-out strong-lens check

The six [H0LiCOW XIII lenses](https://shsuyu.github.io/H0LiCOW/site/h0licow_data.html)
were not used in the Pantheon+ / DESI BAO-13 / compressed Planck training
fit. Using the released per-lens distance likelihoods and the same flat
distance-sum rule for both fixed predictions, the smooth mirage CPL
template scores slightly better than the linear void optical template:
**ln L(void) − ln L(mirage) = −0.341**. Their largest fractional
separation is 0.504% (0.104 of a corresponding marginal distance standard
deviation). The ranking survives the predeclared smoothing and
leave-one-lens-out checks. The result is not decisive and does not test
direction-dependent void transport, curved Timescape, or the entire
class of smooth dark-energy models. The point-likelihood ratio is not a
Bayes factor.

- [Protocol](data/derived/h0licow_holdout_protocol_v1.json)
- [Frozen predictions](data/derived/h0licow_holdout_predictions_v1.json)
- [Scores, controls and six upstream file checksums](data/derived/h0licow_holdout_score_v1.json)

With Python 3.10+, NumPy 2.2.6 and SciPy 1.15.3 (about 122 MB of pinned
public lens inputs are fetched at score time; inputs are not redistributed):

~~~bash
python3 -W error scripts/score_h0licow_holdout.py predict --output /tmp/p23_lens_predictions.json
python3 -W error scripts/score_h0licow_holdout.py score --output /tmp/p23_lens_score.json
~~~

The first command does not access lens data. The second uses and verifies
the shipped frozen predictions by default; pass `--predictions` to use the
newly generated copy. Do not use existing output filenames for a rerun.

## 2026-11-23: matched same-data validation

The English working manuscript includes a separate full-vector re-fit of
Pantheon+, DESI DR2 BAO-13 and the compressed Planck calibration.
The linear optical template gains 9.826 in chi-squared; the smooth mirage
CPL line gains 8.078. The three-template tangent null gives tail 0.0064145
on the local score, not a global nonlinear detection or independent data
test. The smooth comparator is highly aligned (0.927) with the optical
shape after nuisance projection. No mechanism is selected.

Reproduce with Python 3.10 or newer, NumPy 2.2.6 and SciPy 1.15.3:

~~~bash
python3 -W error scripts/validate_pantheon_joint_tangent.py \
  --source '<working Sub4>/supplementary_code_and_data' \
  --output /tmp/pantheon_matched_new.json
~~~

The script verifies the original input hashes; no additional catalog is
pooled. The accepted receipt is
[pantheon_joint_tangent_20261123_strict.json](data/derived/pantheon_joint_tangent_20261123_strict.json).
The previously submitted PRD version remains unchanged.

---

## 1. Repository Structure (100% Standalone Referee Bundle)
- `Tau_Core_Paper_23_Void_Clocks_DES2026_and_Galactic_Dip_EN.pdf` — Compiled English Letter manuscript (`manuscript_en/main.tex`, v5).
- `Tau_Core_Paper_23_Void_Clocks_DES2026_and_Galactic_Dip_HU.pdf` — Compiled Hungarian Letter manuscript (`manuscript_hu/main_HU.tex`, v5).
- `Sub4_MNRAS_JCAP_Letter_Referee_Package.zip` — Self-contained referee audit zip bundle (includes all 9 raw observational datasets in `supplementary_code_and_data/data/`).
- `COVER_LETTER.txt` — Journal submission cover letter (*MNRAS Letters* / *JCAP*, v5).
- `README_FOR_REFEREE_AI.md` — Structured guide for automated referee auditing (v5: exact published Wiltshire 2009 / Lane+ 2025 dressed distance formula with primary-literature unit tests, zero-choice $p=0$ phenomenological baselines outperforming $\Lambda$CDM on `Pantheon+`, optional `SymPy` dependency, and full internal numerical/bibliographic cleanup).
- `figures/` — 300 DPI bilingual 3-panel figures (`fig_sub4_void_clock_des2026_and_galactic_dip_en.png` and `_hu.png`).
- `scripts/` — Standalone `NumPy`/`SciPy` verification scripts (`verify_C9_void_clock_timescape_apparent_w0_wa_audit.py`, `verify_C5_kh_background_dependence_audit.py`, `verify_C3_C4_paper10_nogo_and_des2026_cpl_audit.py`, `verify_C6_C7_canonical_cosmological_closure_and_likelihoods.py`) and bundled raw data (`scripts/data/`).
- `data/derived/` — Machine-verified JSON and CSV outputs.

## 2. Quick Standalone Verification (4/4 Suites PASS)
```bash
python3 scripts/verify_C9_void_clock_timescape_apparent_w0_wa_audit.py
python3 scripts/verify_C5_kh_background_dependence_audit.py
python3 scripts/verify_C3_C4_paper10_nogo_and_des2026_cpl_audit.py
python3 scripts/verify_C6_C7_canonical_cosmological_closure_and_likelihoods.py
```
