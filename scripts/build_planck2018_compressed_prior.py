#!/usr/bin/env python3
"""
build_planck2018_compressed_prior.py
====================================
Computes the exact weighted posterior mean and 3x3 covariance matrix for
  (omega_b, omega_cdm, 100 * theta_*)
directly from the four official Planck 2018 PR3 MCMC chains:
  base/plikHM_TTTEEE_lowl_lowE_lensing/base_plikHM_TTTEEE_lowl_lowE_lensing_{1..4}.txt
(from COM_CosmoParams_base-plikHM_R3.01.zip, IRSA / Planck Legacy Archive)
and updates `data/planck2018_compressed_priors.json`.

Usage:
  python3 build_planck2018_compressed_prior.py /path/to/COM_CosmoParams_base-plikHM_R3.01.zip
"""

from __future__ import annotations

import json
import sys
import zipfile
from pathlib import Path
from typing import Dict, List

import numpy as np

SUP_DIR = Path(__file__).resolve().parent
PRIOR_JSON = SUP_DIR / "data" / "planck2018_compressed_priors.json"


def compute_weighted_planck_3x3(chain_zip_path: Path) -> Dict[str, object]:
    prefix = "base/plikHM_TTTEEE_lowl_lowE_lensing/base_plikHM_TTTEEE_lowl_lowE_lensing"
    with zipfile.ZipFile(chain_zip_path, "r") as zf:
        pnames_raw = zf.read(f"{prefix}.paramnames").decode("utf-8").splitlines()
        pnames = [line.split()[0] for line in pnames_raw if line.strip()]
        # In CosmoMC chain files: col 0 = weight, col 1 = -ln(L), cols 2.. = paramnames
        idx_ob = 2 + pnames.index("omegabh2")
        idx_oc = 2 + pnames.index("omegach2")
        # Prefer exact recombination 100*theta* (`thetastar*`) rather than `theta` (100*theta_MC):
        theta_col_name = "thetastar*" if "thetastar*" in pnames else "thetastar"
        idx_th = 2 + pnames.index(theta_col_name)

        weights_list: List[np.ndarray] = []
        samples_list: List[np.ndarray] = []
        for k in range(1, 5):
            raw = zf.read(f"{prefix}_{k}.txt").decode("utf-8").splitlines()
            arr = np.loadtxt(raw, usecols=(0, idx_ob, idx_oc, idx_th), dtype=np.float64)
            weights_list.append(arr[:, 0])
            samples_list.append(arr[:, 1:4])

    w = np.concatenate(weights_list)
    X = np.vstack(samples_list)
    w_sum = float(np.sum(w))
    mean = np.sum(w[:, None] * X, axis=0) / w_sum
    dx = X - mean[None, :]
    cov = np.einsum("ni,n,nj->ij", dx, w, dx) / w_sum
    std = np.sqrt(np.diag(cov))
    corr = cov / np.outer(std, std)
    return {
        "source": (
            "Planck Collaboration 2020 (A&A 641, A6) base_plikHM_TTTEEE_lowl_lowE_lensing PR3 MCMC "
            f"chains 1..4 weighted posterior (column `{theta_col_name}` = 100*theta_*, N_samples={len(w)})"
        ),
        "param_order": ["omega_b", "omega_cdm", "100_theta_star"],
        "mean": [float(f"{v:.8g}") for v in mean],
        "cov": [[float(f"{cov[i, j]:.8g}") for j in range(3)] for i in range(3)],
        "std_diagnostics": [float(f"{v:.6g}") for v in std],
        "corr_diagnostics": [[float(f"{corr[i, j]:.5f}") for j in range(3)] for i in range(3)],
    }


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python3 build_planck2018_compressed_prior.py <COM_CosmoParams_base-plikHM_R3.01.zip>")
        sys.exit(1)
    zip_p = Path(sys.argv[1]).expanduser().resolve()
    res = compute_weighted_planck_3x3(zip_p)
    doc = json.loads(PRIOR_JSON.read_text(encoding="utf-8"))
    doc["planck2018_mcmc_3x3_omegab_omegac_100thetastar"] = {
        "source": res["source"],
        "param_order": res["param_order"],
        "mean": res["mean"],
        "cov": res["cov"],
    }
    PRIOR_JSON.write_text(json.dumps(doc, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(res, indent=2))
