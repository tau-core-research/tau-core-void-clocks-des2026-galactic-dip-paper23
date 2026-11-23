# Paper 23, v15.1-lens-holdout

This evidence update contains the revised EN, HU and EPJ C working
preprints. The previously submitted PRD manuscript is unchanged.

The external test uses the six H0LiCOW XIII lens distance likelihoods
that were not in the training fit. At the fixed forecasts, the smooth
mirage CPL template slightly outscores the linear void optical
template (log-likelihood difference optical minus smooth: -0.341).
This is not decisive: the largest distance separation is 0.504% or
0.104 times the corresponding marginal distance width. The two templates
use the same additional flat effective lens-distance sum rule and the
released lens mass-model assumptions. The result neither tests curved
Timescape nor identifies either physical mechanism.

## Evidence and reproduction

See [README.md](README.md) for commands and the precise evidence scope.
The point forecasts are frozen to the same-data fit receipt, not derived
from a cosmology fit to the lenses. Re-scoring obtains the six distance
inputs from the pinned H0LiCOW repository; those files are not redistributed.
The old matched training refit remains a separate validation step.

The release ZIP includes the EN/HU main sources and figures and the
EPJ C working source and figures under `epjc/`. To build the latter,
use the journal's Springer SVJour3/EPJC class files from the
[EPJ C submission instructions](https://link.springer.com/journal/10052/submission-guidelines).
The archived EN/HU sources use standard LaTeX packages.
