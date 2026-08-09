"""borehole_geometry — well-interference modelling from 3-D drilling geometry.

Pipeline (one module per phase; see docs/RECIPE.md):
    cohort -> legs -> pairs -> context -> dataset -> baselines -> model/train -> score -> report

Deterministic geometry (cohort..context) produces parquet that is useful on its own;
the PyTorch set-encoder (model/train/score) adds the counterfactual interference estimate.
"""

__version__ = "0.1.0"
