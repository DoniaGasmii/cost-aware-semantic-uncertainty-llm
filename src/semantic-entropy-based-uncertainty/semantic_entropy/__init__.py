"""Semantic entropy-based uncertainty quantification for LLMs."""

from .estimator import SemanticUncertaintyEstimator
from .clustering import SemanticClusterer
from .utils import (
    compute_entropy,
    normalize_entropy,
    cluster_probabilities_uniform,
    cluster_probabilities_weighted,
    format_uncertainty_report
)

__version__ = "0.1.0"

__all__ = [
    "SemanticUncertaintyEstimator",
    "SemanticClusterer",
    "compute_entropy",
    "normalize_entropy",
    "cluster_probabilities_uniform",
    "cluster_probabilities_weighted",
    "format_uncertainty_report",
]