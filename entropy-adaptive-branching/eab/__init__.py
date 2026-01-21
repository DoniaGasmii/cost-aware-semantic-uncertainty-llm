"""
Entropy-Adaptive Branching for Efficient Multi-Sample Generation

This package provides an efficient alternative to naive multi-sample generation
by branching only when the model is uncertain (high entropy), while sharing
computation for confident predictions.
"""

from .core import EntropyAdaptiveBranching
from .path import GenerationPath
from .entropy import compute_entropy, normalize_entropy
from .cache import deep_copy_cache, merge_caches
from .cache_cow import CopyOnWriteCache
from .utils import set_seed, format_results, compute_statistics

__version__ = "0.1.0"
__all__ = [
    "EntropyAdaptiveBranching",
    "GenerationPath",
    "CopyOnWriteCache",
    "compute_entropy",
    "normalize_entropy",
    "deep_copy_cache",
    "merge_caches",
    "set_seed",
    "format_results",
    "compute_statistics",
]