"""
Analysis utilities for EAB experiments.
"""

from .metrics import (
    MetricsTracker,
    compute_token_steps,
    compute_speedup,
    compute_efficiency_metrics
)

from .data_utils import (
    save_results,
    load_results,
    save_json,
    load_json
)

from .plotting import (
    plot_speedup_vs_length,
    plot_cost_breakdown,
    plot_branching_analysis
)

__all__ = [
    'MetricsTracker',
    'compute_token_steps',
    'compute_speedup',
    'compute_efficiency_metrics',
    'save_results',
    'load_results',
    'save_json',
    'load_json',
    'plot_speedup_vs_length',
    'plot_cost_breakdown',
    'plot_branching_analysis'
]
