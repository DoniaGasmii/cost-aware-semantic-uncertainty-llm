"""
Generic plotting utilities that work with different experiment types.
"""

import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from typing import Dict, List, Any, Union


def set_publication_style():
    """Set matplotlib style for publication-quality figures."""
    plt.rcParams.update({
        'font.size': 13,
        'axes.labelsize': 15,
        'axes.titlesize': 16,
        'xtick.labelsize': 12,
        'ytick.labelsize': 12,
        'legend.fontsize': 12,
        'figure.titlesize': 18,
        'figure.dpi': 100,
        'savefig.dpi': 300,
        'savefig.bbox': 'tight',
        'axes.grid': True,
        'grid.alpha': 0.3,
        'grid.linestyle': ':',
    })


def plot_metric_generic(
    summary: Dict[Any, Dict[str, Any]],
    output_path: Union[str, Path],
    metric: str = 'speedup_token_steps',
    x_label: str = 'X-axis',
    title: str = 'Plot Title',
    show_plot: bool = False,
    add_values: bool = True,
    x_values: List[Any] = None,
    x_labels: List[str] = None
):
    """
    Generic plot for any metric across any grouping variable.

    Args:
        summary: Summary statistics grouped by some variable
        output_path: Path to save figure
        metric: Metric to plot
        x_label: Label for x-axis
        title: Plot title
        show_plot: Whether to display the plot
        add_values: Whether to add value labels
        x_values: Optional explicit x-values (for numerical x-axis)
        x_labels: Optional explicit x-labels (for categorical x-axis)
    """
    set_publication_style()

    fig, ax = plt.subplots(figsize=(12, 7))

    # Get keys
    keys = sorted(summary.keys()) if x_values is None else x_values

    # Extract data
    means = [summary[k][metric]['mean'] for k in keys]
    stds = [summary[k][metric]['std'] for k in keys]

    # Metric-specific styling
    metric_info = {
        'speedup_token_steps': {
            'color': '#2E86AB',
            'ylabel': 'Speedup Factor (Naive / EAB)',
            'baseline': 1.0,
            'format': '{:.2f}×'
        },
        'speedup_time': {
            'color': '#A23B72',
            'ylabel': 'Speedup Factor (Naive / EAB)',
            'baseline': 1.0,
            'format': '{:.2f}×'
        },
        'speedup_memory': {
            'color': '#F18F01',
            'ylabel': 'Memory Ratio (EAB / Naive)',
            'baseline': 1.0,
            'format': '{:.2f}×'
        }
    }

    info = metric_info.get(metric, {
        'color': '#2E86AB',
        'ylabel': 'Value',
        'baseline': None,
        'format': '{:.2f}'
    })

    # Decide if x-axis is numerical or categorical
    if x_values is None and all(isinstance(k, (int, float)) for k in keys):
        # Numerical x-axis
        x_plot = keys
    else:
        # Categorical x-axis
        x_plot = range(len(keys))

    # Plot with error bars
    ax.errorbar(
        x_plot, means, yerr=stds,
        color=info['color'],
        marker='o',
        markersize=10,
        linewidth=2.5,
        capsize=5,
        capthick=2,
        alpha=0.9
    )

    # Add value labels if requested
    if add_values:
        for x, y in zip(x_plot, means):
            ax.annotate(
                info['format'].format(y),
                xy=(x, y),
                xytext=(0, 10),
                textcoords='offset points',
                ha='center',
                va='bottom',
                fontsize=10,
                fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor='gray', alpha=0.8)
            )

    # Reference line
    if info['baseline'] is not None:
        ax.axhline(
            y=info['baseline'],
            color='red',
            linestyle='--',
            linewidth=2,
            alpha=0.7,
            label='No speedup baseline'
        )

    # Labels and title
    ax.set_xlabel(x_label, fontweight='bold', fontsize=15)
    ax.set_ylabel(info['ylabel'], fontweight='bold', fontsize=15)
    ax.set_title(title, fontweight='bold', pad=20, fontsize=16)
    ax.legend(loc='best', framealpha=0.95, fontsize=12)
    ax.grid(True, alpha=0.3, linestyle=':', linewidth=1)

    # Set y-axis to start at 0
    ax.set_ylim(bottom=0)

    # Handle x-axis labels
    if x_labels is not None:
        ax.set_xticks(x_plot)
        ax.set_xticklabels(x_labels, rotation=45 if len(x_labels) > 5 else 0)
    elif not all(isinstance(k, (int, float)) for k in keys):
        ax.set_xticks(x_plot)
        ax.set_xticklabels(keys, rotation=45)

    # Add subtle background
    ax.set_facecolor('#f9f9f9')

    # Save
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches='tight', dpi=300, facecolor='white')
    print(f"  ✓ Saved: {output_path.name}")

    if show_plot:
        plt.show()
    else:
        plt.close()


def generate_clean_plots_for_experiment(
    summary: Dict[Any, Dict[str, Any]],
    output_dir: Union[str, Path],
    experiment_name: str,
    x_label: str,
    grouping_key: str = None,
    x_values: List[Any] = None,
    x_labels: List[str] = None
):
    """
    Generate all three metric plots for any experiment.

    Args:
        summary: Summary statistics
        output_dir: Directory to save plots
        experiment_name: Name for titles (e.g., "Prompt Length", "Model Size")
        x_label: Label for x-axis
        grouping_key: Key used for grouping in summary (e.g., 'by_length', 'by_model')
        x_values: Optional explicit x-values
        x_labels: Optional explicit x-labels
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nGenerating plots for {experiment_name}...")

    # 1. Token-steps speedup
    plot_metric_generic(
        summary,
        output_dir / "speedup_token_steps.png",
        metric='speedup_token_steps',
        x_label=x_label,
        title=f'EAB Token-Step Speedup vs {experiment_name}',
        add_values=True,
        x_values=x_values,
        x_labels=x_labels
    )

    # 2. Wall-clock speedup
    plot_metric_generic(
        summary,
        output_dir / "speedup_wallclock.png",
        metric='speedup_time',
        x_label=x_label,
        title=f'EAB Wall-Clock Speedup vs {experiment_name}',
        add_values=True,
        x_values=x_values,
        x_labels=x_labels
    )

    # 3. Memory overhead
    plot_metric_generic(
        summary,
        output_dir / "memory_overhead.png",
        metric='speedup_memory',
        x_label=x_label,
        title=f'EAB Memory Overhead vs {experiment_name}',
        add_values=True,
        x_values=x_values,
        x_labels=x_labels
    )

    print(f"✓ Generated 3 plots in {output_dir}")
