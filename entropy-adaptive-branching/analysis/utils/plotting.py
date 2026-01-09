"""
Plotting utilities for EAB experiments.
"""

import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from typing import Dict, List, Any, Union, Optional


def set_publication_style():
    """Set matplotlib style for publication-quality figures."""
    plt.rcParams.update({
        'font.size': 12,
        'axes.labelsize': 14,
        'axes.titlesize': 14,
        'xtick.labelsize': 11,
        'ytick.labelsize': 11,
        'legend.fontsize': 11,
        'figure.titlesize': 16,
        'figure.figsize': (10, 6),
        'figure.dpi': 100,
        'savefig.dpi': 300,
        'savefig.bbox': 'tight',
    })


def plot_speedup_vs_length(
    summary: Dict[int, Dict[str, Any]],
    output_path: Union[str, Path],
    metrics: List[str] = ['speedup_token_steps', 'speedup_time'],
    show_plot: bool = False
):
    """
    Plot speedup vs prompt length with error bars.

    Args:
        summary: Summary statistics by prompt length
        output_path: Path to save figure
        metrics: List of metrics to plot
        show_plot: Whether to display the plot
    """
    set_publication_style()

    fig, ax = plt.subplots(figsize=(10, 6))

    # Sort lengths
    lengths = sorted(summary.keys())

    # Metric display names and colors
    metric_info = {
        'speedup_token_steps': {'label': 'Token-steps', 'color': '#2E86AB', 'marker': 'o'},
        'speedup_time': {'label': 'Wall-clock time', 'color': '#A23B72', 'marker': 's'},
        'speedup_memory': {'label': 'Memory', 'color': '#F18F01', 'marker': '^'}
    }

    # Plot each metric
    for metric in metrics:
        if metric not in metric_info:
            continue

        means = [summary[l][metric]['mean'] for l in lengths]
        stds = [summary[l][metric]['std'] for l in lengths]

        info = metric_info[metric]
        ax.plot(
            lengths, means,
            label=info['label'],
            color=info['color'],
            marker=info['marker'],
            markersize=8,
            linewidth=2,
            linestyle='-'
        )
        ax.fill_between(
            lengths,
            np.array(means) - np.array(stds),
            np.array(means) + np.array(stds),
            alpha=0.2,
            color=info['color']
        )

    # Reference line at y=1 (no speedup)
    ax.axhline(y=1.0, color='gray', linestyle='--', linewidth=1, alpha=0.5, label='No speedup')

    # Labels and title
    ax.set_xlabel('Prompt Length (tokens)', fontweight='bold')
    ax.set_ylabel('Speedup Factor (Naive / EAB)', fontweight='bold')
    ax.set_title('EAB Speedup vs Prompt Length', fontweight='bold', pad=20)
    ax.legend(loc='best', framealpha=0.9)
    ax.grid(True, alpha=0.3, linestyle=':')

    # Set y-axis to start at 0
    ax.set_ylim(bottom=0)

    # Save
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, bbox_inches='tight', dpi=300)
    print(f"✓ Saved plot to {output_path}")

    if show_plot:
        plt.show()
    else:
        plt.close()


def plot_cost_breakdown(
    results: List[Dict[str, Any]],
    output_path: Union[str, Path],
    show_plot: bool = False
):
    """
    Plot stacked bar chart showing cost breakdown (Naive vs EAB).

    Args:
        results: List of result dictionaries
        output_path: Path to save figure
        show_plot: Whether to display the plot
    """
    set_publication_style()

    # Group by length
    by_length = {}
    for result in results:
        length = result['prompt_length']
        if length not in by_length:
            by_length[length] = {'naive': [], 'eab': []}
        by_length[length]['naive'].append(result['naive_metrics']['token_steps'])
        by_length[length]['eab'].append(result['eab_metrics']['token_steps'])

    # Compute means
    lengths = sorted(by_length.keys())
    naive_means = [np.mean(by_length[l]['naive']) for l in lengths]
    eab_means = [np.mean(by_length[l]['eab']) for l in lengths]

    # Plot
    fig, ax = plt.subplots(figsize=(10, 6))

    x = np.arange(len(lengths))
    width = 0.35

    bars1 = ax.bar(x - width/2, naive_means, width, label='Naive', color='#E63946', alpha=0.8)
    bars2 = ax.bar(x + width/2, eab_means, width, label='EAB', color='#06A77D', alpha=0.8)

    # Add value labels on bars
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width()/2., height,
                f'{int(height):,}',
                ha='center', va='bottom', fontsize=9
            )

    # Labels
    ax.set_xlabel('Prompt Length (tokens)', fontweight='bold')
    ax.set_ylabel('Total Token-Steps', fontweight='bold')
    ax.set_title('Computational Cost: Naive vs EAB', fontweight='bold', pad=20)
    ax.set_xticks(x)
    ax.set_xticklabels(lengths)
    ax.legend(loc='best', framealpha=0.9)
    ax.grid(True, alpha=0.3, axis='y', linestyle=':')

    # Save
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, bbox_inches='tight', dpi=300)
    print(f"✓ Saved plot to {output_path}")

    if show_plot:
        plt.show()
    else:
        plt.close()


def plot_branching_analysis(
    results: List[Dict[str, Any]],
    output_path: Union[str, Path],
    show_plot: bool = False
):
    """
    Plot branching behavior analysis (EAB only).

    Args:
        results: List of result dictionaries
        output_path: Path to save figure
        show_plot: Whether to display the plot
    """
    set_publication_style()

    # Group by length
    by_length = {}
    for result in results:
        length = result['prompt_length']
        if length not in by_length:
            by_length[length] = {
                'branch_counts': [],
                'avg_positions': [],
                'final_paths': []
            }

        branching_stats = result.get('branching_stats', {})
        by_length[length]['branch_counts'].append(branching_stats.get('total_branches', 0))
        by_length[length]['final_paths'].append(result['eab_metrics'].get('final_path_count', 0))

        avg_pos = branching_stats.get('avg_branch_position')
        if avg_pos is not None:
            by_length[length]['avg_positions'].append(avg_pos)

    # Compute means
    lengths = sorted(by_length.keys())

    # Create subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # Plot 1: Branch count and final paths
    branch_means = [np.mean(by_length[l]['branch_counts']) for l in lengths]
    branch_stds = [np.std(by_length[l]['branch_counts']) for l in lengths]
    path_means = [np.mean(by_length[l]['final_paths']) for l in lengths]

    ax1.plot(lengths, branch_means, marker='o', label='Branch count', color='#2E86AB', linewidth=2, markersize=8)
    ax1.fill_between(
        lengths,
        np.array(branch_means) - np.array(branch_stds),
        np.array(branch_means) + np.array(branch_stds),
        alpha=0.2, color='#2E86AB'
    )
    ax1.plot(lengths, path_means, marker='s', label='Final paths', color='#A23B72', linewidth=2, markersize=8)

    ax1.set_xlabel('Prompt Length (tokens)', fontweight='bold')
    ax1.set_ylabel('Count', fontweight='bold')
    ax1.set_title('Branching Frequency', fontweight='bold')
    ax1.legend(loc='best', framealpha=0.9)
    ax1.grid(True, alpha=0.3, linestyle=':')

    # Plot 2: Average branch position
    avg_pos_means = []
    avg_pos_stds = []
    for l in lengths:
        positions = by_length[l]['avg_positions']
        if positions:
            avg_pos_means.append(np.mean(positions))
            avg_pos_stds.append(np.std(positions))
        else:
            avg_pos_means.append(0)
            avg_pos_stds.append(0)

    ax2.plot(lengths, avg_pos_means, marker='o', color='#F18F01', linewidth=2, markersize=8)
    ax2.fill_between(
        lengths,
        np.array(avg_pos_means) - np.array(avg_pos_stds),
        np.array(avg_pos_means) + np.array(avg_pos_stds),
        alpha=0.2, color='#F18F01'
    )

    ax2.set_xlabel('Prompt Length (tokens)', fontweight='bold')
    ax2.set_ylabel('Token Position', fontweight='bold')
    ax2.set_title('Average Branch Position', fontweight='bold')
    ax2.grid(True, alpha=0.3, linestyle=':')

    plt.tight_layout()

    # Save
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, bbox_inches='tight', dpi=300)
    print(f"✓ Saved plot to {output_path}")

    if show_plot:
        plt.show()
    else:
        plt.close()


def plot_all_results(
    results: List[Dict[str, Any]],
    summary: Dict[int, Dict[str, Any]],
    output_dir: Union[str, Path],
    show_plots: bool = False
):
    """
    Generate all plots for experiment 1.A.

    Args:
        results: List of result dictionaries
        summary: Summary statistics
        output_dir: Directory to save plots
        show_plots: Whether to display plots
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("\nGenerating plots...")

    # Plot 1: Speedup vs length (ALL 3 METRICS)
    plot_speedup_vs_length(
        summary,
        output_dir / "speedup_vs_length.png",
        metrics=['speedup_token_steps', 'speedup_time', 'speedup_memory'],  # All 3!
        show_plot=show_plots
    )

    # Plot 2: Cost breakdown
    plot_cost_breakdown(
        results,
        output_dir / "cost_breakdown.png",
        show_plot=show_plots
    )

    # Plot 3: Branching analysis
    plot_branching_analysis(
        results,
        output_dir / "branching_analysis.png",
        show_plot=show_plots
    )

    print("✓ All plots generated successfully!")
