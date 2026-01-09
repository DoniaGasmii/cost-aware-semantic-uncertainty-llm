"""
Generate plots for Experiment 1.C.2: Max Paths Sensitivity
"""

import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

experiment_dir = Path(__file__).parent


def set_style():
    plt.rcParams.update({
        'font.size': 12,
        'axes.labelsize': 14,
        'figure.figsize': (10, 6),
        'savefig.dpi': 300,
        'savefig.bbox': 'tight',
    })


def plot_speedup_vs_max_paths(summary, output_dir):
    """Plot speedup vs max_paths."""
    set_style()
    fig, ax = plt.subplots(figsize=(10, 6))

    paths_values = sorted(summary.keys())
    means = [summary[p]['speedup_token_steps']['mean'] for p in paths_values]
    stds = [summary[p]['speedup_token_steps']['std'] for p in paths_values]

    ax.plot(paths_values, means, 'o-', color='#2E86AB', markersize=10, linewidth=2)
    ax.fill_between(paths_values, np.array(means) - np.array(stds), np.array(means) + np.array(stds),
                     alpha=0.2, color='#2E86AB')

    ax.axhline(y=1.0, color='gray', linestyle='--', linewidth=1, alpha=0.5)
    ax.set_xlabel('Max Paths', fontweight='bold')
    ax.set_ylabel('Speedup Factor (Naive / EAB)', fontweight='bold')
    ax.set_title('Speedup vs Max Paths (Budget)', fontweight='bold', pad=20)
    ax.set_xticks(paths_values)
    ax.grid(True, alpha=0.3, linestyle=':')
    ax.set_ylim(bottom=0)

    plt.savefig(output_dir / "speedup_vs_max_paths.png")
    print(f"   ✓ Saved: speedup_vs_max_paths.png")
    plt.close()


def plot_diminishing_returns(summary, output_dir):
    """Plot diminishing returns: samples vs max_paths."""
    set_style()
    fig, ax = plt.subplots(figsize=(10, 6))

    paths_values = sorted(summary.keys())
    samples = [summary[p]['num_samples']['mean'] for p in paths_values]

    ax.plot(paths_values, samples, 'o-', color='#A23B72', markersize=10, linewidth=2)

    # Add 45-degree line (ideal scaling)
    ideal = paths_values
    ax.plot(paths_values, ideal, '--', color='gray', alpha=0.5, label='Ideal (linear scaling)')

    ax.set_xlabel('Max Paths (Budget)', fontweight='bold')
    ax.set_ylabel('Number of Samples Generated', fontweight='bold')
    ax.set_title('Sample Count vs Max Paths (Diminishing Returns)', fontweight='bold', pad=20)
    ax.set_xticks(paths_values)
    ax.legend()
    ax.grid(True, alpha=0.3, linestyle=':')

    plt.savefig(output_dir / "diminishing_returns.png")
    print(f"   ✓ Saved: diminishing_returns.png")
    plt.close()


def plot_cost_quality_tradeoff(summary, output_dir):
    """Plot cost-quality tradeoff."""
    set_style()
    fig, ax = plt.subplots(figsize=(10, 6))

    paths_values = sorted(summary.keys())
    speedups = [summary[p]['speedup_token_steps']['mean'] for p in paths_values]
    diversities = [summary[p]['eab_diversity']['mean'] for p in paths_values]

    ax.scatter(speedups, diversities, s=200, c=paths_values, cmap='viridis', edgecolors='black', linewidth=2)

    for i, p in enumerate(paths_values):
        ax.annotate(f"MP={p}", (speedups[i], diversities[i]), xytext=(10, 10),
                    textcoords='offset points', fontsize=10, fontweight='bold')

    ax.set_xlabel('Speedup Factor (Higher = Better)', fontweight='bold')
    ax.set_ylabel('Diversity (Unique Tokens Ratio)', fontweight='bold')
    ax.set_title('Cost-Quality Tradeoff: Max Paths', fontweight='bold', pad=20)
    ax.grid(True, alpha=0.3, linestyle=':')

    cbar = plt.colorbar(ax.collections[0], ax=ax)
    cbar.set_label('Max Paths', fontweight='bold')

    plt.savefig(output_dir / "cost_quality_tradeoff.png")
    print(f"   ✓ Saved: cost_quality_tradeoff.png")
    plt.close()


def main():
    print("=" * 70)
    print("GENERATING PLOTS FOR EXPERIMENT 1.C.2")
    print("=" * 70)

    print("\n1. Loading summary statistics...")
    summary_file = experiment_dir / "results" / "summary_stats.json"
    with open(summary_file, 'r') as f:
        summary_data = json.load(f)
    summary = {int(k): v for k, v in summary_data['summary_by_paths'].items()}
    print(f"   ✓ Loaded summary for {len(summary)} max_paths values")

    print("\n2. Generating plots...")
    figures_dir = experiment_dir / "results" / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    plot_speedup_vs_max_paths(summary, figures_dir)
    plot_diminishing_returns(summary, figures_dir)
    plot_cost_quality_tradeoff(summary, figures_dir)

    print("\n" + "=" * 70)
    print("PLOTTING COMPLETE")
    print("=" * 70)
    print(f"\nGenerated figures in: {figures_dir}/")
    print("  • speedup_vs_max_paths.png")
    print("  • diminishing_returns.png")
    print("  • cost_quality_tradeoff.png")
    print("=" * 70)


if __name__ == "__main__":
    main()
