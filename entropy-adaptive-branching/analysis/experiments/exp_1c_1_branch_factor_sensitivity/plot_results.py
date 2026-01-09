"""
Generate plots for Experiment 1.C.1: Branch Factor Sensitivity
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


def plot_speedup_vs_factor(summary, output_dir):
    """Plot speedup vs branch_factor."""
    set_style()
    fig, ax = plt.subplots(figsize=(10, 6))

    factors = sorted(summary.keys())
    means = [summary[f]['speedup_token_steps']['mean'] for f in factors]
    stds = [summary[f]['speedup_token_steps']['std'] for f in factors]

    ax.plot(factors, means, 'o-', color='#2E86AB', markersize=10, linewidth=2)
    ax.fill_between(factors, np.array(means) - np.array(stds), np.array(means) + np.array(stds),
                     alpha=0.2, color='#2E86AB')

    ax.axhline(y=1.0, color='gray', linestyle='--', linewidth=1, alpha=0.5)
    ax.set_xlabel('Branch Factor', fontweight='bold')
    ax.set_ylabel('Speedup Factor (Naive / EAB)', fontweight='bold')
    ax.set_title('Speedup vs Branch Factor', fontweight='bold', pad=20)
    ax.set_xticks(factors)
    ax.grid(True, alpha=0.3, linestyle=':')
    ax.set_ylim(bottom=0)

    plt.savefig(output_dir / "speedup_vs_branch_factor.png")
    print(f"   ✓ Saved: speedup_vs_branch_factor.png")
    plt.close()


def plot_cost_quality_tradeoff(summary, output_dir):
    """Plot cost-quality tradeoff: speedup vs diversity."""
    set_style()
    fig, ax = plt.subplots(figsize=(10, 6))

    factors = sorted(summary.keys())
    speedups = [summary[f]['speedup_token_steps']['mean'] for f in factors]
    diversities = [summary[f]['eab_diversity']['mean'] for f in factors]

    ax.scatter(speedups, diversities, s=200, c=factors, cmap='viridis', edgecolors='black', linewidth=2)

    for i, f in enumerate(factors):
        ax.annotate(f"BF={f}", (speedups[i], diversities[i]), xytext=(10, 10),
                    textcoords='offset points', fontsize=10, fontweight='bold')

    ax.set_xlabel('Speedup Factor (Higher = Better)', fontweight='bold')
    ax.set_ylabel('Diversity (Unique Tokens Ratio)', fontweight='bold')
    ax.set_title('Cost-Quality Tradeoff: Branch Factor', fontweight='bold', pad=20)
    ax.grid(True, alpha=0.3, linestyle=':')

    cbar = plt.colorbar(ax.collections[0], ax=ax)
    cbar.set_label('Branch Factor', fontweight='bold')

    plt.savefig(output_dir / "cost_quality_tradeoff.png")
    print(f"   ✓ Saved: cost_quality_tradeoff.png")
    plt.close()


def plot_branching_behavior(summary, output_dir):
    """Plot branching behavior vs branch_factor."""
    set_style()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    factors = sorted(summary.keys())
    branches = [summary[f]['branch_count']['mean'] for f in factors]
    samples = [summary[f]['num_samples']['mean'] for f in factors]

    # Branches
    ax1.plot(factors, branches, 'o-', color='#F18F01', markersize=10, linewidth=2)
    ax1.set_xlabel('Branch Factor', fontweight='bold')
    ax1.set_ylabel('Number of Branches', fontweight='bold')
    ax1.set_title('Branches vs Branch Factor', fontweight='bold')
    ax1.set_xticks(factors)
    ax1.grid(True, alpha=0.3, linestyle=':')

    # Samples generated
    ax2.plot(factors, samples, 'o-', color='#A23B72', markersize=10, linewidth=2)
    ax2.set_xlabel('Branch Factor', fontweight='bold')
    ax2.set_ylabel('Number of Samples Generated', fontweight='bold')
    ax2.set_title('Sample Count vs Branch Factor', fontweight='bold')
    ax2.set_xticks(factors)
    ax2.grid(True, alpha=0.3, linestyle=':')

    plt.tight_layout()
    plt.savefig(output_dir / "branching_behavior.png")
    print(f"   ✓ Saved: branching_behavior.png")
    plt.close()


def main():
    print("=" * 70)
    print("GENERATING PLOTS FOR EXPERIMENT 1.C.1")
    print("=" * 70)

    print("\n1. Loading summary statistics...")
    summary_file = experiment_dir / "results" / "summary_stats.json"
    with open(summary_file, 'r') as f:
        summary_data = json.load(f)
    summary = {int(k): v for k, v in summary_data['summary_by_factor'].items()}
    print(f"   ✓ Loaded summary for {len(summary)} branch factors")

    print("\n2. Generating plots...")
    figures_dir = experiment_dir / "results" / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    plot_speedup_vs_factor(summary, figures_dir)
    plot_cost_quality_tradeoff(summary, figures_dir)
    plot_branching_behavior(summary, figures_dir)

    print("\n" + "=" * 70)
    print("PLOTTING COMPLETE")
    print("=" * 70)
    print(f"\nGenerated figures in: {figures_dir}/")
    print("  • speedup_vs_branch_factor.png")
    print("  • cost_quality_tradeoff.png")
    print("  • branching_behavior.png")
    print("=" * 70)


if __name__ == "__main__":
    main()
