"""
Generate plots for Experiment 1.A.5: Speedup vs Generation Length
"""

import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from typing import Dict, Any

experiment_dir = Path(__file__).parent


def set_style():
    """Set matplotlib style."""
    plt.rcParams.update({
        'font.size': 12,
        'axes.labelsize': 14,
        'xtick.labelsize': 11,
        'ytick.labelsize': 11,
        'legend.fontsize': 11,
        'figure.figsize': (10, 6),
        'savefig.dpi': 300,
        'savefig.bbox': 'tight',
    })


def plot_speedup_vs_generation_length(summary: Dict[int, Dict], output_dir: Path):
    """Plot speedup vs generation length."""
    set_style()

    fig, ax = plt.subplots(figsize=(10, 6))

    lengths = sorted(summary.keys())
    means = [summary[l]['speedup_token_steps']['mean'] for l in lengths]
    stds = [summary[l]['speedup_token_steps']['std'] for l in lengths]

    ax.plot(lengths, means, 'o-', color='#2E86AB', markersize=10, linewidth=2, label='Token-steps speedup')
    ax.fill_between(lengths, np.array(means) - np.array(stds), np.array(means) + np.array(stds),
                     alpha=0.2, color='#2E86AB')

    ax.axhline(y=1.0, color='gray', linestyle='--', linewidth=1, alpha=0.5, label='No speedup')
    ax.set_xlabel('Generation Length (max_new_tokens)', fontweight='bold')
    ax.set_ylabel('Speedup Factor (Naive / EAB)', fontweight='bold')
    ax.set_title('EAB Speedup vs Generation Length', fontweight='bold', pad=20)
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3, linestyle=':')
    ax.set_ylim(bottom=0)

    output_file = output_dir / "speedup_vs_generation_length.png"
    plt.savefig(output_file)
    print(f"   ✓ Saved: {output_file.name}")
    plt.close()


def plot_cost_breakdown(summary: Dict[int, Dict], output_dir: Path):
    """Plot cost breakdown: Naive vs EAB."""
    set_style()

    fig, ax = plt.subplots(figsize=(10, 6))

    lengths = sorted(summary.keys())
    naive_means = [summary[l]['naive_token_steps']['mean'] for l in lengths]
    eab_means = [summary[l]['eab_token_steps']['mean'] for l in lengths]

    x = np.arange(len(lengths))
    width = 0.35

    ax.bar(x - width/2, naive_means, width, label='Naive', color='#E63946', alpha=0.8)
    ax.bar(x + width/2, eab_means, width, label='EAB', color='#06A77D', alpha=0.8)

    ax.set_xlabel('Generation Length (tokens)', fontweight='bold')
    ax.set_ylabel('Token Steps (FLOPs proxy)', fontweight='bold')
    ax.set_title('Cost Comparison: Naive vs EAB', fontweight='bold', pad=20)
    ax.set_xticks(x)
    ax.set_xticklabels([str(l) for l in lengths])
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3, axis='y', linestyle=':')

    output_file = output_dir / "cost_breakdown.png"
    plt.savefig(output_file)
    print(f"   ✓ Saved: {output_file.name}")
    plt.close()


def plot_branching_analysis(summary: Dict[int, Dict], output_dir: Path):
    """Plot branching behavior vs generation length."""
    set_style()

    fig, ax = plt.subplots(figsize=(10, 6))

    lengths = sorted(summary.keys())
    branch_means = [summary[l]['branch_count']['mean'] for l in lengths]
    branch_stds = [summary[l]['branch_count']['std'] for l in lengths]

    ax.plot(lengths, branch_means, 'o-', color='#F18F01', markersize=10, linewidth=2)
    ax.fill_between(lengths, np.array(branch_means) - np.array(branch_stds),
                     np.array(branch_means) + np.array(branch_stds), alpha=0.2, color='#F18F01')

    ax.set_xlabel('Generation Length (tokens)', fontweight='bold')
    ax.set_ylabel('Number of Branches', fontweight='bold')
    ax.set_title('Branching Behavior vs Generation Length', fontweight='bold', pad=20)
    ax.grid(True, alpha=0.3, linestyle=':')
    ax.set_ylim(bottom=0)

    output_file = output_dir / "branching_analysis.png"
    plt.savefig(output_file)
    print(f"   ✓ Saved: {output_file.name}")
    plt.close()


def main():
    print("=" * 70)
    print("GENERATING PLOTS FOR EXPERIMENT 1.A.5")
    print("=" * 70)

    # Load summary
    print("\n1. Loading summary statistics...")
    summary_file = experiment_dir / "results" / "summary_stats.json"
    with open(summary_file, 'r') as f:
        summary_data = json.load(f)
    summary = {int(k): v for k, v in summary_data['summary_by_length'].items()}
    print(f"   ✓ Loaded summary for {len(summary)} generation lengths")

    # Generate plots
    print("\n2. Generating plots...")
    figures_dir = experiment_dir / "results" / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    plot_speedup_vs_generation_length(summary, figures_dir)
    plot_cost_breakdown(summary, figures_dir)
    plot_branching_analysis(summary, figures_dir)

    print("\n" + "=" * 70)
    print("PLOTTING COMPLETE")
    print("=" * 70)
    print(f"\nGenerated figures in: {figures_dir}/")
    print("  • speedup_vs_generation_length.png")
    print("  • cost_breakdown.png")
    print("  • branching_analysis.png")
    print("=" * 70)


if __name__ == "__main__":
    main()
