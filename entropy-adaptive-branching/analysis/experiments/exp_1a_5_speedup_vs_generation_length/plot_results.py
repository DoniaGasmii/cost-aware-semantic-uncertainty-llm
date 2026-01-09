"""
Generate plots for Experiment 1.A.5: Speedup vs Generation Length
Shows ALL 3 metrics: Token-steps, Wall-clock time, Peak memory
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


def plot_speedup_all_metrics(summary: Dict[int, Dict], output_dir: Path):
    """Plot ALL 3 speedup metrics together."""
    set_style()

    fig, ax = plt.subplots(figsize=(12, 7))

    lengths = sorted(summary.keys())

    # Metric configurations
    metrics = {
        'speedup_token_steps': {
            'label': 'Token-steps (algorithmic)',
            'color': '#2E86AB',
            'marker': 'o'
        },
        'speedup_time': {
            'label': 'Wall-clock time (practical)',
            'color': '#A23B72',
            'marker': 's'
        },
        'speedup_memory': {
            'label': 'Peak memory',
            'color': '#F18F01',
            'marker': '^'
        }
    }

    # Plot each metric
    for metric_name, metric_info in metrics.items():
        means = [summary[l][metric_name]['mean'] for l in lengths]
        stds = [summary[l][metric_name]['std'] for l in lengths]

        ax.plot(lengths, means, marker=metric_info['marker'],
                color=metric_info['color'], markersize=10, linewidth=2,
                label=metric_info['label'])
        ax.fill_between(lengths,
                        np.array(means) - np.array(stds),
                        np.array(means) + np.array(stds),
                        alpha=0.2, color=metric_info['color'])

    # Reference line
    ax.axhline(y=1.0, color='gray', linestyle='--', linewidth=1, alpha=0.5, label='No speedup (1.0×)')

    ax.set_xlabel('Generation Length (max_new_tokens)', fontweight='bold', fontsize=14)
    ax.set_ylabel('Speedup Factor (Naive / EAB)', fontweight='bold', fontsize=14)
    ax.set_title('EAB Efficiency: All Metrics vs Generation Length', fontweight='bold', pad=20, fontsize=16)
    ax.legend(loc='best', framealpha=0.95, fontsize=12)
    ax.grid(True, alpha=0.3, linestyle=':')
    ax.set_ylim(bottom=0)

    output_file = output_dir / "speedup_all_metrics.png"
    plt.savefig(output_file)
    print(f"   ✓ Saved: {output_file.name}")
    plt.close()


def plot_cost_breakdown(summary: Dict[int, Dict], output_dir: Path):
    """Plot cost breakdown: Naive vs EAB (token-steps only)."""
    set_style()

    fig, ax = plt.subplots(figsize=(10, 6))

    lengths = sorted(summary.keys())
    naive_means = [summary[l]['naive_token_steps']['mean'] for l in lengths]
    eab_means = [summary[l]['eab_token_steps']['mean'] for l in lengths]

    x = np.arange(len(lengths))
    width = 0.35

    ax.bar(x - width/2, naive_means, width, label='Naive (Sequential)', color='#E63946', alpha=0.8)
    ax.bar(x + width/2, eab_means, width, label='EAB (Adaptive)', color='#06A77D', alpha=0.8)

    ax.set_xlabel('Generation Length (tokens)', fontweight='bold')
    ax.set_ylabel('Token Steps (Forward Passes)', fontweight='bold')
    ax.set_title('Computational Cost: Naive vs EAB', fontweight='bold', pad=20)
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

    plot_speedup_all_metrics(summary, figures_dir)  # Main plot with all 3 metrics!
    plot_cost_breakdown(summary, figures_dir)
    plot_branching_analysis(summary, figures_dir)

    print("\n" + "=" * 70)
    print("PLOTTING COMPLETE")
    print("=" * 70)
    print(f"\nGenerated figures in: {figures_dir}/")
    print("  • speedup_all_metrics.png  ← ALL 3 METRICS!")
    print("  • cost_breakdown.png")
    print("  • branching_analysis.png")
    print("=" * 70)


if __name__ == "__main__":
    main()
