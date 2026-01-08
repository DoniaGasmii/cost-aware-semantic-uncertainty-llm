"""
Generate plots for Experiment 1.A.2: Speedup vs Sample Count
"""

import json
import sys
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# Add utils to path
experiment_dir = Path(__file__).parent
analysis_dir = experiment_dir.parent.parent
sys.path.insert(0, str(analysis_dir))

from utils.plotting import set_publication_style


def plot_speedup_vs_sample_count(summary, output_path):
    """Plot speedup vs sample count."""
    set_publication_style()

    fig, ax = plt.subplots(figsize=(10, 6))

    # Sort by sample count
    counts = sorted(summary.keys())

    # Extract metrics
    token_means = [summary[c]['speedup_token_steps']['mean'] for c in counts]
    token_stds = [summary[c]['speedup_token_steps']['std'] for c in counts]
    time_means = [summary[c]['speedup_time']['mean'] for c in counts]
    time_stds = [summary[c]['speedup_time']['std'] for c in counts]

    # Plot token-steps speedup
    ax.plot(counts, token_means, label='Token-steps', color='#2E86AB',
            marker='o', markersize=8, linewidth=2, linestyle='-')
    ax.fill_between(counts,
                     np.array(token_means) - np.array(token_stds),
                     np.array(token_means) + np.array(token_stds),
                     alpha=0.2, color='#2E86AB')

    # Plot time speedup
    ax.plot(counts, time_means, label='Wall-clock time', color='#A23B72',
            marker='s', markersize=8, linewidth=2, linestyle='-')
    ax.fill_between(counts,
                     np.array(time_means) - np.array(time_stds),
                     np.array(time_means) + np.array(time_stds),
                     alpha=0.2, color='#A23B72')

    # Reference line
    ax.axhline(y=1.0, color='gray', linestyle='--', linewidth=1, alpha=0.5, label='No speedup')

    # Labels
    ax.set_xlabel('Target Sample Count (max_paths)', fontweight='bold')
    ax.set_ylabel('Speedup Factor (Naive / EAB)', fontweight='bold')
    ax.set_title('EAB Speedup vs Sample Count', fontweight='bold', pad=20)
    ax.legend(loc='best', framealpha=0.9)
    ax.grid(True, alpha=0.3, linestyle=':')
    ax.set_ylim(bottom=0)

    # Save
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, bbox_inches='tight', dpi=300)
    print(f"✓ Saved speedup plot to {output_path}")
    plt.close()


def plot_cost_breakdown(results, output_path):
    """Plot cost breakdown by sample count."""
    set_publication_style()

    # Group by target count
    by_count = {}
    for result in results:
        target = result['target_sample_count']
        if target not in by_count:
            by_count[target] = {'naive': [], 'eab': []}
        by_count[target]['naive'].append(result['naive_metrics']['token_steps'])
        by_count[target]['eab'].append(result['eab_metrics']['token_steps'])

    # Compute means
    counts = sorted(by_count.keys())
    naive_means = [np.mean(by_count[c]['naive']) for c in counts]
    eab_means = [np.mean(by_count[c]['eab']) for c in counts]

    # Plot
    fig, ax = plt.subplots(figsize=(10, 6))

    x = np.arange(len(counts))
    width = 0.35

    bars1 = ax.bar(x - width/2, naive_means, width, label='Naive', color='#E63946', alpha=0.8)
    bars2 = ax.bar(x + width/2, eab_means, width, label='EAB', color='#06A77D', alpha=0.8)

    # Add value labels
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{int(height):,}', ha='center', va='bottom', fontsize=9)

    # Labels
    ax.set_xlabel('Target Sample Count (max_paths)', fontweight='bold')
    ax.set_ylabel('Total Token-Steps', fontweight='bold')
    ax.set_title('Computational Cost: Naive vs EAB', fontweight='bold', pad=20)
    ax.set_xticks(x)
    ax.set_xticklabels(counts)
    ax.legend(loc='best', framealpha=0.9)
    ax.grid(True, alpha=0.3, axis='y', linestyle=':')

    # Save
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, bbox_inches='tight', dpi=300)
    print(f"✓ Saved cost breakdown plot to {output_path}")
    plt.close()


def plot_samples_vs_target(summary, output_path):
    """Plot actual EAB samples vs target."""
    set_publication_style()

    fig, ax = plt.subplots(figsize=(10, 6))

    counts = sorted(summary.keys())
    actual_means = [summary[c]['actual_eab_samples']['mean'] for c in counts]
    actual_stds = [summary[c]['actual_eab_samples']['std'] for c in counts]

    # Plot actual vs target
    ax.plot(counts, actual_means, label='Actual EAB samples', color='#2E86AB',
            marker='o', markersize=8, linewidth=2, linestyle='-')
    ax.fill_between(counts,
                     np.array(actual_means) - np.array(actual_stds),
                     np.array(actual_means) + np.array(actual_stds),
                     alpha=0.2, color='#2E86AB')

    # Reference line (target = actual)
    ax.plot(counts, counts, color='gray', linestyle='--', linewidth=1, alpha=0.5, label='Target')

    # Labels
    ax.set_xlabel('Target Sample Count (max_paths)', fontweight='bold')
    ax.set_ylabel('Actual Samples Generated', fontweight='bold')
    ax.set_title('EAB Samples: Target vs Actual', fontweight='bold', pad=20)
    ax.legend(loc='best', framealpha=0.9)
    ax.grid(True, alpha=0.3, linestyle=':')

    # Save
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, bbox_inches='tight', dpi=300)
    print(f"✓ Saved samples comparison plot to {output_path}")
    plt.close()


def main():
    """Generate all plots."""
    print("=" * 70)
    print("GENERATING PLOTS FOR EXPERIMENT 1.A.2")
    print("=" * 70)

    # Load results
    print("\n1. Loading results...")
    results_file = experiment_dir / "results" / "raw_results.json"
    with open(results_file, 'r') as f:
        data = json.load(f)
    results = data['results']
    print(f"   ✓ Loaded {len(results)} experiments")

    # Load summary
    print("\n2. Loading summary statistics...")
    summary_file = experiment_dir / "results" / "summary_stats.json"
    with open(summary_file, 'r') as f:
        summary_data = json.load(f)
    summary = {int(k): v for k, v in summary_data['summary_by_count'].items()}
    print(f"   ✓ Loaded summary for {len(summary)} sample counts")

    # Generate plots
    print("\n3. Generating plots...")
    figures_dir = experiment_dir / "results" / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    plot_speedup_vs_sample_count(summary, figures_dir / "speedup_vs_count.png")
    plot_cost_breakdown(results, figures_dir / "cost_breakdown.png")
    plot_samples_vs_target(summary, figures_dir / "samples_comparison.png")

    # Done
    print("\n" + "=" * 70)
    print("PLOTTING COMPLETE")
    print("=" * 70)
    print(f"\nGenerated figures saved to: {figures_dir}/")
    print("  • speedup_vs_count.png")
    print("  • cost_breakdown.png")
    print("  • samples_comparison.png")
    print("=" * 70)


if __name__ == "__main__":
    main()
