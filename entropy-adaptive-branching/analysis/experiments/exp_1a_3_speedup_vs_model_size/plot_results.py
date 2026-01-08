"""
Generate plots for Experiment 1.A.3: Speedup vs Model Size
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


def plot_speedup_vs_model_size(summary, output_path):
    """Plot speedup vs model size."""
    set_publication_style()

    fig, ax = plt.subplots(figsize=(10, 6))

    # Sort by model size
    model_order = {"0.5B": 0.5, "1.5B": 1.5, "3B": 3.0, "7B": 7.0}
    model_params = sorted(summary.keys(), key=lambda x: model_order.get(x, 0))
    model_sizes = [model_order[m] for m in model_params]

    # Extract metrics
    token_means = [summary[m]['speedup_token_steps']['mean'] for m in model_params]
    token_stds = [summary[m]['speedup_token_steps']['std'] for m in model_params]
    time_means = [summary[m]['speedup_time']['mean'] for m in model_params]
    time_stds = [summary[m]['speedup_time']['std'] for m in model_params]

    # Plot token-steps speedup
    ax.plot(model_sizes, token_means, label='Token-steps', color='#2E86AB',
            marker='o', markersize=8, linewidth=2, linestyle='-')
    ax.fill_between(model_sizes,
                     np.array(token_means) - np.array(token_stds),
                     np.array(token_means) + np.array(token_stds),
                     alpha=0.2, color='#2E86AB')

    # Plot time speedup
    ax.plot(model_sizes, time_means, label='Wall-clock time', color='#A23B72',
            marker='s', markersize=8, linewidth=2, linestyle='-')
    ax.fill_between(model_sizes,
                     np.array(time_means) - np.array(time_stds),
                     np.array(time_means) + np.array(time_stds),
                     alpha=0.2, color='#A23B72')

    # Reference line
    ax.axhline(y=1.0, color='gray', linestyle='--', linewidth=1, alpha=0.5, label='No speedup')

    # Labels
    ax.set_xlabel('Model Size (Billion Parameters)', fontweight='bold')
    ax.set_ylabel('Speedup Factor (Naive / EAB)', fontweight='bold')
    ax.set_title('EAB Speedup vs Model Size', fontweight='bold', pad=20)
    ax.legend(loc='best', framealpha=0.9)
    ax.grid(True, alpha=0.3, linestyle=':')
    ax.set_ylim(bottom=0)

    # Use log scale for x-axis
    ax.set_xscale('log')
    ax.set_xticks(model_sizes)
    ax.set_xticklabels([f'{s:.1f}B' for s in model_sizes])

    # Save
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, bbox_inches='tight', dpi=300)
    print(f"✓ Saved speedup plot to {output_path}")
    plt.close()


def plot_absolute_time_savings(summary, output_path):
    """Plot absolute time savings vs model size."""
    set_publication_style()

    fig, ax = plt.subplots(figsize=(10, 6))

    # Sort by model size
    model_order = {"0.5B": 0.5, "1.5B": 1.5, "3B": 3.0, "7B": 7.0}
    model_params = sorted(summary.keys(), key=lambda x: model_order.get(x, 0))
    model_sizes = [model_order[m] for m in model_params]

    # Extract times
    eab_times = [summary[m]['eab_time']['mean'] for m in model_params]
    naive_times = [summary[m]['naive_time']['mean'] for m in model_params]
    time_saved = [n - e for n, e in zip(naive_times, eab_times)]

    # Plot
    ax.bar(range(len(model_params)), time_saved, color='#06A77D', alpha=0.8)

    # Add value labels
    for i, saved in enumerate(time_saved):
        ax.text(i, saved, f'{saved:.1f}s', ha='center', va='bottom', fontweight='bold')

    # Labels
    ax.set_xlabel('Model Size', fontweight='bold')
    ax.set_ylabel('Time Saved (seconds)', fontweight='bold')
    ax.set_title('Absolute Time Savings: Naive - EAB', fontweight='bold', pad=20)
    ax.set_xticks(range(len(model_params)))
    ax.set_xticklabels(model_params)
    ax.grid(True, alpha=0.3, axis='y', linestyle=':')

    # Save
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, bbox_inches='tight', dpi=300)
    print(f"✓ Saved time savings plot to {output_path}")
    plt.close()


def plot_cost_comparison(results, output_path):
    """Plot cost comparison by model size."""
    set_publication_style()

    # Group by model params
    by_model = {}
    for result in results:
        model_params = result['model_params']
        if model_params not in by_model:
            by_model[model_params] = {'naive': [], 'eab': []}
        by_model[model_params]['naive'].append(result['naive_metrics']['token_steps'])
        by_model[model_params]['eab'].append(result['eab_metrics']['token_steps'])

    # Sort and compute means
    model_order = {"0.5B": 0, "1.5B": 1, "3B": 2, "7B": 3}
    model_params = sorted(by_model.keys(), key=lambda x: model_order.get(x, 999))
    naive_means = [np.mean(by_model[m]['naive']) for m in model_params]
    eab_means = [np.mean(by_model[m]['eab']) for m in model_params]

    # Plot
    fig, ax = plt.subplots(figsize=(10, 6))

    x = np.arange(len(model_params))
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
    ax.set_xlabel('Model Size', fontweight='bold')
    ax.set_ylabel('Total Token-Steps', fontweight='bold')
    ax.set_title('Computational Cost by Model Size', fontweight='bold', pad=20)
    ax.set_xticks(x)
    ax.set_xticklabels(model_params)
    ax.legend(loc='best', framealpha=0.9)
    ax.grid(True, alpha=0.3, axis='y', linestyle=':')

    # Save
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, bbox_inches='tight', dpi=300)
    print(f"✓ Saved cost comparison plot to {output_path}")
    plt.close()


def main():
    """Generate all plots."""
    print("=" * 70)
    print("GENERATING PLOTS FOR EXPERIMENT 1.A.3")
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
    summary = summary_data['summary_by_model']
    print(f"   ✓ Loaded summary for {len(summary)} model sizes")

    # Generate plots
    print("\n3. Generating plots...")
    figures_dir = experiment_dir / "results" / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    plot_speedup_vs_model_size(summary, figures_dir / "speedup_vs_model_size.png")
    plot_absolute_time_savings(summary, figures_dir / "time_savings.png")
    plot_cost_comparison(results, figures_dir / "cost_comparison.png")

    # Done
    print("\n" + "=" * 70)
    print("PLOTTING COMPLETE")
    print("=" * 70)
    print(f"\nGenerated figures saved to: {figures_dir}/")
    print("  • speedup_vs_model_size.png")
    print("  • time_savings.png")
    print("  • cost_comparison.png")
    print("=" * 70)


if __name__ == "__main__":
    main()
