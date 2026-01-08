"""
Generate plots for Experiment 1.A.4: Speedup vs Domain
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


def plot_speedup_by_domain(summary, output_path):
    """Plot speedup comparison across domains."""
    set_publication_style()

    fig, ax = plt.subplots(figsize=(10, 6))

    # Sort domains by speedup (descending)
    domains = sorted(summary.keys(), key=lambda d: summary[d]['speedup_token_steps']['mean'], reverse=True)

    # Extract metrics
    token_means = [summary[d]['speedup_token_steps']['mean'] for d in domains]
    token_stds = [summary[d]['speedup_token_steps']['std'] for d in domains]
    token_cis = [summary[d]['speedup_token_steps']['ci_95'] for d in domains]

    # Calculate error bars (95% CI)
    errors_lower = [m - ci[0] for m, ci in zip(token_means, token_cis)]
    errors_upper = [ci[1] - m for m, ci in zip(token_means, token_cis)]

    # Plot bars
    x = np.arange(len(domains))
    bars = ax.bar(x, token_means, yerr=[errors_lower, errors_upper],
                   color=['#2E86AB', '#A23B72', '#F18F01'][:len(domains)],
                   alpha=0.8, capsize=5)

    # Add value labels on bars
    for i, (bar, mean) in enumerate(zip(bars, token_means)):
        ax.text(bar.get_x() + bar.get_width()/2., mean,
               f'{mean:.2f}×', ha='center', va='bottom', fontweight='bold')

    # Reference line at y=1
    ax.axhline(y=1.0, color='gray', linestyle='--', linewidth=1, alpha=0.5, label='No speedup')

    # Labels
    ax.set_xlabel('Domain', fontweight='bold')
    ax.set_ylabel('Speedup Factor (Naive / EAB)', fontweight='bold')
    ax.set_title('EAB Speedup by Domain', fontweight='bold', pad=20)
    ax.set_xticks(x)
    ax.set_xticklabels([d.replace('_', ' ').title() for d in domains])
    ax.legend(loc='best', framealpha=0.9)
    ax.grid(True, alpha=0.3, axis='y', linestyle=':')
    ax.set_ylim(bottom=0)

    # Save
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, bbox_inches='tight', dpi=300)
    print(f"✓ Saved speedup by domain plot to {output_path}")
    plt.close()


def plot_branching_by_domain(summary, output_path):
    """Plot branching behavior by domain."""
    set_publication_style()

    fig, ax = plt.subplots(figsize=(10, 6))

    # Sort domains by branch count
    domains = sorted(summary.keys(), key=lambda d: summary[d]['branch_count']['mean'], reverse=True)

    # Extract metrics
    branch_means = [summary[d]['branch_count']['mean'] for d in domains]
    branch_stds = [summary[d]['branch_count']['std'] for d in domains]

    # Plot bars
    x = np.arange(len(domains))
    bars = ax.bar(x, branch_means, yerr=branch_stds,
                   color=['#06A77D', '#F18F01', '#E63946'][:len(domains)],
                   alpha=0.8, capsize=5)

    # Add value labels
    for bar, mean in zip(bars, branch_means):
        ax.text(bar.get_x() + bar.get_width()/2., mean,
               f'{mean:.1f}', ha='center', va='bottom', fontweight='bold')

    # Labels
    ax.set_xlabel('Domain', fontweight='bold')
    ax.set_ylabel('Average Branch Count', fontweight='bold')
    ax.set_title('Branching Frequency by Domain', fontweight='bold', pad=20)
    ax.set_xticks(x)
    ax.set_xticklabels([d.replace('_', ' ').title() for d in domains])
    ax.grid(True, alpha=0.3, axis='y', linestyle=':')

    # Save
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, bbox_inches='tight', dpi=300)
    print(f"✓ Saved branching by domain plot to {output_path}")
    plt.close()


def plot_speedup_vs_branching(summary, output_path):
    """Scatter plot: speedup vs branch count (to show correlation)."""
    set_publication_style()

    fig, ax = plt.subplots(figsize=(10, 6))

    domains = list(summary.keys())

    # Extract data
    speedups = [summary[d]['speedup_token_steps']['mean'] for d in domains]
    branches = [summary[d]['branch_count']['mean'] for d in domains]

    # Plot scatter
    colors = ['#2E86AB', '#A23B72', '#F18F01']
    for i, domain in enumerate(domains):
        ax.scatter(branches[i], speedups[i], s=200, alpha=0.7,
                  color=colors[i % len(colors)], label=domain.replace('_', ' ').title(),
                  edgecolors='black', linewidth=1.5)

    # Add trend line
    if len(speedups) > 1:
        z = np.polyfit(branches, speedups, 1)
        p = np.poly1d(z)
        x_trend = np.linspace(min(branches), max(branches), 100)
        ax.plot(x_trend, p(x_trend), color='gray', linestyle='--', linewidth=2,
               alpha=0.5, label=f'Trend (slope={z[0]:.3f})')

    # Calculate correlation
    if len(speedups) > 2:
        from scipy import stats
        corr, p_value = stats.pearsonr(branches, speedups)
        ax.text(0.05, 0.95, f'Pearson r = {corr:.3f}\np = {p_value:.3f}',
               transform=ax.transAxes, fontsize=11, verticalalignment='top',
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    # Labels
    ax.set_xlabel('Average Branch Count', fontweight='bold')
    ax.set_ylabel('Speedup Factor', fontweight='bold')
    ax.set_title('Speedup vs Branching Frequency', fontweight='bold', pad=20)
    ax.legend(loc='best', framealpha=0.9)
    ax.grid(True, alpha=0.3, linestyle=':')

    # Save
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, bbox_inches='tight', dpi=300)
    print(f"✓ Saved speedup vs branching plot to {output_path}")
    plt.close()


def plot_cost_breakdown_by_domain(results, output_path):
    """Plot cost breakdown by domain."""
    set_publication_style()

    # Group by domain
    by_domain = {}
    for result in results:
        domain = result['domain']
        if domain not in by_domain:
            by_domain[domain] = {'naive': [], 'eab': []}
        by_domain[domain]['naive'].append(result['naive_metrics']['token_steps'])
        by_domain[domain]['eab'].append(result['eab_metrics']['token_steps'])

    # Sort domains
    domains = sorted(by_domain.keys())
    naive_means = [np.mean(by_domain[d]['naive']) for d in domains]
    eab_means = [np.mean(by_domain[d]['eab']) for d in domains]

    # Plot
    fig, ax = plt.subplots(figsize=(10, 6))

    x = np.arange(len(domains))
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
    ax.set_xlabel('Domain', fontweight='bold')
    ax.set_ylabel('Total Token-Steps', fontweight='bold')
    ax.set_title('Computational Cost by Domain', fontweight='bold', pad=20)
    ax.set_xticks(x)
    ax.set_xticklabels([d.replace('_', ' ').title() for d in domains])
    ax.legend(loc='best', framealpha=0.9)
    ax.grid(True, alpha=0.3, axis='y', linestyle=':')

    # Save
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, bbox_inches='tight', dpi=300)
    print(f"✓ Saved cost breakdown plot to {output_path}")
    plt.close()


def main():
    """Generate all plots."""
    print("=" * 70)
    print("GENERATING PLOTS FOR EXPERIMENT 1.A.4")
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
    summary = summary_data['summary_by_domain']
    print(f"   ✓ Loaded summary for {len(summary)} domains")

    # Generate plots
    print("\n3. Generating plots...")
    figures_dir = experiment_dir / "results" / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    plot_speedup_by_domain(summary, figures_dir / "speedup_by_domain.png")
    plot_branching_by_domain(summary, figures_dir / "branching_by_domain.png")
    plot_speedup_vs_branching(summary, figures_dir / "speedup_vs_branching.png")
    plot_cost_breakdown_by_domain(results, figures_dir / "cost_breakdown.png")

    # Done
    print("\n" + "=" * 70)
    print("PLOTTING COMPLETE")
    print("=" * 70)
    print(f"\nGenerated figures saved to: {figures_dir}/")
    print("  • speedup_by_domain.png")
    print("  • branching_by_domain.png")
    print("  • speedup_vs_branching.png")
    print("  • cost_breakdown.png")
    print("=" * 70)


if __name__ == "__main__":
    main()
