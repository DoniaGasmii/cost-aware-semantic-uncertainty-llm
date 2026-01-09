"""
Generate plots for Experiment 1.C.3: Temperature Sensitivity
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


def plot_speedup_vs_temperature(summary, output_dir):
    """Plot speedup vs temperature."""
    set_style()
    fig, ax = plt.subplots(figsize=(10, 6))

    temps = sorted(summary.keys())
    means = [summary[t]['speedup_token_steps']['mean'] for t in temps]
    stds = [summary[t]['speedup_token_steps']['std'] for t in temps]

    ax.plot(temps, means, 'o-', color='#2E86AB', markersize=10, linewidth=2)
    ax.fill_between(temps, np.array(means) - np.array(stds), np.array(means) + np.array(stds),
                     alpha=0.2, color='#2E86AB')

    ax.axhline(y=1.0, color='gray', linestyle='--', linewidth=1, alpha=0.5)
    ax.set_xlabel('Temperature', fontweight='bold')
    ax.set_ylabel('Speedup Factor (Naive / EAB)', fontweight='bold')
    ax.set_title('Speedup vs Temperature', fontweight='bold', pad=20)
    ax.grid(True, alpha=0.3, linestyle=':')
    ax.set_ylim(bottom=0)

    plt.savefig(output_dir / "speedup_vs_temperature.png")
    print(f"   ✓ Saved: speedup_vs_temperature.png")
    plt.close()


def plot_entropy_vs_temperature(summary, output_dir):
    """Plot avg entropy vs temperature (key insight!)."""
    set_style()
    fig, ax = plt.subplots(figsize=(10, 6))

    temps = sorted(summary.keys())
    entropies = [summary[t]['avg_entropy']['mean'] for t in temps]
    stds = [summary[t]['avg_entropy']['std'] for t in temps]

    ax.plot(temps, entropies, 'o-', color='#F18F01', markersize=10, linewidth=2)
    ax.fill_between(temps, np.array(entropies) - np.array(stds), np.array(entropies) + np.array(stds),
                     alpha=0.2, color='#F18F01')

    ax.set_xlabel('Temperature', fontweight='bold')
    ax.set_ylabel('Average Entropy', fontweight='bold')
    ax.set_title('Temperature-Entropy Relationship', fontweight='bold', pad=20)
    ax.grid(True, alpha=0.3, linestyle=':')

    plt.savefig(output_dir / "entropy_vs_temperature.png")
    print(f"   ✓ Saved: entropy_vs_temperature.png")
    plt.close()


def plot_branching_vs_temperature(summary, output_dir):
    """Plot branching behavior vs temperature."""
    set_style()
    fig, ax = plt.subplots(figsize=(10, 6))

    temps = sorted(summary.keys())
    branches = [summary[t]['branch_count']['mean'] for t in temps]
    stds = [summary[t]['branch_count']['std'] for t in temps]

    ax.plot(temps, branches, 'o-', color='#A23B72', markersize=10, linewidth=2)
    ax.fill_between(temps, np.array(branches) - np.array(stds), np.array(branches) + np.array(stds),
                     alpha=0.2, color='#A23B72')

    ax.set_xlabel('Temperature', fontweight='bold')
    ax.set_ylabel('Number of Branches', fontweight='bold')
    ax.set_title('Branching Behavior vs Temperature', fontweight='bold', pad=20)
    ax.grid(True, alpha=0.3, linestyle=':')

    plt.savefig(output_dir / "branching_vs_temperature.png")
    print(f"   ✓ Saved: branching_vs_temperature.png")
    plt.close()


def plot_combined_analysis(summary, output_dir):
    """Combined plot: speedup, entropy, and branching."""
    set_style()
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 14))

    temps = sorted(summary.keys())

    # Speedup
    speedups = [summary[t]['speedup_token_steps']['mean'] for t in temps]
    ax1.plot(temps, speedups, 'o-', color='#2E86AB', markersize=8, linewidth=2)
    ax1.axhline(y=1.0, color='gray', linestyle='--', linewidth=1, alpha=0.5)
    ax1.set_ylabel('Speedup Factor', fontweight='bold')
    ax1.set_title('Combined Analysis: Temperature Effects', fontweight='bold', pad=20)
    ax1.grid(True, alpha=0.3, linestyle=':')

    # Entropy
    entropies = [summary[t]['avg_entropy']['mean'] for t in temps]
    ax2.plot(temps, entropies, 'o-', color='#F18F01', markersize=8, linewidth=2)
    ax2.set_ylabel('Average Entropy', fontweight='bold')
    ax2.grid(True, alpha=0.3, linestyle=':')

    # Branches
    branches = [summary[t]['branch_count']['mean'] for t in temps]
    ax3.plot(temps, branches, 'o-', color='#A23B72', markersize=8, linewidth=2)
    ax3.set_xlabel('Temperature', fontweight='bold')
    ax3.set_ylabel('Number of Branches', fontweight='bold')
    ax3.grid(True, alpha=0.3, linestyle=':')

    plt.tight_layout()
    plt.savefig(output_dir / "combined_analysis.png")
    print(f"   ✓ Saved: combined_analysis.png")
    plt.close()


def main():
    print("=" * 70)
    print("GENERATING PLOTS FOR EXPERIMENT 1.C.3")
    print("=" * 70)

    print("\n1. Loading summary statistics...")
    summary_file = experiment_dir / "results" / "summary_stats.json"
    with open(summary_file, 'r') as f:
        summary_data = json.load(f)
    summary = {float(k): v for k, v in summary_data['summary_by_temperature'].items()}
    print(f"   ✓ Loaded summary for {len(summary)} temperatures")

    print("\n2. Generating plots...")
    figures_dir = experiment_dir / "results" / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    plot_speedup_vs_temperature(summary, figures_dir)
    plot_entropy_vs_temperature(summary, figures_dir)
    plot_branching_vs_temperature(summary, figures_dir)
    plot_combined_analysis(summary, figures_dir)

    print("\n" + "=" * 70)
    print("PLOTTING COMPLETE")
    print("=" * 70)
    print(f"\nGenerated figures in: {figures_dir}/")
    print("  • speedup_vs_temperature.png")
    print("  • entropy_vs_temperature.png")
    print("  • branching_vs_temperature.png")
    print("  • combined_analysis.png")
    print("=" * 70)


if __name__ == "__main__":
    main()
