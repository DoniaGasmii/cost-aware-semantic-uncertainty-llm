"""
Generate plots for Experiment 1.C.4: Temperature × Threshold Interaction (2D Heatmaps)
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
        'figure.figsize': (10, 8),
        'savefig.dpi': 300,
        'savefig.bbox': 'tight',
    })


def parse_summary(summary_data):
    """Parse summary data into (temp, threshold) format."""
    summary = {}
    for key, value in summary_data['summary_by_combo'].items():
        temp_str, threshold_str = key.split('_')
        temp = float(temp_str)
        threshold = float(threshold_str)
        summary[(temp, threshold)] = value
    return summary


def build_grid(summary, metric):
    """Build 2D grid for heatmap."""
    temps = sorted(set(k[0] for k in summary.keys()))
    thresholds = sorted(set(k[1] for k in summary.keys()))

    grid = np.zeros((len(thresholds), len(temps)))

    for i, threshold in enumerate(thresholds):
        for j, temp in enumerate(temps):
            if (temp, threshold) in summary:
                grid[i, j] = summary[(temp, threshold)][metric]['mean']
            else:
                grid[i, j] = np.nan

    return temps, thresholds, grid


def plot_heatmap_speedup(summary, output_dir):
    """2D heatmap: Speedup vs (temperature, threshold)."""
    set_style()
    fig, ax = plt.subplots(figsize=(10, 8))

    temps, thresholds, grid = build_grid(summary, 'speedup_token_steps')

    im = ax.imshow(grid, cmap='RdYlGn', aspect='auto', origin='lower')

    # Labels
    ax.set_xticks(range(len(temps)))
    ax.set_xticklabels([f"{t:.1f}" for t in temps])
    ax.set_yticks(range(len(thresholds)))
    ax.set_yticklabels([f"{t:.3f}" for t in thresholds])

    ax.set_xlabel('Temperature', fontweight='bold')
    ax.set_ylabel('Entropy Threshold', fontweight='bold')
    ax.set_title('Speedup Heatmap: Temperature × Threshold', fontweight='bold', pad=20)

    # Colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Speedup Factor', fontweight='bold')

    # Annotate cells with values
    for i in range(len(thresholds)):
        for j in range(len(temps)):
            if not np.isnan(grid[i, j]):
                text = ax.text(j, i, f'{grid[i, j]:.2f}',
                              ha="center", va="center", color="black", fontsize=9, fontweight='bold')

    plt.tight_layout()
    plt.savefig(output_dir / "heatmap_speedup.png")
    print(f"   ✓ Saved: heatmap_speedup.png")
    plt.close()


def plot_heatmap_diversity(summary, output_dir):
    """2D heatmap: Diversity vs (temperature, threshold)."""
    set_style()
    fig, ax = plt.subplots(figsize=(10, 8))

    temps, thresholds, grid = build_grid(summary, 'eab_diversity')

    im = ax.imshow(grid, cmap='viridis', aspect='auto', origin='lower')

    ax.set_xticks(range(len(temps)))
    ax.set_xticklabels([f"{t:.1f}" for t in temps])
    ax.set_yticks(range(len(thresholds)))
    ax.set_yticklabels([f"{t:.3f}" for t in thresholds])

    ax.set_xlabel('Temperature', fontweight='bold')
    ax.set_ylabel('Entropy Threshold', fontweight='bold')
    ax.set_title('Diversity Heatmap: Temperature × Threshold', fontweight='bold', pad=20)

    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Diversity (Unique Tokens Ratio)', fontweight='bold')

    for i in range(len(thresholds)):
        for j in range(len(temps)):
            if not np.isnan(grid[i, j]):
                text = ax.text(j, i, f'{grid[i, j]:.3f}',
                              ha="center", va="center", color="white", fontsize=9, fontweight='bold')

    plt.tight_layout()
    plt.savefig(output_dir / "heatmap_diversity.png")
    print(f"   ✓ Saved: heatmap_diversity.png")
    plt.close()


def plot_heatmap_branching(summary, output_dir):
    """2D heatmap: Branching count vs (temperature, threshold)."""
    set_style()
    fig, ax = plt.subplots(figsize=(10, 8))

    temps, thresholds, grid = build_grid(summary, 'branch_count')

    im = ax.imshow(grid, cmap='plasma', aspect='auto', origin='lower')

    ax.set_xticks(range(len(temps)))
    ax.set_xticklabels([f"{t:.1f}" for t in temps])
    ax.set_yticks(range(len(thresholds)))
    ax.set_yticklabels([f"{t:.3f}" for t in thresholds])

    ax.set_xlabel('Temperature', fontweight='bold')
    ax.set_ylabel('Entropy Threshold', fontweight='bold')
    ax.set_title('Branching Heatmap: Temperature × Threshold', fontweight='bold', pad=20)

    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Number of Branches', fontweight='bold')

    for i in range(len(thresholds)):
        for j in range(len(temps)):
            if not np.isnan(grid[i, j]):
                text = ax.text(j, i, f'{grid[i, j]:.0f}',
                              ha="center", va="center", color="white", fontsize=9, fontweight='bold')

    plt.tight_layout()
    plt.savefig(output_dir / "heatmap_branching.png")
    print(f"   ✓ Saved: heatmap_branching.png")
    plt.close()


def plot_pareto_frontier(summary, output_dir):
    """Pareto frontier: Cost vs Quality."""
    set_style()
    fig, ax = plt.subplots(figsize=(10, 6))

    speedups = []
    diversities = []
    labels = []

    for (temp, threshold), stats in summary.items():
        speedups.append(stats['speedup_token_steps']['mean'])
        diversities.append(stats['eab_diversity']['mean'])
        labels.append(f"T={temp:.1f}, τ={threshold:.3f}")

    scatter = ax.scatter(speedups, diversities, s=150, c=range(len(speedups)),
                        cmap='coolwarm', edgecolors='black', linewidth=1.5)

    ax.set_xlabel('Speedup Factor (Lower Cost)', fontweight='bold')
    ax.set_ylabel('Diversity (Higher Quality)', fontweight='bold')
    ax.set_title('Pareto Frontier: Cost vs Quality', fontweight='bold', pad=20)
    ax.grid(True, alpha=0.3, linestyle=':')

    plt.colorbar(scatter, ax=ax, label='Configuration Index')

    plt.tight_layout()
    plt.savefig(output_dir / "pareto_frontier.png")
    print(f"   ✓ Saved: pareto_frontier.png")
    plt.close()


def main():
    print("=" * 70)
    print("GENERATING PLOTS FOR EXPERIMENT 1.C.4")
    print("=" * 70)

    print("\n1. Loading summary statistics...")
    summary_file = experiment_dir / "results" / "summary_stats.json"
    with open(summary_file, 'r') as f:
        summary_data = json.load(f)
    summary = parse_summary(summary_data)
    print(f"   ✓ Loaded summary for {len(summary)} (temp, threshold) combinations")

    print("\n2. Generating 2D heatmaps...")
    figures_dir = experiment_dir / "results" / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    plot_heatmap_speedup(summary, figures_dir)
    plot_heatmap_diversity(summary, figures_dir)
    plot_heatmap_branching(summary, figures_dir)
    plot_pareto_frontier(summary, figures_dir)

    print("\n" + "=" * 70)
    print("PLOTTING COMPLETE")
    print("=" * 70)
    print(f"\nGenerated figures in: {figures_dir}/")
    print("  • heatmap_speedup.png (2D)")
    print("  • heatmap_diversity.png (2D)")
    print("  • heatmap_branching.png (2D)")
    print("  • pareto_frontier.png")
    print("=" * 70)


if __name__ == "__main__":
    main()
