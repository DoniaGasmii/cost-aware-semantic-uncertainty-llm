#!/usr/bin/env python3
"""
Visualize Entropy Distributions

Creates publication-quality plots showing:
1. Overlaid distribution curves for 3 confidence levels
2. Box plots by confidence level
3. CDF plots showing percentiles
4. Branching behavior vs threshold
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from scipy import stats
from scipy.stats import gaussian_kde

# Set style for publication-quality plots
plt.style.use('seaborn-v0_8-paper')
sns.set_palette("husl")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 11
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['xtick.labelsize'] = 10
plt.rcParams['ytick.labelsize'] = 10
plt.rcParams['legend.fontsize'] = 11


def load_data():
    """Load pilot study results."""
    results_file = Path(__file__).parent.parent / 'results' / 'pilot_summary.csv'
    if not results_file.exists():
        raise FileNotFoundError(f"Results file not found: {results_file}\nRun pilot study first!")

    df = pd.read_csv(results_file)
    return df


def plot_overlaid_distributions(df, output_dir):
    """Plot overlaid KDE distributions for the 3 confidence levels."""
    fig, ax = plt.subplots(figsize=(12, 7))

    colors = {'high': '#2ecc71', 'medium': '#f39c12', 'low': '#e74c3c'}
    labels = {'high': 'High Confidence', 'medium': 'Medium Confidence', 'low': 'Low Confidence'}

    for level in ['high', 'medium', 'low']:
        level_data = df[df['confidence_level'] == level]['avg_entropy'].values

        # Create KDE
        kde = gaussian_kde(level_data)
        x_range = np.linspace(0, max(df['avg_entropy']), 500)
        density = kde(x_range)

        # Plot
        ax.plot(x_range, density, label=labels[level], color=colors[level], linewidth=2.5)
        ax.fill_between(x_range, density, alpha=0.2, color=colors[level])

        # Add mean line
        mean_val = level_data.mean()
        ax.axvline(mean_val, color=colors[level], linestyle='--', alpha=0.7, linewidth=1.5)

    ax.set_xlabel('Average Normalized Entropy', fontsize=13, fontweight='bold')
    ax.set_ylabel('Density', fontsize=13, fontweight='bold')
    ax.set_title('Entropy Distribution by Prompt Confidence Level', fontsize=15, fontweight='bold', pad=20)
    ax.legend(loc='upper right', framealpha=0.9)
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.set_xlim(0, None)

    plt.tight_layout()
    output_file = output_dir / 'entropy_distributions.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"  ✓ Saved: {output_file}")
    plt.close()


def plot_box_plots(df, output_dir):
    """Create box plots for entropy by confidence level."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # Plot 1: Average Entropy
    order = ['high', 'medium', 'low']
    colors = ['#2ecc71', '#f39c12', '#e74c3c']

    box_data_avg = [df[df['confidence_level'] == level]['avg_entropy'].values for level in order]
    bp1 = ax1.boxplot(box_data_avg, labels=['High', 'Medium', 'Low'],
                      patch_artist=True, widths=0.6)

    for patch, color in zip(bp1['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)

    ax1.set_ylabel('Average Normalized Entropy', fontsize=12, fontweight='bold')
    ax1.set_xlabel('Confidence Level', fontsize=12, fontweight='bold')
    ax1.set_title('Average Entropy Distribution', fontsize=13, fontweight='bold')
    ax1.grid(True, alpha=0.3, axis='y')

    # Plot 2: Max Entropy
    box_data_max = [df[df['confidence_level'] == level]['max_entropy'].values for level in order]
    bp2 = ax2.boxplot(box_data_max, labels=['High', 'Medium', 'Low'],
                      patch_artist=True, widths=0.6)

    for patch, color in zip(bp2['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)

    ax2.set_ylabel('Maximum Normalized Entropy', fontsize=12, fontweight='bold')
    ax2.set_xlabel('Confidence Level', fontsize=12, fontweight='bold')
    ax2.set_title('Maximum Entropy Distribution', fontsize=13, fontweight='bold')
    ax2.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    output_file = output_dir / 'entropy_boxplots.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"  ✓ Saved: {output_file}")
    plt.close()


def plot_cdf(df, output_dir):
    """Plot cumulative distribution functions with percentile markers."""
    fig, ax = plt.subplots(figsize=(12, 7))

    colors = {'high': '#2ecc71', 'medium': '#f39c12', 'low': '#e74c3c'}
    labels = {'high': 'High Confidence', 'medium': 'Medium Confidence', 'low': 'Low Confidence'}

    percentiles_to_mark = [25, 50, 75, 90]

    for level in ['high', 'medium', 'low']:
        level_data = df[df['confidence_level'] == level]['avg_entropy'].values
        sorted_data = np.sort(level_data)
        cdf = np.arange(1, len(sorted_data) + 1) / len(sorted_data)

        ax.plot(sorted_data, cdf, label=labels[level], color=colors[level], linewidth=2.5)

        # Mark percentiles
        for p in percentiles_to_mark:
            percentile_val = np.percentile(level_data, p)
            if level == 'medium':  # Only mark for medium to avoid clutter
                ax.axvline(percentile_val, color=colors[level], linestyle=':', alpha=0.5, linewidth=1)
                ax.text(percentile_val, 0.05, f'{p}th', rotation=90,
                       verticalalignment='bottom', fontsize=9, color=colors[level])

    ax.set_xlabel('Average Normalized Entropy', fontsize=13, fontweight='bold')
    ax.set_ylabel('Cumulative Probability', fontsize=13, fontweight='bold')
    ax.set_title('Cumulative Distribution Functions by Confidence Level', fontsize=15, fontweight='bold', pad=20)
    ax.legend(loc='lower right', framealpha=0.9)
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.set_xlim(0, None)
    ax.set_ylim(0, 1)

    plt.tight_layout()
    output_file = output_dir / 'entropy_cdf.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"  ✓ Saved: {output_file}")
    plt.close()


def plot_branching_behavior(df, output_dir):
    """Plot branching behavior: samples and branches by confidence level."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    colors = {'high': '#2ecc71', 'medium': '#f39c12', 'low': '#e74c3c'}
    levels = ['high', 'medium', 'low']
    x_pos = np.arange(len(levels))

    # Plot 1: Average number of samples
    samples_mean = [df[df['confidence_level'] == level]['num_samples'].mean() for level in levels]
    samples_std = [df[df['confidence_level'] == level]['num_samples'].std() for level in levels]

    bars1 = ax1.bar(x_pos, samples_mean, yerr=samples_std, capsize=5,
                    color=[colors[l] for l in levels], alpha=0.7, edgecolor='black', linewidth=1.5)

    ax1.set_xlabel('Confidence Level', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Average Number of Samples', fontsize=12, fontweight='bold')
    ax1.set_title('Samples Generated by Confidence Level', fontsize=13, fontweight='bold')
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels(['High', 'Medium', 'Low'])
    ax1.grid(True, alpha=0.3, axis='y')

    # Plot 2: Average number of branches
    branches_mean = [df[df['confidence_level'] == level]['num_branches'].mean() for level in levels]
    branches_std = [df[df['confidence_level'] == level]['num_branches'].std() for level in levels]

    bars2 = ax2.bar(x_pos, branches_mean, yerr=branches_std, capsize=5,
                    color=[colors[l] for l in levels], alpha=0.7, edgecolor='black', linewidth=1.5)

    ax2.set_xlabel('Confidence Level', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Average Number of Branches', fontsize=12, fontweight='bold')
    ax2.set_title('Branching Points by Confidence Level', fontsize=13, fontweight='bold')
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(['High', 'Medium', 'Low'])
    ax2.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    output_file = output_dir / 'branching_behavior.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"  ✓ Saved: {output_file}")
    plt.close()


def plot_threshold_sweep(df, output_dir):
    """Simulate threshold sweep showing branching behavior."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    thresholds = np.linspace(0.01, 0.30, 50)
    colors = {'high': '#2ecc71', 'medium': '#f39c12', 'low': '#e74c3c'}

    for level in ['high', 'medium', 'low']:
        level_data = df[df['confidence_level'] == level]

        # For each threshold, count how many prompts would branch
        branch_rates = []
        avg_branches = []

        for threshold in thresholds:
            # Count prompts with max_entropy > threshold
            would_branch = (level_data['max_entropy'] > threshold).sum()
            branch_rate = would_branch / len(level_data) * 100
            branch_rates.append(branch_rate)

            # Average branches for prompts that would branch
            branching_prompts = level_data[level_data['max_entropy'] > threshold]
            avg_branch = branching_prompts['num_branches'].mean() if len(branching_prompts) > 0 else 0
            avg_branches.append(avg_branch)

        ax1.plot(thresholds, branch_rates, label=level.capitalize(),
                color=colors[level], linewidth=2.5)
        ax2.plot(thresholds, avg_branches, label=level.capitalize(),
                color=colors[level], linewidth=2.5)

    # Plot 1: Branch rate
    ax1.set_xlabel('Entropy Threshold', fontsize=12, fontweight='bold')
    ax1.set_ylabel('% of Prompts That Would Branch', fontsize=12, fontweight='bold')
    ax1.set_title('Branching Rate vs Threshold', fontsize=13, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(0.01, 0.30)

    # Plot 2: Average branches
    ax2.set_xlabel('Entropy Threshold', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Average Number of Branches', fontsize=12, fontweight='bold')
    ax2.set_title('Branch Count vs Threshold', fontsize=13, fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(0.01, 0.30)

    plt.tight_layout()
    output_file = output_dir / 'threshold_sweep.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"  ✓ Saved: {output_file}")
    plt.close()


def main():
    print("="*80)
    print("Visualizing Entropy Distributions")
    print("="*80)

    # Load data
    print("\n[1/2] Loading data...")
    df = load_data()
    print(f"  ✓ Loaded {len(df)} prompt results")

    # Create output directory
    output_dir = Path(__file__).parent.parent / 'plots'
    output_dir.mkdir(exist_ok=True)

    # Generate plots
    print("\n[2/2] Generating plots...")

    plot_overlaid_distributions(df, output_dir)
    plot_box_plots(df, output_dir)
    plot_cdf(df, output_dir)
    plot_branching_behavior(df, output_dir)
    plot_threshold_sweep(df, output_dir)

    print("\n" + "="*80)
    print(f"✓ All plots saved to: {output_dir}")
    print("="*80)
    print("\nGenerated plots:")
    print("  1. entropy_distributions.png - Overlaid KDE distributions")
    print("  2. entropy_boxplots.png - Box plots by confidence level")
    print("  3. entropy_cdf.png - Cumulative distribution functions")
    print("  4. branching_behavior.png - Samples and branches by level")
    print("  5. threshold_sweep.png - Branching behavior vs threshold")
    print("\nThese plots are publication-ready (300 DPI) for your report!")
    print("="*80)


if __name__ == '__main__':
    main()
