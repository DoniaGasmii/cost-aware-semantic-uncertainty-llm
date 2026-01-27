"""
Create clear, concise visualizations for diversity comparison.

Clean pastel colors, straightforward message.
"""

import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

experiment_dir = Path(__file__).parent


def set_style():
    """Set clean plotting style with pastel colors."""
    sns.set_palette("pastel")
    plt.rcParams.update({
        'font.size': 13,
        'axes.labelsize': 14,
        'axes.titlesize': 15,
        'xtick.labelsize': 12,
        'ytick.labelsize': 12,
        'legend.fontsize': 12,
        'figure.dpi': 100,
        'savefig.dpi': 300,
        'savefig.bbox': 'tight',
    })


def plot_comparison_bars(results: dict, output_path: Path):
    """
    Create clean bar chart comparing EAB vs Naive.

    Shows Self-BLEU, Distinct-n, and Lexical Diversity.
    """
    set_style()

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    eab = results['eab_aggregated']
    naive = results['naive_aggregated']

    # Pastel colors
    eab_color = '#AEC6CF'  # Pastel blue
    naive_color = '#FFB6C1'  # Pastel pink

    # Plot 1: Self-BLEU (lower = more diverse)
    ax1 = axes[0]
    metrics = ['self_bleu_1', 'self_bleu_2', 'self_bleu_3']
    labels = ['1-gram', '2-gram', '3-gram']
    x = np.arange(len(labels))
    width = 0.35

    eab_vals = [eab[m]['mean'] for m in metrics]
    naive_vals = [naive[m]['mean'] for m in metrics]
    eab_stds = [eab[m]['std'] for m in metrics]
    naive_stds = [naive[m]['std'] for m in metrics]

    ax1.bar(x - width/2, eab_vals, width, label='EAB', color=eab_color,
            yerr=eab_stds, capsize=5, edgecolor='black', linewidth=1.2)
    ax1.bar(x + width/2, naive_vals, width, label='Naive', color=naive_color,
            yerr=naive_stds, capsize=5, edgecolor='black', linewidth=1.2)

    ax1.set_xlabel('N-gram Size', fontweight='bold')
    ax1.set_ylabel('Self-BLEU Score', fontweight='bold')
    ax1.set_title('Self-BLEU\n(Lower = More Diverse)', fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels)
    ax1.legend()
    ax1.grid(axis='y', alpha=0.3, linestyle=':')

    # Plot 2: Distinct-n (higher = more diverse)
    ax2 = axes[1]
    metrics = ['distinct_1', 'distinct_2', 'distinct_3']

    eab_vals = [eab[m]['mean'] for m in metrics]
    naive_vals = [naive[m]['mean'] for m in metrics]
    eab_stds = [eab[m]['std'] for m in metrics]
    naive_stds = [naive[m]['std'] for m in metrics]

    ax2.bar(x - width/2, eab_vals, width, label='EAB', color=eab_color,
            yerr=eab_stds, capsize=5, edgecolor='black', linewidth=1.2)
    ax2.bar(x + width/2, naive_vals, width, label='Naive', color=naive_color,
            yerr=naive_stds, capsize=5, edgecolor='black', linewidth=1.2)

    ax2.set_xlabel('N-gram Size', fontweight='bold')
    ax2.set_ylabel('Distinct-n Score', fontweight='bold')
    ax2.set_title('Distinct-n\n(Higher = More Diverse)', fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels)
    ax2.legend()
    ax2.grid(axis='y', alpha=0.3, linestyle=':')

    # Plot 3: Lexical Diversity
    ax3 = axes[2]
    eab_lex = eab['lexical_diversity']['mean']
    naive_lex = naive['lexical_diversity']['mean']
    eab_std = eab['lexical_diversity']['std']
    naive_std = naive['lexical_diversity']['std']

    bars = ax3.bar(['EAB', 'Naive'], [eab_lex, naive_lex],
                   color=[eab_color, naive_color],
                   yerr=[eab_std, naive_std],
                   capsize=8, edgecolor='black', linewidth=1.2, width=0.6)

    # Add values on bars
    for bar, val in zip(bars, [eab_lex, naive_lex]):
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., height,
                f'{val:.3f}',
                ha='center', va='bottom', fontweight='bold', fontsize=11)

    ax3.set_ylabel('Lexical Diversity', fontweight='bold')
    ax3.set_title('Lexical Diversity\n(Higher = More Diverse)', fontweight='bold')
    ax3.set_ylim(0, max(eab_lex, naive_lex) * 1.2)
    ax3.grid(axis='y', alpha=0.3, linestyle=':')

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"✓ Saved: {output_path}")
    plt.close()


def plot_diversity_ratios(results: dict, output_path: Path):
    """
    Create a clean ratio chart: How many times more diverse is Naive vs EAB?
    """
    set_style()

    comparison = results['comparison']

    # Select key metrics
    metrics = {
        'Self-BLEU-2': comparison['self_bleu_2'],
        'Distinct-2': comparison['distinct_2'],
        'Lexical Div.': comparison['lexical_diversity']
    }

    labels = list(metrics.keys())
    ratios = [metrics[k]['ratio'] for k in labels]

    fig, ax = plt.subplots(figsize=(8, 6))

    bars = ax.barh(labels, ratios, color='#B19CD9', edgecolor='black', linewidth=1.2)

    # Add value labels
    for i, (bar, ratio) in enumerate(zip(bars, ratios)):
        ax.text(ratio + 0.1, bar.get_y() + bar.get_height()/2,
                f'{ratio:.1f}×',
                ha='left', va='center', fontweight='bold', fontsize=12)

    # Baseline at 1.0
    ax.axvline(1.0, color='red', linestyle='--', linewidth=2, alpha=0.7,
               label='Equal diversity')

    ax.set_xlabel('Diversity Ratio (Naive / EAB)', fontweight='bold', fontsize=14)
    ax.set_title('How Much More Diverse is Naive Sampling?', fontweight='bold', fontsize=15)
    ax.legend()
    ax.grid(axis='x', alpha=0.3, linestyle=':')
    ax.set_xlim(0, max(ratios) * 1.15)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"✓ Saved: {output_path}")
    plt.close()


def create_summary_table(results: dict, output_path: Path):
    """Create a clean summary table."""
    set_style()

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.axis('tight')
    ax.axis('off')

    eab = results['eab_aggregated']
    naive = results['naive_aggregated']
    comp = results['comparison']

    # Table data
    data = [
        ['Metric', 'EAB', 'Naive', 'Ratio (Naive/EAB)'],
        ['Self-BLEU-2 ↓',
         f"{eab['self_bleu_2']['mean']:.3f}",
         f"{naive['self_bleu_2']['mean']:.3f}",
         f"{comp['self_bleu_2']['ratio']:.1f}× more diverse"],
        ['Distinct-2 ↑',
         f"{eab['distinct_2']['mean']:.3f}",
         f"{naive['distinct_2']['mean']:.3f}",
         f"{comp['distinct_2']['ratio']:.1f}× more diverse"],
        ['Lexical Div. ↑',
         f"{eab['lexical_diversity']['mean']:.3f}",
         f"{naive['lexical_diversity']['mean']:.3f}",
         f"{comp['lexical_diversity']['ratio']:.1f}× more diverse"],
    ]

    table = ax.table(cellText=data, cellLoc='center', loc='center',
                     colWidths=[0.3, 0.2, 0.2, 0.3])

    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1, 2.2)

    # Style header
    for i in range(4):
        table[(0, i)].set_facecolor('#4A90E2')
        table[(0, i)].set_text_props(weight='bold', color='white')

    # Alternate row colors
    for i in range(1, 4):
        for j in range(4):
            if i % 2 == 0:
                table[(i, j)].set_facecolor('#F0F0F0')

    plt.title('Diversity Comparison: EAB vs Naive Sampling',
              fontsize=15, fontweight='bold', pad=15)

    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"✓ Saved: {output_path}")
    plt.close()


def main():
    """Generate all plots."""
    print("="*70)
    print("CREATING DIVERSITY VISUALIZATIONS")
    print("="*70)

    # Load results
    results_file = experiment_dir / "results" / "diversity_metrics.json"
    with open(results_file) as f:
        results = json.load(f)

    output_dir = experiment_dir / "results" / "figures"
    output_dir.mkdir(exist_ok=True)

    print("\nGenerating plots...")

    # 1. Comparison bars
    plot_comparison_bars(results, output_dir / "diversity_comparison_bars.png")

    # 2. Ratio chart
    plot_diversity_ratios(results, output_dir / "diversity_ratios.png")

    # 3. Summary table
    create_summary_table(results, output_dir / "diversity_summary_table.png")

    print("\n" + "="*70)
    print("✓ ALL PLOTS GENERATED")
    print("="*70)
    print(f"\nSaved to: {output_dir}/")
    print("  • diversity_comparison_bars.png")
    print("  • diversity_ratios.png")
    print("  • diversity_summary_table.png")
    print("="*70)


if __name__ == "__main__":
    main()
