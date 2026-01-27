"""
Improved plotting utilities for EAB experiments with clearer visualizations.
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Any, Union, Optional
import seaborn as sns


def set_publication_style():
    """Set matplotlib style for publication-quality figures."""
    plt.rcParams.update({
        'font.size': 13,
        'axes.labelsize': 15,
        'axes.titlesize': 16,
        'xtick.labelsize': 12,
        'ytick.labelsize': 12,
        'legend.fontsize': 12,
        'figure.titlesize': 18,
        'figure.dpi': 100,
        'savefig.dpi': 300,
        'savefig.bbox': 'tight',
        'axes.grid': True,
        'grid.alpha': 0.3,
        'grid.linestyle': ':',
    })
    sns.set_palette("husl")


def plot_speedup_vs_length_improved(
    summary: Dict[int, Dict[str, Any]],
    output_path: Union[str, Path],
    metric: str = 'speedup_token_steps',
    show_plot: bool = False,
    add_values: bool = True
):
    """
    Plot speedup vs prompt length with clear numbers.

    Args:
        summary: Summary statistics by prompt length
        output_path: Path to save figure
        metric: Metric to plot ('speedup_token_steps', 'speedup_time', or 'speedup_memory')
        show_plot: Whether to display the plot
        add_values: Whether to add value labels to data points
    """
    set_publication_style()

    fig, ax = plt.subplots(figsize=(12, 7))

    # Sort lengths
    lengths = sorted(summary.keys())

    # Extract data
    means = [summary[l][metric]['mean'] for l in lengths]
    stds = [summary[l][metric]['std'] for l in lengths]

    # Metric-specific styling
    metric_info = {
        'speedup_token_steps': {
            'label': 'Token-Step Speedup',
            'color': '#2E86AB',
            'ylabel': 'Speedup Factor (Naive / EAB)',
            'title': 'EAB Computational Efficiency: Token-Step Speedup vs Prompt Length'
        },
        'speedup_time': {
            'label': 'Wall-Clock Speedup',
            'color': '#A23B72',
            'ylabel': 'Speedup Factor (Naive / EAB)',
            'title': 'EAB Wall-Clock Time Speedup vs Prompt Length'
        },
        'speedup_memory': {
            'label': 'Memory Overhead',
            'color': '#F18F01',
            'ylabel': 'Memory Ratio (EAB / Naive)',
            'title': 'EAB Memory Overhead vs Prompt Length'
        }
    }

    info = metric_info.get(metric, metric_info['speedup_token_steps'])

    # Plot with error bars
    ax.errorbar(
        lengths, means, yerr=stds,
        color=info['color'],
        marker='o',
        markersize=10,
        linewidth=2.5,
        capsize=5,
        capthick=2,
        label=info['label'],
        alpha=0.9
    )

    # Add value labels if requested
    if add_values:
        for x, y in zip(lengths, means):
            ax.annotate(
                f'{y:.2f}×',
                xy=(x, y),
                xytext=(0, 10),
                textcoords='offset points',
                ha='center',
                va='bottom',
                fontsize=10,
                fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor='gray', alpha=0.8)
            )

    # Reference line at y=1 (no speedup)
    ax.axhline(y=1.0, color='red', linestyle='--', linewidth=2, alpha=0.7, label='No speedup baseline')

    # Labels and title
    ax.set_xlabel('Prompt Length (tokens)', fontweight='bold', fontsize=15)
    ax.set_ylabel(info['ylabel'], fontweight='bold', fontsize=15)
    ax.set_title(info['title'], fontweight='bold', pad=20, fontsize=16)
    ax.legend(loc='best', framealpha=0.95, fontsize=12)
    ax.grid(True, alpha=0.3, linestyle=':', linewidth=1)

    # Set y-axis to start at 0
    ax.set_ylim(bottom=0)

    # Add subtle background
    ax.set_facecolor('#f9f9f9')

    # Save
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches='tight', dpi=300, facecolor='white')
    print(f"✓ Saved improved plot to {output_path}")

    if show_plot:
        plt.show()
    else:
        plt.close()


def create_comparison_table_3way(
    summary: Dict[int, Dict[str, Any]],
    output_path: Union[str, Path]
):
    """
    Create a 3-way comparison table: Sequential Naive vs Parallel Naive vs EAB.

    Args:
        summary: Summary statistics by prompt length
        output_path: Path to save table (PNG and CSV)
    """
    set_publication_style()

    # Prepare data
    data = []
    for length in sorted(summary.keys())[:10]:  # Limit to first 10 for readability
        s = summary[length]

        # Sequential Naive (baseline)
        seq_naive_tokens = s['naive_token_steps']['mean']

        # Parallel Naive (theoretical: same as sequential for token-steps, but different for time)
        # For token-steps: same as sequential
        # For wall-clock time: divide by num_samples (ideal parallelization)
        parallel_naive_tokens = seq_naive_tokens  # Same computational cost

        # EAB
        eab_tokens = s['eab_token_steps']['mean']

        # Speedup calculations
        speedup_vs_seq = seq_naive_tokens / eab_tokens
        speedup_vs_parallel = parallel_naive_tokens / eab_tokens  # Same as speedup_vs_seq for token-steps

        data.append({
            'Prompt Length': length,
            'Seq. Naive': f"{int(seq_naive_tokens):,}",
            'Par. Naive': f"{int(parallel_naive_tokens):,}",
            'EAB': f"{int(eab_tokens):,}",
            'Speedup (vs Seq)': f"{speedup_vs_seq:.2f}×",
            'Speedup (vs Par)': f"{speedup_vs_parallel:.2f}×",
            'Branches': f"{s['branch_count']['mean']:.1f}"
        })

    df = pd.DataFrame(data)

    # Save as CSV
    csv_path = Path(output_path).with_suffix('.csv')
    df.to_csv(csv_path, index=False)
    print(f"✓ Saved comparison table (CSV) to {csv_path}")

    # Create figure with table
    fig, ax = plt.subplots(figsize=(14, len(df) * 0.6 + 2))
    ax.axis('tight')
    ax.axis('off')

    # Create table
    table = ax.table(
        cellText=df.values,
        colLabels=df.columns,
        cellLoc='center',
        loc='center',
        colWidths=[0.12, 0.14, 0.14, 0.14, 0.16, 0.16, 0.12]
    )

    # Style table
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1, 2)

    # Header styling
    for i in range(len(df.columns)):
        cell = table[(0, i)]
        cell.set_facecolor('#2E86AB')
        cell.set_text_props(weight='bold', color='white')

    # Alternate row colors
    for i in range(1, len(df) + 1):
        for j in range(len(df.columns)):
            cell = table[(i, j)]
            if i % 2 == 0:
                cell.set_facecolor('#f0f0f0')
            else:
                cell.set_facecolor('#ffffff')

    plt.title('3-Way Comparison: Token-Step Cost Analysis', fontweight='bold', fontsize=16, pad=20)

    # Save
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches='tight', dpi=300, facecolor='white')
    print(f"✓ Saved comparison table (PNG) to {output_path}")
    plt.close()


def create_cost_analysis_table(
    summary: Dict[int, Dict[str, Any]],
    output_path: Union[str, Path],
    cost_per_1k_tokens: float = 0.001,  # Example: $0.001 per 1K tokens
    num_prompts: int = 1000  # Example scenario
):
    """
    Create a practical cost analysis table with dollar amounts.

    Args:
        summary: Summary statistics by prompt length
        output_path: Path to save table
        cost_per_1k_tokens: Cost per 1000 tokens (e.g., API pricing)
        num_prompts: Number of prompts in scenario
    """
    set_publication_style()

    # Calculate aggregate statistics across all prompt lengths
    total_naive_tokens = sum(s['naive_token_steps']['mean'] for s in summary.values())
    total_eab_tokens = sum(s['eab_token_steps']['mean'] for s in summary.values())
    avg_speedup = sum(s['speedup_token_steps']['mean'] for s in summary.values()) / len(summary)

    # Scale to scenario
    naive_cost = (total_naive_tokens / 1000) * cost_per_1k_tokens * num_prompts / len(summary)
    eab_cost = (total_eab_tokens / 1000) * cost_per_1k_tokens * num_prompts / len(summary)
    savings = naive_cost - eab_cost
    savings_pct = (savings / naive_cost) * 100

    # Create summary data
    data = [
        ['Method', 'Tokens/Prompt (avg)', 'Total Tokens', 'Cost (USD)', 'Savings'],
        ['Sequential Naive', f'{int(total_naive_tokens/len(summary)):,}',
         f'{int(total_naive_tokens/len(summary) * num_prompts):,}',
         f'${naive_cost:,.2f}', '-'],
        ['EAB', f'{int(total_eab_tokens/len(summary)):,}',
         f'{int(total_eab_tokens/len(summary) * num_prompts):,}',
         f'${eab_cost:,.2f}',
         f'${savings:,.2f} ({savings_pct:.1f}%)'],
        ['', '', '', '', ''],
        ['Average Speedup', f'{avg_speedup:.2f}×', '', '', ''],
        ['Scenario', f'{num_prompts:,} prompts', f'@ ${cost_per_1k_tokens}/1K tokens', '', '']
    ]

    # Create figure
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.axis('tight')
    ax.axis('off')

    # Create table
    table = ax.table(
        cellText=data,
        cellLoc='center',
        loc='center',
        colWidths=[0.25, 0.25, 0.20, 0.15, 0.20]
    )

    # Style table
    table.auto_set_font_size(False)
    table.set_fontsize(12)
    table.scale(1, 2.5)

    # Header styling
    for i in range(5):
        cell = table[(0, i)]
        cell.set_facecolor('#2E86AB')
        cell.set_text_props(weight='bold', color='white', fontsize=13)

    # EAB row (highlight savings)
    for i in range(5):
        cell = table[(2, i)]
        cell.set_facecolor('#e8f5e9')
        if i == 4:  # Savings column
            cell.set_text_props(weight='bold', color='#2e7d32')

    # Summary rows
    for row_idx in [4, 5]:
        for i in range(5):
            cell = table[(row_idx, i)]
            cell.set_facecolor('#f5f5f5')
            cell.set_text_props(style='italic')

    plt.title('Practical Cost Analysis: Real-World Savings Estimate',
              fontweight='bold', fontsize=16, pad=20)

    # Save
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches='tight', dpi=300, facecolor='white')
    print(f"✓ Saved cost analysis table to {output_path}")
    plt.close()


def create_comprehensive_bar_chart(
    summary: Dict[int, Dict[str, Any]],
    output_path: Union[str, Path],
    show_plot: bool = False
):
    """
    Create a clean bar chart comparing EAB vs Naive with actual numbers.

    Args:
        summary: Summary statistics by prompt length
        output_path: Path to save figure
        show_plot: Whether to display the plot
    """
    set_publication_style()

    # Select representative prompt lengths (avoid too many bars)
    lengths = sorted(summary.keys())
    # Select every Nth length to avoid overcrowding
    if len(lengths) > 10:
        step = len(lengths) // 10
        selected_lengths = [lengths[i] for i in range(0, len(lengths), step)][:10]
    else:
        selected_lengths = lengths

    # Extract data
    speedups = [summary[l]['speedup_token_steps']['mean'] for l in selected_lengths]
    naive_tokens = [summary[l]['naive_token_steps']['mean'] for l in selected_lengths]
    eab_tokens = [summary[l]['eab_token_steps']['mean'] for l in selected_lengths]

    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    # Plot 1: Speedup bar chart
    bars = ax1.bar(range(len(selected_lengths)), speedups, color='#2E86AB', alpha=0.8, edgecolor='black', linewidth=1.2)
    ax1.axhline(y=1.0, color='red', linestyle='--', linewidth=2, alpha=0.7, label='No speedup baseline')

    # Add value labels on bars
    for i, (bar, speedup) in enumerate(zip(bars, speedups)):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                f'{speedup:.2f}×',
                ha='center', va='bottom', fontsize=11, fontweight='bold')

    ax1.set_xlabel('Prompt Length (tokens)', fontweight='bold', fontsize=14)
    ax1.set_ylabel('Speedup Factor', fontweight='bold', fontsize=14)
    ax1.set_title('Token-Step Speedup by Prompt Length', fontweight='bold', fontsize=15)
    ax1.set_xticks(range(len(selected_lengths)))
    ax1.set_xticklabels(selected_lengths, rotation=45)
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3, axis='y')
    ax1.set_ylim(bottom=0, top=max(speedups) * 1.2)

    # Plot 2: Side-by-side comparison
    x = np.arange(len(selected_lengths))
    width = 0.35

    bars1 = ax2.bar(x - width/2, naive_tokens, width, label='Naive', color='#E63946', alpha=0.8, edgecolor='black')
    bars2 = ax2.bar(x + width/2, eab_tokens, width, label='EAB', color='#06A77D', alpha=0.8, edgecolor='black')

    ax2.set_xlabel('Prompt Length (tokens)', fontweight='bold', fontsize=14)
    ax2.set_ylabel('Token-Steps', fontweight='bold', fontsize=14)
    ax2.set_title('Computational Cost: Naive vs EAB', fontweight='bold', fontsize=15)
    ax2.set_xticks(x)
    ax2.set_xticklabels(selected_lengths, rotation=45)
    ax2.legend(fontsize=12, loc='upper left')
    ax2.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()

    # Save
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, bbox_inches='tight', dpi=300, facecolor='white')
    print(f"✓ Saved comprehensive bar chart to {output_path}")

    if show_plot:
        plt.show()
    else:
        plt.close()


def generate_all_improved_plots(
    results: List[Dict[str, Any]],
    summary: Dict[int, Dict[str, Any]],
    output_dir: Union[str, Path],
    show_plots: bool = False
):
    """
    Generate all improved plots for efficiency experiments.

    Args:
        results: List of result dictionaries
        summary: Summary statistics
        output_dir: Directory to save plots
        show_plots: Whether to display plots
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "="*70)
    print("GENERATING IMPROVED VISUALIZATIONS")
    print("="*70)

    # 1. Token-step speedup with clear numbers
    print("\n1. Creating token-step speedup plot...")
    plot_speedup_vs_length_improved(
        summary,
        output_dir / "speedup_token_steps_improved.png",
        metric='speedup_token_steps',
        show_plot=show_plots,
        add_values=True
    )

    # 2. Comprehensive bar chart
    print("\n2. Creating comprehensive bar chart...")
    create_comprehensive_bar_chart(
        summary,
        output_dir / "speedup_comparison_bars.png",
        show_plot=show_plots
    )

    # 3. 3-way comparison table
    print("\n3. Creating 3-way comparison table...")
    create_comparison_table_3way(
        summary,
        output_dir / "comparison_table_3way.png"
    )

    # 4. Cost analysis table
    print("\n4. Creating cost analysis table...")
    create_cost_analysis_table(
        summary,
        output_dir / "cost_analysis.png",
        cost_per_1k_tokens=0.001,  # Example API pricing
        num_prompts=1000
    )

    print("\n" + "="*70)
    print("✓ ALL IMPROVED VISUALIZATIONS GENERATED SUCCESSFULLY!")
    print("="*70)
    print(f"\nSaved to: {output_dir}/")
    print("  • speedup_token_steps_improved.png")
    print("  • speedup_comparison_bars.png")
    print("  • comparison_table_3way.png")
    print("  • comparison_table_3way.csv")
    print("  • cost_analysis.png")
    print("="*70)
