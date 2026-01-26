"""
Analyze correlation between EAB branching and human ambiguity ratings.
"""

import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, Any
from scipy.stats import spearmanr, pearsonr

# Set plotting style
sns.set_palette("pastel")
plt.rcParams.update({
    'font.size': 12,
    'axes.titlesize': 14,
    'axes.labelsize': 12,
    'legend.fontsize': 11,
    'figure.titlesize': 16,
})

SCRIPT_DIR = Path(__file__).parent
RESULTS_DIR = SCRIPT_DIR / "results"


def load_results() -> pd.DataFrame:
    """Load experimental results and convert to DataFrame."""
    results_file = RESULTS_DIR / "raw_results.json"

    with open(results_file, 'r') as f:
        data = json.load(f)

    results = data['results']

    # Convert to DataFrame
    df = pd.DataFrame([{
        'prompt_id': r['prompt_id'],
        'prompt': r['prompt'],
        'human_ambiguity_score': r['human_ambiguity_score'],
        'num_branches': r['num_branches'],
        'avg_branch_entropy': r['avg_branch_entropy'],
        'branching_frequency': r['branching_frequency'],
        'did_branch': r['did_branch'],
        'total_tokens': r['total_tokens']
    } for r in results])

    return df


def compute_correlations(df: pd.DataFrame) -> Dict[str, Any]:
    """Compute correlation metrics."""
    human_scores = df['human_ambiguity_score'].values

    correlations = {}

    # Correlation with number of branches
    rho_branches, p_branches = spearmanr(human_scores, df['num_branches'])
    r_branches, _ = pearsonr(human_scores, df['num_branches'])

    correlations['num_branches'] = {
        'spearman_rho': rho_branches,
        'spearman_p': p_branches,
        'pearson_r': r_branches
    }

    # Correlation with average branch entropy (filter out non-branching)
    branched_df = df[df['did_branch']]
    if len(branched_df) > 2:
        rho_entropy, p_entropy = spearmanr(
            branched_df['human_ambiguity_score'],
            branched_df['avg_branch_entropy']
        )
        r_entropy, _ = pearsonr(
            branched_df['human_ambiguity_score'],
            branched_df['avg_branch_entropy']
        )

        correlations['avg_branch_entropy'] = {
            'spearman_rho': rho_entropy,
            'spearman_p': p_entropy,
            'pearson_r': r_entropy,
            'n_samples': len(branched_df)
        }

    # Correlation with branching frequency
    rho_freq, p_freq = spearmanr(human_scores, df['branching_frequency'])
    r_freq, _ = pearsonr(human_scores, df['branching_frequency'])

    correlations['branching_frequency'] = {
        'spearman_rho': rho_freq,
        'spearman_p': p_freq,
        'pearson_r': r_freq
    }

    return correlations


def generate_latex_table(df: pd.DataFrame, correlations: Dict) -> str:
    """Generate LaTeX table showing prompts grouped by ambiguity."""
    # Group by ambiguity level
    df_sorted = df.sort_values('human_ambiguity_score')

    # Categorize
    low = df_sorted[df_sorted['human_ambiguity_score'] < 1.5]
    medium = df_sorted[(df_sorted['human_ambiguity_score'] >= 1.5) &
                       (df_sorted['human_ambiguity_score'] < 2.0)]
    high = df_sorted[df_sorted['human_ambiguity_score'] >= 2.0]

    latex = "\\begin{table}[h]\n"
    latex += "\\centering\n"
    latex += "\\caption{EAB branching behavior by human ambiguity level.}\n"
    latex += "\\label{tab:eab-human-correlation}\n"
    latex += "\\begin{tabular}{lccc}\n"
    latex += "\\toprule\n"
    latex += "\\textbf{Ambiguity} & \\textbf{n} & \\textbf{Avg Branches} & \\textbf{Branch Rate} \\\\\n"
    latex += "\\midrule\n"

    for category_df, label in [
        (low, "Low (< 1.5)"),
        (medium, "Medium (1.5-2.0)"),
        (high, "High (≥ 2.0)")
    ]:
        if len(category_df) > 0:
            avg_branches = category_df['num_branches'].mean()
            branch_rate = category_df['did_branch'].mean() * 100
            latex += f"{label:20s} & {len(category_df):2d} & {avg_branches:4.2f} & {branch_rate:4.1f}\\% \\\\\n"

    latex += "\\midrule\n"
    latex += f"Overall & {len(df):2d} & {df['num_branches'].mean():4.2f} & {df['did_branch'].mean()*100:4.1f}\\% \\\\\n"
    latex += "\\bottomrule\n"
    latex += "\\end{tabular}\n"
    latex += "\\end{table}\n\n"

    # Add correlation results
    latex += "\\textbf{Correlation with human ambiguity:}\n"
    latex += "\\begin{itemize}[noitemsep]\n"

    nb_corr = correlations['num_branches']
    latex += f"\\item Number of branches: $\\rho = {nb_corr['spearman_rho']:.3f}$ "
    latex += f"(p {'<' if nb_corr['spearman_p'] < 0.05 else '='} {nb_corr['spearman_p']:.3f})\n"

    bf_corr = correlations['branching_frequency']
    latex += f"\\item Branching frequency: $\\rho = {bf_corr['spearman_rho']:.3f}$ "
    latex += f"(p {'<' if bf_corr['spearman_p'] < 0.05 else '='} {bf_corr['spearman_p']:.3f})\n"

    latex += "\\end{itemize}\n"

    return latex


def plot_correlation_scatter(df: pd.DataFrame, save_path: Path):
    """Create scatter plot of human ambiguity vs EAB branches."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Plot 1: Number of branches
    ax1.scatter(df['human_ambiguity_score'], df['num_branches'],
                alpha=0.6, s=100, color=sns.color_palette()[0])

    # Fit trendline
    z = np.polyfit(df['human_ambiguity_score'], df['num_branches'], 1)
    p = np.poly1d(z)
    x_trend = np.linspace(df['human_ambiguity_score'].min(),
                          df['human_ambiguity_score'].max(), 100)
    ax1.plot(x_trend, p(x_trend), "r--", alpha=0.8, linewidth=2)

    # Compute correlation
    rho, p_val = spearmanr(df['human_ambiguity_score'], df['num_branches'])

    ax1.set_xlabel('Human Ambiguity Score (1-3)', fontsize=13)
    ax1.set_ylabel('Number of EAB Branches', fontsize=13)
    ax1.set_title(f'Branching vs Ambiguity (ρ={rho:.3f}, p={p_val:.3f})',
                  fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)

    # Plot 2: Branching frequency
    ax2.scatter(df['human_ambiguity_score'], df['branching_frequency'],
                alpha=0.6, s=100, color=sns.color_palette()[1])

    # Fit trendline
    z2 = np.polyfit(df['human_ambiguity_score'], df['branching_frequency'], 1)
    p2 = np.poly1d(z2)
    ax2.plot(x_trend, p2(x_trend), "r--", alpha=0.8, linewidth=2)

    # Compute correlation
    rho2, p_val2 = spearmanr(df['human_ambiguity_score'], df['branching_frequency'])

    ax2.set_xlabel('Human Ambiguity Score (1-3)', fontsize=13)
    ax2.set_ylabel('Branching Frequency (branches/token)', fontsize=13)
    ax2.set_title(f'Branch Frequency vs Ambiguity (ρ={rho2:.3f}, p={p_val2:.3f})',
                  fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"📊 Correlation plot saved to: {save_path}")
    plt.close()


def plot_ambiguity_groups(df: pd.DataFrame, save_path: Path):
    """Plot branching behavior grouped by ambiguity level."""
    # Categorize prompts
    df['ambiguity_category'] = pd.cut(
        df['human_ambiguity_score'],
        bins=[0, 1.5, 2.0, 3.0],
        labels=['Low\n(<1.5)', 'Medium\n(1.5-2.0)', 'High\n(≥2.0)']
    )

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Plot 1: Average branches by category
    category_means = df.groupby('ambiguity_category')['num_branches'].agg(['mean', 'std', 'count'])

    colors = sns.color_palette("pastel", 3)
    bars1 = ax1.bar(range(len(category_means)), category_means['mean'],
                    yerr=category_means['std'], capsize=5,
                    color=colors, edgecolor='gray', linewidth=1.5)

    # Add count labels
    for i, (idx, row) in enumerate(category_means.iterrows()):
        ax1.text(i, row['mean'] + row['std'] + 0.2,
                f"n={int(row['count'])}", ha='center', fontsize=10)

    ax1.set_xticks(range(len(category_means)))
    ax1.set_xticklabels(category_means.index)
    ax1.set_ylabel('Average Number of Branches', fontsize=13)
    ax1.set_xlabel('Human Ambiguity Level', fontsize=13)
    ax1.set_title('Branching by Ambiguity Category', fontsize=14, fontweight='bold')
    ax1.grid(axis='y', alpha=0.3)

    # Plot 2: Branching rate (% of prompts that branched)
    branch_rates = df.groupby('ambiguity_category')['did_branch'].mean() * 100

    bars2 = ax2.bar(range(len(branch_rates)), branch_rates,
                    color=colors, edgecolor='gray', linewidth=1.5)

    ax2.set_xticks(range(len(branch_rates)))
    ax2.set_xticklabels(branch_rates.index)
    ax2.set_ylabel('Branching Rate (%)', fontsize=13)
    ax2.set_xlabel('Human Ambiguity Level', fontsize=13)
    ax2.set_title('% of Prompts with Branching', fontsize=14, fontweight='bold')
    ax2.grid(axis='y', alpha=0.3)

    # Add percentage labels on bars
    for i, (idx, val) in enumerate(branch_rates.items()):
        ax2.text(i, val + 2, f"{val:.1f}%", ha='center', fontsize=10)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"📊 Category plot saved to: {save_path}")
    plt.close()


def plot_summary_figure(df: pd.DataFrame, correlations: Dict, save_path: Path):
    """Create a comprehensive summary figure showing all key results."""
    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.3)

    # 1. Main correlation scatter (top left)
    ax1 = fig.add_subplot(gs[0, 0])
    colors = plt.cm.RdYlGn_r(df['human_ambiguity_score'] / 3.0)  # Color by ambiguity
    scatter = ax1.scatter(df['human_ambiguity_score'], df['num_branches'],
                          c=df['human_ambiguity_score'], cmap='RdYlGn_r',
                          s=150, alpha=0.7, edgecolors='black', linewidth=1.5)

    # Trend line
    z = np.polyfit(df['human_ambiguity_score'], df['num_branches'], 1)
    p = np.poly1d(z)
    x_line = np.linspace(df['human_ambiguity_score'].min(), df['human_ambiguity_score'].max(), 100)
    ax1.plot(x_line, p(x_line), "r--", linewidth=3, alpha=0.8, label='Trend')

    rho, p_val = spearmanr(df['human_ambiguity_score'], df['num_branches'])
    ax1.set_xlabel('Human Ambiguity Score (1-3)', fontsize=13, fontweight='bold')
    ax1.set_ylabel('EAB Branches', fontsize=13, fontweight='bold')
    ax1.set_title(f'Correlation: ρ = {rho:.3f}, p = {p_val:.3f}',
                  fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.legend(fontsize=11)

    # Add colorbar
    cbar = plt.colorbar(scatter, ax=ax1)
    cbar.set_label('Ambiguity', fontsize=11)

    # 2. Category comparison (top right)
    ax2 = fig.add_subplot(gs[0, 1])
    df['category'] = pd.cut(df['human_ambiguity_score'],
                            bins=[0, 1.5, 2.0, 3.0],
                            labels=['Low\n(<1.5)', 'Med\n(1.5-2.0)', 'High\n(≥2.0)'])

    category_means = df.groupby('category')['num_branches'].agg(['mean', 'std', 'count'])
    colors_cat = ['#90EE90', '#FFD700', '#FF6B6B']  # Green, Yellow, Red

    bars = ax2.bar(range(len(category_means)), category_means['mean'],
                   yerr=category_means['std'], capsize=8,
                   color=colors_cat, edgecolor='black', linewidth=2, alpha=0.8)

    # Add sample counts
    for i, (idx, row) in enumerate(category_means.iterrows()):
        ax2.text(i, row['mean'] + row['std'] + 1,
                f"n={int(row['count'])}", ha='center', fontsize=12, fontweight='bold')

    ax2.set_xticks(range(len(category_means)))
    ax2.set_xticklabels(category_means.index, fontsize=12)
    ax2.set_ylabel('Average Branches', fontsize=13, fontweight='bold')
    ax2.set_title('Branching by Ambiguity Level', fontsize=14, fontweight='bold')
    ax2.grid(axis='y', alpha=0.3)

    # 3. Branching frequency heatmap (bottom left)
    ax3 = fig.add_subplot(gs[1, 0])

    # Create a matrix showing prompts vs branching
    sorted_df = df.sort_values('human_ambiguity_score')
    prompt_labels = [p[:30] + '...' if len(p) > 30 else p for p in sorted_df['prompt']]

    # Branching data
    branch_data = sorted_df[['num_branches', 'branching_frequency']].T

    im = ax3.imshow(branch_data, aspect='auto', cmap='YlOrRd', interpolation='nearest')
    ax3.set_yticks([0, 1])
    ax3.set_yticklabels(['# Branches', 'Branch Freq'], fontsize=11)
    ax3.set_xticks(range(len(sorted_df)))
    ax3.set_xticklabels([f"{score:.2f}" for score in sorted_df['human_ambiguity_score']],
                        rotation=45, ha='right', fontsize=9)
    ax3.set_xlabel('Prompts (sorted by ambiguity)', fontsize=12, fontweight='bold')
    ax3.set_title('Branching Patterns', fontsize=14, fontweight='bold')

    # Colorbar
    cbar2 = plt.colorbar(im, ax=ax3)
    cbar2.set_label('Value', fontsize=11)

    # 4. Statistical summary (bottom right)
    ax4 = fig.add_subplot(gs[1, 1])
    ax4.axis('off')

    # Create summary text
    summary_text = "📊 SUMMARY STATISTICS\n" + "="*40 + "\n\n"
    summary_text += f"Total prompts: {len(df)}\n\n"

    summary_text += "Correlation Results:\n"
    for metric, corr in correlations.items():
        metric_name = metric.replace('_', ' ').title()
        sig = "✓" if corr['spearman_p'] < 0.05 else "✗"
        summary_text += f"  {sig} {metric_name}:\n"
        summary_text += f"     ρ = {corr['spearman_rho']:.3f}\n"
        summary_text += f"     p = {corr['spearman_p']:.4f}\n\n"

    summary_text += "\nBy Ambiguity Level:\n"
    for cat, group in df.groupby('category'):
        summary_text += f"  {cat}: {len(group)} prompts\n"
        summary_text += f"    Avg branches: {group['num_branches'].mean():.2f}\n"
        summary_text += f"    Branch rate: {group['did_branch'].mean()*100:.1f}%\n\n"

    ax4.text(0.1, 0.95, summary_text, transform=ax4.transAxes,
             fontsize=11, verticalalignment='top', family='monospace',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))

    plt.suptitle('EAB Branching vs Human Ambiguity: Complete Analysis',
                 fontsize=18, fontweight='bold', y=0.98)

    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"📊 Summary figure saved to: {save_path}")
    plt.close()


def main():
    print("=" * 70)
    print("ANALYZING EAB vs HUMAN AMBIGUITY CORRELATION")
    print("=" * 70)

    # Load results
    df = load_results()
    print(f"\nLoaded {len(df)} prompt results")

    # Display data
    print("\n" + "=" * 70)
    print("PROMPT RESULTS")
    print("=" * 70)
    print(df[['prompt', 'human_ambiguity_score', 'num_branches', 'branching_frequency']].to_string(index=False))

    # Compute correlations
    print("\n" + "=" * 70)
    print("CORRELATION ANALYSIS")
    print("=" * 70)

    correlations = compute_correlations(df)

    for metric, corr_data in correlations.items():
        print(f"\n{metric}:")
        print(f"  Spearman ρ = {corr_data['spearman_rho']:.4f} (p = {corr_data['spearman_p']:.4f})")
        print(f"  Pearson r  = {corr_data['pearson_r']:.4f}")
        if 'n_samples' in corr_data:
            print(f"  (n = {corr_data['n_samples']} prompts with branching)")

        if corr_data['spearman_p'] < 0.05:
            print("  ✓ Statistically significant (p < 0.05)")
        else:
            print("  ✗ Not statistically significant")

    # Save correlations
    corr_file = RESULTS_DIR / "correlations.json"
    with open(corr_file, 'w') as f:
        json.dump(correlations, f, indent=2)
    print(f"\n✓ Correlations saved to: {corr_file}")

    # Generate LaTeX table
    latex_table = generate_latex_table(df, correlations)
    latex_path = RESULTS_DIR / "correlation_table.tex"
    with open(latex_path, 'w') as f:
        f.write(latex_table)
    print(f"✓ LaTeX table saved to: {latex_path}")

    # Print LaTeX
    print("\n" + "=" * 70)
    print("LATEX TABLE")
    print("=" * 70)
    print(latex_table)

    # Generate plots
    figures_dir = RESULTS_DIR / "figures"
    figures_dir.mkdir(exist_ok=True, parents=True)

    plot_correlation_scatter(df, figures_dir / "correlation_scatter.png")
    plot_ambiguity_groups(df, figures_dir / "ambiguity_groups.png")
    plot_summary_figure(df, correlations, figures_dir / "summary_analysis.png")

    # Summary statistics
    print("\n" + "=" * 70)
    print("SUMMARY STATISTICS")
    print("=" * 70)
    print(f"Total prompts: {len(df)}")
    print(f"Prompts with branching: {df['did_branch'].sum()} ({df['did_branch'].mean()*100:.1f}%)")
    print(f"Average branches per prompt: {df['num_branches'].mean():.2f} (±{df['num_branches'].std():.2f})")
    print(f"Average branching frequency: {df['branching_frequency'].mean():.4f}")

    # Group stats
    print("\nBy ambiguity level:")
    for ambig_min, ambig_max, label in [(0, 1.5, "Low"), (1.5, 2.0, "Medium"), (2.0, 3.1, "High")]:
        subset = df[(df['human_ambiguity_score'] >= ambig_min) &
                    (df['human_ambiguity_score'] < ambig_max)]
        if len(subset) > 0:
            print(f"  {label:8s}: {len(subset):2d} prompts, "
                  f"{subset['num_branches'].mean():.2f} avg branches, "
                  f"{subset['did_branch'].mean()*100:.1f}% branching rate")

    print("\n✅ Analysis complete!")


if __name__ == "__main__":
    main()
