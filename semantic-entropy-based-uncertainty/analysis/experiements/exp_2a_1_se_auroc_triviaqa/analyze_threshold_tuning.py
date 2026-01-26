"""
Analyze threshold tuning results and generate LaTeX table + plots.
"""

import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

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


def load_threshold_data(threshold: float) -> dict:
    """Load data for a single threshold."""
    thresh_str = f"{threshold:.2f}".replace(".", "_")
    thresh_dir = RESULTS_DIR / f"threshold_{thresh_str}"

    if not thresh_dir.exists():
        return None

    # Load summary stats
    summary_file = thresh_dir / "summary_stats.json"
    if not summary_file.exists():
        return None

    with open(summary_file, 'r') as f:
        summary = json.load(f)

    # Load raw results to get cluster counts
    raw_file = thresh_dir / "raw_results.json"
    with open(raw_file, 'r') as f:
        raw_data = json.load(f)

    # Calculate average clusters
    cluster_counts = [r['se_n_clusters'] for r in raw_data['results']]
    avg_clusters = np.mean(cluster_counts)

    # Extract AUROC
    auroc = summary['auroc_metrics']['se_uncertainty_score']['best_incorrect']['auroc']
    accuracy = summary['accuracy']
    avg_se = summary['avg_se']

    return {
        'threshold': threshold,
        'avg_clusters': avg_clusters,
        'auroc': auroc,
        'accuracy': accuracy,
        'avg_se': avg_se,
        'num_questions': len(raw_data['results'])
    }


def aggregate_all_thresholds(thresholds):
    """Aggregate data from all thresholds."""
    data = []
    for thresh in thresholds:
        result = load_threshold_data(thresh)
        if result:
            data.append(result)
        else:
            print(f"⚠️  No data for threshold {thresh:.2f}")

    return pd.DataFrame(data)


def generate_latex_table(df: pd.DataFrame) -> str:
    """Generate LaTeX table from results."""
    # Find best AUROC
    best_idx = df['auroc'].idxmax()

    latex = "\\begin{table}[h]\n"
    latex += "\\centering\n"
    latex += "\\caption{AUROC sensitivity to clustering distance threshold $\\delta$.}\n"
    latex += "\\label{tab:threshold-sensitivity}\n"
    latex += "\\begin{tabular}{ccc}\n"
    latex += "\\toprule\n"
    latex += "\\textbf{Threshold $\\delta$} & \\textbf{Avg Clusters} & \\textbf{AUROC} \\\\\n"
    latex += "\\midrule\n"

    for idx, row in df.iterrows():
        thresh = row['threshold']

        # Add label for special thresholds
        if thresh == 0.05:
            label = " (strict)"
        elif thresh == df.loc[best_idx, 'threshold']:
            label = " (default)"
        elif thresh >= 0.25:
            label = " (loose)"
        else:
            label = ""

        # Bold AUROC if best
        auroc_str = f"{row['auroc']:.3f}"
        if idx == best_idx:
            auroc_str = f"\\textbf{{{auroc_str}}}"

        latex += f"{thresh:.2f}{label:11s} & {row['avg_clusters']:.1f} & {auroc_str} \\\\\n"

    latex += "\\bottomrule\n"
    latex += "\\end{tabular}\n"
    latex += "\\end{table}\n\n"

    # Add observation
    best_row = df.loc[best_idx]
    min_auroc = df['auroc'].min()
    max_auroc = df['auroc'].max()

    latex += "\\textbf{Observation.} "
    latex += f"AUROC varies from {min_auroc:.3f} to {max_auroc:.3f} across thresholds. "
    latex += f"The {'default ' if best_row['threshold'] == 0.15 else ''}$\\delta={best_row['threshold']:.2f}$ "
    latex += f"achieves {'near-' if max_auroc - best_row['auroc'] < 0.01 else ''}optimal performance.\n"

    return latex


def plot_threshold_sensitivity(df: pd.DataFrame, save_path: Path):
    """Plot AUROC and cluster count vs threshold."""
    fig, ax1 = plt.subplots(figsize=(10, 6))

    # Plot AUROC on primary y-axis
    color1 = sns.color_palette()[0]
    ax1.set_xlabel('Clustering Distance Threshold ($\\delta$)', fontsize=13)
    ax1.set_ylabel('AUROC', color=color1, fontsize=13)
    line1 = ax1.plot(df['threshold'], df['auroc'], 'o-', color=color1,
                     linewidth=2.5, markersize=8, label='AUROC')
    ax1.tick_params(axis='y', labelcolor=color1)
    ax1.grid(True, alpha=0.3)

    # Mark best threshold
    best_idx = df['auroc'].idxmax()
    ax1.axvline(df.loc[best_idx, 'threshold'], color='red', linestyle='--',
                alpha=0.5, linewidth=2, label=f'Best: $\\delta$={df.loc[best_idx, "threshold"]:.2f}')

    # Plot average clusters on secondary y-axis
    ax2 = ax1.twinx()
    color2 = sns.color_palette()[3]
    ax2.set_ylabel('Average Number of Clusters', color=color2, fontsize=13)
    line2 = ax2.plot(df['threshold'], df['avg_clusters'], 's--', color=color2,
                     linewidth=2, markersize=8, alpha=0.7, label='Avg Clusters')
    ax2.tick_params(axis='y', labelcolor=color2)

    # Combine legends
    lines = line1 + line2 + [plt.Line2D([0], [0], color='red', linestyle='--', linewidth=2)]
    labels = ['AUROC', 'Avg Clusters', f'Best: $\\delta$={df.loc[best_idx, "threshold"]:.2f}']
    ax1.legend(lines, labels, loc='upper left', fontsize=11)

    plt.title('Threshold Sensitivity Analysis', fontsize=15, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"📊 Plot saved to: {save_path}")
    plt.close()


def main():
    print("=" * 70)
    print("THRESHOLD TUNING ANALYSIS")
    print("=" * 70)

    # Define thresholds that were tested
    thresholds = [0.05, 0.10, 0.15, 0.20, 0.25]

    # Aggregate results
    print("\nAggregating results from all thresholds...")
    df = aggregate_all_thresholds(thresholds)

    if df.empty:
        print("❌ No threshold data found!")
        return

    # Save CSV summary
    csv_path = RESULTS_DIR / "threshold_tuning_summary.csv"
    df.to_csv(csv_path, index=False, float_format='%.4f')
    print(f"✓ CSV summary saved to: {csv_path}")

    # Display summary table
    print("\n" + "=" * 70)
    print("THRESHOLD SENSITIVITY SUMMARY")
    print("=" * 70)
    print(df.to_string(index=False, float_format=lambda x: f'{x:.3f}'))

    # Find best threshold
    best_idx = df['auroc'].idxmax()
    best_row = df.loc[best_idx]
    print(f"\n🏆 Best threshold: δ = {best_row['threshold']:.2f}")
    print(f"   AUROC: {best_row['auroc']:.3f}")
    print(f"   Avg Clusters: {best_row['avg_clusters']:.1f}")
    print(f"   Accuracy: {best_row['accuracy']:.1%}")

    # Generate LaTeX table
    latex_table = generate_latex_table(df)
    latex_path = RESULTS_DIR / "threshold_table.tex"
    with open(latex_path, 'w') as f:
        f.write(latex_table)
    print(f"\n✓ LaTeX table saved to: {latex_path}")

    # Print LaTeX to console
    print("\n" + "=" * 70)
    print("LATEX TABLE")
    print("=" * 70)
    print(latex_table)

    # Generate plot
    plot_path = RESULTS_DIR / "threshold_sensitivity.png"
    plot_threshold_sensitivity(df, plot_path)

    print("\n✅ Analysis complete!")
    print(f"   Best threshold for your report: δ = {best_row['threshold']:.2f}")
    thresh_str = f"{best_row['threshold']:.2f}".replace('.', '_')
    print(f"   Results directory: {RESULTS_DIR / f'threshold_{thresh_str}'}")


if __name__ == "__main__":
    main()
