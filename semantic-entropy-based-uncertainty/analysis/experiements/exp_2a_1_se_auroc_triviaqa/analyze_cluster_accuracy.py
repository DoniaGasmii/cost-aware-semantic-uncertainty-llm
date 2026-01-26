"""
Analyze the relationship between cluster count and accuracy (RQ2).
"""

import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from collections import defaultdict

# Set plotting style
sns.set_palette("pastel")
plt.rcParams.update({
    'font.size': 12,
    'axes.titlesize': 14,
    'axes.labelsize': 12,
    'legend.fontsize': 11,
})

SCRIPT_DIR = Path(__file__).parent


def load_results(threshold: float = 0.05):
    """Load results for the best threshold."""
    thresh_str = f"{threshold:.2f}".replace(".", "_")
    results_dir = SCRIPT_DIR / "results" / f"threshold_{thresh_str}"
    raw_file = results_dir / "raw_results.json"

    with open(raw_file, 'r') as f:
        data = json.load(f)

    return data['results'], results_dir


def analyze_cluster_accuracy(results):
    """Analyze accuracy stratified by cluster count."""
    # Group by cluster count
    cluster_groups = defaultdict(list)

    for r in results:
        n_clusters = r['se_n_clusters']
        is_correct = r['best_sample_correct']
        cluster_groups[n_clusters].append(is_correct)

    # Calculate statistics per cluster count
    cluster_stats = []
    for n_clusters in sorted(cluster_groups.keys()):
        correctness = cluster_groups[n_clusters]
        cluster_stats.append({
            'n_clusters': n_clusters,
            'count': len(correctness),
            'accuracy': np.mean(correctness),
            'num_correct': sum(correctness)
        })

    return pd.DataFrame(cluster_stats)


def generate_latex_table(df: pd.DataFrame, overall_accuracy: float) -> str:
    """Generate LaTeX table for cluster accuracy."""
    # Aggregate into bins
    bin1 = df[df['n_clusters'] == 1]
    bin2_3 = df[df['n_clusters'].isin([2, 3])]
    bin4plus = df[df['n_clusters'] >= 4]

    def aggregate_bin(bin_df):
        if bin_df.empty:
            return {'count': 0, 'accuracy': 0.0}
        total_count = bin_df['count'].sum()
        total_correct = bin_df['num_correct'].sum()
        accuracy = total_correct / total_count if total_count > 0 else 0
        return {'count': total_count, 'accuracy': accuracy}

    stats1 = aggregate_bin(bin1)
    stats2_3 = aggregate_bin(bin2_3)
    stats4plus = aggregate_bin(bin4plus)

    latex = "\\begin{table}[h]\n"
    latex += "\\centering\n"
    latex += "\\caption{Accuracy by number of semantic clusters.}\n"
    latex += "\\label{tab:cluster-accuracy}\n"
    latex += "\\begin{tabular}{cccc}\n"
    latex += "\\toprule\n"
    latex += "\\textbf{Clusters} & \\textbf{n} & \\textbf{Accuracy} & \\textbf{Interpretation} \\\\\n"
    latex += "\\midrule\n"
    latex += f"1 (full agreement) & {stats1['count']:3d} & {stats1['accuracy']*100:4.1f}\\% & High confidence \\\\\n"
    latex += f"2-3                & {stats2_3['count']:3d} & {stats2_3['accuracy']*100:4.1f}\\% & Moderate uncertainty \\\\\n"
    latex += f"4+                 & {stats4plus['count']:3d} & {stats4plus['accuracy']*100:4.1f}\\% & High uncertainty \\\\\n"
    latex += "\\midrule\n"
    latex += f"Overall            & {stats1['count'] + stats2_3['count'] + stats4plus['count']:3d} & {overall_accuracy*100:4.1f}\\% & --- \\\\\n"
    latex += "\\bottomrule\n"
    latex += "\\end{tabular}\n"
    latex += "\\end{table}\n\n"

    # Add observation
    gap = (stats1['accuracy'] - stats4plus['accuracy']) * 100
    latex += f"\\textbf{{Observation.}} Questions with 1 cluster achieve {stats1['accuracy']*100:.1f}\\% accuracy "
    latex += f"vs {stats4plus['accuracy']*100:.1f}\\% for 4+ clusters—a {gap:.1f} percentage point gap.\n\n"

    latex += "\\textbf{Interpretation.} Cluster count provides an intuitive uncertainty signal: "
    latex += "unanimous agreement correlates with correctness, while disagreement signals potential errors.\n"

    return latex, (stats1, stats2_3, stats4plus)


def plot_cluster_accuracy(df: pd.DataFrame, overall_accuracy: float, save_path: Path):
    """Plot accuracy vs cluster count."""
    fig, ax = plt.subplots(figsize=(10, 6))

    # Plot accuracy by cluster count
    colors = sns.color_palette("pastel", len(df))
    bars = ax.bar(df['n_clusters'], df['accuracy'] * 100,
                   color=colors, edgecolor='gray', linewidth=1.5)

    # Add count labels on bars
    for i, (idx, row) in enumerate(df.iterrows()):
        ax.text(row['n_clusters'], row['accuracy'] * 100 + 1.5,
                f"n={row['count']}", ha='center', va='bottom', fontsize=10)

    # Add overall accuracy line
    ax.axhline(overall_accuracy * 100, color='red', linestyle='--',
               linewidth=2, alpha=0.7, label=f'Overall accuracy ({overall_accuracy*100:.1f}%)')

    ax.set_xlabel('Number of Semantic Clusters', fontsize=13)
    ax.set_ylabel('Accuracy (%)', fontsize=13)
    ax.set_title('Accuracy by Number of Semantic Clusters', fontsize=15, fontweight='bold')
    ax.set_xticks(df['n_clusters'])
    ax.grid(axis='y', alpha=0.3)
    ax.legend(loc='upper right', fontsize=11)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"📊 Plot saved to: {save_path}")
    plt.close()


def plot_cluster_distribution(results, save_path: Path):
    """Plot distribution of cluster counts."""
    cluster_counts = [r['se_n_clusters'] for r in results]

    fig, ax = plt.subplots(figsize=(8, 5))

    unique_clusters = sorted(set(cluster_counts))
    counts = [cluster_counts.count(c) for c in unique_clusters]

    colors = sns.color_palette("pastel", len(unique_clusters))
    ax.bar(unique_clusters, counts, color=colors, edgecolor='gray', linewidth=1.5)

    ax.set_xlabel('Number of Clusters', fontsize=13)
    ax.set_ylabel('Frequency', fontsize=13)
    ax.set_title('Distribution of Cluster Counts', fontsize=15, fontweight='bold')
    ax.set_xticks(unique_clusters)
    ax.grid(axis='y', alpha=0.3)

    # Add percentage labels
    total = len(cluster_counts)
    for i, (c, cnt) in enumerate(zip(unique_clusters, counts)):
        pct = (cnt / total) * 100
        ax.text(c, cnt + max(counts)*0.02, f'{pct:.1f}%',
                ha='center', va='bottom', fontsize=10)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"📊 Cluster distribution plot saved to: {save_path}")
    plt.close()


def main():
    print("=" * 70)
    print("CLUSTER-ACCURACY ANALYSIS (RQ2)")
    print("=" * 70)

    # Load results from best threshold
    results, results_dir = load_results(threshold=0.05)
    print(f"\nLoaded {len(results)} results from threshold δ=0.05")

    # Analyze cluster-accuracy relationship
    df = analyze_cluster_accuracy(results)
    overall_accuracy = np.mean([r['best_sample_correct'] for r in results])

    print("\n" + "=" * 70)
    print("ACCURACY BY CLUSTER COUNT")
    print("=" * 70)
    print(df.to_string(index=False, float_format=lambda x: f'{x:.3f}'))
    print(f"\nOverall accuracy: {overall_accuracy:.1%}")

    # Generate LaTeX table
    latex_table, (stats1, stats2_3, stats4plus) = generate_latex_table(df, overall_accuracy)
    latex_path = results_dir / "cluster_accuracy_table.tex"
    with open(latex_path, 'w') as f:
        f.write(latex_table)
    print(f"\n✓ LaTeX table saved to: {latex_path}")

    # Print LaTeX
    print("\n" + "=" * 70)
    print("LATEX TABLE")
    print("=" * 70)
    print(latex_table)

    # Generate plots
    figures_dir = results_dir / "figures"
    figures_dir.mkdir(exist_ok=True, parents=True)

    plot_path = figures_dir / "cluster_accuracy.png"
    plot_cluster_accuracy(df, overall_accuracy, plot_path)

    dist_plot_path = figures_dir / "cluster_distribution.png"
    plot_cluster_distribution(results, dist_plot_path)

    # Print key findings
    print("\n" + "=" * 70)
    print("KEY FINDINGS")
    print("=" * 70)
    print(f"✓ 1 cluster (unanimous):  {stats1['count']:3d} questions, {stats1['accuracy']*100:4.1f}% accuracy")
    print(f"✓ 2-3 clusters (mixed):   {stats2_3['count']:3d} questions, {stats2_3['accuracy']*100:4.1f}% accuracy")
    print(f"✓ 4+ clusters (diverse):  {stats4plus['count']:3d} questions, {stats4plus['accuracy']*100:4.1f}% accuracy")
    gap = (stats1['accuracy'] - stats4plus['accuracy']) * 100
    print(f"\n🎯 Accuracy gap (1 cluster vs 4+): {gap:+.1f} percentage points")

    if gap > 0:
        print("   → Cluster count inversely correlates with accuracy (as expected)")
    else:
        print("   → Unexpected: more clusters associated with higher/equal accuracy")

    print("\n✅ Analysis complete!")


if __name__ == "__main__":
    main()
