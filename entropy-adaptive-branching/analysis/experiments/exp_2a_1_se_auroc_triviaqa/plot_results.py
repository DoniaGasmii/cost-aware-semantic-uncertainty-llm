"""
Plot results from Experiment 2.A.1: SE AUROC on TriviaQA

This script generates:
1. ROC curve for SE predicting incorrectness
2. Distribution of SE scores for correct vs incorrect
3. Scatter plot of SE vs RougeL score
4. Cluster count distribution
"""

import sys
import json
import yaml
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.metrics import roc_curve, auc

# Set style
plt.style.use('seaborn-v0_8-whitegrid')

experiment_dir = Path(__file__).parent


def load_config():
    """Load experiment configuration."""
    config_path = experiment_dir / "config.yaml"
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def load_results():
    """Load raw results from experiment."""
    results_dir = experiment_dir / "results"
    results_file = results_dir / "raw_results.json"
    with open(results_file, 'r') as f:
        return json.load(f)


def load_analysis():
    """Load analysis results."""
    results_dir = experiment_dir / "results"
    analysis_file = results_dir / "summary_stats.json"
    with open(analysis_file, 'r') as f:
        return json.load(f)


def plot_roc_curve(results, save_dir):
    """Plot ROC curve for SE predicting incorrectness."""
    fig, ax = plt.subplots(figsize=(8, 6))

    # Extract data
    se_uncertainty = np.array([r['se_uncertainty_score'] for r in results])
    any_incorrect = np.array([1 - r['any_correct'] for r in results])

    # Compute ROC curve
    fpr, tpr, thresholds = roc_curve(any_incorrect, se_uncertainty)
    roc_auc = auc(fpr, tpr)

    # Plot
    ax.plot(fpr, tpr, 'b-', linewidth=2, label=f'SE Uncertainty (AUROC = {roc_auc:.3f})')
    ax.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random (AUROC = 0.500)')

    ax.set_xlabel('False Positive Rate', fontsize=12)
    ax.set_ylabel('True Positive Rate', fontsize=12)
    ax.set_title('ROC Curve: Semantic Entropy Predicting Incorrect Answers', fontsize=14)
    ax.legend(loc='lower right', fontsize=11)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])

    # Add grid
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_dir / 'roc_curve.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"   Saved: roc_curve.png")


def plot_distribution_comparison(results, save_dir):
    """Plot SE distribution for correct vs incorrect answers."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Separate correct and incorrect
    correct = [r for r in results if r['any_correct']]
    incorrect = [r for r in results if not r['any_correct']]

    metrics = [
        ('se_uncertainty_score', 'SE Uncertainty Score'),
        ('se_n_clusters', 'Number of Semantic Clusters'),
    ]

    for ax, (metric, label) in zip(axes, metrics):
        correct_vals = [r[metric] for r in correct]
        incorrect_vals = [r[metric] for r in incorrect]

        # Plot histograms
        bins = 20 if metric == 'se_uncertainty_score' else range(1, max(incorrect_vals + correct_vals) + 2)

        ax.hist(correct_vals, bins=bins, alpha=0.6, label=f'Correct (n={len(correct)})',
                color='green', density=True)
        ax.hist(incorrect_vals, bins=bins, alpha=0.6, label=f'Incorrect (n={len(incorrect)})',
                color='red', density=True)

        # Add mean lines
        ax.axvline(np.mean(correct_vals), color='green', linestyle='--', linewidth=2,
                   label=f'Correct mean: {np.mean(correct_vals):.2f}')
        ax.axvline(np.mean(incorrect_vals), color='red', linestyle='--', linewidth=2,
                   label=f'Incorrect mean: {np.mean(incorrect_vals):.2f}')

        ax.set_xlabel(label, fontsize=12)
        ax.set_ylabel('Density', fontsize=12)
        ax.legend(fontsize=10)
        ax.set_title(f'{label} by Correctness', fontsize=12)

    plt.tight_layout()
    plt.savefig(save_dir / 'distribution_comparison.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"   Saved: distribution_comparison.png")


def plot_se_vs_rouge(results, save_dir):
    """Scatter plot of SE uncertainty vs RougeL score."""
    fig, ax = plt.subplots(figsize=(10, 6))

    se_uncertainty = [r['se_uncertainty_score'] for r in results]
    rouge_scores = [r['best_rouge_l'] for r in results]
    correct = [r['any_correct'] for r in results]

    # Color by correctness
    colors = ['green' if c else 'red' for c in correct]

    scatter = ax.scatter(rouge_scores, se_uncertainty, c=colors, alpha=0.6, s=50)

    # Add threshold line
    ax.axvline(0.3, color='gray', linestyle='--', linewidth=1, label='Correctness threshold (RougeL=0.3)')

    ax.set_xlabel('Best RougeL Score', fontsize=12)
    ax.set_ylabel('SE Uncertainty Score', fontsize=12)
    ax.set_title('Semantic Uncertainty vs Answer Quality', fontsize=14)

    # Custom legend
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor='green', markersize=10, label='Correct'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='red', markersize=10, label='Incorrect'),
        Line2D([0], [0], color='gray', linestyle='--', label='Threshold'),
    ]
    ax.legend(handles=legend_elements, loc='upper right', fontsize=11)

    plt.tight_layout()
    plt.savefig(save_dir / 'se_vs_rouge.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"   Saved: se_vs_rouge.png")


def plot_cluster_analysis(results, save_dir):
    """Plot cluster count distribution and its relation to correctness."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    n_clusters = [r['se_n_clusters'] for r in results]
    correct = [r['any_correct'] for r in results]

    # Plot 1: Cluster count distribution
    ax1 = axes[0]
    cluster_counts = np.bincount(n_clusters)
    ax1.bar(range(len(cluster_counts)), cluster_counts, color='steelblue', alpha=0.7)
    ax1.set_xlabel('Number of Clusters', fontsize=12)
    ax1.set_ylabel('Count', fontsize=12)
    ax1.set_title('Distribution of Semantic Clusters', fontsize=12)

    # Plot 2: Accuracy by cluster count
    ax2 = axes[1]
    unique_clusters = sorted(set(n_clusters))
    accuracies = []
    counts = []

    for nc in unique_clusters:
        mask = [n == nc for n in n_clusters]
        correct_at_nc = [c for c, m in zip(correct, mask) if m]
        if correct_at_nc:
            accuracies.append(np.mean(correct_at_nc))
            counts.append(sum(mask))
        else:
            accuracies.append(0)
            counts.append(0)

    bars = ax2.bar(unique_clusters, accuracies, color='steelblue', alpha=0.7)

    # Add count labels
    for bar, count in zip(bars, counts):
        if count > 0:
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                    f'n={count}', ha='center', fontsize=9)

    ax2.set_xlabel('Number of Clusters', fontsize=12)
    ax2.set_ylabel('Accuracy', fontsize=12)
    ax2.set_title('Accuracy by Number of Semantic Clusters', fontsize=12)
    ax2.set_ylim([0, 1.1])
    ax2.axhline(np.mean(correct), color='red', linestyle='--', label=f'Overall accuracy: {np.mean(correct):.1%}')
    ax2.legend()

    plt.tight_layout()
    plt.savefig(save_dir / 'cluster_analysis.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"   Saved: cluster_analysis.png")


def plot_summary_figure(results, analysis, save_dir):
    """Create a summary figure with key results."""
    fig = plt.figure(figsize=(16, 10))

    # Layout: 2x2 grid
    gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.3)

    # Top-left: ROC curve
    ax1 = fig.add_subplot(gs[0, 0])
    se_uncertainty = np.array([r['se_uncertainty_score'] for r in results])
    any_incorrect = np.array([1 - r['any_correct'] for r in results])
    fpr, tpr, _ = roc_curve(any_incorrect, se_uncertainty)
    roc_auc = auc(fpr, tpr)

    ax1.plot(fpr, tpr, 'b-', linewidth=2, label=f'AUROC = {roc_auc:.3f}')
    ax1.plot([0, 1], [0, 1], 'k--', linewidth=1, alpha=0.5)
    ax1.set_xlabel('False Positive Rate')
    ax1.set_ylabel('True Positive Rate')
    ax1.set_title('(a) ROC Curve: SE → Incorrectness')
    ax1.legend(loc='lower right')
    ax1.set_xlim([0, 1])
    ax1.set_ylim([0, 1])

    # Top-right: SE distribution by correctness
    ax2 = fig.add_subplot(gs[0, 1])
    correct = [r['se_uncertainty_score'] for r in results if r['any_correct']]
    incorrect = [r['se_uncertainty_score'] for r in results if not r['any_correct']]

    ax2.hist(correct, bins=20, alpha=0.6, label=f'Correct (n={len(correct)})', color='green', density=True)
    ax2.hist(incorrect, bins=20, alpha=0.6, label=f'Incorrect (n={len(incorrect)})', color='red', density=True)
    ax2.axvline(np.mean(correct), color='green', linestyle='--', linewidth=2)
    ax2.axvline(np.mean(incorrect), color='red', linestyle='--', linewidth=2)
    ax2.set_xlabel('SE Uncertainty Score')
    ax2.set_ylabel('Density')
    ax2.set_title('(b) SE Distribution by Correctness')
    ax2.legend()

    # Bottom-left: Accuracy by cluster count
    ax3 = fig.add_subplot(gs[1, 0])
    n_clusters = [r['se_n_clusters'] for r in results]
    correct_list = [r['any_correct'] for r in results]
    unique_clusters = sorted(set(n_clusters))

    accuracies = []
    counts = []
    for nc in unique_clusters:
        mask = [n == nc for n in n_clusters]
        correct_at_nc = [c for c, m in zip(correct_list, mask) if m]
        accuracies.append(np.mean(correct_at_nc) if correct_at_nc else 0)
        counts.append(sum(mask))

    bars = ax3.bar(unique_clusters, accuracies, color='steelblue', alpha=0.7)
    for bar, count in zip(bars, counts):
        if count > 2:
            ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02, f'n={count}', ha='center', fontsize=8)

    ax3.axhline(np.mean(correct_list), color='red', linestyle='--', label=f'Overall: {np.mean(correct_list):.1%}')
    ax3.set_xlabel('Number of Clusters')
    ax3.set_ylabel('Accuracy')
    ax3.set_title('(c) Accuracy vs Cluster Count')
    ax3.legend()
    ax3.set_ylim([0, 1.1])

    # Bottom-right: Key statistics text box
    ax4 = fig.add_subplot(gs[1, 1])
    ax4.axis('off')

    stats_text = f"""
    KEY RESULTS
    {'='*40}

    Dataset:
      • Questions: {len(results)}
      • Samples per question: {results[0]['num_samples']}

    Accuracy:
      • Any correct: {np.mean([r['any_correct'] for r in results]):.1%}
      • Majority correct: {np.mean([r['majority_correct'] for r in results]):.1%}

    Semantic Entropy:
      • Avg clusters: {np.mean(n_clusters):.1f}
      • Avg SE uncertainty: {np.mean(se_uncertainty):.3f}

    Uncertainty Estimation:
      • AUROC: {roc_auc:.3f}
      • SE higher for incorrect: {np.mean(incorrect) > np.mean(correct)}

    Interpretation:
      {'✓ SE reliably predicts errors' if roc_auc > 0.6 else '○ SE moderately predictive' if roc_auc > 0.5 else '✗ SE not predictive'}
    """

    ax4.text(0.1, 0.9, stats_text, transform=ax4.transAxes, fontsize=11,
             verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    ax4.set_title('(d) Summary Statistics')

    plt.suptitle('Semantic Entropy Evaluation on TriviaQA', fontsize=16, fontweight='bold', y=1.02)
    plt.savefig(save_dir / 'summary_figure.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"   Saved: summary_figure.png")


def main():
    """Main plotting function."""
    print("=" * 70)
    print("PLOTTING EXPERIMENT 2.A.1 RESULTS")
    print("=" * 70)

    # Load data
    print("\n1. Loading data...")
    data = load_results()
    results = data['results']
    print(f"   Loaded {len(results)} results")

    # Try to load analysis
    try:
        analysis = load_analysis()
    except FileNotFoundError:
        print("   Warning: summary_stats.json not found, running with limited analysis")
        analysis = None

    # Create figures directory
    save_dir = experiment_dir / "results" / "figures"
    save_dir.mkdir(exist_ok=True, parents=True)

    # Generate plots
    print("\n2. Generating plots...")

    plot_roc_curve(results, save_dir)
    plot_distribution_comparison(results, save_dir)
    plot_se_vs_rouge(results, save_dir)
    plot_cluster_analysis(results, save_dir)
    plot_summary_figure(results, analysis, save_dir)

    print("\n" + "=" * 70)
    print("PLOTTING COMPLETE")
    print(f"All figures saved to: {save_dir}")
    print("=" * 70)


if __name__ == "__main__":
    main()
