"""
Threshold Tuning for EAB-SE Results

This script loads pre-generated EAB samples and recomputes semantic entropy
with different clustering thresholds to find the optimal threshold that
maximizes AUROC for predicting answer correctness.

Since samples are already generated, this is very fast!
"""

import json
import sys
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from tqdm import tqdm
from sklearn.metrics import roc_auc_score, roc_curve
from typing import Dict, List, Any

# Add parent directories to path
experiment_dir = Path(__file__).parent
analysis_dir = experiment_dir.parent.parent
project_root = analysis_dir.parent.parent

sys.path.insert(0, str(project_root / "semantic-entropy-based-uncertainty"))
from semantic_entropy.estimator import SemanticUncertaintyEstimator


def load_results(results_path: Path) -> Dict[str, Any]:
    """Load pre-generated EAB results."""
    print(f"Loading results from: {results_path}")
    with open(results_path, 'r') as f:
        data = json.load(f)
    print(f"Loaded {len(data['results'])} questions")
    return data


def compute_auroc_for_threshold(
    results: List[Dict],
    threshold: float,
    encoder_model: str,
    linkage: str,
    device: str,
    correctness_key: str = 'best_sample_correct'
) -> Dict[str, float]:
    """
    Recompute SE with given threshold and compute AUROC.

    Args:
        results: List of question results (with generated_samples)
        threshold: Distance threshold for clustering
        encoder_model: Encoder model name
        linkage: Linkage method for clustering
        device: Device to run on
        correctness_key: Which correctness metric to use

    Returns:
        Dict with AUROC, TPR@0.5, FPR@0.5, etc.
    """
    # Create fresh SE estimator with this threshold
    se_estimator = SemanticUncertaintyEstimator(
        encoder_model=encoder_model,
        distance_threshold=threshold,
        linkage=linkage,
        device=device
    )

    se_scores = []
    correctness = []

    for result in results:
        samples = result['generated_samples']

        # Skip if no samples
        if not samples or len(samples) < 2:
            continue

        # Compute SE with this threshold
        se_result = se_estimator.compute(samples, return_details=False)

        se_scores.append(se_result['uncertainty_score'])
        correctness.append(1 if result[correctness_key] else 0)

    # Convert to numpy
    se_scores = np.array(se_scores)
    correctness = np.array(correctness)

    # Compute AUROC (SE predicting INcorrectness, so we use 1-correctness)
    incorrectness = 1 - correctness

    try:
        auroc = roc_auc_score(incorrectness, se_scores)
        fpr, tpr, thresholds_roc = roc_curve(incorrectness, se_scores)

        # Find TPR/FPR at SE threshold of 0.5
        idx = np.argmin(np.abs(thresholds_roc - 0.5))
        tpr_at_05 = tpr[idx]
        fpr_at_05 = fpr[idx]

    except ValueError:
        # Handle case where all labels are the same
        auroc = 0.5
        tpr_at_05 = 0.0
        fpr_at_05 = 0.0

    return {
        'auroc': auroc,
        'tpr_at_05': tpr_at_05,
        'fpr_at_05': fpr_at_05,
        'mean_se': np.mean(se_scores),
        'std_se': np.std(se_scores),
        'accuracy': np.mean(correctness),
        'n_samples': len(se_scores)
    }


def tune_threshold(
    results: List[Dict],
    thresholds: np.ndarray,
    encoder_model: str = "all-mpnet-base-v2",
    linkage: str = "average",
    device: str = "cpu"
) -> Dict[str, Any]:
    """
    Try different thresholds and compute AUROC for each.

    Returns dict with thresholds and corresponding metrics.
    """
    print(f"\nTuning threshold over {len(thresholds)} values...")
    print(f"Encoder: {encoder_model}")
    print(f"Linkage: {linkage}")
    print(f"Device: {device}")

    # Store results
    auroc_scores = []
    tpr_scores = []
    fpr_scores = []
    mean_se_scores = []

    # Try each threshold (create fresh estimator each time)
    for threshold in tqdm(thresholds, desc="Testing thresholds"):
        print(f"\n  Testing threshold: {threshold}")

        metrics = compute_auroc_for_threshold(
            results=results,
            threshold=threshold,
            encoder_model=encoder_model,
            linkage=linkage,
            device=device,
            correctness_key='best_sample_correct'
        )

        auroc_scores.append(metrics['auroc'])
        tpr_scores.append(metrics['tpr_at_05'])
        fpr_scores.append(metrics['fpr_at_05'])
        mean_se_scores.append(metrics['mean_se'])

        print(f"    AUROC: {metrics['auroc']:.4f}, Mean SE: {metrics['mean_se']:.4f}")

    # Find optimal threshold
    best_idx = np.argmax(auroc_scores)
    optimal_threshold = thresholds[best_idx]
    best_auroc = auroc_scores[best_idx]

    return {
        'thresholds': thresholds.tolist(),
        'auroc_scores': auroc_scores,
        'tpr_scores': tpr_scores,
        'fpr_scores': fpr_scores,
        'mean_se_scores': mean_se_scores,
        'optimal_threshold': optimal_threshold,
        'best_auroc': best_auroc,
        'best_idx': best_idx
    }


def plot_threshold_tuning_results(
    tuning_results: Dict[str, Any],
    save_path: Path
):
    """Create visualization of threshold tuning results."""

    # Set plotting style
    sns.set_palette("pastel")
    plt.rcParams.update({
        'font.size': 12,
        'axes.titlesize': 14,
        'axes.labelsize': 12,
        'legend.fontsize': 11,
        'figure.titlesize': 16,
    })

    thresholds = np.array(tuning_results['thresholds'])
    auroc_scores = np.array(tuning_results['auroc_scores'])
    tpr_scores = np.array(tuning_results['tpr_scores'])
    fpr_scores = np.array(tuning_results['fpr_scores'])
    mean_se_scores = np.array(tuning_results['mean_se_scores'])

    optimal_threshold = tuning_results['optimal_threshold']
    best_auroc = tuning_results['best_auroc']

    # Create figure with 3 subplots
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Plot 1: AUROC vs Threshold
    ax1 = axes[0]
    ax1.plot(thresholds, auroc_scores, linewidth=2.5, marker='o', markersize=5, color='steelblue')
    ax1.axvline(optimal_threshold, color='red', linestyle='--', linewidth=2,
                label=f'Optimal = {optimal_threshold:.3f}')
    ax1.axhline(0.5, color='gray', linestyle=':', linewidth=1.5, alpha=0.5, label='Random')
    ax1.set_xlabel('Distance Threshold', fontweight='bold')
    ax1.set_ylabel('AUROC', fontweight='bold')
    ax1.set_title(f'AUROC vs Threshold\nBest AUROC = {best_auroc:.3f}', fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.legend()

    # Plot 2: TPR and FPR vs Threshold
    ax2 = axes[1]
    ax2.plot(thresholds, tpr_scores, linewidth=2.5, marker='s', markersize=5,
             color='green', label='TPR @ SE=0.5')
    ax2.plot(thresholds, fpr_scores, linewidth=2.5, marker='^', markersize=5,
             color='orange', label='FPR @ SE=0.5')
    ax2.axvline(optimal_threshold, color='red', linestyle='--', linewidth=2, alpha=0.7)
    ax2.set_xlabel('Distance Threshold', fontweight='bold')
    ax2.set_ylabel('Rate', fontweight='bold')
    ax2.set_title('TPR and FPR vs Threshold', fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.legend()

    # Plot 3: Mean SE vs Threshold
    ax3 = axes[2]
    ax3.plot(thresholds, mean_se_scores, linewidth=2.5, marker='D', markersize=5,
             color='purple')
    ax3.axvline(optimal_threshold, color='red', linestyle='--', linewidth=2, alpha=0.7)
    ax3.set_xlabel('Distance Threshold', fontweight='bold')
    ax3.set_ylabel('Mean SE Score', fontweight='bold')
    ax3.set_title('Mean SE Score vs Threshold', fontweight='bold')
    ax3.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"\nPlot saved to: {save_path}")
    plt.close()


def plot_comparison_table(
    baseline_auroc: float,
    tuned_auroc: float,
    baseline_threshold: float,
    optimal_threshold: float,
    save_path: Path
):
    """Create a comparison table figure."""

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.axis('tight')
    ax.axis('off')

    # Create table data
    table_data = [
        ['Metric', 'Reference (0.01)', 'Fine-tuned', 'Improvement'],
        ['Clustering Threshold', f'{baseline_threshold:.3f}', f'{optimal_threshold:.3f}',
         f'{(optimal_threshold - baseline_threshold):.3f}'],
        ['AUROC', f'{baseline_auroc:.3f}', f'{tuned_auroc:.3f}',
         f'+{(tuned_auroc - baseline_auroc):.3f}'],
        ['Relative Improvement', '-', '-',
         f'{((tuned_auroc - baseline_auroc) / baseline_auroc * 100):.1f}%']
    ]

    table = ax.table(cellText=table_data, cellLoc='center', loc='center',
                     colWidths=[0.3, 0.25, 0.25, 0.2])
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1, 2)

    # Style header row
    for i in range(4):
        table[(0, i)].set_facecolor('#4472C4')
        table[(0, i)].set_text_props(weight='bold', color='white')

    # Style improvement column if positive
    improvement = tuned_auroc - baseline_auroc
    if improvement > 0:
        table[(2, 3)].set_facecolor('#90EE90')
        table[(3, 3)].set_facecolor('#90EE90')

    plt.title('EAB-SE Fine-Grained Threshold Tuning Results', fontsize=16, fontweight='bold', pad=20)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Comparison table saved to: {save_path}")
    plt.close()


def main():
    """Main function for threshold tuning."""
    print("=" * 70)
    print("THRESHOLD TUNING FOR EAB-SE")
    print("=" * 70)

    # Configuration
    results_path = experiment_dir / "results_eab" / "raw_results_eab.json"
    output_dir = experiment_dir / "results_eab" / "threshold_tuning_fine"  # New directory to keep old results
    output_dir.mkdir(exist_ok=True, parents=True)

    # Load pre-generated results
    data = load_results(results_path)
    results = data['results']

    # Threshold range to test - fine-grained around 0.01
    thresholds = np.array([0.005, 0.008, 0.01, 0.012, 0.015])

    print(f"\nTesting {len(thresholds)} thresholds: {thresholds}")

    # Run threshold tuning
    tuning_results = tune_threshold(
        results=results,
        thresholds=thresholds,
        encoder_model="all-mpnet-base-v2",
        linkage="average",
        device="cpu"  # Change to "cuda" if available
    )

    # Save tuning results (convert numpy types to native Python for JSON)
    output_json = output_dir / "threshold_tuning_results.json"

    # Convert numpy types
    tuning_results_serializable = {
        'thresholds': [float(x) for x in tuning_results['thresholds']],
        'auroc_scores': [float(x) for x in tuning_results['auroc_scores']],
        'tpr_scores': [float(x) for x in tuning_results['tpr_scores']],
        'fpr_scores': [float(x) for x in tuning_results['fpr_scores']],
        'mean_se_scores': [float(x) for x in tuning_results['mean_se_scores']],
        'optimal_threshold': float(tuning_results['optimal_threshold']),
        'best_auroc': float(tuning_results['best_auroc']),
        'best_idx': int(tuning_results['best_idx'])
    }

    with open(output_json, 'w') as f:
        json.dump(tuning_results_serializable, f, indent=2)
    print(f"\nTuning results saved to: {output_json}")

    # Print summary
    print("\n" + "=" * 70)
    print("THRESHOLD TUNING RESULTS")
    print("=" * 70)

    optimal_threshold = tuning_results['optimal_threshold']
    best_auroc = tuning_results['best_auroc']

    print(f"\nOptimal Threshold: {optimal_threshold:.3f}")
    print(f"Best AUROC: {best_auroc:.3f}")

    # Find AUROC at reference threshold (0.01 - the value we're refining around)
    reference_idx = np.argmin(np.abs(thresholds - 0.01))
    reference_auroc = tuning_results['auroc_scores'][reference_idx]

    print(f"\nComparison:")
    print(f"  Reference threshold (0.01): AUROC = {reference_auroc:.3f}")
    print(f"  Optimal threshold ({optimal_threshold:.3f}): AUROC = {best_auroc:.3f}")
    print(f"  Improvement: {(best_auroc - reference_auroc):.3f} ({((best_auroc - reference_auroc) / reference_auroc * 100):.1f}%)")

    # Create visualizations
    print("\n" + "-" * 70)
    print("Creating visualizations...")

    plot_path = output_dir / "threshold_tuning_plot.png"
    plot_threshold_tuning_results(tuning_results, plot_path)

    comparison_path = output_dir / "comparison_table.png"
    plot_comparison_table(
        baseline_auroc=reference_auroc,
        tuned_auroc=best_auroc,
        baseline_threshold=0.01,
        optimal_threshold=optimal_threshold,
        save_path=comparison_path
    )

    # Show top 5 thresholds
    print("\nTop 5 Thresholds:")
    print("-" * 70)
    sorted_indices = np.argsort(tuning_results['auroc_scores'])[::-1][:5]
    for i, idx in enumerate(sorted_indices, 1):
        threshold = thresholds[idx]
        auroc = tuning_results['auroc_scores'][idx]
        print(f"{i}. Threshold = {threshold:.3f}, AUROC = {auroc:.3f}")

    print("\n" + "=" * 70)
    print("COMPLETE")
    print("=" * 70)
    print(f"\nResults saved to: {output_dir}")
    print("\nNext steps:")
    print(f"  1. Update config_eab.yaml with optimal threshold: {optimal_threshold:.3f}")
    print("  2. Re-run analyze_results.py with optimal threshold")
    print("  3. Compare with naive sampling SE results")


if __name__ == "__main__":
    main()
