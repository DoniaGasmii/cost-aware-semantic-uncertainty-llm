"""
Analyze results from Experiment 2.A.1: SE AUROC on TriviaQA

This script:
1. Loads raw results
2. Computes AUROC for different SE metrics and correctness definitions
3. Analyzes correlation between uncertainty and incorrectness
4. Optionally sweeps clustering thresholds
5. Saves summary statistics
"""

import sys
import json
import yaml
import numpy as np
from pathlib import Path
from typing import Dict, Any, List
from sklearn.metrics import roc_auc_score, roc_curve, precision_recall_curve, average_precision_score
from scipy.stats import pearsonr, spearmanr

experiment_dir = Path(__file__).parent


def load_config() -> Dict[str, Any]:
    """Load experiment configuration."""
    config_path = experiment_dir / "config.yaml"
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def load_results() -> Dict[str, Any]:
    """Load raw results from experiment."""
    results_dir = experiment_dir / "results"
    results_file = results_dir / "raw_results.json"

    with open(results_file, 'r') as f:
        return json.load(f)


def compute_auroc_metrics(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Compute AUROC for different uncertainty metrics and correctness definitions.

    AUROC measures: "If we rank questions by uncertainty, do incorrect ones rank higher?"
    Higher AUROC = better uncertainty estimation.
    """
    # Extract metrics
    se_entropy = np.array([r['se_entropy'] for r in results])
    se_norm_entropy = np.array([r['se_normalized_entropy'] for r in results])
    se_uncertainty = np.array([r['se_uncertainty_score'] for r in results])
    n_clusters = np.array([r['se_n_clusters'] for r in results])

    # Correctness labels (we predict INCORRECTNESS, so invert)
    any_correct = np.array([r['any_correct'] for r in results])
    majority_correct = np.array([r['majority_correct'] for r in results])
    best_correct = np.array([r['best_sample_correct'] for r in results])

    # AUROC: higher uncertainty should predict INCORRECTNESS
    # So we use (1 - correct) as the target
    any_incorrect = 1 - any_correct
    majority_incorrect = 1 - majority_correct
    best_incorrect = 1 - best_correct

    metrics = {}

    # For each uncertainty metric, compute AUROC against each correctness definition
    uncertainty_metrics = {
        'se_entropy': se_entropy,
        'se_normalized_entropy': se_norm_entropy,
        'se_uncertainty_score': se_uncertainty,
        'n_clusters': n_clusters,
    }

    correctness_defs = {
        'any_incorrect': any_incorrect,
        'majority_incorrect': majority_incorrect,
        'best_incorrect': best_incorrect,
    }

    for unc_name, unc_values in uncertainty_metrics.items():
        metrics[unc_name] = {}

        for corr_name, corr_values in correctness_defs.items():
            # Skip if all same class (AUROC undefined)
            if len(np.unique(corr_values)) < 2:
                metrics[unc_name][corr_name] = {
                    'auroc': None,
                    'note': 'All samples same class'
                }
                continue

            try:
                auroc = roc_auc_score(corr_values, unc_values)

                # Also compute AUPR (Area Under Precision-Recall)
                aupr = average_precision_score(corr_values, unc_values)

                # Correlation
                pearson_r, pearson_p = pearsonr(unc_values, corr_values)
                spearman_r, spearman_p = spearmanr(unc_values, corr_values)

                metrics[unc_name][corr_name] = {
                    'auroc': float(auroc),
                    'aupr': float(aupr),
                    'pearson_r': float(pearson_r),
                    'pearson_p': float(pearson_p),
                    'spearman_r': float(spearman_r),
                    'spearman_p': float(spearman_p),
                }
            except Exception as e:
                metrics[unc_name][corr_name] = {
                    'auroc': None,
                    'error': str(e)
                }

    return metrics


def compute_summary_statistics(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Compute summary statistics for the experiment."""

    # Extract arrays
    se_entropy = [r['se_entropy'] for r in results]
    se_norm_entropy = [r['se_normalized_entropy'] for r in results]
    se_uncertainty = [r['se_uncertainty_score'] for r in results]
    n_clusters = [r['se_n_clusters'] for r in results]
    num_samples = [r['num_samples'] for r in results]
    best_rouge = [r['best_rouge_l'] for r in results]

    any_correct = [r['any_correct'] for r in results]
    majority_correct = [r['majority_correct'] for r in results]

    return {
        'num_questions': len(results),

        # Accuracy
        'accuracy_any': float(np.mean(any_correct)),
        'accuracy_majority': float(np.mean(majority_correct)),

        # SE metrics distribution
        'se_entropy': {
            'mean': float(np.mean(se_entropy)),
            'std': float(np.std(se_entropy)),
            'min': float(np.min(se_entropy)),
            'max': float(np.max(se_entropy)),
        },
        'se_normalized_entropy': {
            'mean': float(np.mean(se_norm_entropy)),
            'std': float(np.std(se_norm_entropy)),
            'min': float(np.min(se_norm_entropy)),
            'max': float(np.max(se_norm_entropy)),
        },
        'se_uncertainty_score': {
            'mean': float(np.mean(se_uncertainty)),
            'std': float(np.std(se_uncertainty)),
            'min': float(np.min(se_uncertainty)),
            'max': float(np.max(se_uncertainty)),
        },
        'n_clusters': {
            'mean': float(np.mean(n_clusters)),
            'std': float(np.std(n_clusters)),
            'min': int(np.min(n_clusters)),
            'max': int(np.max(n_clusters)),
            'distribution': dict(zip(*np.unique(n_clusters, return_counts=True))),
        },

        # Sample info
        'avg_samples_per_question': float(np.mean(num_samples)),
        'avg_best_rouge_l': float(np.mean(best_rouge)),
    }


def analyze_by_correctness(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Compare uncertainty metrics for correct vs incorrect answers.

    This helps understand if SE is actually higher for incorrect answers.
    """
    correct = [r for r in results if r['any_correct']]
    incorrect = [r for r in results if not r['any_correct']]

    if not correct or not incorrect:
        return {'error': 'Need both correct and incorrect samples'}

    analysis = {}

    for metric in ['se_entropy', 'se_normalized_entropy', 'se_uncertainty_score', 'se_n_clusters']:
        correct_values = [r[metric] for r in correct]
        incorrect_values = [r[metric] for r in incorrect]

        analysis[metric] = {
            'correct': {
                'mean': float(np.mean(correct_values)),
                'std': float(np.std(correct_values)),
                'n': len(correct_values),
            },
            'incorrect': {
                'mean': float(np.mean(incorrect_values)),
                'std': float(np.std(incorrect_values)),
                'n': len(incorrect_values),
            },
            'difference': float(np.mean(incorrect_values) - np.mean(correct_values)),
            'higher_for_incorrect': float(np.mean(incorrect_values)) > float(np.mean(correct_values)),
        }

    return analysis


def save_json(data: Any, path: Path):
    """Save data to JSON file."""

    def convert(obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj

    with open(path, 'w') as f:
        json.dump(data, f, indent=2, default=convert)


def main():
    """Main analysis function."""
    print("=" * 70)
    print("ANALYZING EXPERIMENT 2.A.1: SE AUROC ON TRIVIAQA")
    print("=" * 70)

    # Load results
    print("\n1. Loading results...")
    data = load_results()
    results = data['results']
    print(f"   Loaded {len(results)} question results")

    # Compute AUROC metrics
    print("\n2. Computing AUROC metrics...")
    auroc_metrics = compute_auroc_metrics(results)

    # Print key AUROC results
    print("\n   AUROC Results (SE uncertainty -> incorrectness):")
    print("   " + "-" * 50)

    for unc_metric in ['se_uncertainty_score', 'se_normalized_entropy', 'n_clusters']:
        if 'any_incorrect' in auroc_metrics[unc_metric]:
            auroc = auroc_metrics[unc_metric]['any_incorrect'].get('auroc')
            if auroc is not None:
                print(f"   {unc_metric:25s} AUROC: {auroc:.3f}")

    # Compute summary statistics
    print("\n3. Computing summary statistics...")
    summary = compute_summary_statistics(results)
    print(f"   Accuracy (any correct): {summary['accuracy_any']:.1%}")
    print(f"   Avg clusters: {summary['n_clusters']['mean']:.1f}")
    print(f"   Avg SE uncertainty: {summary['se_uncertainty_score']['mean']:.3f}")

    # Analyze by correctness
    print("\n4. Comparing correct vs incorrect...")
    correctness_analysis = analyze_by_correctness(results)

    if 'error' not in correctness_analysis:
        metric = 'se_uncertainty_score'
        corr_mean = correctness_analysis[metric]['correct']['mean']
        incorr_mean = correctness_analysis[metric]['incorrect']['mean']
        print(f"   SE uncertainty (correct):   {corr_mean:.3f}")
        print(f"   SE uncertainty (incorrect): {incorr_mean:.3f}")
        if correctness_analysis[metric]['higher_for_incorrect']:
            print(f"   ✓ SE is higher for incorrect answers (good!)")
        else:
            print(f"   ✗ SE is NOT higher for incorrect answers (bad)")

    # Save analysis results
    print("\n5. Saving analysis...")
    results_dir = experiment_dir / "results"

    analysis_output = {
        'auroc_metrics': auroc_metrics,
        'summary_statistics': summary,
        'correctness_analysis': correctness_analysis,
    }

    save_json(analysis_output, results_dir / "summary_stats.json")
    print(f"   Saved to: {results_dir / 'summary_stats.json'}")

    # Print final summary
    print("\n" + "=" * 70)
    print("ANALYSIS COMPLETE")
    print("=" * 70)

    # Highlight key metric
    key_auroc = auroc_metrics['se_uncertainty_score']['any_incorrect'].get('auroc')
    if key_auroc:
        print(f"\n>>> KEY RESULT: AUROC = {key_auroc:.3f} <<<")
        if key_auroc > 0.6:
            print("    (Good: SE reliably identifies incorrect answers)")
        elif key_auroc > 0.5:
            print("    (Moderate: SE is somewhat predictive)")
        else:
            print("    (Poor: SE is not predictive of correctness)")

    print("\nNext step: python plot_results.py")
    print("=" * 70)


if __name__ == "__main__":
    main()
