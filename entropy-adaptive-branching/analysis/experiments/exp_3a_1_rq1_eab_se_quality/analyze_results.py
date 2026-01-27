"""
Analyze results from RQ1: EAB+SE Quality Evaluation
"""

import sys
import json
import yaml
import numpy as np
import argparse
from pathlib import Path
from typing import Dict, Any, List
from sklearn.metrics import roc_auc_score, average_precision_score
from scipy.stats import spearmanr, pearsonr

experiment_dir = Path(__file__).parent


def load_config(config_path: str = None) -> Dict[str, Any]:
    if config_path is None:
        config_path = experiment_dir / "config.yaml"
    else:
        config_path = Path(config_path)
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def load_results(config: Dict[str, Any]) -> Dict[str, Any]:
    results_dir = experiment_dir / config['output']['results_dir']
    results_file = results_dir / "raw_results.json"
    with open(results_file, 'r') as f:
        return json.load(f)


def print_cost_summary(data: Dict[str, Any]):
    cost_stats = data.get('cost_stats', {})
    if not cost_stats:
        print("   ⚠️  No cost statistics found.")
        return

    print("\n   💻 RESOURCE USAGE SUMMARY:")
    print("   " + "-" * 40)
    print(f"   Total time:          {cost_stats.get('total_time_seconds', 0):.1f} s")
    print(f"   Generation time:     {cost_stats.get('total_generation_time', 0):.1f} s")
    print(f"   SE computation time: {cost_stats.get('total_se_time', 0):.1f} s")
    print(f"   Peak GPU memory:     {cost_stats.get('peak_memory_mb', 0):.1f} MB")
    print(f"   Total tokens:        {cost_stats.get('total_tokens_generated', 0):,}")
    print(f"   Total branches:      {cost_stats.get('total_branches', 0):,}")
    if cost_stats.get('tokens_per_second', 0) > 0:
        print(f"   Throughput:          {cost_stats['tokens_per_second']:.1f} tok/s")


def compute_auroc_metrics(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Compute AUROC for multiple uncertainty metrics and correctness definitions."""

    metrics = {}

    # Uncertainty measures to evaluate
    uncertainty_keys = ['se_entropy', 'se_normalized_entropy', 'se_uncertainty_score', 'n_clusters']

    # Correctness definitions
    correctness_defs = {
        'any_incorrect': lambda r: not r['any_correct'],
        'majority_incorrect': lambda r: not r['majority_correct'],
        'best_incorrect': lambda r: not r['best_sample_correct']
    }

    for unc_key in uncertainty_keys:
        # Extract uncertainty scores
        if unc_key == 'n_clusters':
            unc_scores = np.array([r['se_n_clusters'] for r in results])
        else:
            unc_scores = np.array([r[unc_key] for r in results])

        metrics[unc_key] = {}

        for corr_name, corr_fn in correctness_defs.items():
            y_true = np.array([1 if corr_fn(r) else 0 for r in results])

            # Skip if all same class
            if len(np.unique(y_true)) < 2:
                continue

            auroc = roc_auc_score(y_true, unc_scores)
            aupr = average_precision_score(y_true, unc_scores)
            pearson_r, pearson_p = pearsonr(unc_scores, y_true)
            spearman_r, spearman_p = spearmanr(unc_scores, y_true)

            metrics[unc_key][corr_name] = {
                'auroc': float(auroc),
                'aupr': float(aupr),
                'pearson_r': float(pearson_r),
                'pearson_p': float(pearson_p),
                'spearman_r': float(spearman_r),
                'spearman_p': float(spearman_p)
            }

    return metrics


def analyze_by_correctness(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Compare uncertainty metrics between correct and incorrect answers."""

    correct = [r for r in results if r['best_sample_correct']]
    incorrect = [r for r in results if not r['best_sample_correct']]

    if not correct or not incorrect:
        return {'error': 'Need both correct and incorrect samples'}

    analysis = {}

    for metric in ['se_entropy', 'se_normalized_entropy', 'se_uncertainty_score', 'se_n_clusters']:
        corr_vals = [r[metric] for r in correct]
        incorr_vals = [r[metric] for r in incorrect]

        analysis[metric] = {
            'correct': {
                'mean': float(np.mean(corr_vals)),
                'std': float(np.std(corr_vals)),
                'n': len(corr_vals)
            },
            'incorrect': {
                'mean': float(np.mean(incorr_vals)),
                'std': float(np.std(incorr_vals)),
                'n': len(incorr_vals)
            },
            'difference': float(np.mean(incorr_vals) - np.mean(corr_vals)),
            'higher_for_incorrect': np.mean(incorr_vals) > np.mean(corr_vals)
        }

    return analysis


def compute_summary_stats(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Compute summary statistics."""

    # Cluster distribution
    cluster_counts = {}
    for r in results:
        n = r['se_n_clusters']
        cluster_counts[n] = cluster_counts.get(n, 0) + 1

    return {
        'num_questions': len(results),
        'accuracy_any': float(np.mean([r['any_correct'] for r in results])),
        'accuracy_majority': float(np.mean([r['majority_correct'] for r in results])),
        'se_entropy': {
            'mean': float(np.mean([r['se_entropy'] for r in results])),
            'std': float(np.std([r['se_entropy'] for r in results])),
            'min': float(np.min([r['se_entropy'] for r in results])),
            'max': float(np.max([r['se_entropy'] for r in results]))
        },
        'se_normalized_entropy': {
            'mean': float(np.mean([r['se_normalized_entropy'] for r in results])),
            'std': float(np.std([r['se_normalized_entropy'] for r in results])),
            'min': float(np.min([r['se_normalized_entropy'] for r in results])),
            'max': float(np.max([r['se_normalized_entropy'] for r in results]))
        },
        'se_uncertainty_score': {
            'mean': float(np.mean([r['se_uncertainty_score'] for r in results])),
            'std': float(np.std([r['se_uncertainty_score'] for r in results])),
            'min': float(np.min([r['se_uncertainty_score'] for r in results])),
            'max': float(np.max([r['se_uncertainty_score'] for r in results]))
        },
        'n_clusters': {
            'mean': float(np.mean([r['se_n_clusters'] for r in results])),
            'std': float(np.std([r['se_n_clusters'] for r in results])),
            'min': int(np.min([r['se_n_clusters'] for r in results])),
            'max': int(np.max([r['se_n_clusters'] for r in results])),
            'distribution': {str(k): v for k, v in sorted(cluster_counts.items())}
        },
        'avg_samples_per_question': float(np.mean([r['num_samples'] for r in results])),
        'avg_best_rouge_l': float(np.mean([r['best_rouge_l'] for r in results]))
    }


def save_json(data, path):
    def json_serializer(obj):
        if isinstance(obj, (np.floating, np.complexfloating)):
            return float(obj)
        elif isinstance(obj, (np.integer, np.bool_)):
            return int(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj

    with open(path, 'w') as f:
        json.dump(data, f, indent=2, default=json_serializer)


def main():
    parser = argparse.ArgumentParser(description="Analyze RQ1 results")
    parser.add_argument('--config', type=str, default=None, help='Path to config file')
    args = parser.parse_args()

    print("=" * 70)
    print("ANALYZING RQ1: EAB+SE QUALITY")
    print("=" * 70)

    config = load_config(args.config)
    data = load_results(config)
    results = data['results']

    print(f"\nLoaded {len(results)} results")
    print_cost_summary(data)

    # Compute metrics
    print("\nComputing AUROC metrics...")
    auroc_metrics = compute_auroc_metrics(results)

    print("Analyzing correctness patterns...")
    correctness_analysis = analyze_by_correctness(results)

    print("Computing summary statistics...")
    summary_stats = compute_summary_stats(results)

    # Combine into summary
    summary = {
        'auroc_metrics': auroc_metrics,
        'correctness_analysis': correctness_analysis,
        'summary_statistics': summary_stats
    }

    # Save
    results_dir = experiment_dir / config['output']['results_dir']
    save_json(summary, results_dir / "summary_stats.json")

    # Print key results
    print("\n" + "=" * 70)
    print("KEY RESULTS")
    print("=" * 70)

    key_auroc = auroc_metrics.get('se_uncertainty_score', {}).get('best_incorrect', {}).get('auroc')
    if key_auroc:
        print(f"\n>>> AUROC (SE Uncertainty Score): {key_auroc:.3f} <<<")

    print(f"\nAccuracy (best sample): {summary_stats['accuracy_any']:.1%}")
    print(f"Avg SE uncertainty: {summary_stats['se_uncertainty_score']['mean']:.3f}")
    print(f"Avg samples per question: {summary_stats['avg_samples_per_question']:.1f}")
    print(f"Avg clusters: {summary_stats['n_clusters']['mean']:.2f}")

    print(f"\n✓ Analysis saved to: {results_dir / 'summary_stats.json'}")


if __name__ == "__main__":
    main()
