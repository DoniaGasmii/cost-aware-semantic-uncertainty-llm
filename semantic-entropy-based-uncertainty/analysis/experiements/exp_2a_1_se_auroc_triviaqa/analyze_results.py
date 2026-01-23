"""
Analyze results from Experiment 2.A.1: SE AUROC on TriviaQA
"""

import sys
import json
import yaml
import numpy as np
from pathlib import Path
from typing import Dict, Any, List
from sklearn.metrics import roc_auc_score, average_precision_score
from scipy.stats import spearmanr

experiment_dir = Path(__file__).parent


def load_config() -> Dict[str, Any]:
    config_path = experiment_dir / "config.yaml"
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def load_results() -> Dict[str, Any]:
    results_dir = experiment_dir / "results"
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
    if cost_stats.get('tokens_per_second', 0) > 0:
        print(f"   Throughput:          {cost_stats['tokens_per_second']:.1f} tok/s")


def compute_auroc_metrics(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    se_uncertainty = np.array([r['se_uncertainty_score'] for r in results])
    best_incorrect = 1 - np.array([r['best_sample_correct'] for r in results])
    auroc = roc_auc_score(best_incorrect, se_uncertainty)
    aupr = average_precision_score(best_incorrect, se_uncertainty)
    spearman_r, _ = spearmanr(se_uncertainty, best_incorrect)
    return {
        'se_uncertainty_score': {
            'best_incorrect': {
                'auroc': float(auroc),
                'aupr': float(aupr),
                'spearman_r': float(spearman_r)
            }
        }
    }


def analyze_by_correctness(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    correct = [r for r in results if r['best_sample_correct']]
    incorrect = [r for r in results if not r['best_sample_correct']]
    if not correct or not incorrect:
        return {'error': 'Need both classes'}
    metric = 'se_uncertainty_score'
    corr_vals = [r[metric] for r in correct]
    incorr_vals = [r[metric] for r in incorrect]
    return {
        metric: {
            'correct': {'mean': float(np.mean(corr_vals))},
            'incorrect': {'mean': float(np.mean(incorr_vals))},
            'higher_for_incorrect': np.mean(incorr_vals) > np.mean(corr_vals)
        }
    }


def save_json(data, path):
    with open(path, 'w') as f:
        json.dump(data, f, indent=2, default=lambda x: float(x) if isinstance(x, np.floating) else x)


def main():
    print("=" * 70)
    print("ANALYZING EXPERIMENT 2.A.1")
    print("=" * 70)
    data = load_results()
    results = data['results']
    print(f"\nLoaded {len(results)} results")
    print_cost_summary(data)
    auroc_metrics = compute_auroc_metrics(results)
    correctness_analysis = analyze_by_correctness(results)
    summary = {
        'auroc_metrics': auroc_metrics,
        'correctness_analysis': correctness_analysis,
        'accuracy': float(np.mean([r['best_sample_correct'] for r in results])),
        'avg_se': float(np.mean([r['se_uncertainty_score'] for r in results]))
    }
    save_json(summary, experiment_dir / "results" / "summary_stats.json")
    key_auroc = auroc_metrics['se_uncertainty_score']['best_incorrect']['auroc']
    print(f"\n>>> KEY RESULT: AUROC = {key_auroc:.3f} <<<")
    print(f"Accuracy: {summary['accuracy']:.1%}")
    print(f"Avg SE: {summary['avg_se']:.3f}")


if __name__ == "__main__":
    main()