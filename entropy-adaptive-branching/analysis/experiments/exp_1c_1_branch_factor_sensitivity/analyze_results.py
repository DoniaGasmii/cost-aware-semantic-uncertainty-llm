"""
Analyze results from Experiment 1.C.1: Branch Factor Sensitivity
"""

import json
import numpy as np
from pathlib import Path
from scipy import stats
from typing import Dict, List, Any

experiment_dir = Path(__file__).parent


def load_results() -> List[Dict[str, Any]]:
    results_file = experiment_dir / "results" / "raw_results.json"
    with open(results_file, 'r') as f:
        data = json.load(f)
    return data['results']


def compute_summary_stats(results: List[Dict[str, Any]]) -> Dict[int, Dict[str, Any]]:
    """Compute summary statistics grouped by branch_factor."""
    by_factor = {}
    for result in results:
        factor = result['branch_factor']
        if factor not in by_factor:
            by_factor[factor] = []
        by_factor[factor].append(result)

    summary = {}
    for factor, factor_results in sorted(by_factor.items()):
        speedup_token = [r['efficiency']['speedup_token_steps'] for r in factor_results]
        speedup_time = [r['efficiency']['speedup_time'] for r in factor_results]

        num_samples = [r['num_eab_samples'] for r in factor_results]
        branch_counts = [r['branching_stats']['total_branches'] for r in factor_results]

        # Quality metrics
        eab_diversity = [r['eab_quality']['unique_tokens_ratio'] for r in factor_results if 'eab_quality' in r]
        naive_diversity = [r['naive_quality']['unique_tokens_ratio'] for r in factor_results if 'naive_quality' in r]

        summary[factor] = {
            'num_prompts': len(factor_results),
            'speedup_token_steps': {
                'mean': np.mean(speedup_token),
                'std': np.std(speedup_token),
                'ci_95': stats.t.interval(0.95, len(speedup_token) - 1,
                                         loc=np.mean(speedup_token), scale=stats.sem(speedup_token))
            },
            'speedup_time': {
                'mean': np.mean(speedup_time),
                'std': np.std(speedup_time),
            },
            'num_samples': {
                'mean': np.mean(num_samples),
                'std': np.std(num_samples),
            },
            'branch_count': {
                'mean': np.mean(branch_counts),
                'std': np.std(branch_counts),
            },
            'eab_diversity': {
                'mean': np.mean(eab_diversity) if eab_diversity else 0,
                'std': np.std(eab_diversity) if eab_diversity else 0,
            },
            'naive_diversity': {
                'mean': np.mean(naive_diversity) if naive_diversity else 0,
                'std': np.std(naive_diversity) if naive_diversity else 0,
            }
        }

    return summary


def analyze_tradeoff(summary: Dict[int, Dict[str, Any]]) -> Dict[str, Any]:
    """Analyze cost-quality tradeoff."""
    factors = sorted(summary.keys())
    speedups = [summary[f]['speedup_token_steps']['mean'] for f in factors]
    diversities = [summary[f]['eab_diversity']['mean'] for f in factors]

    return {
        'speedup_range': (min(speedups), max(speedups)),
        'diversity_range': (min(diversities), max(diversities)),
        'optimal_factor': factors[np.argmax(speedups)],  # Factor with best speedup
        'interpretation': f"Branch factor {factors[np.argmax(speedups)]} achieves best speedup ({max(speedups):.2f}×)"
    }


def print_summary_table(summary: Dict[int, Dict[str, Any]]):
    print("\n" + "=" * 100)
    print("SUMMARY STATISTICS: BRANCH FACTOR SENSITIVITY")
    print("=" * 100)

    print(f"{'Factor':<10} {'N':<5} {'Speedup':<20} {'Samples':<15} {'Branches':<15} {'Diversity (EAB)':<20}")
    print(f"{'':<10} {'':<5} {'Mean±SD':<20} {'Mean±SD':<15} {'Mean±SD':<15} {'Mean±SD':<20}")
    print("-" * 100)

    for factor in sorted(summary.keys()):
        s = summary[factor]

        speedup = f"{s['speedup_token_steps']['mean']:.2f}±{s['speedup_token_steps']['std']:.2f}"
        samples = f"{s['num_samples']['mean']:.1f}±{s['num_samples']['std']:.1f}"
        branches = f"{s['branch_count']['mean']:.1f}±{s['branch_count']['std']:.1f}"
        diversity = f"{s['eab_diversity']['mean']:.3f}±{s['eab_diversity']['std']:.3f}"

        print(f"{factor:<10} {s['num_prompts']:<5} {speedup:<20} {samples:<15} {branches:<15} {diversity:<20}")

    print("=" * 100)


def main():
    print("=" * 70)
    print("ANALYZING EXPERIMENT 1.C.1 RESULTS")
    print("=" * 70)

    print("\n1. Loading results...")
    results = load_results()
    print(f"   ✓ Loaded {len(results)} experiments")

    print("\n2. Computing summary statistics...")
    summary = compute_summary_stats(results)
    print(f"   ✓ Summarized results for {len(summary)} branch factors")

    print_summary_table(summary)

    print("\n3. Analyzing cost-quality tradeoff...")
    tradeoff = analyze_tradeoff(summary)
    print(f"   Speedup range: {tradeoff['speedup_range'][0]:.2f}× to {tradeoff['speedup_range'][1]:.2f}×")
    print(f"   {tradeoff['interpretation']}")

    print("\n4. Saving summary statistics...")
    summary_file = experiment_dir / "results" / "summary_stats.json"

    def convert(obj):
        if isinstance(obj, (np.integer, np.floating)):
            return float(obj)
        elif isinstance(obj, (np.ndarray, tuple)):
            return [float(x) for x in obj]
        return obj

    summary_serializable = {
        k: {kk: convert(vv) if not isinstance(vv, dict) else {kkk: convert(vvv) for kkk, vvv in vv.items()}
            for kk, vv in v.items()}
        for k, v in summary.items()
    }

    with open(summary_file, 'w') as f:
        json.dump({'summary_by_factor': summary_serializable, 'tradeoff_analysis': tradeoff}, f, indent=2)

    print(f"   ✓ Summary saved to {summary_file}")

    print("\n" + "=" * 70)
    print("ANALYSIS COMPLETE - Next: python plot_results.py")
    print("=" * 70)


if __name__ == "__main__":
    main()
