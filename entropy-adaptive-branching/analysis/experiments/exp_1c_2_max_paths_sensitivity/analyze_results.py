"""
Analyze results from Experiment 1.C.2: Max Paths Sensitivity
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
    """Compute summary statistics grouped by max_paths."""
    by_paths = {}
    for result in results:
        paths = result['max_paths']
        if paths not in by_paths:
            by_paths[paths] = []
        by_paths[paths].append(result)

    summary = {}
    for paths, paths_results in sorted(by_paths.items()):
        speedup_token = [r['efficiency']['speedup_token_steps'] for r in paths_results]
        num_samples = [r['num_eab_samples'] for r in paths_results]
        branch_counts = [r['branching_stats']['total_branches'] for r in paths_results]
        eab_diversity = [r['eab_quality']['unique_tokens_ratio'] for r in paths_results if 'eab_quality' in r]

        summary[paths] = {
            'num_prompts': len(paths_results),
            'speedup_token_steps': {
                'mean': np.mean(speedup_token),
                'std': np.std(speedup_token),
                'ci_95': stats.t.interval(0.95, len(speedup_token) - 1,
                                         loc=np.mean(speedup_token), scale=stats.sem(speedup_token))
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
            }
        }

    return summary


def print_summary_table(summary: Dict[int, Dict[str, Any]]):
    print("\n" + "=" * 100)
    print("SUMMARY STATISTICS: MAX PATHS SENSITIVITY")
    print("=" * 100)

    print(f"{'Max Paths':<12} {'N':<5} {'Speedup':<20} {'Samples':<15} {'Branches':<15} {'Diversity (EAB)':<20}")
    print(f"{'':<12} {'':<5} {'Mean±SD':<20} {'Mean±SD':<15} {'Mean±SD':<15} {'Mean±SD':<20}")
    print("-" * 100)

    for paths_val in sorted(summary.keys()):
        s = summary[paths_val]

        speedup = f"{s['speedup_token_steps']['mean']:.2f}±{s['speedup_token_steps']['std']:.2f}"
        samples = f"{s['num_samples']['mean']:.1f}±{s['num_samples']['std']:.1f}"
        branches = f"{s['branch_count']['mean']:.1f}±{s['branch_count']['std']:.1f}"
        diversity = f"{s['eab_diversity']['mean']:.3f}±{s['eab_diversity']['std']:.3f}"

        print(f"{paths_val:<12} {s['num_prompts']:<5} {speedup:<20} {samples:<15} {branches:<15} {diversity:<20}")

    print("=" * 100)


def main():
    print("=" * 70)
    print("ANALYZING EXPERIMENT 1.C.2 RESULTS")
    print("=" * 70)

    print("\n1. Loading results...")
    results = load_results()
    print(f"   ✓ Loaded {len(results)} experiments")

    print("\n2. Computing summary statistics...")
    summary = compute_summary_stats(results)
    print(f"   ✓ Summarized results for {len(summary)} max_paths values")

    print_summary_table(summary)

    print("\n3. Saving summary statistics...")
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
        json.dump({'summary_by_paths': summary_serializable}, f, indent=2)

    print(f"   ✓ Summary saved to {summary_file}")

    print("\n" + "=" * 70)
    print("ANALYSIS COMPLETE - Next: python plot_results.py")
    print("=" * 70)


if __name__ == "__main__":
    main()
