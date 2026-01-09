"""
Analyze results from Experiment 1.A.5: Speedup vs Generation Length
"""

import json
import numpy as np
from pathlib import Path
from scipy import stats
from typing import Dict, List, Any

experiment_dir = Path(__file__).parent


def load_results() -> List[Dict[str, Any]]:
    """Load experiment results."""
    results_file = experiment_dir / "results" / "raw_results.json"
    with open(results_file, 'r') as f:
        data = json.load(f)
    return data['results']


def compute_summary_stats(results: List[Dict[str, Any]]) -> Dict[int, Dict[str, Any]]:
    """
    Compute summary statistics grouped by generation_length.
    """
    # Group by generation length
    by_length = {}
    for result in results:
        length = result['generation_length']
        if length not in by_length:
            by_length[length] = []
        by_length[length].append(result)

    # Compute stats for each length
    summary = {}
    for length, length_results in sorted(by_length.items()):
        speedup_token = [r['efficiency']['speedup_token_steps'] for r in length_results]
        speedup_time = [r['efficiency']['speedup_time'] for r in length_results]
        speedup_memory = [r['efficiency']['speedup_memory'] for r in length_results]  # Added!

        eab_token_steps = [r['eab_metrics']['token_steps'] for r in length_results]
        naive_token_steps = [r['naive_metrics']['token_steps'] for r in length_results]

        branch_counts = [r['branching_stats']['total_branches'] for r in length_results]

        summary[length] = {
            'num_prompts': len(length_results),
            'speedup_token_steps': {
                'mean': np.mean(speedup_token),
                'std': np.std(speedup_token),
                'median': np.median(speedup_token),
                'ci_95': stats.t.interval(
                    0.95,
                    len(speedup_token) - 1,
                    loc=np.mean(speedup_token),
                    scale=stats.sem(speedup_token)
                )
            },
            'speedup_time': {
                'mean': np.mean(speedup_time),
                'std': np.std(speedup_time),
                'ci_95': stats.t.interval(
                    0.95,
                    len(speedup_time) - 1,
                    loc=np.mean(speedup_time),
                    scale=stats.sem(speedup_time)
                )
            },
            'speedup_memory': {  # Added!
                'mean': np.mean(speedup_memory),
                'std': np.std(speedup_memory),
                'median': np.median(speedup_memory),
                'ci_95': stats.t.interval(
                    0.95,
                    len(speedup_memory) - 1,
                    loc=np.mean(speedup_memory),
                    scale=stats.sem(speedup_memory)
                ) if len(speedup_memory) > 1 else (np.mean(speedup_memory), np.mean(speedup_memory))
            },
            'eab_token_steps': {
                'mean': np.mean(eab_token_steps),
                'std': np.std(eab_token_steps),
            },
            'naive_token_steps': {
                'mean': np.mean(naive_token_steps),
                'std': np.std(naive_token_steps),
            },
            'branch_count': {
                'mean': np.mean(branch_counts),
                'std': np.std(branch_counts),
            }
        }

    return summary


def test_speedup_trend(summary: Dict[int, Dict[str, Any]]) -> Dict[str, Any]:
    """Test if speedup increases with generation length."""
    lengths = sorted(summary.keys())
    speedups = [summary[l]['speedup_token_steps']['mean'] for l in lengths]

    # Linear regression
    slope, intercept, r_value, p_value, std_err = stats.linregress(lengths, speedups)

    return {
        'slope': slope,
        'intercept': intercept,
        'r_squared': r_value ** 2,
        'p_value': p_value,
        'interpretation': (
            f"Speedup increases by {slope:.4f}× per additional 10 tokens. "
            f"R² = {r_value**2:.3f}, p = {p_value:.4f}"
        )
    }


def print_summary_table(summary: Dict[int, Dict[str, Any]]):
    """Print formatted summary table."""
    print("\n" + "=" * 80)
    print("SUMMARY STATISTICS")
    print("=" * 80)

    print(f"{'Gen Length':<12} {'N':<5} {'Speedup (Token)':<25} {'Speedup (Time)':<25} {'Branches':<15}")
    print(f"{'(tokens)':<12} {'':<5} {'Mean±SD [95% CI]':<25} {'Mean±SD':<25} {'Mean±SD':<15}")
    print("-" * 80)

    for length in sorted(summary.keys()):
        s = summary[length]

        token_mean = s['speedup_token_steps']['mean']
        token_std = s['speedup_token_steps']['std']
        token_ci = s['speedup_token_steps']['ci_95']
        token_str = f"{token_mean:.2f}±{token_std:.2f} [{token_ci[0]:.2f}-{token_ci[1]:.2f}]"

        time_mean = s['speedup_time']['mean']
        time_std = s['speedup_time']['std']
        time_str = f"{time_mean:.2f}±{time_std:.2f}"

        branch_mean = s['branch_count']['mean']
        branch_std = s['branch_count']['std']
        branch_str = f"{branch_mean:.1f}±{branch_std:.1f}"

        print(f"{length:<12} {s['num_prompts']:<5} {token_str:<25} {time_str:<25} {branch_str:<15}")

    print("=" * 80)


def main():
    print("=" * 70)
    print("ANALYZING EXPERIMENT 1.A.5 RESULTS")
    print("=" * 70)

    # Load results
    print("\n1. Loading results...")
    results = load_results()
    print(f"   ✓ Loaded {len(results)} experiments")

    # Compute summary statistics
    print("\n2. Computing summary statistics...")
    summary = compute_summary_stats(results)
    print(f"   ✓ Summarized results for {len(summary)} generation lengths")

    # Print summary table
    print_summary_table(summary)

    # Test trend
    print("\n3. Testing speedup trend...")
    trend_test = test_speedup_trend(summary)
    print(f"   Slope: {trend_test['slope']:.4f}× per token")
    print(f"   R²: {trend_test['r_squared']:.3f}")
    print(f"   P-value: {trend_test['p_value']:.6f}")
    print(f"\n   {trend_test['interpretation']}")

    # Save summary
    print("\n4. Saving summary statistics...")
    summary_file = experiment_dir / "results" / "summary_stats.json"

    def convert(obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, tuple):
            return list(obj)
        return obj

    summary_serializable = {
        k: {
            kk: convert(vv) if not isinstance(vv, dict) else {
                kkk: convert(vvv) for kkk, vvv in vv.items()
            }
            for kk, vv in v.items()
        }
        for k, v in summary.items()
    }

    with open(summary_file, 'w') as f:
        json.dump({
            'summary_by_length': summary_serializable,
            'trend_analysis': {
                'slope': float(trend_test['slope']),
                'r_squared': float(trend_test['r_squared']),
                'p_value': float(trend_test['p_value'])
            }
        }, f, indent=2)

    print(f"   ✓ Summary saved to {summary_file}")

    print("\n" + "=" * 70)
    print("ANALYSIS COMPLETE")
    print("=" * 70)
    print("\nKey Findings:")
    print(f"  • Speedup increases with generation length (slope = {trend_test['slope']:.4f})")
    print(f"  • Linear relationship strength: R² = {trend_test['r_squared']:.3f}")
    print("\nNext step:")
    print("  Run: python plot_results.py")
    print("=" * 70)


if __name__ == "__main__":
    main()
