"""
Analyze results from Experiment 1.A.2: Speedup vs Sample Count

Computes summary statistics and statistical significance tests.
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
    Compute summary statistics grouped by target sample count.

    Returns:
        Dictionary mapping target_sample_count -> statistics
    """
    # Group by target sample count
    by_count = {}
    for result in results:
        target = result['target_sample_count']
        if target not in by_count:
            by_count[target] = []
        by_count[target].append(result)

    # Compute stats for each target count
    summary = {}
    for target_count, count_results in sorted(by_count.items()):
        # Extract metrics
        speedup_token = [r['efficiency']['speedup_token_steps'] for r in count_results]
        speedup_time = [r['efficiency']['speedup_time'] for r in count_results]
        speedup_memory = [r['efficiency'].get('speedup_memory', 0) for r in count_results]

        actual_eab_samples = [r['num_eab_samples'] for r in count_results]
        eab_token_steps = [r['eab_metrics']['token_steps'] for r in count_results]
        naive_token_steps = [r['naive_metrics']['token_steps'] for r in count_results]

        branch_counts = [r['branching_stats']['total_branches'] for r in count_results]

        summary[target_count] = {
            'num_prompts': len(count_results),
            'max_paths_used': count_results[0]['max_paths'],  # max_paths setting

            # Speedup metrics
            'speedup_token_steps': {
                'mean': np.mean(speedup_token),
                'std': np.std(speedup_token),
                'median': np.median(speedup_token),
                'min': np.min(speedup_token),
                'max': np.max(speedup_token),
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
                'median': np.median(speedup_time),
                'min': np.min(speedup_time),
                'max': np.max(speedup_time),
                'ci_95': stats.t.interval(
                    0.95,
                    len(speedup_time) - 1,
                    loc=np.mean(speedup_time),
                    scale=stats.sem(speedup_time)
                )
            },

            # Actual samples generated
            'actual_eab_samples': {
                'mean': np.mean(actual_eab_samples),
                'std': np.std(actual_eab_samples),
                'median': np.median(actual_eab_samples),
            },

            # Cost metrics
            'eab_token_steps': {
                'mean': np.mean(eab_token_steps),
                'std': np.std(eab_token_steps),
            },
            'naive_token_steps': {
                'mean': np.mean(naive_token_steps),
                'std': np.std(naive_token_steps),
            },

            # Branching stats
            'branch_count': {
                'mean': np.mean(branch_counts),
                'std': np.std(branch_counts),
            },
        }

    return summary


def test_speedup_trend(summary: Dict[int, Dict[str, Any]]) -> Dict[str, Any]:
    """
    Test if speedup increases with sample count (linear regression).

    Returns:
        Statistical test results
    """
    counts = sorted(summary.keys())
    speedups = [summary[c]['speedup_token_steps']['mean'] for c in counts]

    # Linear regression
    slope, intercept, r_value, p_value, std_err = stats.linregress(counts, speedups)

    return {
        'slope': slope,
        'intercept': intercept,
        'r_squared': r_value ** 2,
        'p_value': p_value,
        'std_err': std_err,
        'interpretation': (
            f"Speedup increases by {slope:.4f}× per additional sample. "
            f"R² = {r_value**2:.3f}, p = {p_value:.4f}"
        )
    }


def print_summary_table(summary: Dict[int, Dict[str, Any]]):
    """Print formatted summary table."""
    print("\n" + "=" * 110)
    print("SUMMARY STATISTICS")
    print("=" * 110)

    # Header
    print(f"{'Target':<8} {'max_paths':<10} {'Actual':<15} {'Speedup (Token)':<25} {'Speedup (Time)':<25} {'Branches':<15}")
    print(f"{'Samples':<8} {'':<10} {'Samples (μ±σ)':<15} {'Mean±SD [95% CI]':<25} {'Mean±SD [95% CI]':<25} {'Mean±SD':<15}")
    print("-" * 110)

    # Rows
    for target_count in sorted(summary.keys()):
        s = summary[target_count]

        # Actual samples
        actual_mean = s['actual_eab_samples']['mean']
        actual_std = s['actual_eab_samples']['std']
        actual_str = f"{actual_mean:.1f}±{actual_std:.1f}"

        # Token speedup
        token_mean = s['speedup_token_steps']['mean']
        token_std = s['speedup_token_steps']['std']
        token_ci = s['speedup_token_steps']['ci_95']
        token_str = f"{token_mean:.2f}±{token_std:.2f} [{token_ci[0]:.2f}-{token_ci[1]:.2f}]"

        # Time speedup
        time_mean = s['speedup_time']['mean']
        time_std = s['speedup_time']['std']
        time_ci = s['speedup_time']['ci_95']
        time_str = f"{time_mean:.2f}±{time_std:.2f} [{time_ci[0]:.2f}-{time_ci[1]:.2f}]"

        # Branches
        branch_mean = s['branch_count']['mean']
        branch_std = s['branch_count']['std']
        branch_str = f"{branch_mean:.1f}±{branch_std:.1f}"

        print(f"{target_count:<8} {s['max_paths_used']:<10} {actual_str:<15} {token_str:<25} {time_str:<25} {branch_str:<15}")

    print("=" * 110)


def main():
    """Main analysis script."""
    print("=" * 70)
    print("ANALYZING EXPERIMENT 1.A.2 RESULTS")
    print("=" * 70)

    # Load results
    print("\n1. Loading results...")
    results = load_results()
    print(f"   ✓ Loaded {len(results)} experiments")

    # Compute summary statistics
    print("\n2. Computing summary statistics...")
    summary = compute_summary_stats(results)
    print(f"   ✓ Summarized results for {len(summary)} sample count targets")

    # Print summary table
    print_summary_table(summary)

    # Test trend
    print("\n3. Testing speedup trend (linear regression)...")
    trend_test = test_speedup_trend(summary)
    print(f"   Slope: {trend_test['slope']:.4f}× per sample")
    print(f"   R²: {trend_test['r_squared']:.3f}")
    print(f"   P-value: {trend_test['p_value']:.6f}")
    print(f"\n   {trend_test['interpretation']}")

    # Save summary
    print("\n4. Saving summary statistics...")
    summary_file = experiment_dir / "results" / "summary_stats.json"
    with open(summary_file, 'w') as f:
        # Convert numpy types to native Python types
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

        json.dump({
            'summary_by_count': summary_serializable,
            'trend_analysis': {
                'slope': float(trend_test['slope']),
                'r_squared': float(trend_test['r_squared']),
                'p_value': float(trend_test['p_value'])
            }
        }, f, indent=2)

    print(f"   ✓ Summary saved to {summary_file}")

    # Conclusion
    print("\n" + "=" * 70)
    print("ANALYSIS COMPLETE")
    print("=" * 70)
    print("\nKey Findings:")
    print(f"  • Speedup increases with sample count (slope = {trend_test['slope']:.4f})")
    print(f"  • Linear relationship strength: R² = {trend_test['r_squared']:.3f}")
    print(f"  • Statistical significance: p = {trend_test['p_value']:.6f}")
    print("\nNext step:")
    print("  Run: python plot_results.py")
    print("=" * 70)


if __name__ == "__main__":
    main()
