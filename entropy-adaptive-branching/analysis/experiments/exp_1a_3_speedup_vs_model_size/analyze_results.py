"""
Analyze results from Experiment 1.A.3: Speedup vs Model Size

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


def compute_summary_stats(results: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    """
    Compute summary statistics grouped by model size.

    Returns:
        Dictionary mapping model_params -> statistics
    """
    # Group by model params
    by_model = {}
    for result in results:
        model_params = result['model_params']
        if model_params not in by_model:
            by_model[model_params] = []
        by_model[model_params].append(result)

    # Define sort order for models (0.5B, 1.5B, 3B, 7B)
    model_order = {"0.5B": 0, "1.5B": 1, "3B": 2, "7B": 3}

    # Compute stats for each model
    summary = {}
    for model_params, model_results in sorted(by_model.items(), key=lambda x: model_order.get(x[0], 999)):
        # Extract metrics
        speedup_token = [r['efficiency']['speedup_token_steps'] for r in model_results]
        speedup_time = [r['efficiency']['speedup_time'] for r in model_results]

        eab_token_steps = [r['eab_metrics']['token_steps'] for r in model_results]
        naive_token_steps = [r['naive_metrics']['token_steps'] for r in model_results]

        eab_time = [r['eab_metrics']['wall_clock_time'] for r in model_results]
        naive_time = [r['naive_metrics']['wall_clock_time'] for r in model_results]

        samples_generated = [r['num_eab_samples'] for r in model_results]
        branch_counts = [r['branching_stats']['total_branches'] for r in model_results]

        summary[model_params] = {
            'model_name': model_results[0]['model_name'],
            'num_prompts': len(model_results),

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

            # Absolute times
            'eab_time': {
                'mean': np.mean(eab_time),
                'std': np.std(eab_time),
            },
            'naive_time': {
                'mean': np.mean(naive_time),
                'std': np.std(naive_time),
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

            # Samples and branches
            'samples_generated': {
                'mean': np.mean(samples_generated),
                'std': np.std(samples_generated),
            },
            'branch_count': {
                'mean': np.mean(branch_counts),
                'std': np.std(branch_counts),
            },
        }

    return summary


def test_speedup_trend(summary: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    """
    Test if speedup increases with model size.

    Returns:
        Statistical test results
    """
    # Convert model sizes to numeric (in billions)
    model_order = {"0.5B": 0.5, "1.5B": 1.5, "3B": 3.0, "7B": 7.0}

    model_params = sorted(summary.keys(), key=lambda x: model_order.get(x, 0))
    model_sizes = [model_order[m] for m in model_params]
    speedups = [summary[m]['speedup_token_steps']['mean'] for m in model_params]

    # Linear regression
    slope, intercept, r_value, p_value, std_err = stats.linregress(model_sizes, speedups)

    # Log-log regression (for power-law relationship)
    log_sizes = np.log(model_sizes)
    log_speedups = np.log(speedups)
    log_slope, log_intercept, log_r_value, log_p_value, log_std_err = stats.linregress(log_sizes, log_speedups)

    return {
        'linear': {
            'slope': slope,
            'intercept': intercept,
            'r_squared': r_value ** 2,
            'p_value': p_value,
            'interpretation': f"Linear: Speedup increases by {slope:.4f}× per 1B parameters (R²={r_value**2:.3f})"
        },
        'log_log': {
            'slope': log_slope,
            'intercept': log_intercept,
            'r_squared': log_r_value ** 2,
            'p_value': log_p_value,
            'interpretation': f"Power-law: Speedup ∝ size^{log_slope:.3f} (R²={log_r_value**2:.3f})"
        }
    }


def print_summary_table(summary: Dict[str, Dict[str, Any]]):
    """Print formatted summary table."""
    print("\n" + "=" * 120)
    print("SUMMARY STATISTICS BY MODEL SIZE")
    print("=" * 120)

    # Header
    print(f"{'Model':<25} {'Size':<8} {'N':<5} {'Speedup (Token)':<25} {'Speedup (Time)':<25} {'Time (s)':<20}")
    print(f"{'':<25} {'':<8} {'':<5} {'Mean±SD [95% CI]':<25} {'Mean±SD [95% CI]':<25} {'EAB | Naive':<20}")
    print("-" * 120)

    # Rows (ordered by size)
    model_order = {"0.5B": 0, "1.5B": 1, "3B": 2, "7B": 3}
    for model_params in sorted(summary.keys(), key=lambda x: model_order.get(x, 999)):
        s = summary[model_params]

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

        # Absolute times
        eab_time = s['eab_time']['mean']
        naive_time = s['naive_time']['mean']
        time_abs_str = f"{eab_time:.1f} | {naive_time:.1f}"

        # Model name (shortened)
        model_name = s['model_name'].split('/')[-1][:24]

        print(f"{model_name:<25} {model_params:<8} {s['num_prompts']:<5} {token_str:<25} {time_str:<25} {time_abs_str:<20}")

    print("=" * 120)


def main():
    """Main analysis script."""
    print("=" * 70)
    print("ANALYZING EXPERIMENT 1.A.3 RESULTS")
    print("=" * 70)

    # Load results
    print("\n1. Loading results...")
    results = load_results()
    print(f"   ✓ Loaded {len(results)} experiments")

    # Compute summary statistics
    print("\n2. Computing summary statistics...")
    summary = compute_summary_stats(results)
    print(f"   ✓ Summarized results for {len(summary)} model sizes")

    # Print summary table
    print_summary_table(summary)

    # Test trend
    print("\n3. Testing speedup trend...")
    trend_test = test_speedup_trend(summary)
    print(f"   {trend_test['linear']['interpretation']}")
    print(f"   {trend_test['log_log']['interpretation']}")

    # Save summary
    print("\n4. Saving summary statistics...")
    summary_file = experiment_dir / "results" / "summary_stats.json"
    with open(summary_file, 'w') as f:
        # Convert numpy types
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
            'summary_by_model': summary_serializable,
            'trend_analysis': {
                'linear': {
                    'slope': float(trend_test['linear']['slope']),
                    'r_squared': float(trend_test['linear']['r_squared']),
                    'p_value': float(trend_test['linear']['p_value'])
                },
                'power_law': {
                    'exponent': float(trend_test['log_log']['slope']),
                    'r_squared': float(trend_test['log_log']['r_squared']),
                    'p_value': float(trend_test['log_log']['p_value'])
                }
            }
        }, f, indent=2)

    print(f"   ✓ Summary saved to {summary_file}")

    # Conclusion
    print("\n" + "=" * 70)
    print("ANALYSIS COMPLETE")
    print("=" * 70)
    print("\nKey Findings:")
    print(f"  • {trend_test['linear']['interpretation']}")
    print(f"  • {trend_test['log_log']['interpretation']}")
    print("\nNext step:")
    print("  Run: python plot_results.py")
    print("=" * 70)


if __name__ == "__main__":
    main()
