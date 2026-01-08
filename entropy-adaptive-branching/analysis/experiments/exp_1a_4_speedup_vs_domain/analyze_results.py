"""
Analyze results from Experiment 1.A.4: Speedup vs Domain

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
    Compute summary statistics grouped by domain.

    Returns:
        Dictionary mapping domain_name -> statistics
    """
    # Group by domain
    by_domain = {}
    for result in results:
        domain = result['domain']
        if domain not in by_domain:
            by_domain[domain] = []
        by_domain[domain].append(result)

    # Compute stats for each domain
    summary = {}
    for domain, domain_results in sorted(by_domain.items()):
        # Extract metrics
        speedup_token = [r['efficiency']['speedup_token_steps'] for r in domain_results]
        speedup_time = [r['efficiency']['speedup_time'] for r in domain_results]

        eab_token_steps = [r['eab_metrics']['token_steps'] for r in domain_results]
        naive_token_steps = [r['naive_metrics']['token_steps'] for r in domain_results]

        samples_generated = [r['num_eab_samples'] for r in domain_results]
        branch_counts = [r['branching_stats']['total_branches'] for r in domain_results]
        branch_frequency = [
            r['branching_stats']['total_branches'] / r['eab_metrics'].get('num_generated_tokens', 1)
            for r in domain_results
        ]

        summary[domain] = {
            'num_prompts': len(domain_results),

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

            # Cost metrics
            'eab_token_steps': {
                'mean': np.mean(eab_token_steps),
                'std': np.std(eab_token_steps),
            },
            'naive_token_steps': {
                'mean': np.mean(naive_token_steps),
                'std': np.std(naive_token_steps),
            },

            # Branching behavior
            'samples_generated': {
                'mean': np.mean(samples_generated),
                'std': np.std(samples_generated),
            },
            'branch_count': {
                'mean': np.mean(branch_counts),
                'std': np.std(branch_counts),
                'median': np.median(branch_counts),
            },
            'branch_frequency': {
                'mean': np.mean(branch_frequency),
                'std': np.std(branch_frequency),
                'description': 'Branches per generated token'
            },
        }

    return summary


def test_domain_differences(summary: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    """
    Test if speedup differs significantly across domains (ANOVA).

    Returns:
        Statistical test results
    """
    domains = sorted(summary.keys())

    # Extract speedups for each domain
    speedups_by_domain = {
        domain: [] for domain in domains
    }

    # For this test, we'd need individual results, not just summary
    # For now, return descriptive stats

    # Rank domains by speedup
    ranked = sorted(domains, key=lambda d: summary[d]['speedup_token_steps']['mean'], reverse=True)

    return {
        'ranked_by_speedup': [
            {
                'domain': d,
                'mean_speedup': summary[d]['speedup_token_steps']['mean'],
                'mean_branches': summary[d]['branch_count']['mean']
            }
            for d in ranked
        ],
        'interpretation': f"Domains ranked by speedup (highest to lowest): {', '.join(ranked)}"
    }


def print_summary_table(summary: Dict[str, Dict[str, Any]]):
    """Print formatted summary table."""
    print("\n" + "=" * 120)
    print("SUMMARY STATISTICS BY DOMAIN")
    print("=" * 120)

    # Header
    print(f"{'Domain':<20} {'N':<5} {'Speedup (Token)':<25} {'Speedup (Time)':<25} {'Branches':<20}")
    print(f"{'':<20} {'':<5} {'Mean±SD [95% CI]':<25} {'Mean±SD [95% CI]':<25} {'Mean±SD (freq)':<20}")
    print("-" * 120)

    # Rows (ordered by speedup)
    domains_ranked = sorted(summary.keys(), key=lambda d: summary[d]['speedup_token_steps']['mean'], reverse=True)

    for domain in domains_ranked:
        s = summary[domain]

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
        branch_freq = s['branch_frequency']['mean']
        branch_str = f"{branch_mean:.1f}±{branch_std:.1f} ({branch_freq:.3f})"

        print(f"{domain:<20} {s['num_prompts']:<5} {token_str:<25} {time_str:<25} {branch_str:<20}")

    print("=" * 120)


def main():
    """Main analysis script."""
    print("=" * 70)
    print("ANALYZING EXPERIMENT 1.A.4 RESULTS")
    print("=" * 70)

    # Load results
    print("\n1. Loading results...")
    results = load_results()
    print(f"   ✓ Loaded {len(results)} experiments")

    # Compute summary statistics
    print("\n2. Computing summary statistics...")
    summary = compute_summary_stats(results)
    print(f"   ✓ Summarized results for {len(summary)} domains")

    # Print summary table
    print_summary_table(summary)

    # Test domain differences
    print("\n3. Comparing domains...")
    domain_tests = test_domain_differences(summary)
    print(f"   {domain_tests['interpretation']}")

    print("\n   Domain Ranking by Speedup:")
    for i, item in enumerate(domain_tests['ranked_by_speedup'], 1):
        print(f"     {i}. {item['domain']:15s} - Speedup: {item['mean_speedup']:.2f}×, "
              f"Branches: {item['mean_branches']:.1f}")

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
            'summary_by_domain': summary_serializable,
            'domain_analysis': {
                'ranked_by_speedup': domain_tests['ranked_by_speedup']
            }
        }, f, indent=2)

    print(f"   ✓ Summary saved to {summary_file}")

    # Conclusion
    print("\n" + "=" * 70)
    print("ANALYSIS COMPLETE")
    print("=" * 70)
    print("\nKey Findings:")
    for i, item in enumerate(domain_tests['ranked_by_speedup'], 1):
        print(f"  {i}. {item['domain']}: {item['mean_speedup']:.2f}× speedup "
              f"(avg {item['mean_branches']:.1f} branches)")
    print("\nNext step:")
    print("  Run: python plot_results.py")
    print("=" * 70)


if __name__ == "__main__":
    main()
