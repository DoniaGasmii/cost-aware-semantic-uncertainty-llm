"""
Analyze results from Experiment 1.C.3: Temperature Sensitivity
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


def compute_summary_stats(results: List[Dict[str, Any]]) -> Dict[float, Dict[str, Any]]:
    """Compute summary statistics grouped by temperature."""
    by_temp = {}
    for result in results:
        temp = result['temperature']
        if temp not in by_temp:
            by_temp[temp] = []
        by_temp[temp].append(result)

    summary = {}
    for temp, temp_results in sorted(by_temp.items()):
        speedup_token = [r['efficiency']['speedup_token_steps'] for r in temp_results]
        branch_counts = [r['branching_stats']['total_branches'] for r in temp_results]

        # Entropy stats (key insight!)
        avg_entropies = [r['entropy_stats'].get('avg_entropy', 0) for r in temp_results if 'entropy_stats' in r]

        eab_diversity = [r['eab_quality']['unique_tokens_ratio'] for r in temp_results if 'eab_quality' in r]

        summary[temp] = {
            'num_prompts': len(temp_results),
            'speedup_token_steps': {
                'mean': np.mean(speedup_token),
                'std': np.std(speedup_token),
                'ci_95': stats.t.interval(0.95, len(speedup_token) - 1,
                                         loc=np.mean(speedup_token), scale=stats.sem(speedup_token))
            },
            'branch_count': {
                'mean': np.mean(branch_counts),
                'std': np.std(branch_counts),
            },
            'avg_entropy': {
                'mean': np.mean(avg_entropies) if avg_entropies else 0,
                'std': np.std(avg_entropies) if avg_entropies else 0,
            },
            'eab_diversity': {
                'mean': np.mean(eab_diversity) if eab_diversity else 0,
                'std': np.std(eab_diversity) if eab_diversity else 0,
            }
        }

    return summary


def print_summary_table(summary: Dict[float, Dict[str, Any]]):
    print("\n" + "=" * 110)
    print("SUMMARY STATISTICS: TEMPERATURE SENSITIVITY")
    print("=" * 110)

    print(f"{'Temp':<8} {'N':<5} {'Speedup':<20} {'Branches':<15} {'Avg Entropy':<20} {'Diversity (EAB)':<20}")
    print(f"{'':<8} {'':<5} {'Mean±SD':<20} {'Mean±SD':<15} {'Mean±SD':<20} {'Mean±SD':<20}")
    print("-" * 110)

    for temp in sorted(summary.keys()):
        s = summary[temp]

        speedup = f"{s['speedup_token_steps']['mean']:.2f}±{s['speedup_token_steps']['std']:.2f}"
        branches = f"{s['branch_count']['mean']:.1f}±{s['branch_count']['std']:.1f}"
        entropy = f"{s['avg_entropy']['mean']:.4f}±{s['avg_entropy']['std']:.4f}"
        diversity = f"{s['eab_diversity']['mean']:.3f}±{s['eab_diversity']['std']:.3f}"

        print(f"{temp:<8.1f} {s['num_prompts']:<5} {speedup:<20} {branches:<15} {entropy:<20} {diversity:<20}")

    print("=" * 110)


def main():
    print("=" * 70)
    print("ANALYZING EXPERIMENT 1.C.3 RESULTS")
    print("=" * 70)

    print("\n1. Loading results...")
    results = load_results()
    print(f"   ✓ Loaded {len(results)} experiments")

    print("\n2. Computing summary statistics...")
    summary = compute_summary_stats(results)
    print(f"   ✓ Summarized results for {len(summary)} temperatures")

    print_summary_table(summary)

    print("\n3. Key insight: Temperature-Entropy Relationship")
    temps = sorted(summary.keys())
    entropies = [summary[t]['avg_entropy']['mean'] for t in temps]
    print(f"   Temperature range: {min(temps):.1f} to {max(temps):.1f}")
    print(f"   Entropy range: {min(entropies):.4f} to {max(entropies):.4f}")
    print(f"   → Higher temperature = higher entropy = more branching")

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
        json.dump({'summary_by_temperature': summary_serializable}, f, indent=2)

    print(f"   ✓ Summary saved to {summary_file}")

    print("\n" + "=" * 70)
    print("ANALYSIS COMPLETE - Next: python plot_results.py")
    print("=" * 70)


if __name__ == "__main__":
    main()
