"""
Analyze results from Experiment 1.C.4: Temperature × Threshold Interaction
"""

import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Any, Tuple

experiment_dir = Path(__file__).parent


def load_results() -> List[Dict[str, Any]]:
    results_file = experiment_dir / "results" / "raw_results.json"
    with open(results_file, 'r') as f:
        data = json.load(f)
    return data['results']


def compute_summary_stats(results: List[Dict[str, Any]]) -> Dict[Tuple[float, float], Dict[str, Any]]:
    """Compute summary statistics grouped by (temperature, threshold) pairs."""
    by_combo = {}
    for result in results:
        temp = result['temperature']
        threshold = result['entropy_threshold']
        key = (temp, threshold)
        if key not in by_combo:
            by_combo[key] = []
        by_combo[key].append(result)

    summary = {}
    for (temp, threshold), combo_results in sorted(by_combo.items()):
        speedup_token = [r['efficiency']['speedup_token_steps'] for r in combo_results]
        branch_counts = [r['branching_stats']['total_branches'] for r in combo_results]
        avg_entropies = [r['entropy_stats'].get('avg_entropy', 0) for r in combo_results if 'entropy_stats' in r]
        eab_diversity = [r['eab_quality']['unique_tokens_ratio'] for r in combo_results if 'eab_quality' in r]

        summary[(temp, threshold)] = {
            'num_prompts': len(combo_results),
            'speedup_token_steps': {
                'mean': np.mean(speedup_token),
                'std': np.std(speedup_token),
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


def build_grid_data(summary: Dict[Tuple[float, float], Dict], metric: str) -> Tuple[List, List, np.ndarray]:
    """Build 2D grid data for heatmap plotting."""
    temps = sorted(set(k[0] for k in summary.keys()))
    thresholds = sorted(set(k[1] for k in summary.keys()))

    grid = np.zeros((len(thresholds), len(temps)))

    for i, threshold in enumerate(thresholds):
        for j, temp in enumerate(temps):
            if (temp, threshold) in summary:
                grid[i, j] = summary[(temp, threshold)][metric]['mean']

    return temps, thresholds, grid


def print_summary_table(summary: Dict[Tuple[float, float], Dict]):
    print("\n" + "=" * 120)
    print("SUMMARY STATISTICS: TEMPERATURE × THRESHOLD INTERACTION")
    print("=" * 120)

    print(f"{'Temp':<8} {'Threshold':<12} {'N':<5} {'Speedup':<18} {'Branches':<15} {'Entropy':<18} {'Diversity':<18}")
    print(f"{'':<8} {'':<12} {'':<5} {'Mean±SD':<18} {'Mean±SD':<15} {'Mean±SD':<18} {'Mean±SD':<18}")
    print("-" * 120)

    for (temp, threshold) in sorted(summary.keys()):
        s = summary[(temp, threshold)]

        speedup = f"{s['speedup_token_steps']['mean']:.2f}±{s['speedup_token_steps']['std']:.2f}"
        branches = f"{s['branch_count']['mean']:.1f}±{s['branch_count']['std']:.1f}"
        entropy = f"{s['avg_entropy']['mean']:.4f}±{s['avg_entropy']['std']:.4f}"
        diversity = f"{s['eab_diversity']['mean']:.3f}±{s['eab_diversity']['std']:.3f}"

        print(f"{temp:<8.1f} {threshold:<12.3f} {s['num_prompts']:<5} {speedup:<18} {branches:<15} {entropy:<18} {diversity:<18}")

    print("=" * 120)


def main():
    print("=" * 70)
    print("ANALYZING EXPERIMENT 1.C.4 RESULTS")
    print("=" * 70)

    print("\n1. Loading results...")
    results = load_results()
    print(f"   ✓ Loaded {len(results)} experiments")

    print("\n2. Computing summary statistics...")
    summary = compute_summary_stats(results)
    print(f"   ✓ Summarized results for {len(summary)} (temp, threshold) combinations")

    print_summary_table(summary)

    print("\n3. Analyzing interaction effects...")
    temps, thresholds, speedup_grid = build_grid_data(summary, 'speedup_token_steps')
    print(f"   Temperature range: {min(temps):.1f} to {max(temps):.1f}")
    print(f"   Threshold range: {min(thresholds):.3f} to {max(thresholds):.3f}")
    print(f"   Speedup range: {speedup_grid.min():.2f}× to {speedup_grid.max():.2f}×")

    print("\n4. Saving summary statistics...")
    summary_file = experiment_dir / "results" / "summary_stats.json"

    def convert(obj):
        if isinstance(obj, (np.integer, np.floating)):
            return float(obj)
        elif isinstance(obj, (np.ndarray, tuple)):
            return [float(x) if isinstance(x, (np.integer, np.floating)) else x for x in obj]
        return obj

    summary_serializable = {
        f"{k[0]}_{k[1]}": {
            kk: convert(vv) if not isinstance(vv, dict) else {kkk: convert(vvv) for kkk, vvv in vv.items()}
            for kk, vv in v.items()
        }
        for k, v in summary.items()
    }

    with open(summary_file, 'w') as f:
        json.dump({'summary_by_combo': summary_serializable}, f, indent=2)

    print(f"   ✓ Summary saved to {summary_file}")

    print("\n" + "=" * 70)
    print("ANALYSIS COMPLETE - Next: python plot_results.py (2D heatmaps!)")
    print("=" * 70)


if __name__ == "__main__":
    main()
