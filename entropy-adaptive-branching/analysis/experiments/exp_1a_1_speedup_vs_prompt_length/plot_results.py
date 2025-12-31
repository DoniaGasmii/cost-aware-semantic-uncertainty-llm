"""
Generate plots for Experiment 1.A results.
"""

import json
import sys
from pathlib import Path

# Add utils to path
experiment_dir = Path(__file__).parent
analysis_dir = experiment_dir.parent.parent
sys.path.insert(0, str(analysis_dir))

from utils.plotting import plot_all_results
from utils.data_utils import load_results


def main():
    """Generate all plots."""
    print("=" * 70)
    print("GENERATING PLOTS FOR EXPERIMENT 1.A")
    print("=" * 70)

    # Load results
    print("\n1. Loading results...")
    results_file = experiment_dir / "results" / "raw_results.json"
    with open(results_file, 'r') as f:
        data = json.load(f)
    results = data['results']
    print(f"   ✓ Loaded {len(results)} experiments")

    # Load summary
    print("\n2. Loading summary statistics...")
    summary_file = experiment_dir / "results" / "summary_stats.json"
    with open(summary_file, 'r') as f:
        summary_data = json.load(f)
    summary = {int(k): v for k, v in summary_data['summary_by_length'].items()}
    print(f"   ✓ Loaded summary for {len(summary)} prompt lengths")

    # Generate plots
    print("\n3. Generating plots...")
    figures_dir = experiment_dir / "results" / "figures"
    plot_all_results(results, summary, figures_dir, show_plots=False)

    # Done
    print("\n" + "=" * 70)
    print("PLOTTING COMPLETE")
    print("=" * 70)
    print(f"\nGenerated figures saved to: {figures_dir}/")
    print("  • speedup_vs_length.png")
    print("  • cost_breakdown.png")
    print("  • branching_analysis.png")
    print("=" * 70)


if __name__ == "__main__":
    main()
