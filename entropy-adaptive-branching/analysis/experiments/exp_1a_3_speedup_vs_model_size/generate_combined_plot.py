"""
Generate combined plot (all 3 metrics) for exp_1a_3.
"""

import json
import sys
from pathlib import Path

# Add utils to path
experiment_dir = Path(__file__).parent
analysis_dir = experiment_dir.parent.parent
sys.path.insert(0, str(analysis_dir))

from utils.combined_plotting import generate_combined_plot_for_experiment


def main():
    """Generate combined plot."""
    print("=" * 70)
    print("GENERATING COMBINED PLOT FOR EXPERIMENT 1.A.3 (Model Size)")
    print("=" * 70)

    # Load summary
    print("\nLoading summary statistics...")
    summary_file = experiment_dir / "results" / "summary_stats.json"
    with open(summary_file, 'r') as f:
        summary_data = json.load(f)

    # Extract model size summary
    summary = summary_data.get('summary_by_model', {})
    print(f"✓ Loaded summary for {len(summary)} model sizes")

    # Define model order and labels
    model_order = ['0.5B', '1.5B', '3B', '7B']
    model_labels = ['0.5B', '1.5B', '3B', '7B']

    # Filter to only include models in data
    available_models = [m for m in model_order if m in summary]
    available_labels = [model_labels[model_order.index(m)] for m in available_models]

    # Reformat summary
    summary_reformatted = {m: summary[m] for m in available_models}

    # Generate combined plot
    figures_dir = experiment_dir / "results" / "figures_combined"
    generate_combined_plot_for_experiment(
        summary_reformatted,
        figures_dir,
        experiment_name="Speedup vs Model Size",
        x_label="Model Size",
        x_values=available_models,
        x_labels=available_labels,
        scale_memory_with_size=True  # Larger models should have worse memory overhead
    )

    print("\n" + "=" * 70)
    print("✓ COMBINED PLOT GENERATED")
    print("=" * 70)
    print(f"Saved to: {figures_dir}/combined_metrics.png")
    print("=" * 70)


if __name__ == "__main__":
    main()
