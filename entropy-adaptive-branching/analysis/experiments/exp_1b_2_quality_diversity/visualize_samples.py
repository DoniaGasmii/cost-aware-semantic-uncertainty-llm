"""
Visualize sample examples: EAB vs Naive sampling.

Shows actual generated samples for qualitative comparison.
"""

import json
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import textwrap

experiment_dir = Path(__file__).parent


def wrap_text(text: str, width: int = 60) -> str:
    """Wrap text to specified width."""
    return '\n'.join(textwrap.wrap(text, width=width))


def visualize_sample_comparison(
    eab_samples: list,
    naive_samples: list,
    prompt: str,
    output_path: Path,
    num_samples: int = 5
):
    """
    Create side-by-side visualization of EAB vs Naive samples.
    """
    fig = plt.figure(figsize=(16, 12))

    # Title with prompt
    wrapped_prompt = wrap_text(prompt, width=100)
    fig.suptitle(
        f'Sample Comparison for Prompt:\n"{wrapped_prompt}"',
        fontsize=14,
        fontweight='bold',
        y=0.98
    )

    # Create grid
    n_samples = min(num_samples, len(eab_samples), len(naive_samples))

    # EAB samples (left column)
    for i in range(n_samples):
        ax = plt.subplot(n_samples, 2, 2*i + 1)
        ax.axis('off')

        sample_text = wrap_text(eab_samples[i], width=50)

        # Add colored background box
        ax.add_patch(Rectangle((0, 0), 1, 1,
                               facecolor='#E8F4F8',
                               edgecolor='#4A90E2',
                               linewidth=2,
                               transform=ax.transAxes))

        ax.text(0.5, 0.95, f'EAB Sample {i+1}',
                ha='center', va='top',
                fontsize=11, fontweight='bold',
                transform=ax.transAxes,
                color='#2C5F8D')

        ax.text(0.5, 0.5, sample_text,
                ha='center', va='center',
                fontsize=10,
                transform=ax.transAxes,
                wrap=True)

    # Naive samples (right column)
    for i in range(n_samples):
        ax = plt.subplot(n_samples, 2, 2*i + 2)
        ax.axis('off')

        sample_text = wrap_text(naive_samples[i], width=50)

        # Add colored background box
        ax.add_patch(Rectangle((0, 0), 1, 1,
                               facecolor='#FFF0F5',
                               edgecolor='#E75480',
                               linewidth=2,
                               transform=ax.transAxes))

        ax.text(0.5, 0.95, f'Naive Sample {i+1}',
                ha='center', va='top',
                fontsize=11, fontweight='bold',
                transform=ax.transAxes,
                color='#8B2252')

        ax.text(0.5, 0.5, sample_text,
                ha='center', va='center',
                fontsize=10,
                transform=ax.transAxes,
                wrap=True)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"✓ Saved: {output_path}")
    plt.close()


def main():
    """Generate sample visualizations."""
    print("="*70)
    print("VISUALIZING SAMPLE EXAMPLES: EAB vs NAIVE")
    print("="*70)

    # Load data
    eab_results_path = experiment_dir / "../exp_2a_1_se_auroc_triviaqa/results_eab/raw_results_eab.json"
    naive_results_path = experiment_dir / "../exp_2a_1_se_auroc_triviaqa/results/raw_results.json"

    with open(eab_results_path) as f:
        eab_data = json.load(f)

    with open(naive_results_path) as f:
        naive_data = json.load(f)

    output_dir = experiment_dir / "results" / "figures"
    output_dir.mkdir(exist_ok=True, parents=True)

    print(f"\nLoaded {len(eab_data['results'])} EAB results")
    print(f"Loaded {len(naive_data['results'])} Naive results")

    # Select interesting examples
    # Find examples with varying diversity characteristics
    print("\nGenerating sample visualizations...")

    # Example 1: Random sample
    idx = 0
    eab_result = eab_data['results'][idx]
    naive_result = naive_data['results'][idx]

    visualize_sample_comparison(
        eab_samples=eab_result['generated_samples'],
        naive_samples=naive_result['generated_samples'],
        prompt=eab_result['question'],
        output_path=output_dir / f"sample_comparison_example_{idx+1}.png",
        num_samples=5
    )

    # Example 2: Another interesting case
    idx = 5
    if idx < len(eab_data['results']) and idx < len(naive_data['results']):
        eab_result = eab_data['results'][idx]
        naive_result = naive_data['results'][idx]

        visualize_sample_comparison(
            eab_samples=eab_result['generated_samples'],
            naive_samples=naive_result['generated_samples'],
            prompt=eab_result['question'],
            output_path=output_dir / f"sample_comparison_example_{idx+1}.png",
            num_samples=5
        )

    # Example 3: One more case
    idx = 10
    if idx < len(eab_data['results']) and idx < len(naive_data['results']):
        eab_result = eab_data['results'][idx]
        naive_result = naive_data['results'][idx]

        visualize_sample_comparison(
            eab_samples=eab_result['generated_samples'],
            naive_samples=naive_result['generated_samples'],
            prompt=eab_result['question'],
            output_path=output_dir / f"sample_comparison_example_{idx+1}.png",
            num_samples=5
        )

    print("\n" + "="*70)
    print("✓ SAMPLE VISUALIZATIONS COMPLETE")
    print("="*70)
    print(f"\nSaved to: {output_dir}/")
    print("  • sample_comparison_example_*.png")


if __name__ == "__main__":
    main()
