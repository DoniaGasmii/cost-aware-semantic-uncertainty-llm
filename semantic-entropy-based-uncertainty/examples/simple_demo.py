#!/usr/bin/env python3
"""
Simple example demonstrating the experiment pipeline with dummy data.
This runs immediately without needing any API setup.
"""

import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Create example data
def create_demo_batches():
    """Create two batches with different difficulty levels"""
    
    # Easy batch (high scores, low variance)
    easy_batch = {
        "name": "Easy Questions",
        "prompts": [
            {"id": f"easy_{i}", "prompt": f"Easy question {i}", "metadata": {"difficulty": "easy"}}
            for i in range(30)
        ]
    }
    
    # Hard batch (low scores, higher variance)
    hard_batch = {
        "name": "Hard Questions",
        "prompts": [
            {"id": f"hard_{i}", "prompt": f"Hard question {i}", "metadata": {"difficulty": "hard"}}
            for i in range(30)
        ]
    }
    
    return [easy_batch, hard_batch]


def generate_dummy_scores(difficulty, n_samples=30):
    """Generate dummy scores based on difficulty"""
    if difficulty == "easy":
        mean, std = 0.80, 0.10
    else:  # hard
        mean, std = 0.35, 0.15
    
    return np.clip(np.random.normal(mean, std, n_samples), 0, 1)


def plot_distributions(batch_data):
    """Create the distribution plot similar to your image"""
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    colors = ['#FF1493', '#4169E1']  # Pink and Blue like your image
    
    for idx, (batch_name, scores) in enumerate(batch_data.items()):
        mean = np.mean(scores)
        std = np.std(scores)
        
        # Create smooth distribution curve
        x = np.linspace(0, 1, 1000)
        y = (1 / (std * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x - mean) / std) ** 2)
        
        # Plot
        ax.plot(x, y, linewidth=3, color=colors[idx], label=batch_name, alpha=0.9)
        ax.fill_between(x, y, alpha=0.3, color=colors[idx])
        
        # Mean line (dashed vertical)
        ax.axvline(mean, color=colors[idx], linestyle='--', linewidth=2, alpha=0.7)
        
        # Label with μ and σ
        y_pos = 0.95 - (idx * 0.1)
        ax.text(mean, ax.get_ylim()[1] * y_pos, 
               f'μ = {mean:.2f}, σ = {std:.2f}',
               ha='center', fontsize=12, color=colors[idx],
               bbox=dict(boxstyle='round', facecolor='white', edgecolor=colors[idx], linewidth=2))
    
    ax.set_xlabel('Score', fontsize=14, fontweight='bold')
    ax.set_ylabel('Density', fontsize=14, fontweight='bold')
    ax.set_title('Score Distribution Comparison Across Batches', fontsize=16, fontweight='bold', pad=20)
    ax.legend(fontsize=12, loc='upper right', framealpha=0.9)
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.set_xlim(0, 1)
    ax.set_ylim(0, None)
    
    # Remove top and right spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    plt.tight_layout()
    return fig


def create_comprehensive_plots(batch_data):
    """Create multiple comparison plots"""
    
    fig = plt.figure(figsize=(15, 10))
    gs = fig.add_gridspec(3, 2, hspace=0.3, wspace=0.3)
    
    colors = ['#FF1493', '#4169E1']
    batch_names = list(batch_data.keys())
    
    # 1. Distribution curves (main plot - larger)
    ax1 = fig.add_subplot(gs[0, :])
    for idx, (batch_name, scores) in enumerate(batch_data.items()):
        mean, std = np.mean(scores), np.std(scores)
        x = np.linspace(0, 1, 1000)
        y = (1 / (std * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x - mean) / std) ** 2)
        ax1.plot(x, y, linewidth=3, color=colors[idx], label=batch_name)
        ax1.fill_between(x, y, alpha=0.2, color=colors[idx])
        ax1.axvline(mean, color=colors[idx], linestyle='--', linewidth=1.5)
    ax1.set_xlabel('Score', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Density', fontsize=12, fontweight='bold')
    ax1.set_title('Normal Distribution Comparison', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3)
    
    # 2. Histogram comparison
    ax2 = fig.add_subplot(gs[1, 0])
    for idx, (batch_name, scores) in enumerate(batch_data.items()):
        ax2.hist(scores, bins=15, alpha=0.6, color=colors[idx], label=batch_name, edgecolor='black')
    ax2.set_xlabel('Score', fontsize=11, fontweight='bold')
    ax2.set_ylabel('Frequency', fontsize=11, fontweight='bold')
    ax2.set_title('Score Histograms', fontsize=12, fontweight='bold')
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3, axis='y')
    
    # 3. Box plots
    ax3 = fig.add_subplot(gs[1, 1])
    bp = ax3.boxplot([scores for scores in batch_data.values()], 
                      labels=batch_names, patch_artist=True,
                      boxprops=dict(linewidth=2),
                      medianprops=dict(linewidth=2, color='red'),
                      whiskerprops=dict(linewidth=1.5),
                      capprops=dict(linewidth=1.5))
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)
    ax3.set_ylabel('Score', fontsize=11, fontweight='bold')
    ax3.set_title('Score Distribution (Box Plot)', fontsize=12, fontweight='bold')
    ax3.grid(True, alpha=0.3, axis='y')
    
    # 4. Mean comparison with error bars
    ax4 = fig.add_subplot(gs[2, 0])
    means = [np.mean(scores) for scores in batch_data.values()]
    stds = [np.std(scores) for scores in batch_data.values()]
    x_pos = np.arange(len(batch_names))
    bars = ax4.bar(x_pos, means, yerr=stds, alpha=0.7, color=colors, 
                   capsize=10, width=0.6, edgecolor='black', linewidth=2)
    ax4.set_xticks(x_pos)
    ax4.set_xticklabels(batch_names)
    ax4.set_ylabel('Mean Score', fontsize=11, fontweight='bold')
    ax4.set_title('Mean Scores ± Std Dev', fontsize=12, fontweight='bold')
    ax4.grid(True, alpha=0.3, axis='y')
    ax4.set_ylim(0, 1)
    
    # 5. Statistics table
    ax5 = fig.add_subplot(gs[2, 1])
    ax5.axis('off')
    
    stats_text = "Statistical Summary\n" + "="*40 + "\n\n"
    for batch_name, scores in batch_data.items():
        stats_text += f"{batch_name}:\n"
        stats_text += f"  Mean:   {np.mean(scores):.3f}\n"
        stats_text += f"  Median: {np.median(scores):.3f}\n"
        stats_text += f"  Std:    {np.std(scores):.3f}\n"
        stats_text += f"  Min:    {np.min(scores):.3f}\n"
        stats_text += f"  Max:    {np.max(scores):.3f}\n\n"
    
    ax5.text(0.1, 0.9, stats_text, transform=ax5.transAxes,
            fontsize=10, verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    return fig


def main():
    """Run the demo"""
    print("=" * 80)
    print("EXPERIMENT PIPELINE DEMO")
    print("=" * 80)
    print("\nGenerating dummy data for two batches...")
    
    # Create batches
    batches = create_demo_batches()
    
    # Generate scores
    batch_data = {}
    for batch in batches:
        batch_name = batch["name"]
        difficulty = batch["prompts"][0]["metadata"]["difficulty"]
        scores = generate_dummy_scores(difficulty, len(batch["prompts"]))
        batch_data[batch_name] = scores
        
        print(f"\n{batch_name}:")
        print(f"  Samples: {len(scores)}")
        print(f"  Mean:    {np.mean(scores):.3f}")
        print(f"  Std:     {np.std(scores):.3f}")
    
    # Create plots
    print("\nGenerating visualizations...")
    
    # Simple distribution plot
    fig1 = plot_distributions(batch_data)
    plt.savefig('demo_distribution.png', dpi=300, bbox_inches='tight')
    print("  ✓ Saved: demo_distribution.png")
    
    # Comprehensive comparison
    fig2 = create_comprehensive_plots(batch_data)
    plt.savefig('demo_comprehensive.png', dpi=300, bbox_inches='tight')
    print("  ✓ Saved: demo_comprehensive.png")
    
    # Show plots
    plt.show()
    
    print("\n" + "=" * 80)
    print("DEMO COMPLETE!")
    print("=" * 80)
    print("\nThis demonstrates how your pipeline will visualize:")
    print("  • Different batches with different difficulty levels")
    print("  • Normal distribution curves (like your image)")
    print("  • Statistical comparisons")
    print("\nNext steps:")
    print("  1. Replace dummy score generation with real API calls")
    print("  2. Replace scoring function with your evaluation logic")
    print("  3. Create JSON batch files with your actual prompts")


if __name__ == "__main__":
    main()