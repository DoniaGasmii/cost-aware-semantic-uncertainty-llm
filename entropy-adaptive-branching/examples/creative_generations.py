"""
Creative Text Generation with Entropy-Adaptive Branching.

This example demonstrates using EAB for creative tasks where we expect
high branching rates due to diverse possibilities.
"""

from eab import EntropyAdaptiveBranching
from eab.utils import compute_diversity_metrics, save_results, print_generation_tree


def main():
    print("=" * 70)
    print("Creative Text Generation with Entropy-Adaptive Branching")
    print("=" * 70)
    
    # Initialize EAB with aggressive settings for creative generation
    eab = EntropyAdaptiveBranching(
        model_name="gpt2",
        entropy_threshold=0.3,  # Lower threshold = more branching
        branch_factor=4,  # More branches for diversity
        max_paths=20
    )
    
    # Creative prompts
    prompts = [
        "Once upon a time, in a magical forest,",
        "The last thing I expected to find in my backyard was",
        "She opened the mysterious box and found",
    ]
    
    for i, prompt in enumerate(prompts, 1):
        print(f"\n{'='*70}")
        print(f"Prompt {i}/{len(prompts)}")
        print('='*70)
        print(f"Prompt: '{prompt}'")
        
        # Generate creative completions
        results = eab.generate(
            prompt=prompt,
            max_new_tokens=40,
            temperature=1.2,  # Higher temperature for creativity
            seed=42 + i
        )
        
        # Display diverse samples
        print(f"\nTop 5 creative completions:")
        print("-" * 70)
        for j, result in enumerate(results[:5], 1):
            text = result['text'].strip()
            prob = result['probability']
            branches = result.get('num_branches', 0)
            print(f"\n{j}. (p={prob:.4f}, branches={branches})")
            print(f"   {text}")
        
        # Branching stats
        stats = eab.get_entropy_history()['statistics']
        print(f"\n{'='*70}")
        print("Branching Statistics")
        print('='*70)
        print(f"  Total paths generated: {len(results)}")
        print(f"  Total branch points: {stats['num_branches']}")
        print(f"  Branch rate: {stats['branch_rate']:.1%}")
        print(f"  Avg entropy: {stats['mean_entropy']:.3f}")
        print(f"  Max entropy: {stats['max_entropy']:.3f}")
        
        # Diversity metrics
        texts = [r['text'] for r in results]
        diversity = compute_diversity_metrics(texts)
        print(f"\nDiversity Metrics")
        print('-'*70)
        print(f"  Unique completions: {diversity['num_unique']}/{len(texts)}")
        print(f"  Unique ratio: {diversity['unique_ratio']:.1%}")
        print(f"  Vocabulary diversity: {diversity['vocab_diversity']:.3f}")
        
        # Show generation tree structure
        if i == 1:  # Only for first prompt to avoid clutter
            print(f"\n{'='*70}")
            print("Generation Tree Structure")
            print('='*70)
            # Convert results back to paths for tree visualization
            # Note: This is a simplified view
            print("(Showing first 3 paths)")
            for j, result in enumerate(results[:3], 1):
                branches = result.get('branch_points', [])
                print(f"Path {j}: {len(branches)} branch points at positions {branches}")
        
        # Plot entropy evolution
        if i == 1:
            print(f"\n{'='*70}")
            print("Entropy Evolution")
            print('='*70)
            try:
                eab.plot_entropy(figsize=(14, 6))
            except Exception as e:
                print(f"Could not display plot: {e}")
    
    # Overall summary
    print(f"\n{'='*70}")
    print("Creative Generation Summary")
    print('='*70)
    print("\nObservations:")
    print("  ✓ High branch rate (many creative possibilities)")
    print("  ✓ High entropy (diverse token distributions)")
    print("  ✓ High vocabulary diversity")
    print("  ✓ Still efficient: shared computation for early tokens")
    print("\nUse cases:")
    print("  • Story generation")
    print("  • Brainstorming")
    print("  • Multiple alternative completions")
    print("  • Exploring diverse continuations")


if __name__ == "__main__":
    main()