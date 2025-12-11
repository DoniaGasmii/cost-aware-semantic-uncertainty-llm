"""
Basic usage example for Entropy-Adaptive Branching.

This example demonstrates the simplest way to use EAB for multi-sample generation.
"""

from eab import EntropyAdaptiveBranching


def main():
    # Initialize EAB with a small model
    print("=" * 60)
    print("Basic Usage Example: Entropy-Adaptive Branching")
    print("=" * 60)
    
    eab = EntropyAdaptiveBranching(
        model_name="gpt2",  # Small model for quick testing
        entropy_threshold=0.4,
        branch_factor=3,
        max_paths=20
    )
    
    # Example 1: Factual question (low uncertainty)
    print("\n" + "=" * 60)
    print("Example 1: Factual Question")
    print("=" * 60)
    
    prompt1 = "The capital of France is"
    results1 = eab.generate(
        prompt=prompt1,
        max_new_tokens=10,
        temperature=0.8,
        seed=42
    )
    
    print("\nTop 3 completions:")
    for i, result in enumerate(results1[:3], 1):
        print(f"{i}. {result['text']} (p={result['probability']:.4f})")
    
    print(f"\nBranching statistics:")
    print(f"  Total paths: {len(results1)}")
    stats1 = eab.get_entropy_history()['statistics']
    print(f"  Branch rate: {stats1['branch_rate']:.1%}")
    print(f"  Expected: Low branching for factual questions")
    
    # Example 2: Creative prompt (high uncertainty)
    print("\n" + "=" * 60)
    print("Example 2: Creative Prompt")
    print("=" * 60)
    
    prompt2 = "Once upon a time, in a land far away,"
    results2 = eab.generate(
        prompt=prompt2,
        max_new_tokens=30,
        temperature=1.0,
        seed=42
    )
    
    print("\nTop 5 completions:")
    for i, result in enumerate(results2[:5], 1):
        print(f"{i}. {result['text'][:100]}... (p={result['probability']:.4f})")
    
    print(f"\nBranching statistics:")
    print(f"  Total paths: {len(results2)}")
    stats2 = eab.get_entropy_history()['statistics']
    print(f"  Branch rate: {stats2['branch_rate']:.1%}")
    print(f"  Expected: Higher branching for creative prompts")
    
    # Example 3: Ambiguous question (medium uncertainty)
    print("\n" + "=" * 60)
    print("Example 3: Ambiguous Question")
    print("=" * 60)
    
    prompt3 = "The best programming language for beginners is"
    results3 = eab.generate(
        prompt=prompt3,
        max_new_tokens=20,
        temperature=0.9,
        seed=42
    )
    
    print("\nTop 5 completions:")
    for i, result in enumerate(results3[:5], 1):
        print(f"{i}. {result['text']} (p={result['probability']:.4f})")
    
    print(f"\nBranching statistics:")
    print(f"  Total paths: {len(results3)}")
    stats3 = eab.get_entropy_history()['statistics']
    print(f"  Branch rate: {stats3['branch_rate']:.1%}")
    
    # Plot entropy for the last example
    print("\n" + "=" * 60)
    print("Entropy Visualization")
    print("=" * 60)
    print("Displaying entropy plot for Example 3...")
    try:
        eab.plot_entropy()
    except Exception as e:
        print(f"Could not display plot: {e}")
        print("(This is normal if running without display)")
    
    print("\n" + "=" * 60)
    print("Examples complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()