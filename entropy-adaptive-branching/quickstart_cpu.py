#!/usr/bin/env python3
"""
Quick start script for Entropy-Adaptive Branching (CPU-only version).

This version explicitly uses CPU to avoid CUDA issues.
Run this if you encounter CUDA library errors.
"""

import os
# Force CPU-only mode before importing torch
os.environ['CUDA_VISIBLE_DEVICES'] = ''

from eab import EntropyAdaptiveBranching


def main():
    print("=" * 70)
    print("Entropy-Adaptive Branching - Quick Start (CPU Mode)")
    print("=" * 70)
    print("\nThis script demonstrates basic EAB functionality using CPU.")
    print("Loading model (this may take a moment)...\n")
    
    # Initialize with explicit CPU device
    eab = EntropyAdaptiveBranching(
        model_name="gpt2",
        device="cpu",  # Explicitly use CPU
        entropy_threshold=0.4,
        branch_factor=3,
        max_paths=10
    )
    
    # Simple example
    prompt = "The capital of France is"
    print(f"\nGenerating completions for: '{prompt}'")
    
    results = eab.generate(
        prompt=prompt,
        max_new_tokens=10,
        temperature=0.8,
        seed=42
    )
    
    print(f"\nTop 5 completions:")
    print("-" * 70)
    for i, result in enumerate(results[:5], 1):
        print(f"{i}. {result['text']}")
        print(f"   Probability: {result['probability']:.4f}")
        print()
    
    # Show statistics
    stats = eab.get_entropy_history()['statistics']
    print("=" * 70)
    print("Statistics:")
    print("-" * 70)
    print(f"Total paths generated: {len(results)}")
    print(f"Average entropy: {stats['mean_entropy']:.3f}")
    print(f"Branch rate: {stats['branch_rate']:.1%}")
    print(f"Total branches: {stats['num_branches']}")
    
    print("\n" + "=" * 70)
    print("Success! EAB is working correctly on CPU.")
    print("=" * 70)
    print("\nNext steps:")
    print("  • Run examples: python examples/basic_usage.py")
    print("  • Try the notebook: jupyter notebook notebooks/tutorial.ipynb")
    print("  • Read the docs: see README.md")


if __name__ == "__main__":
    main()