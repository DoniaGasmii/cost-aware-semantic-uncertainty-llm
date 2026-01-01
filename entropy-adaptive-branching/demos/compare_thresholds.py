#!/usr/bin/env python3
"""
Compare Thresholds - Parameter Sweep

Tests EAB with different entropy thresholds to understand their effect
on sample count, branching behavior, and resource usage.

Usage:
    python compare_thresholds.py --prompt "Your prompt here"
    python compare_thresholds.py --prompt "The capital of France is" --thresholds 0.2 0.3 0.4 0.5
"""

import sys
import argparse
import time
import traceback
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from eab import EntropyAdaptiveBranching


DEFAULT_THRESHOLDS = [0.2, 0.3, 0.4, 0.5, 0.6]


def test_threshold(prompt, threshold, branch_factor=3, max_tokens=20, max_paths=20, temperature=0.8):
    """Test EAB with a specific threshold."""
    print(f"\n{'='*70}")
    print(f"Testing threshold: {threshold}")
    print(f"{'='*70}")

    try:
        # Initialize EAB with this threshold
        eab = EntropyAdaptiveBranching(
            model_name='Qwen/Qwen2.5-3B-Instruct',
            entropy_threshold=threshold,
            branch_factor=branch_factor,
            max_paths=max_paths,
            device='cpu'
        )

        # Generate
        start_time = time.time()
        samples = eab.generate(
            prompt=prompt,
            max_new_tokens=max_tokens,
            temperature=temperature
        )
        elapsed_time = time.time() - start_time

        # Extract metrics
        num_samples = len(samples)
        all_branch_points = set()
        total_tokens = 0

        for sample in samples:
            all_branch_points.update(sample.get('branch_points', []))
            total_tokens += sample.get('length', len(sample.get('tokens', [])))

        num_branches = len(all_branch_points)

        print(f"\nResults:")
        print(f"  Samples generated: {num_samples}")
        print(f"  Total branches: {num_branches}")
        print(f"  Branch positions: {sorted(all_branch_points)[:10]}{'...' if len(all_branch_points) > 10 else ''}")
        print(f"  Total tokens: {total_tokens}")
        print(f"  Time: {elapsed_time:.2f}s")

        # Show sample diversity (first 3)
        print(f"\n  Sample previews:")
        for i, sample in enumerate(samples[:3], 1):
            text = sample.get('text', sample.get('generated_only', 'N/A'))
            if len(text) > 80:
                text = text[:80] + '...'
            print(f"    {i}. {text}")

        if len(samples) > 3:
            print(f"    ... and {len(samples) - 3} more")

        return {
            'threshold': threshold,
            'num_samples': num_samples,
            'num_branches': num_branches,
            'total_tokens': total_tokens,
            'time': elapsed_time,
            'branch_positions': sorted(all_branch_points)
        }

    except Exception as e:
        print(f"\n  ✗ ERROR: {e}")
        import traceback
        traceback.print_exc()
        return {
            'threshold': threshold,
            'error': str(e)
        }


def display_comparison(results):
    """Display comparison table of all thresholds."""
    print(f"\n{'='*70}")
    print("  COMPARISON SUMMARY")
    print(f"{'='*70}\n")

    # Filter out errors
    valid_results = [r for r in results if 'error' not in r]

    if not valid_results:
        print("  ✗ No valid results to compare")
        return

    # Table header
    print(f"{'Threshold':<12} {'Samples':<10} {'Branches':<10} {'Tokens':<10} {'Time (s)':<10}")
    print("-" * 70)

    for result in valid_results:
        print(f"{result['threshold']:<12.1f} "
              f"{result['num_samples']:<10} "
              f"{result['num_branches']:<10} "
              f"{result['total_tokens']:<10} "
              f"{result['time']:<10.2f}")

    print("\n" + "="*70)

    # Analysis
    print("\nAnalysis:")

    # Find threshold with most branching
    max_branches = max(valid_results, key=lambda r: r['num_branches'])
    print(f"  Most branching: threshold={max_branches['threshold']} "
          f"({max_branches['num_branches']} branches, {max_branches['num_samples']} samples)")

    # Find threshold with least branching
    min_branches = min(valid_results, key=lambda r: r['num_branches'])
    print(f"  Least branching: threshold={min_branches['threshold']} "
          f"({min_branches['num_branches']} branches, {min_branches['num_samples']} samples)")

    # Find most efficient (least tokens for similar samples)
    min_tokens = min(valid_results, key=lambda r: r['total_tokens'])
    print(f"  Most efficient: threshold={min_tokens['threshold']} "
          f"({min_tokens['total_tokens']} tokens, {min_tokens['num_samples']} samples)")

    print("\nRecommendations:")
    print(f"  - For maximum diversity: Use threshold {max_branches['threshold']}")
    print(f"  - For efficiency: Use threshold {min_tokens['threshold']}")
    print(f"  - For balanced approach: Use threshold 0.4 (default)")

    print("="*70 + "\n")


def main():
    parser = argparse.ArgumentParser(description='Compare EAB thresholds')
    parser.add_argument('--prompt', type=str, required=True, help='Input prompt to test')
    parser.add_argument('--thresholds', type=float, nargs='+', default=DEFAULT_THRESHOLDS,
                        help='List of thresholds to test (default: 0.2 0.3 0.4 0.5 0.6)')
    parser.add_argument('--branch-factor', type=int, default=3, help='Branch factor (default: 3)')
    parser.add_argument('--max-tokens', type=int, default=20, help='Max new tokens (default: 20)')
    parser.add_argument('--max-paths', type=int, default=20, help='Max paths (default: 20)')
    parser.add_argument('--temperature', type=float, default=0.8, help='Temperature (default: 0.8)')

    args = parser.parse_args()

    print("\n" + "="*70)
    print("  EAB Threshold Comparison")
    print("="*70)
    print(f"\nPrompt: '{args.prompt}'")
    print(f"Thresholds to test: {args.thresholds}")
    print(f"Branch factor: {args.branch_factor}")
    print(f"Max tokens: {args.max_tokens}")
    print(f"Temperature: {args.temperature}")

    # Run tests for each threshold
    results = []
    for threshold in args.thresholds:
        result = test_threshold(
            prompt=args.prompt,
            threshold=threshold,
            branch_factor=args.branch_factor,
            max_tokens=args.max_tokens,
            max_paths=args.max_paths,
            temperature=args.temperature
        )
        results.append(result)

    # Display comparison
    display_comparison(results)


if __name__ == '__main__':
    main()
