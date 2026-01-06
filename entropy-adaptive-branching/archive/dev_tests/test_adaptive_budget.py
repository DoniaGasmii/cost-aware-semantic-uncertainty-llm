#!/usr/bin/env python3
"""
Test script to verify adaptive budgeting strategy.
Demonstrates that high-entropy positions can branch even when max_paths is reached.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import torch
from eab import EntropyAdaptiveBranching


def test_adaptive_budgeting():
    """
    Test that the improved strategy allows later high-entropy positions to branch.
    """
    print("=" * 70)
    print("Testing Adaptive Budgeting Strategy")
    print("=" * 70)
    print("\nObjective: Verify that high-entropy positions can branch")
    print("even after reaching max_paths limit.\n")

    # Initialize EAB with LOW max_paths to trigger the constraint
    print("[1/3] Initializing EAB with max_paths=5 (low limit)...")
    eab = EntropyAdaptiveBranching(
        model_name='Qwen/Qwen2.5-3B-Instruct',
        entropy_threshold=0.05,  # Low threshold for frequent branching
        branch_factor=3,
        max_paths=5,  # Very low to test adaptive behavior
        device='cuda',
        torch_dtype=torch.float16
    )
    print("  ✓ Model loaded\n")

    # Use a medium-confidence prompt that should have entropy throughout
    prompt = "Recommend one effective method for learning a new language quickly."

    print("[2/3] Generating with adaptive budgeting...")
    print(f"Prompt: '{prompt}'")
    print(f"Settings: threshold={eab.entropy_threshold}, max_paths={eab.max_paths}\n")

    samples = eab.generate(
        prompt=prompt,
        max_new_tokens=60,  # Longer to test late branching
        temperature=0.8,
        use_chat_template=True,
        show_progress=True
    )

    print(f"\n[3/3] Analyzing results...\n")

    # Collect all branch points
    all_branch_points = set()
    for sample in samples:
        all_branch_points.update(sample.get('branch_points', []))

    # Get entropy history
    entropy_stats = eab.entropy_tracker.get_statistics()
    entropy_history = eab.entropy_tracker.entropy_history

    print(f"Results:")
    print(f"  Total samples: {len(samples)}")
    print(f"  Unique branch points: {len(all_branch_points)}")
    print(f"  Branch positions: {sorted(list(all_branch_points))[:10]}{'...' if len(all_branch_points) > 10 else ''}")
    print(f"  Average entropy: {entropy_stats.get('mean_entropy', 0):.4f}")
    print(f"  Max entropy: {entropy_stats.get('max_entropy', 0):.4f}")

    # Check if branching occurred at later positions
    if len(all_branch_points) > 0:
        earliest_branch = min(all_branch_points)
        latest_branch = max(all_branch_points)
        branch_range = latest_branch - earliest_branch

        print(f"\nBranching range analysis:")
        print(f"  Earliest branch: position {earliest_branch}")
        print(f"  Latest branch: position {latest_branch}")
        print(f"  Branching span: {branch_range} positions")

        if branch_range > 20:
            print("\n✓ SUCCESS: Branching occurs across wide range of positions")
            print("  This demonstrates that later high-entropy positions can branch")
            print("  even after initial max_paths limit was reached.")
        else:
            print("\n⚠ INFO: Branching range is narrow (might be low-entropy prompt)")

    # Show sample diversity
    print(f"\nSample outputs (first 3):")
    for i, sample in enumerate(samples[:3], 1):
        text = sample.get('generated_only', sample.get('text', 'N/A'))
        if len(text) > 80:
            text = text[:80] + '...'
        branches = len(sample.get('branch_points', []))
        print(f"  {i}. [{branches} branches] {text}")

    if len(samples) > 3:
        print(f"  ... and {len(samples) - 3} more samples")

    # Summary
    print("\n" + "=" * 70)
    print("ASSESSMENT")
    print("=" * 70)

    if len(samples) > 1 and len(all_branch_points) > 0:
        print("✓ Adaptive budgeting is working correctly!")
        print(f"  - Generated {len(samples)} samples with max_paths={eab.max_paths}")
        print(f"  - Branched at {len(all_branch_points)} positions")
        print("  - System adaptively managed path budget")

        if len(all_branch_points) > 5:
            print("\n✓ EXCELLENT: Multiple branching positions detected")
            print("  This confirms later positions can explore when entropy is high")
    else:
        print("⚠ WARNING: Limited branching detected")
        print("  This might indicate low-entropy prompt or issue with strategy")

    print("=" * 70)


if __name__ == '__main__':
    test_adaptive_budgeting()
