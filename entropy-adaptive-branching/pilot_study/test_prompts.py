#!/usr/bin/env python3
"""
Quick test to verify new prompts produce expected branching behavior.
Tests 3 prompts (one per confidence level) before running full pilot.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
from eab import EntropyAdaptiveBranching


def test_prompt(eab, prompt, expected_level):
    """Test a single prompt and report results."""
    print(f"\n{'='*70}")
    print(f"Testing: {expected_level.upper()} Confidence")
    print(f"{'='*70}")
    print(f"Prompt: '{prompt}'")

    samples = eab.generate(
        prompt=prompt,
        max_new_tokens=50,
        temperature=0.8,
        use_chat_template=True,
        show_progress=False
    )

    # Get metrics
    all_branches = set()
    for sample in samples:
        all_branches.update(sample.get('branch_points', []))

    # Get entropy statistics
    entropy_stats = eab.entropy_tracker.get_statistics()
    avg_entropy = entropy_stats.get('mean_entropy', 0)
    max_entropy = entropy_stats.get('max_entropy', 0)

    print(f"\nResults:")
    print(f"  Samples generated: {len(samples)}")
    print(f"  Branching points: {len(all_branches)}")
    print(f"  Average entropy: {avg_entropy:.4f}")
    print(f"  Max entropy: {max_entropy:.4f}")

    print(f"\nSample outputs (first 3):")
    for i, sample in enumerate(samples[:3], 1):
        text = sample.get('generated_only', sample.get('text', 'N/A'))
        # Truncate if too long
        if len(text) > 100:
            text = text[:100] + '...'
        print(f"  {i}. {text}")

    if len(samples) > 3:
        print(f"  ... and {len(samples) - 3} more samples")

    # Assessment
    print(f"\nAssessment:")
    if expected_level == 'high':
        if len(samples) <= 3 and len(all_branches) <= 2:
            print("  ✓ PASS - Low branching as expected for high confidence")
        else:
            print(f"  ⚠ WARNING - More branching than expected ({len(samples)} samples, {len(all_branches)} branches)")
    elif expected_level == 'medium':
        if 3 <= len(samples) <= 20 and len(all_branches) >= 2:
            print("  ✓ PASS - Moderate branching as expected")
        else:
            print(f"  ⚠ WARNING - Branching outside expected range")
    elif expected_level == 'low':
        if len(samples) >= 8 and len(all_branches) >= 4:
            print("  ✓ PASS - High branching as expected for low confidence")
        else:
            print(f"  ⚠ WARNING - Less branching than expected")

    return {
        'samples': len(samples),
        'branches': len(all_branches),
        'avg_entropy': avg_entropy,
        'max_entropy': max_entropy
    }


def main():
    print("="*70)
    print("Quick Prompt Test - Verify Branching Behavior")
    print("="*70)
    print("\nThis tests 3 prompts (one per confidence level) to verify")
    print("the new prompts produce expected entropy/branching behavior.\n")

    # Initialize EAB
    print("[1/2] Initializing EAB...")
    eab = EntropyAdaptiveBranching(
        model_name='Qwen/Qwen2.5-3B-Instruct',
        entropy_threshold=0.1,  # Moderate threshold for testing
        branch_factor=3,
        max_paths=20,
        device='cuda',
        torch_dtype=torch.float16
    )
    print("  ✓ Model loaded with threshold=0.1\n")

    # Test prompts
    print("[2/2] Testing prompts...")

    test_prompts = [
        ("What is the capital of France?", "high"),
        ("Recommend one effective method for learning programming.", "medium"),
        ("Write a short poem about forbidden love.", "low")
    ]

    results = []
    for prompt, level in test_prompts:
        result = test_prompt(eab, prompt, level)
        result['level'] = level
        results.append(result)

    # Summary
    print(f"\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}\n")

    print("Entropy by confidence level:")
    for r in results:
        print(f"  {r['level'].upper():8s}: avg={r['avg_entropy']:.4f}, max={r['max_entropy']:.4f}, "
              f"samples={r['samples']}, branches={r['branches']}")

    print(f"\n{'='*70}")
    print("Assessment:")
    print(f"{'='*70}\n")

    high_result = results[0]
    med_result = results[1]
    low_result = results[2]

    if (high_result['avg_entropy'] < med_result['avg_entropy'] < low_result['avg_entropy']):
        print("✓ GOOD: Entropy increases with confidence level (high < medium < low)")
    else:
        print("⚠ WARNING: Entropy ordering not as expected")
        print("  This may indicate prompts need further tuning")

    if (high_result['samples'] < med_result['samples'] <= low_result['samples']):
        print("✓ GOOD: Sample count increases with uncertainty")
    else:
        print("⚠ WARNING: Sample counts don't follow expected pattern")

    print("\n" + "="*70)
    if all(r['avg_entropy'] > 0 for r in results):
        print("✓ Ready to run full pilot study!")
        print("\nNext: ./run_all.sh  (or python3 run_pilot.py)")
    else:
        print("⚠ Review results above before running full pilot study")
    print("="*70)


if __name__ == '__main__':
    main()
