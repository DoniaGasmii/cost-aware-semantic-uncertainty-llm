#!/usr/bin/env python3
"""
Quick Test - Validate EAB Behavior

Tests EAB on predefined prompts with different confidence levels:
- High confidence: Should produce few samples (1-3)
- Medium confidence: Should produce moderate samples (5-10)
- Low confidence: Should produce many samples (10-20)

Usage:
    python quick_test.py
"""

import sys
import traceback
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from eab import EntropyAdaptiveBranching


# Test prompts with expected behavior
TEST_CASES = [
    {
        'name': 'High Confidence - Math',
        'prompt': '2 + 2 =',
        'expected_samples': (1, 3),
        'expected_branches': (0, 2),
        'description': 'Deterministic answer, should have minimal branching'
    },
    {
        'name': 'High Confidence - Factual',
        'prompt': 'The capital of France is',
        'expected_samples': (1, 5),
        'expected_branches': (0, 3),
        'description': 'Well-known fact, low uncertainty'
    },
    {
        'name': 'Medium Confidence - Completion',
        'prompt': 'The best way to learn programming is',
        'expected_samples': (5, 15),
        'expected_branches': (3, 10),
        'description': 'Multiple valid answers, moderate uncertainty'
    },
    {
        'name': 'Low Confidence - Creative',
        'prompt': 'Once upon a time in a magical forest',
        'expected_samples': (10, 20),
        'expected_branches': (5, 20),
        'description': 'Creative prompt, high uncertainty and many continuations'
    },
    {
        'name': 'Low Confidence - Opinion',
        'prompt': 'The most important quality in a leader is',
        'expected_samples': (8, 20),
        'expected_branches': (4, 15),
        'description': 'Subjective topic, many valid perspectives'
    }
]


def run_test_case(eab, test_case):
    """Run a single test case and return results."""
    print(f"\n{'='*70}")
    print(f"Test: {test_case['name']}")
    print(f"{'='*70}")
    print(f"Prompt: '{test_case['prompt']}'")
    print(f"Expected: {test_case['description']}")

    try:
        # Generate samples
        samples = eab.generate(
            prompt=test_case['prompt'],
            max_new_tokens=50,  # Increased for more branching opportunities
            temperature=0.9
        )

        # Extract metrics
        num_samples = len(samples)
        all_branch_points = set()
        for sample in samples:
            all_branch_points.update(sample.get('branch_points', []))
        num_branches = len(all_branch_points)

        # Check against expectations
        sample_min, sample_max = test_case['expected_samples']
        branch_min, branch_max = test_case['expected_branches']

        samples_ok = sample_min <= num_samples <= sample_max
        branches_ok = branch_min <= num_branches <= branch_max

        # Display results
        print(f"\nResults:")
        print(f"  Samples: {num_samples} {'✓' if samples_ok else '✗ (expected ' + str(test_case['expected_samples']) + ')'}")
        print(f"  Branches: {num_branches} {'✓' if branches_ok else '✗ (expected ' + str(test_case['expected_branches']) + ')'}")

        if num_branches > 0:
            print(f"  Branch positions: {sorted(all_branch_points)[:10]}{'...' if len(all_branch_points) > 10 else ''}")

        # Show first 3 samples
        print(f"\n  Sample outputs (showing first 3):")
        for i, sample in enumerate(samples[:3], 1):
            text = sample.get('text', sample.get('generated_only', 'N/A'))
            # Truncate if too long
            if len(text) > 100:
                text = text[:100] + '...'
            print(f"    {i}. {text}")

        if len(samples) > 3:
            print(f"    ... and {len(samples) - 3} more samples")

        return {
            'name': test_case['name'],
            'passed': samples_ok and branches_ok,
            'num_samples': num_samples,
            'num_branches': num_branches
        }

    except Exception as e:
        print(f"\n  ✗ ERROR: {e}")
        import traceback
        traceback.print_exc()
        return {
            'name': test_case['name'],
            'passed': False,
            'num_samples': 0,
            'num_branches': 0,
            'error': str(e)
        }


def main():
    print("\n" + "="*70)
    print("  EAB Quick Test - Validate Branching Behavior")
    print("="*70)

    print("\nInitializing EAB...")
    print("  Model: Qwen/Qwen2.5-3B-Instruct")
    print("  Threshold: 0.1 (lower for more branching)")
    print("  Branch factor: 3")
    print("  Max tokens: 50")

    try:
        import torch
        eab = EntropyAdaptiveBranching(
            model_name='Qwen/Qwen2.5-3B-Instruct',
            entropy_threshold=0.1,  # Lower threshold for more branching
            branch_factor=3,        # Increased from 2
            max_paths=20,           # Increased from 15
            device='cuda',
            torch_dtype=torch.float16  # Use FP16 to reduce memory by ~50%
        )
        print("  ✓ EAB initialized successfully")
    except Exception as e:
        print(f"  ✗ Failed to initialize EAB: {e}")
        return

    # Run all test cases
    results = []
    for test_case in TEST_CASES:
        result = run_test_case(eab, test_case)
        results.append(result)

    # Summary
    print(f"\n{'='*70}")
    print("  SUMMARY")
    print(f"{'='*70}\n")

    passed = sum(1 for r in results if r['passed'])
    total = len(results)

    print(f"Tests passed: {passed}/{total}\n")

    for result in results:
        status = "✓ PASS" if result['passed'] else "✗ FAIL"
        print(f"  {status} - {result['name']}")
        print(f"         Samples: {result['num_samples']}, Branches: {result['num_branches']}")

    print(f"\n{'='*70}")

    if passed == total:
        print("  ✓ All tests passed! EAB is working as expected.")
    else:
        print(f"  ⚠ {total - passed} test(s) failed. Review results above.")
        print("\nPossible issues:")
        print("  - Threshold too high (try 0.1 or 0.15)")
        print("  - Temperature too low (should be >= 0.7)")
        print("  - Model is too confident (try larger model)")

    print("="*70 + "\n")


if __name__ == '__main__':
    main()
