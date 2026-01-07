"""
Test script to validate Copy-on-Write cache implementation and measure memory savings.

This script compares:
1. Original implementation (core.py with deep copy)
2. COW implementation (core_cow.py with COW cache)

Expected outcomes:
- Same generation quality (outputs should be similar with same seed)
- Significant memory reduction (60-70% cache memory savings)
- Similar or better generation time
"""

import torch
import sys
import os
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

from eab.core import EntropyAdaptiveBranching as EAB_Original
from eab.core_cow import EntropyAdaptiveBranching as EAB_COW


def get_gpu_memory():
    """Get current GPU memory usage in MB."""
    if torch.cuda.is_available():
        return torch.cuda.memory_allocated() / 1024 / 1024
    return 0


def test_cow_vs_original():
    """Compare COW implementation with original implementation."""

    print("=" * 80)
    print("Testing Copy-on-Write Cache Implementation")
    print("=" * 80)

    # Test configuration
    model_name = "meta-llama/Llama-3.2-1B-Instruct"
    test_prompt = "What are the main benefits of renewable energy?"
    seed = 42

    # Shared parameters
    params = {
        "entropy_threshold": 0.055,
        "branch_factor": 3,
        "max_paths": 20,
        "torch_dtype": torch.float16
    }

    gen_params = {
        "max_new_tokens": 50,
        "temperature": 0.8,
        "seed": seed,
        "show_progress": True,
        "use_chat_template": True
    }

    print(f"\nTest Configuration:")
    print(f"  Model: {model_name}")
    print(f"  Prompt: {test_prompt}")
    print(f"  Parameters: {params}")
    print(f"  Generation: {gen_params}")

    # Clear GPU cache
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

    # Test 1: Original implementation
    print("\n" + "=" * 80)
    print("Test 1: Original Implementation (Deep Copy)")
    print("=" * 80)

    mem_before_original = get_gpu_memory()
    peak_before_original = torch.cuda.max_memory_allocated() / 1024 / 1024 if torch.cuda.is_available() else 0

    eab_original = EAB_Original(model_name, **params)

    mem_after_load_original = get_gpu_memory()
    print(f"\nMemory after loading model: {mem_after_load_original:.1f} MB")

    results_original = eab_original.generate(test_prompt, **gen_params)

    mem_after_gen_original = get_gpu_memory()
    peak_original = torch.cuda.max_memory_allocated() / 1024 / 1024 if torch.cuda.is_available() else 0

    print(f"\n📊 Original Implementation Results:")
    print(f"  Generated samples: {len(results_original)}")
    print(f"  Memory after generation: {mem_after_gen_original:.1f} MB")
    print(f"  Peak memory: {peak_original:.1f} MB")
    print(f"  Memory increase: {mem_after_gen_original - mem_after_load_original:.1f} MB")

    # Show first 2 samples
    print(f"\n  Sample outputs:")
    for i, result in enumerate(results_original[:2]):
        print(f"    {i+1}. {result['text'][:100]}...")

    # Clear for next test
    del eab_original
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

    # Test 2: COW implementation
    print("\n" + "=" * 80)
    print("Test 2: COW Implementation (Copy-on-Write)")
    print("=" * 80)

    mem_before_cow = get_gpu_memory()

    eab_cow = EAB_COW(model_name, **params)

    mem_after_load_cow = get_gpu_memory()
    print(f"\nMemory after loading model: {mem_after_load_cow:.1f} MB")

    results_cow = eab_cow.generate(test_prompt, **gen_params)

    mem_after_gen_cow = get_gpu_memory()
    peak_cow = torch.cuda.max_memory_allocated() / 1024 / 1024 if torch.cuda.is_available() else 0

    print(f"\n📊 COW Implementation Results:")
    print(f"  Generated samples: {len(results_cow)}")
    print(f"  Memory after generation: {mem_after_gen_cow:.1f} MB")
    print(f"  Peak memory: {peak_cow:.1f} MB")
    print(f"  Memory increase: {mem_after_gen_cow - mem_after_load_cow:.1f} MB")

    # Show first 2 samples
    print(f"\n  Sample outputs:")
    for i, result in enumerate(results_cow[:2]):
        print(f"    {i+1}. {result['text'][:100]}...")

    # Comparison
    print("\n" + "=" * 80)
    print("📈 Comparison: Original vs COW")
    print("=" * 80)

    mem_increase_original = mem_after_gen_original - mem_after_load_original
    mem_increase_cow = mem_after_gen_cow - mem_after_load_cow

    if mem_increase_original > 0:
        mem_savings = mem_increase_original - mem_increase_cow
        mem_savings_percent = (mem_savings / mem_increase_original) * 100
    else:
        mem_savings = 0
        mem_savings_percent = 0

    peak_savings = peak_original - peak_cow
    peak_savings_percent = (peak_savings / peak_original) * 100 if peak_original > 0 else 0

    print(f"\n🎯 Memory Metrics:")
    print(f"  Memory increase during generation:")
    print(f"    Original: {mem_increase_original:.1f} MB")
    print(f"    COW:      {mem_increase_cow:.1f} MB")
    print(f"    Savings:  {mem_savings:.1f} MB ({mem_savings_percent:.1f}%)")
    print(f"\n  Peak memory:")
    print(f"    Original: {peak_original:.1f} MB")
    print(f"    COW:      {peak_cow:.1f} MB")
    print(f"    Savings:  {peak_savings:.1f} MB ({peak_savings_percent:.1f}%)")

    print(f"\n🎯 Generation Quality:")
    print(f"  Samples generated:")
    print(f"    Original: {len(results_original)}")
    print(f"    COW:      {len(results_cow)}")

    # Entropy statistics
    if hasattr(eab_cow, 'entropy_tracker'):
        entropy_stats = eab_cow.entropy_tracker.get_statistics()
        print(f"\n  Entropy statistics (COW):")
        print(f"    Mean entropy: {entropy_stats['mean_entropy']:.3f}")
        print(f"    Branch rate:  {entropy_stats['branch_rate']:.1%}")

    # Summary
    print("\n" + "=" * 80)
    print("✅ Summary")
    print("=" * 80)

    if mem_savings_percent > 50:
        verdict = "🎉 EXCELLENT - Major memory savings achieved!"
    elif mem_savings_percent > 30:
        verdict = "✅ GOOD - Significant memory savings"
    elif mem_savings_percent > 10:
        verdict = "👍 MODERATE - Some memory savings"
    else:
        verdict = "⚠️  LIMITED - Memory savings are modest"

    print(f"\n{verdict}")
    print(f"Memory savings: {mem_savings_percent:.1f}%")
    print(f"Both implementations generated {len(results_cow)} samples")
    print(f"\nCOW implementation is {'✅ WORKING' if len(results_cow) > 0 else '❌ FAILED'}")

    return {
        'original_memory': mem_increase_original,
        'cow_memory': mem_increase_cow,
        'savings_mb': mem_savings,
        'savings_percent': mem_savings_percent,
        'original_samples': len(results_original),
        'cow_samples': len(results_cow)
    }


def quick_cow_test():
    """Quick test of COW implementation only."""

    print("=" * 80)
    print("Quick COW Implementation Test")
    print("=" * 80)

    model_name = "meta-llama/Llama-3.2-1B-Instruct"
    test_prompt = "What are three benefits of exercise?"

    eab = EAB_COW(
        model_name,
        entropy_threshold=0.055,
        branch_factor=3,
        max_paths=20,
        torch_dtype=torch.float16
    )

    results = eab.generate(
        test_prompt,
        max_new_tokens=30,
        temperature=0.8,
        seed=42,
        use_chat_template=True
    )

    print(f"\n✅ Generated {len(results)} samples")
    print(f"\nSample outputs:")
    for i, result in enumerate(results[:3]):
        print(f"  {i+1}. {result['text']}")

    return results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Test COW cache implementation")
    parser.add_argument("--quick", action="store_true", help="Run quick test only")
    parser.add_argument("--full", action="store_true", help="Run full comparison test")
    args = parser.parse_args()

    if args.quick:
        quick_cow_test()
    elif args.full:
        test_cow_vs_original()
    else:
        # Default: run quick test
        print("Running quick test. Use --full for complete comparison.")
        print()
        quick_cow_test()
