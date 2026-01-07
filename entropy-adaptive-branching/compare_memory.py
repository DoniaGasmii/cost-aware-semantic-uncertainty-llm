"""
Simple script to compare memory usage between original and COW implementations.

This script runs both implementations on the same prompt and measures:
- Peak GPU memory usage
- Memory increase during generation
- Cache memory specifically
"""

import torch
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from eab.core import EntropyAdaptiveBranching as EAB_Original
from eab.core_cow import EntropyAdaptiveBranching as EAB_COW


def format_memory(bytes_val):
    """Format memory in MB."""
    return f"{bytes_val / 1024 / 1024:.1f} MB"


def measure_memory(implementation_name, eab_class, test_config):
    """Measure memory for a single implementation."""

    print(f"\n{'='*80}")
    print(f"Testing: {implementation_name}")
    print(f"{'='*80}")

    # Clear GPU memory
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

    mem_before = torch.cuda.memory_allocated() if torch.cuda.is_available() else 0
    print(f"Memory before loading: {format_memory(mem_before)}")

    # Load model
    eab = eab_class(
        test_config['model'],
        entropy_threshold=test_config['threshold'],
        branch_factor=test_config['branch_factor'],
        max_paths=test_config['max_paths'],
        torch_dtype=torch.float16
    )

    mem_after_load = torch.cuda.memory_allocated() if torch.cuda.is_available() else 0
    model_memory = mem_after_load - mem_before
    print(f"Memory after loading: {format_memory(mem_after_load)}")
    print(f"Model memory: {format_memory(model_memory)}")

    # Generate
    print(f"\nGenerating with prompt: '{test_config['prompt']}'")
    results = eab.generate(
        test_config['prompt'],
        max_new_tokens=test_config['max_tokens'],
        temperature=test_config['temperature'],
        seed=test_config['seed'],
        show_progress=False,
        use_chat_template=True
    )

    mem_after_gen = torch.cuda.memory_allocated() if torch.cuda.is_available() else 0
    peak_memory = torch.cuda.max_memory_allocated() if torch.cuda.is_available() else 0
    generation_memory = mem_after_gen - mem_after_load

    print(f"\nMemory after generation: {format_memory(mem_after_gen)}")
    print(f"Peak memory: {format_memory(peak_memory)}")
    print(f"Generation overhead: {format_memory(generation_memory)}")
    print(f"Generated {len(results)} samples")

    # Clean up
    del eab
    torch.cuda.empty_cache()

    return {
        'model_memory': model_memory,
        'generation_memory': generation_memory,
        'peak_memory': peak_memory,
        'total_memory': mem_after_gen,
        'num_samples': len(results)
    }


def main():
    """Run comparison."""

    print("="*80)
    print("Memory Comparison: Original vs Copy-on-Write (COW)")
    print("="*80)

    test_config = {
        'model': 'meta-llama/Llama-3.2-1B-Instruct',
        'prompt': 'What are the main benefits of renewable energy?',
        'threshold': 0.055,
        'branch_factor': 3,
        'max_paths': 20,
        'max_tokens': 40,
        'temperature': 0.8,
        'seed': 42
    }

    print(f"\nTest Configuration:")
    print(f"  Model: {test_config['model']}")
    print(f"  Prompt: {test_config['prompt']}")
    print(f"  Max paths: {test_config['max_paths']}")
    print(f"  Branch factor: {test_config['branch_factor']}")
    print(f"  Max tokens: {test_config['max_tokens']}")

    # Test original
    original_stats = measure_memory("Original (Deep Copy)", EAB_Original, test_config)

    # Test COW
    cow_stats = measure_memory("COW (Copy-on-Write)", EAB_COW, test_config)

    # Comparison
    print(f"\n{'='*80}")
    print("COMPARISON RESULTS")
    print(f"{'='*80}")

    print(f"\n📊 Memory Usage:")
    print(f"  Model memory:")
    print(f"    Original: {format_memory(original_stats['model_memory'])}")
    print(f"    COW:      {format_memory(cow_stats['model_memory'])}")

    print(f"\n  Generation overhead (cache + paths):")
    print(f"    Original: {format_memory(original_stats['generation_memory'])}")
    print(f"    COW:      {format_memory(cow_stats['generation_memory'])}")

    if original_stats['generation_memory'] > 0:
        savings = original_stats['generation_memory'] - cow_stats['generation_memory']
        savings_pct = (savings / original_stats['generation_memory']) * 100
        print(f"    Savings:  {format_memory(savings)} ({savings_pct:.1f}%)")

    print(f"\n  Peak memory:")
    print(f"    Original: {format_memory(original_stats['peak_memory'])}")
    print(f"    COW:      {format_memory(cow_stats['peak_memory'])}")

    if original_stats['peak_memory'] > 0:
        peak_savings = original_stats['peak_memory'] - cow_stats['peak_memory']
        peak_savings_pct = (peak_savings / original_stats['peak_memory']) * 100
        print(f"    Savings:  {format_memory(peak_savings)} ({peak_savings_pct:.1f}%)")

    print(f"\n📈 Generation:")
    print(f"  Samples generated:")
    print(f"    Original: {original_stats['num_samples']}")
    print(f"    COW:      {cow_stats['num_samples']}")

    # Summary
    print(f"\n{'='*80}")
    print("SUMMARY")
    print(f"{'='*80}")

    if original_stats['generation_memory'] > 0:
        if savings_pct > 50:
            verdict = "🎉 EXCELLENT - Major memory savings!"
        elif savings_pct > 30:
            verdict = "✅ GOOD - Significant savings"
        elif savings_pct > 10:
            verdict = "👍 MODERATE - Some savings"
        elif savings_pct > 0:
            verdict = "✅ POSITIVE - Modest savings"
        else:
            verdict = "⚠️ LIMITED - Minimal difference"

        print(f"\n{verdict}")
        print(f"Generation overhead reduced by {savings_pct:.1f}%")
        print(f"Peak memory reduced by {peak_savings_pct:.1f}%")
    else:
        print("\n⚠️  Could not measure savings (generation overhead was 0)")

    print(f"\nBoth implementations generated {cow_stats['num_samples']} samples successfully.")


if __name__ == "__main__":
    main()
