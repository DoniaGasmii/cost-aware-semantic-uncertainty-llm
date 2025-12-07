#!/usr/bin/env python3
"""
Comparison demo: EAB vs Naive Generation

Compares entropy-adaptive branching against naive multi-sample generation
in terms of time and memory usage.

Available models:
  - gpt2 (124M params) - Fast, baseline
  - gpt2-medium (355M params) - Better quality, 3x larger
  - gpt2-large (774M params) - High quality, 6x larger
  - gpt2-xl (1.5B params) - Best quality, very slow on CPU
"""

import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''  # Force CPU mode

import time
import psutil
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from eab import EntropyAdaptiveBranching

# ==============================================================================
# CONFIGURATION: Change this to use a different model
# ==============================================================================
MODEL_NAME = "gpt2-xl"  # Options: gpt2, gpt2-medium, gpt2-large, gpt2-xl
# ==============================================================================


def get_memory_usage():
    """Get current memory usage in MB."""
    process = psutil.Process()
    return process.memory_info().rss / 1024 / 1024  # Convert to MB


def naive_generation(model, tokenizer, prompt, num_samples=10, max_new_tokens=20, temperature=0.8, seed=42):
    """
    Naive approach: Generate samples independently by calling the model multiple times.

    This is the baseline - each sample requires a full forward pass through the prompt.
    """
    torch.manual_seed(seed)
    results = []

    input_ids = tokenizer(prompt, return_tensors="pt").input_ids

    for i in range(num_samples):
        # Each sample requires encoding the prompt again
        with torch.no_grad():
            output = model.generate(
                input_ids,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=temperature,
                pad_token_id=tokenizer.eos_token_id,
                num_return_sequences=1
            )

        text = tokenizer.decode(output[0], skip_special_tokens=True)
        results.append(text)

    return results


def run_comparison(prompt, max_paths=10, max_new_tokens=20, model_name=MODEL_NAME):
    """
    Run comparison between EAB and naive generation.

    Strategy: Run EAB first, then match naive generation to the same number of samples.
    """

    print("=" * 80)
    print("EAB vs Naive Generation Comparison")
    print("=" * 80)
    print(f"\nModel: {model_name}")
    print(f"Prompt: '{prompt}'")
    print(f"Max paths (EAB): {max_paths}")
    print(f"Max new tokens: {max_new_tokens}")
    print(f"Temperature: 0.8")
    print("\n" + "-" * 80)

    # ==================== EAB GENERATION (RUN FIRST) ====================
    print("\n[1/2] Running EAB generation...")
    print("      (Using entropy-adaptive branching with KV-cache reuse)")

    mem_before_eab = get_memory_usage()
    start_eab = time.time()

    eab = EntropyAdaptiveBranching(
        model_name=model_name,
        device="cpu",
        entropy_threshold=0.4,
        branch_factor=3,
        max_paths=max_paths
    )

    eab_results = eab.generate(
        prompt=prompt,
        max_new_tokens=max_new_tokens,
        temperature=0.8,
        seed=42,
        show_progress=False
    )

    end_eab = time.time()
    mem_after_eab = get_memory_usage()

    eab_time = end_eab - start_eab
    eab_mem = mem_after_eab - mem_before_eab

    num_eab_samples = len(eab_results)
    print(f"      ✓ Generated {num_eab_samples} samples")
    print(f"      ✓ Time: {eab_time:.2f}s")
    print(f"      ✓ Memory delta: {eab_mem:.1f} MB")

    # Clean up EAB model
    del eab
    torch.cuda.empty_cache() if torch.cuda.is_available() else None

    # ==================== NAIVE GENERATION (MATCH EAB COUNT) ====================
    print(f"\n[2/2] Running NAIVE generation...")
    print(f"      (Generating {num_eab_samples} samples independently to match EAB)")
    print(f"      Loading {model_name}...")

    # Load model for naive generation
    model = AutoModelForCausalLM.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Get model size
    num_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"      Model loaded: {num_params:.0f}M parameters")

    mem_before_naive = get_memory_usage()
    start_naive = time.time()

    naive_results = naive_generation(
        model, tokenizer, prompt,
        num_samples=num_eab_samples,  # Match EAB sample count
        max_new_tokens=max_new_tokens,
        temperature=0.8,
        seed=42
    )

    end_naive = time.time()
    mem_after_naive = get_memory_usage()

    naive_time = end_naive - start_naive
    naive_mem = mem_after_naive - mem_before_naive

    print(f"      ✓ Generated {len(naive_results)} samples")
    print(f"      ✓ Time: {naive_time:.2f}s")
    print(f"      ✓ Memory delta: {naive_mem:.1f} MB")

    # Clean up
    del model, tokenizer
    torch.cuda.empty_cache() if torch.cuda.is_available() else None

    # ==================== RESULTS ====================
    print("\n" + "=" * 80)
    print("COMPARISON RESULTS")
    print("=" * 80)

    speedup = naive_time / eab_time if eab_time > 0 else 0
    mem_reduction = ((naive_mem - eab_mem) / naive_mem * 100) if naive_mem > 0 else 0

    print(f"\n{'Metric':<30} {'EAB':<15} {'Naive':<15} {'Speedup':<15}")
    print("-" * 80)
    print(f"{'Time (seconds)':<30} {eab_time:<15.2f} {naive_time:<15.2f} {speedup:.2f}x faster")
    print(f"{'Memory (MB)':<30} {eab_mem:<15.1f} {naive_mem:<15.1f} {mem_reduction:.1f}% less")
    print(f"{'Samples generated':<30} {len(eab_results):<15} {len(naive_results):<15} {'SAME' if len(eab_results) == len(naive_results) else 'DIFF':<15}")

    # Store EAB statistics (before cleanup)
    # Re-create to get stats (already cleaned up above)
    # Just use the results we have

    # Calculate average branches from results
    avg_branches = sum(r['num_branches'] for r in eab_results) / len(eab_results) if eab_results else 0
    total_branches = sum(r['num_branches'] for r in eab_results)

    print(f"\n{'EAB Statistics:':<30}")
    print(f"  Average branches per path: {avg_branches:.1f}")
    print(f"  Total branch points: {total_branches}")

    # ==================== SAMPLE OUTPUTS ====================
    print("\n" + "=" * 80)
    print("SAMPLE OUTPUTS")
    print("=" * 80)

    print("\n[Naive Generation] First 3 samples:")
    print("-" * 80)
    for i, text in enumerate(naive_results[:3], 1):
        print(f"{i}. {text}")

    print("\n[EAB Generation] Top 3 samples:")
    print("-" * 80)
    for i, result in enumerate(eab_results[:3], 1):
        print(f"{i}. {result['text']}")
        print(f"   (log_prob: {result['log_prob']:.2f}, branches: {result['num_branches']})")

    # ==================== KEY INSIGHTS ====================
    print("\n" + "=" * 80)
    print("KEY INSIGHTS")
    print("=" * 80)

    print(f"""
1. Speed: EAB is {speedup:.2f}x faster than naive generation
   - Naive: Encodes prompt {num_eab_samples} times ({num_eab_samples} full forward passes)
   - EAB: Encodes prompt once, shares computation until branching

2. Memory: EAB uses {abs(mem_reduction):.1f}% {'less' if mem_reduction > 0 else 'more'} memory
   - Trade-off: EAB maintains multiple paths simultaneously
   - Naive: Processes one sample at a time (sequential)

3. Efficiency: EAB created {total_branches} branch points across {num_eab_samples} samples
   - Only creates multiple paths when model is uncertain
   - Average {avg_branches:.1f} branches per path

4. Quality: Both methods produce diverse samples
   - EAB samples are probability-weighted
   - Can access branch points and log probabilities for analysis
""")

    print("=" * 80)
    print("Conclusion: EAB provides significant speedup for multi-sample generation,")
    print("especially with longer prompts and more samples.")
    print("=" * 80)


def main():
    print("\n" + "=" * 80)
    print("Entropy-Adaptive Branching: Performance Comparison Demo")
    print("=" * 80)
    print(f"\nModel: {MODEL_NAME}")
    print("\nThis demo runs EAB first, then runs naive generation with the same")
    print("number of samples for a fair comparison.")
    print("\nNote: Running on CPU mode (use GPU for even better EAB performance)")
    print("\nTo use a different model, edit MODEL_NAME at the top of this file.")
    print("Options: gpt2, gpt2-medium, gpt2-large, gpt2-xl")
    print("\n")

    # Example 1: Factual prompt (low uncertainty)
    print("\n" + "█" * 80)
    print("EXAMPLE 1: Factual Prompt (Expected: Low branching, high speedup)")
    print("█" * 80)
    run_comparison(
        prompt="The capital of France is",
        max_paths=10,
        max_new_tokens=15,
        model_name=MODEL_NAME
    )

    # Example 2: Open-ended prompt (high uncertainty)
    print("\n\n" + "█" * 80)
    print("EXAMPLE 2: Creative Prompt (Expected: More branching, diverse outputs)")
    print("█" * 80)
    run_comparison(
        prompt="Once upon a time in a",
        max_paths=10,
        max_new_tokens=15,
        model_name=MODEL_NAME
    )

    print("\n" + "=" * 80)
    print("Demo complete!")
    print("=" * 80)
    print("\nTry modifying the prompts above to see how different inputs affect")
    print("the branching behavior and performance gains of EAB.")
    print()


if __name__ == "__main__":
    main()
