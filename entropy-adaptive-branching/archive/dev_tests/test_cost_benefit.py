#!/usr/bin/env python3
"""
Cost-benefit analysis: Does EAB achieve same diversity with fewer samples?
This is the REAL value proposition test.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import torch
from eab import EntropyAdaptiveBranching
from collections import Counter
import time


def measure_semantic_diversity(samples, model_tokenizer):
    """
    Measure semantic diversity by clustering outputs.
    Simple approach: unique first 50 characters (captures main semantic difference).
    """
    texts = [s.get('generated_only', s.get('text', '')) for s in samples]

    # Extract semantic signatures
    signatures = []
    for text in texts:
        # Normalize: remove minor variations
        normalized = text.lower().strip()
        # Get first 50 chars as semantic signature
        sig = normalized[:50]
        signatures.append(sig)

    # Count unique semantic variants
    unique_variants = len(set(signatures))

    # Get most common variants
    variant_counts = Counter(signatures)

    return {
        'total_samples': len(samples),
        'unique_variants': unique_variants,
        'diversity_ratio': unique_variants / len(samples) if samples else 0,
        'variant_distribution': variant_counts
    }


def test_eab_with_different_budgets(prompt, budgets=[5, 10, 15, 20]):
    """Test EAB with different max_paths to find optimal cost/quality tradeoff."""
    print("=" * 80)
    print("COST-BENEFIT ANALYSIS: EAB vs Naive Sampling")
    print("=" * 80)
    print("\nTesting: Can EAB achieve same diversity with fewer samples?\n")

    results = {}

    # Load model once
    print("[Initializing model...]")
    eab = EntropyAdaptiveBranching(
        model_name='Qwen/Qwen2.5-3B-Instruct',
        entropy_threshold=0.055,
        branch_factor=3,
        max_paths=20,  # Will update this
        device='cuda',
        torch_dtype=torch.float16
    )
    print("✓ Model loaded\n")

    for max_paths in budgets:
        print(f"\n[Testing EAB with max_paths={max_paths}]")

        # Update max_paths
        eab.max_paths = max_paths

        start_time = time.time()
        samples = eab.generate(
            prompt=prompt,
            max_new_tokens=30,
            temperature=0.8,
            use_chat_template=True,
            show_progress=False
        )
        generation_time = time.time() - start_time

        diversity = measure_semantic_diversity(samples, eab.tokenizer)

        results[f'EAB_{max_paths}'] = {
            'method': 'EAB',
            'max_paths': max_paths,
            'samples': len(samples),
            'unique_variants': diversity['unique_variants'],
            'diversity_ratio': diversity['diversity_ratio'],
            'time': generation_time,
            'variant_dist': diversity['variant_distribution']
        }

        print(f"  Samples: {len(samples)}")
        print(f"  Unique variants: {diversity['unique_variants']}")
        print(f"  Diversity: {diversity['diversity_ratio']*100:.1f}%")
        print(f"  Time: {generation_time:.2f}s")

    return results, eab


def test_naive_with_different_counts(eab, prompt, counts=[5, 10, 15, 20]):
    """Test naive sampling with different sample counts."""
    print(f"\n\n[Testing Naive Sampling]")

    results = {}

    for n_samples in counts:
        print(f"\n[Naive with {n_samples} samples]")

        start_time = time.time()
        samples = []

        for i in range(n_samples):
            # Encode prompt with chat template
            messages = [{"role": "user", "content": prompt}]
            formatted_prompt = eab.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
            inputs = eab.tokenizer(formatted_prompt, return_tensors='pt').to(eab.device)

            # Generate
            with torch.no_grad():
                outputs = eab.model.generate(
                    **inputs,
                    max_new_tokens=30,
                    temperature=0.8,
                    do_sample=True,
                    pad_token_id=eab.tokenizer.eos_token_id
                )

            # Decode
            full_text = eab.tokenizer.decode(outputs[0], skip_special_tokens=True)
            generated_only = full_text[len(formatted_prompt):].strip()

            samples.append({
                'text': full_text,
                'generated_only': generated_only
            })

        generation_time = time.time() - start_time

        diversity = measure_semantic_diversity(samples, eab.tokenizer)

        results[f'Naive_{n_samples}'] = {
            'method': 'Naive',
            'samples': n_samples,
            'unique_variants': diversity['unique_variants'],
            'diversity_ratio': diversity['diversity_ratio'],
            'time': generation_time,
            'variant_dist': diversity['variant_distribution']
        }

        print(f"  Unique variants: {diversity['unique_variants']}")
        print(f"  Diversity: {diversity['diversity_ratio']*100:.1f}%")
        print(f"  Time: {generation_time:.2f}s")

    return results


def analyze_cost_benefit(eab_results, naive_results):
    """Analyze which approach gives best cost/quality tradeoff."""
    print("\n\n" + "=" * 80)
    print("COST-BENEFIT ANALYSIS")
    print("=" * 80)

    print("\n📊 Results Summary:")
    print("-" * 80)
    print(f"{'Method':<15} {'Samples':<10} {'Unique':<10} {'Diversity':<12} {'Time (s)':<10}")
    print("-" * 80)

    all_results = {**eab_results, **naive_results}

    for key, result in sorted(all_results.items()):
        method = result['method']
        samples = result['samples']
        unique = result['unique_variants']
        diversity = result['diversity_ratio'] * 100
        time_taken = result['time']

        print(f"{key:<15} {samples:<10} {unique:<10} {diversity:<11.1f}% {time_taken:<10.2f}")

    # Find efficiency sweet spots
    print("\n\n💡 COST-EFFICIENCY ANALYSIS")
    print("=" * 80)

    # Target: Find minimum samples needed for each unique variant count
    variant_counts = {}
    for key, result in all_results.items():
        unique = result['unique_variants']
        samples = result['samples']
        method = result['method']

        if unique not in variant_counts:
            variant_counts[unique] = []
        variant_counts[unique].append({
            'key': key,
            'method': method,
            'samples': samples,
            'time': result['time']
        })

    print("\nTo achieve N unique variants, minimum samples needed:\n")
    print(f"{'Target Variants':<18} {'EAB':<20} {'Naive':<20} {'Winner':<15}")
    print("-" * 80)

    for variant_count in sorted(variant_counts.keys()):
        methods = variant_counts[variant_count]

        eab_entries = [m for m in methods if m['method'] == 'EAB']
        naive_entries = [m for m in methods if m['method'] == 'Naive']

        eab_min = min(eab_entries, key=lambda x: x['samples']) if eab_entries else None
        naive_min = min(naive_entries, key=lambda x: x['samples']) if naive_entries else None

        eab_str = f"{eab_min['samples']} samples" if eab_min else "N/A"
        naive_str = f"{naive_min['samples']} samples" if naive_min else "N/A"

        if eab_min and naive_min:
            if eab_min['samples'] < naive_min['samples']:
                winner = f"EAB (-{naive_min['samples']-eab_min['samples']})"
            elif eab_min['samples'] > naive_min['samples']:
                winner = f"Naive (-{eab_min['samples']-naive_min['samples']})"
            else:
                winner = "Tie"
        else:
            winner = "Incomplete"

        print(f"{variant_count:<18} {eab_str:<20} {naive_str:<20} {winner:<15}")

    # Calculate cost savings
    print("\n\n💰 POTENTIAL COST SAVINGS")
    print("=" * 80)

    # Compare: Can EAB with fewer samples match naive diversity?
    best_naive = max(naive_results.values(), key=lambda x: x['unique_variants'])
    target_diversity = best_naive['unique_variants']

    print(f"\nNaive best result: {best_naive['samples']} samples → {target_diversity} unique variants")

    # Find cheapest EAB config that matches this diversity
    matching_eab = [r for r in eab_results.values() if r['unique_variants'] >= target_diversity]

    if matching_eab:
        cheapest_eab = min(matching_eab, key=lambda x: x['samples'])
        savings = best_naive['samples'] - cheapest_eab['samples']
        savings_pct = (savings / best_naive['samples']) * 100

        print(f"EAB match: {cheapest_eab['samples']} samples → {cheapest_eab['unique_variants']} unique variants")
        print(f"\n✓ Cost savings: {savings} samples ({savings_pct:.1f}% reduction)")

        if savings > 0:
            print(f"  → EAB is MORE EFFICIENT")
        elif savings < 0:
            print(f"  → Naive is MORE EFFICIENT")
        else:
            print(f"  → Same cost")
    else:
        print(f"\n✗ EAB did not match naive diversity with any budget tested")

    print("\n" + "=" * 80)
    print("CONCLUSION")
    print("=" * 80)

    # Determine if EAB provides cost advantage
    if matching_eab and savings > 0:
        print(f"""
✓ EAB PROVIDES COST ADVANTAGE:
  - Achieves same diversity ({target_diversity} variants) with {savings_pct:.1f}% fewer samples
  - Cost: {cheapest_eab['samples']} EAB samples vs {best_naive['samples']} naive samples
  - Savings come from intelligent branching at uncertainty points

RECOMMENDATION: Use EAB with max_paths={cheapest_eab['max_paths']} for this type of prompt.
""")
    elif matching_eab and savings == 0:
        print(f"""
⚠ EAB SAME COST AS NAIVE:
  - Both require {best_naive['samples']} samples for {target_diversity} variants
  - EAB advantage: Memory efficiency through KV-cache sharing
  - EAB advantage: Coherent exploration (shared branching tree)

RECOMMENDATION: Use EAB for memory savings, not sample count savings.
""")
    else:
        print(f"""
✗ EAB MORE EXPENSIVE THAN NAIVE:
  - EAB needs more samples to match naive diversity
  - Possible reasons:
    1. Threshold too high (not branching enough)
    2. Prompt not suitable for branching approach
    3. Naive sampling naturally diverse for this prompt

RECOMMENDATION:
  - Lower entropy threshold (current: 0.055)
  - Or use naive sampling for this type of prompt
  - Or value EAB's coherence over cost
""")

    print("=" * 80)


if __name__ == '__main__':
    # Use a prompt known to have multiple valid answers
    prompt = "Name one important skill students should develop to succeed in the future."

    print(f"Test Prompt: '{prompt}'")
    print(f"This prompt should have multiple valid answers (adaptability, critical thinking, etc.)\n")

    # Test EAB with different budgets
    eab_results, eab_instance = test_eab_with_different_budgets(prompt)

    # Test naive with different sample counts
    naive_results = test_naive_with_different_counts(eab_instance, prompt)

    # Analyze cost-benefit
    analyze_cost_benefit(eab_results, naive_results)
