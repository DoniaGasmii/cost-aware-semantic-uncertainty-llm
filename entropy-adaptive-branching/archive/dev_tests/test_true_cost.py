#!/usr/bin/env python3
"""
TRUE cost comparison: Count actual model forward passes.
This is the correct way to measure computational cost.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import torch
from eab import EntropyAdaptiveBranching
import time


class ForwardPassCounter:
    """Wrapper to count forward passes."""
    def __init__(self, model):
        self.model = model
        self.count = 0
        self._original_forward = model.forward

    def __enter__(self):
        def counting_forward(*args, **kwargs):
            self.count += 1
            return self._original_forward(*args, **kwargs)

        self.model.forward = counting_forward
        return self

    def __exit__(self, *args):
        self.model.forward = self._original_forward

    def reset(self):
        self.count = 0


def test_eab_cost(prompt, n_samples, max_tokens):
    """Measure EAB cost by counting forward passes."""
    print(f"\n[Testing EAB: {n_samples} samples]")

    eab = EntropyAdaptiveBranching(
        model_name='Qwen/Qwen2.5-3B-Instruct',
        entropy_threshold=0.055,
        branch_factor=3,
        max_paths=n_samples,
        device='cuda',
        torch_dtype=torch.float16
    )

    # Count forward passes
    counter = ForwardPassCounter(eab.model)

    with counter:
        start_time = time.time()
        samples = eab.generate(
            prompt=prompt,
            max_new_tokens=max_tokens,
            temperature=0.8,
            use_chat_template=True,
            show_progress=False
        )
        generation_time = time.time() - start_time

    forward_passes = counter.count

    print(f"  Samples generated: {len(samples)}")
    print(f"  Forward passes: {forward_passes}")
    print(f"  Time: {generation_time:.2f}s")
    print(f"  Avg passes/sample: {forward_passes/len(samples):.1f}")

    return {
        'method': 'EAB',
        'samples': len(samples),
        'forward_passes': forward_passes,
        'time': generation_time,
        'passes_per_sample': forward_passes / len(samples)
    }, eab


def test_naive_cost(eab, prompt, n_samples, max_tokens):
    """Measure naive sampling cost by counting forward passes."""
    print(f"\n[Testing Naive: {n_samples} samples]")

    # Encode prompt once
    messages = [{"role": "user", "content": prompt}]
    formatted_prompt = eab.tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    inputs = eab.tokenizer(formatted_prompt, return_tensors='pt').to(eab.device)

    # Count forward passes
    counter = ForwardPassCounter(eab.model)

    samples = []
    with counter:
        start_time = time.time()

        for i in range(n_samples):
            # Generate
            with torch.no_grad():
                outputs = eab.model.generate(
                    **inputs,
                    max_new_tokens=max_tokens,
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

    forward_passes = counter.count

    print(f"  Samples generated: {len(samples)}")
    print(f"  Forward passes: {forward_passes}")
    print(f"  Time: {generation_time:.2f}s")
    print(f"  Avg passes/sample: {forward_passes/len(samples):.1f}")

    return {
        'method': 'Naive',
        'samples': len(samples),
        'forward_passes': forward_passes,
        'time': generation_time,
        'passes_per_sample': forward_passes / len(samples)
    }


def compare_costs():
    """Compare EAB vs Naive computational costs."""
    print("=" * 80)
    print("TRUE COST COMPARISON: Forward Pass Counting")
    print("=" * 80)
    print("\nMeasuring ACTUAL computational cost (forward passes), not wall-clock time.\n")

    prompt = "Name one important skill students should develop to succeed in the future."
    n_samples = 10
    max_tokens = 30

    print(f"Configuration:")
    print(f"  Prompt: '{prompt}'")
    print(f"  Target samples: {n_samples}")
    print(f"  Max tokens: {max_tokens}")

    # Test EAB
    eab_result, eab_instance = test_eab_cost(prompt, n_samples, max_tokens)

    # Test Naive
    naive_result = test_naive_cost(eab_instance, prompt, n_samples, max_tokens)

    # Analysis
    print("\n\n" + "=" * 80)
    print("COST ANALYSIS")
    print("=" * 80)

    print(f"\n📊 Results:")
    print("-" * 80)
    print(f"{'Method':<10} {'Samples':<10} {'Forward Passes':<18} {'Time (s)':<12} {'Passes/Sample':<15}")
    print("-" * 80)
    print(f"{'EAB':<10} {eab_result['samples']:<10} {eab_result['forward_passes']:<18} "
          f"{eab_result['time']:<12.2f} {eab_result['passes_per_sample']:<15.1f}")
    print(f"{'Naive':<10} {naive_result['samples']:<10} {naive_result['forward_passes']:<18} "
          f"{naive_result['time']:<12.2f} {naive_result['passes_per_sample']:<15.1f}")
    print("-" * 80)

    # Calculate savings
    forward_pass_savings = naive_result['forward_passes'] - eab_result['forward_passes']
    savings_pct = (forward_pass_savings / naive_result['forward_passes']) * 100

    time_savings = naive_result['time'] - eab_result['time']
    time_savings_pct = (time_savings / naive_result['time']) * 100

    print(f"\n💰 COST SAVINGS:")
    print(f"  Forward passes saved: {forward_pass_savings} ({savings_pct:.1f}%)")
    print(f"  Time saved: {time_savings:.2f}s ({time_savings_pct:.1f}%)")

    # Theoretical vs actual
    theoretical_naive_passes = n_samples * max_tokens
    print(f"\n🧮 THEORETICAL vs ACTUAL:")
    print(f"  Naive (theoretical): {n_samples} × {max_tokens} = {theoretical_naive_passes} passes")
    print(f"  Naive (actual): {naive_result['forward_passes']} passes")
    print(f"  EAB (actual): {eab_result['forward_passes']} passes")

    sharing_efficiency = 1 - (eab_result['forward_passes'] / naive_result['forward_passes'])
    print(f"\n✓ KV-Cache Sharing Efficiency: {sharing_efficiency*100:.1f}%")
    print(f"  (EAB saves {sharing_efficiency*100:.1f}% of computation through shared prefix)")

    print("\n\n" + "=" * 80)
    print("CONCLUSION")
    print("=" * 80)

    if savings_pct > 0:
        print(f"""
✓ EAB IS MORE EFFICIENT:
  - For {n_samples} samples, EAB saves {savings_pct:.1f}% compute ({forward_pass_savings} forward passes)
  - Savings come from KV-cache sharing (shared branching tree)
  - Wall-clock time also faster by {time_savings_pct:.1f}%

WHY THIS WORKS:
  - Naive: Each sample independently generates all {max_tokens} tokens
  - EAB: Samples share prefix computation, only branch at high-entropy positions
  - Shared computation = fewer total forward passes

RECOMMENDATION:
  ✓ Use EAB for generating multiple diverse samples
  ✓ Savings increase with more samples (more sharing opportunities)
  ✓ Best for prompts with uncertainty (branching worthwhile)
""")
    else:
        print(f"""
⚠ EAB IS LESS EFFICIENT:
  - EAB used {abs(savings_pct):.1f}% MORE compute than naive
  - Possible reasons:
    1. Overhead from branching/pruning
    2. Not enough shared computation (low entropy prompt)
    3. Implementation overhead

RECOMMENDATION:
  - Check entropy distribution (was branching worthwhile?)
  - Consider using naive for this type of prompt
  - Or optimize EAB implementation
""")

    print("=" * 80)


if __name__ == '__main__':
    compare_costs()
