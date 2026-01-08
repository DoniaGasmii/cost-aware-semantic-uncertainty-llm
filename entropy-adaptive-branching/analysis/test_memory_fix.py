#!/usr/bin/env python3
"""
Quick test to verify memory tracking fix.
This should show non-zero memory for both EAB and naive.
"""

import sys
import torch
from pathlib import Path

# Add paths
analysis_dir = Path(__file__).parent
eab_dir = analysis_dir.parent
sys.path.insert(0, str(analysis_dir))
sys.path.insert(0, str(eab_dir))

from transformers import AutoModelForCausalLM, AutoTokenizer
from eab import EntropyAdaptiveBranching
from utils.metrics import MetricsTracker

def test_memory_tracking():
    """Test that memory tracking works for both EAB and naive."""
    print("Testing Memory Tracking Fix")
    print("=" * 60)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_name = "Qwen/Qwen2.5-3B-Instruct"
    prompt = "What is the capital of France?"

    print(f"\nDevice: {device}")
    print(f"Model: {model_name}")
    print(f"Prompt: {prompt}\n")

    # Load model for naive
    print("Loading model...")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32
    ).to(device)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Test 1: Naive generation memory tracking
    print("\n--- Test 1: Naive Generation ---")
    tracker = MetricsTracker(device=device)
    tracker.start()

    prompt_ids = tokenizer.encode(prompt, return_tensors='pt').to(device)

    for i in range(3):
        with torch.no_grad():
            output = model.generate(
                prompt_ids,
                max_new_tokens=20,
                temperature=0.8,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id
            )
        print(f"  Generated sample {i+1}")

    tracker.record_samples(3)
    metrics = tracker.stop()

    print(f"\nNaive Results:")
    print(f"  Memory peak: {metrics.memory_peak_mb:.2f} MB")
    print(f"  Wall time: {metrics.wall_clock_time:.2f}s")

    if metrics.memory_peak_mb == 0.0:
        print("  ❌ FAIL: Memory is 0 MB (bug still present)")
    else:
        print(f"  ✓ PASS: Memory tracking works!")

    # Test 2: EAB generation memory tracking
    print("\n--- Test 2: EAB Generation ---")
    eab = EntropyAdaptiveBranching(
        model_name=model_name,
        entropy_threshold=0.055,
        branch_factor=3,
        max_paths=20,
        device=device,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32
    )

    tracker = MetricsTracker(device=device)
    tracker.start()

    samples = eab.generate(
        prompt=prompt,
        max_new_tokens=20,
        temperature=0.8,
        use_chat_template=True
    )

    tracker.record_samples(len(samples))
    tracker.update_memory()
    metrics = tracker.stop()

    print(f"\nEAB Results:")
    print(f"  Samples: {len(samples)}")
    print(f"  Memory peak: {metrics.memory_peak_mb:.2f} MB")
    print(f"  Wall time: {metrics.wall_clock_time:.2f}s")

    if metrics.memory_peak_mb == 0.0:
        print("  ❌ FAIL: Memory is 0 MB")
    else:
        print(f"  ✓ PASS: Memory tracking works!")

    print("\n" + "=" * 60)
    print("Test Complete!")
    print("=" * 60)

if __name__ == "__main__":
    test_memory_tracking()
