#!/usr/bin/env python3
"""
Test script to demonstrate chat template vs raw prompts.
This shows the quality difference for instruction-tuned models.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from eab import EntropyAdaptiveBranching

def main():
    print("="*80)
    print("Chat Template Quality Test")
    print("="*80)

    # Initialize EAB with a smaller model
    print("\nInitializing EAB with Qwen2.5-3B-Instruct...")
    import torch
    eab = EntropyAdaptiveBranching(
        model_name='Qwen/Qwen2.5-3B-Instruct',
        entropy_threshold=0.1,
        branch_factor=2,
        max_paths=10,
        device='cuda',
        torch_dtype=torch.float16
    )

    # Test prompt
    test_prompt = "What is the best programming language?"

    print("\n" + "="*80)
    print("TEST 1: WITHOUT Chat Template (raw text continuation)")
    print("="*80)

    samples_raw = eab.generate(
        prompt=test_prompt,
        max_new_tokens=20,
        temperature=0.7,
        use_chat_template=False  # Disable chat template
    )

    print("\nGenerated samples (raw):")
    for i, sample in enumerate(samples_raw[:5], 1):
        text = sample.get('generated_only', sample.get('text', 'N/A'))
        print(f"\n{i}. {text}")

    print("\n" + "="*80)
    print("TEST 2: WITH Chat Template (proper instruction format)")
    print("="*80)

    samples_chat = eab.generate(
        prompt=test_prompt,
        max_new_tokens=20,
        temperature=0.7,
        use_chat_template=True  # Enable chat template (default)
    )

    print("\nGenerated samples (with chat template):")
    for i, sample in enumerate(samples_chat[:5], 1):
        text = sample.get('generated_only', sample.get('text', 'N/A'))
        print(f"\n{i}. {text}")

    print("\n" + "="*80)
    print("COMPARISON")
    print("="*80)
    print("\nKey observations:")
    print("- WITHOUT chat template: Model treats prompt as text to continue")
    print("  → May generate more questions or random text")
    print("- WITH chat template: Model understands it's answering a question")
    print("  → Should generate coherent answers")
    print("\nRecommendation: Always use chat template for instruct models!")
    print("="*80)

if __name__ == '__main__':
    main()
