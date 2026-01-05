#!/usr/bin/env python3
"""
Quick test to verify EAB produces coherent branching.
This uses a creative prompt that should trigger branching.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
from eab import EntropyAdaptiveBranching

def main():
    print("="*80)
    print("EAB Coherent Branching Demo")
    print("="*80)

    # Initialize
    print("\n[1/3] Initializing EAB with Qwen2.5-3B-Instruct...")
    eab = EntropyAdaptiveBranching(
        model_name='Qwen/Qwen2.5-3B-Instruct',
        entropy_threshold=0.15,      # Lower threshold = more branching
        branch_factor=3,             # 3-way branches
        max_paths=15,
        device='cuda',
        torch_dtype=torch.float16
    )
    print("✓ Model loaded successfully")

    # Use a creative prompt that should branch
    creative_prompt = "In the year 2050, the most significant technological advancement will be"

    print(f"\n[2/3] Generating samples...")
    print(f"Prompt: '{creative_prompt}'")
    print(f"Expected: Multiple branches (high uncertainty prompt)")

    samples = eab.generate(
        prompt=creative_prompt,
        max_new_tokens=40,
        temperature=0.8,
        use_chat_template=True  # CRITICAL for quality
    )

    print(f"\n[3/3] Results:")
    print(f"="*80)
    print(f"Generated {len(samples)} samples")

    # Count total branches
    all_branches = set()
    for sample in samples:
        all_branches.update(sample.get('branch_points', []))

    print(f"Total branching points: {len(all_branches)}")
    if all_branches:
        print(f"Branch positions: {sorted(all_branches)}")

    print(f"\n{'='*80}")
    print("Sample Outputs:")
    print(f"{'='*80}\n")

    for i, sample in enumerate(samples[:10], 1):  # Show first 10
        text = sample.get('generated_only', sample.get('text', 'N/A'))
        branches = sample.get('branch_points', [])

        print(f"{i}. {text}")
        if branches:
            print(f"   [Branched at positions: {branches}]")
        print()

    if len(samples) > 10:
        print(f"... and {len(samples) - 10} more samples\n")

    # Quality assessment
    print("="*80)
    print("Quality Assessment:")
    print("="*80)

    avg_entropy = sum(s.get('avg_entropy', 0) for s in samples) / len(samples) if samples else 0

    print(f"✓ Average entropy: {avg_entropy:.3f}")
    print(f"✓ Branching rate: {len(all_branches)/40*100:.1f}% of positions")

    if len(samples) > 5 and len(all_branches) > 2:
        print(f"✓ SUCCESS: Good branching behavior!")
        print(f"  - Generated {len(samples)} diverse samples")
        print(f"  - Branched at {len(all_branches)} positions")
    elif len(samples) <= 2:
        print("⚠ WARNING: Low sample count")
        print("  Suggestions:")
        print("  - Lower entropy threshold (try 0.1)")
        print("  - Use more creative/uncertain prompt")
        print("  - Increase temperature to 0.9")
    else:
        print("✓ Samples generated successfully")

    print("\n" + "="*80)
    print("If outputs are coherent and relevant → Ready for experiments!")
    print("If outputs are random/incoherent → Check QUALITY_GUIDE.md")
    print("="*80)

if __name__ == '__main__':
    main()
