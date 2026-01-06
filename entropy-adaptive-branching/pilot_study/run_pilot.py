#!/usr/bin/env python3
"""
Pilot Study: Entropy Distribution Analysis

This script runs 200 diverse prompts across 3 confidence levels to:
1. Measure entropy distributions per confidence level
2. Understand branching behavior
3. Recommend optimal threshold settings

Results are saved to results/ and plots to plots/
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import pandas as pd
import json
import time
from tqdm import tqdm
from eab import EntropyAdaptiveBranching


def load_prompts(confidence_level):
    """Load prompts from file."""
    prompt_file = Path(__file__).parent / 'prompts' / f'{confidence_level}_confidence.txt'

    prompts = []
    with open(prompt_file, 'r') as f:
        for line in f:
            line = line.strip()
            # Skip empty lines and comments
            if line and not line.startswith('#'):
                prompts.append(line)

    return prompts


def run_single_prompt(eab, prompt, confidence_level):
    """Run a single prompt and collect metrics."""
    try:
        start_time = time.time()

        samples = eab.generate(
            prompt=prompt,
            max_new_tokens=50,
            temperature=0.8,
            use_chat_template=True,
            show_progress=False  # Disable inner progress bar
        )

        generation_time = time.time() - start_time

        # Collect branching points
        all_branches = set()
        for sample in samples:
            all_branches.update(sample.get('branch_points', []))

        # Get entropy statistics
        entropy_stats = eab.entropy_tracker.get_statistics()
        entropy_values = eab.entropy_tracker.entropy_history  # All entropy values during generation

        result = {
            'prompt': prompt,
            'confidence_level': confidence_level,
            'num_samples': len(samples),
            'num_branches': len(all_branches),
            'branch_positions': sorted(list(all_branches)),
            'generation_time': generation_time,
            'entropy_values': entropy_values,  # All entropy values during generation
            'avg_entropy': entropy_stats.get('mean_entropy', 0),
            'max_entropy': entropy_stats.get('max_entropy', 0),
            'min_entropy': entropy_stats.get('min_entropy', 0),
            'successful': True
        }

        return result

    except Exception as e:
        return {
            'prompt': prompt,
            'confidence_level': confidence_level,
            'error': str(e),
            'successful': False
        }


def main():
    print("="*80)
    print("EAB Pilot Study: Entropy Distribution Analysis")
    print("="*80)
    print(f"\nGoal: Analyze 200 prompts across 3 confidence levels")
    print(f"Output: Statistical analysis + threshold recommendation\n")

    # Initialize EAB with VERY LOW threshold to capture all entropy
    print("[1/4] Initializing EAB...")
    print("  Using very low threshold (0.05) to observe all branching opportunities")

    eab = EntropyAdaptiveBranching(
        model_name='Qwen/Qwen2.5-3B-Instruct',
        entropy_threshold=0.05,  # Very low to capture everything
        branch_factor=3,
        max_paths=20,
        device='cuda',
        torch_dtype=torch.float16
    )
    print("  ✓ Model loaded\n")

    # Load all prompts
    print("[2/4] Loading prompts...")
    prompts_by_level = {
        'high': load_prompts('high'),
        'medium': load_prompts('medium'),
        'low': load_prompts('low')
    }

    total_prompts = sum(len(p) for p in prompts_by_level.values())
    print(f"  High confidence: {len(prompts_by_level['high'])} prompts")
    print(f"  Medium confidence: {len(prompts_by_level['medium'])} prompts")
    print(f"  Low confidence: {len(prompts_by_level['low'])} prompts")
    print(f"  Total: {total_prompts} prompts\n")

    # Run all prompts
    print("[3/4] Running experiments...")
    print("  This will take ~15-30 minutes depending on GPU speed\n")

    results = []

    for confidence_level, prompts in prompts_by_level.items():
        print(f"\n  Processing {confidence_level} confidence prompts...")

        for prompt in tqdm(prompts, desc=f"  {confidence_level.capitalize()}", ncols=80):
            result = run_single_prompt(eab, prompt, confidence_level)
            results.append(result)

            # Small delay to prevent overheating
            time.sleep(0.1)

    # Save results
    print("\n[4/4] Saving results...")

    # Save detailed JSON
    output_dir = Path(__file__).parent / 'results'
    output_dir.mkdir(exist_ok=True)

    json_file = output_dir / 'pilot_results.json'
    with open(json_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"  ✓ Detailed results: {json_file}")

    # Create summary DataFrame
    successful_results = [r for r in results if r.get('successful', False)]

    df = pd.DataFrame([{
        'prompt': r['prompt'][:50] + '...' if len(r['prompt']) > 50 else r['prompt'],
        'confidence_level': r['confidence_level'],
        'num_samples': r['num_samples'],
        'num_branches': r['num_branches'],
        'avg_entropy': r['avg_entropy'],
        'max_entropy': r['max_entropy'],
        'min_entropy': r['min_entropy'],
        'generation_time': r['generation_time']
    } for r in successful_results])

    csv_file = output_dir / 'pilot_summary.csv'
    df.to_csv(csv_file, index=False)
    print(f"  ✓ Summary CSV: {csv_file}")

    # Print quick summary
    print("\n" + "="*80)
    print("Quick Summary")
    print("="*80)

    for level in ['high', 'medium', 'low']:
        level_data = df[df['confidence_level'] == level]
        print(f"\n{level.upper()} Confidence:")
        print(f"  Samples: {level_data['num_samples'].describe()[['mean', '50%', 'std']].to_dict()}")
        print(f"  Avg Entropy: {level_data['avg_entropy'].mean():.4f} (±{level_data['avg_entropy'].std():.4f})")
        print(f"  Max Entropy: {level_data['max_entropy'].mean():.4f} (±{level_data['max_entropy'].std():.4f})")
        print(f"  Branches: {level_data['num_branches'].mean():.1f} (±{level_data['num_branches'].std():.1f})")

    print("\n" + "="*80)
    print("Next Steps:")
    print("="*80)
    print("1. Run threshold analysis:")
    print("   python threshold/analyze_threshold.py")
    print("\n2. Generate visualizations:")
    print("   python threshold/visualize_distributions.py")
    print("\n3. Get threshold recommendation:")
    print("   python threshold/recommend_threshold.py")
    print("="*80)


if __name__ == '__main__':
    main()
