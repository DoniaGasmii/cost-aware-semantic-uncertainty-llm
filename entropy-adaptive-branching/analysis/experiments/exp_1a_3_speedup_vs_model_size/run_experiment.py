"""
Run Experiment 1.A.3: Speedup vs Model Size

This script:
1. Loads prompts of fixed length (200 tokens)
2. Tests multiple model sizes (0.5B, 1.5B, 3B, 7B)
3. Runs EAB generation with metrics tracking
4. Runs naive generation with same sample count as EAB
5. Computes efficiency metrics
6. Saves results incrementally
"""

import sys
import json
import yaml
import torch
from pathlib import Path
from tqdm import tqdm
from typing import Dict, Any, List
import gc

# Add parent directories to path
experiment_dir = Path(__file__).parent
analysis_dir = experiment_dir.parent.parent
eab_dir = analysis_dir.parent
sys.path.insert(0, str(analysis_dir))
sys.path.insert(0, str(eab_dir))

# Import EAB
from eab import EntropyAdaptiveBranching

# Import analysis utils
from utils.metrics import (
    MetricsTracker,
    compute_efficiency_metrics,
    compute_branching_stats
)
from utils.data_utils import save_json, append_result


def load_config() -> Dict[str, Any]:
    """Load experiment configuration."""
    config_path = experiment_dir / "config.yaml"
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def load_prompts() -> List[Dict[str, Any]]:
    """Load prompts from directory."""
    prompts_file = experiment_dir / "prompts" / "prompts_200.json"
    with open(prompts_file, 'r') as f:
        data = json.load(f)
    return data['prompts']


def run_eab_generation(
    prompt_text: str,
    eab: EntropyAdaptiveBranching,
    config: Dict[str, Any],
    tracker: MetricsTracker
) -> tuple:
    """
    Run EAB generation with metrics tracking.

    Returns:
        (samples, metrics_snapshot)
    """
    tracker.start()

    # Generate with EAB (using chat template for instruct models)
    samples = eab.generate(
        prompt=prompt_text,
        max_new_tokens=config['generation']['max_new_tokens'],
        temperature=config['generation']['temperature'],
        use_chat_template=True  # Format as chat for instruct models
    )

    # Record metrics from EAB samples
    num_samples = len(samples)
    tracker.record_samples(num_samples)

    # Calculate total token steps from EAB samples
    total_tokens = sum(s.get('length', len(s['tokens'])) for s in samples)
    tracker.record_token_steps(total_tokens)

    # Extract branching info from samples
    all_branch_points = set()
    for sample in samples:
        branch_points = sample.get('branch_points', [])
        all_branch_points.update(branch_points)

    # Record unique branch points
    for bp in sorted(all_branch_points):
        tracker.record_branch(bp)

    tracker.record_final_paths(num_samples)

    # Update memory
    tracker.update_memory()

    # Stop tracking
    metrics = tracker.stop()

    return samples, metrics


def run_naive_generation(
    prompt_text: str,
    num_samples: int,
    model,
    tokenizer,
    config: Dict[str, Any],
    device: str,
    tracker: MetricsTracker
) -> tuple:
    """
    Run naive sampling with metrics tracking.

    Args:
        prompt_text: Input prompt
        num_samples: Number of samples to generate (match EAB)
        model: Language model
        tokenizer: Tokenizer
        config: Configuration dict
        device: Device string
        tracker: MetricsTracker instance

    Returns:
        (samples, metrics_snapshot)
    """
    tracker.start()

    samples = []

    # Tokenize prompt once
    prompt_ids = tokenizer.encode(prompt_text, return_tensors='pt').to(device)
    prompt_length = prompt_ids.shape[1]

    # Generate each sample independently
    for _ in range(num_samples):
        with torch.no_grad():
            output_ids = model.generate(
                prompt_ids,
                max_new_tokens=config['generation']['max_new_tokens'],
                temperature=config['generation']['temperature'],
                do_sample=True,
                top_p=config['generation']['top_p'],
                pad_token_id=tokenizer.eos_token_id
            )

        # Decode ONLY the generated part
        generated_ids = output_ids[0][prompt_ids.shape[1]:]
        generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True)
        full_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)

        samples.append({
            'text': full_text,
            'generated_only': generated_text,
            'tokens': output_ids[0].tolist(),
            'num_generated_tokens': len(generated_ids)
        })

        # Track token steps
        total_length = len(output_ids[0])
        tracker.record_token_steps(total_length)

        # Update memory
        tracker.update_memory()

    tracker.record_samples(num_samples)
    metrics = tracker.stop()

    return samples, metrics


def run_single_experiment(
    prompt: Dict[str, Any],
    eab: EntropyAdaptiveBranching,
    model,
    tokenizer,
    config: Dict[str, Any],
    model_name: str,
    model_params: str,
    device: str
) -> Dict[str, Any]:
    """
    Run experiment on a single prompt following the Fair Comparison Protocol.

    Protocol (from README):
    1. Run EAB with natural behavior → generates N samples (varies by prompt)
    2. Run Naive N times → match EAB's sample count
    3. Compare costs fairly

    Args:
        prompt: Prompt dictionary
        eab: EAB instance
        model: Language model
        tokenizer: Tokenizer
        config: Configuration
        model_name: Name of the model being tested
        model_params: Model size (e.g., "3B")
        device: Device string

    Returns:
        Result dictionary with all metrics
    """
    prompt_text = prompt['text']
    prompt_length = prompt['actual_length']

    print(f"  Prompt: {prompt['id']} (length: {prompt_length})")

    # Step 1: Run EAB FIRST with natural behavior
    print(f"    Running EAB (natural behavior)...")
    eab_tracker = MetricsTracker(device=device)
    eab_samples, eab_metrics = run_eab_generation(
        prompt_text, eab, config, eab_tracker
    )
    num_eab_samples = len(eab_samples)
    print(f"      ✓ Generated {num_eab_samples} samples")

    # Step 2: Run Naive with EXACT same count as EAB for fair comparison
    print(f"    Running Naive ({num_eab_samples} samples to match EAB)...")
    naive_tracker = MetricsTracker(device=device)
    naive_samples, naive_metrics = run_naive_generation(
        prompt_text, num_eab_samples, model, tokenizer, config, device, naive_tracker
    )
    print(f"      ✓ Generated {len(naive_samples)} samples")
    print(f"      ✓ Avg generated tokens: {sum(s['num_generated_tokens'] for s in naive_samples) / len(naive_samples):.1f}")

    # Compute efficiency metrics
    efficiency = compute_efficiency_metrics(naive_metrics, eab_metrics)

    # Compute branching stats
    branching_stats = compute_branching_stats(eab_metrics)

    # Compile result
    result = {
        'prompt_id': prompt['id'],
        'prompt_length': prompt_length,
        'model_name': model_name,
        'model_params': model_params,
        'num_eab_samples': num_eab_samples,  # Actual EAB output
        'num_naive_samples': num_eab_samples,  # Naive matched to EAB

        # EAB metrics
        'eab_metrics': eab_metrics.to_dict(),

        # Naive metrics
        'naive_metrics': naive_metrics.to_dict(),

        # Efficiency
        'efficiency': efficiency,

        # Branching behavior
        'branching_stats': branching_stats,
    }

    print(f"      → Speedup (token-steps): {efficiency['speedup_token_steps']:.2f}×")
    print(f"      → Speedup (time): {efficiency['speedup_time']:.2f}×")

    return result


def main():
    """Main experiment runner."""
    print("=" * 70)
    print("EXPERIMENT 1.A.3: SPEEDUP VS MODEL SIZE")
    print("=" * 70)

    # Load configuration
    print("\n1. Loading configuration...")
    config = load_config()

    # Set random seed
    torch.manual_seed(config['seed'])
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config['seed'])

    # Determine mode and extract parameters
    if config['debug']['enabled']:
        print("   🔧 DEBUG MODE")
        model_configs = config['debug']['model_configs']
        prompts_count = config['debug']['prompts_count']
    else:
        print("   🚀 FULL EXPERIMENT MODE")
        model_configs = config['model_configs']
        prompts_count = config['prompts_count']

    print(f"   Models to test: {[cfg['params'] for cfg in model_configs]}")
    print(f"   Prompts per model: {prompts_count}")
    print(f"   Prompt length: {config['prompt_length']} tokens (FIXED)")

    # Load prompts once (reuse for all models)
    print("\n2. Loading prompts...")
    all_prompts = load_prompts()[:prompts_count]
    print(f"   ✓ Loaded {len(all_prompts)} prompts")

    # Prepare results directory
    results_dir = experiment_dir / config['output']['results_dir']
    results_dir.mkdir(exist_ok=True, parents=True)
    results_file = results_dir / "raw_results.json"

    # Initialize results file
    save_json({
        'metadata': {
            'config': config,
            'total_experiments': len(model_configs) * len(all_prompts)
        },
        'results': []
    }, results_file)

    # Run experiments
    print("\n3. Running experiments...")
    print("-" * 70)

    all_results = []
    total_experiments = len(model_configs) * len(all_prompts)
    experiment_count = 0

    for model_cfg in model_configs:
        model_name = model_cfg['name']
        model_params = model_cfg['params']
        device = model_cfg['device']

        print(f"\n🤖 Model: {model_name} ({model_params})")
        print("-" * 70)

        # Initialize model and EAB for this size
        print(f"   Loading model...")
        eab = EntropyAdaptiveBranching(
            model_name=model_name,
            entropy_threshold=config['eab']['entropy_threshold'],
            branch_factor=config['eab']['branch_factor'],
            max_paths=config['eab']['max_paths'],
            device=device
        )
        print(f"   ✓ Model loaded on {device}")

        # Get model and tokenizer for naive sampling
        model = eab.model
        tokenizer = eab.tokenizer

        # Run on each prompt
        for prompt in all_prompts:
            experiment_count += 1
            print(f"\n[{experiment_count}/{total_experiments}]")

            try:
                result = run_single_experiment(
                    prompt, eab, model, tokenizer, config,
                    model_name, model_params, device
                )
                all_results.append(result)

                # Save incrementally
                if config['output']['save_intermediate']:
                    append_result(result, results_file)

            except Exception as e:
                print(f"    ✗ Error: {e}")
                import traceback
                traceback.print_exc()
                continue

        # Cleanup to free memory before loading next model
        print(f"\n   Cleaning up {model_name}...")
        del eab, model, tokenizer
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        print(f"   ✓ Memory cleaned")

    # Final save
    print("\n" + "=" * 70)
    print("4. Saving final results...")
    save_json({
        'metadata': {
            'config': config,
            'total_experiments': len(all_results),
            'completed_successfully': len(all_results),
        },
        'results': all_results
    }, results_file)
    print(f"   ✓ Results saved to {results_file}")

    # Quick summary
    print("\n" + "=" * 70)
    print("EXPERIMENT COMPLETE")
    print("=" * 70)
    print(f"Total experiments: {len(all_results)}")
    print(f"Results saved to: {results_file}")
    print("\nNext steps:")
    print("  1. Run: python analyze_results.py")
    print("  2. Run: python plot_results.py")
    print("=" * 70)


if __name__ == "__main__":
    main()
