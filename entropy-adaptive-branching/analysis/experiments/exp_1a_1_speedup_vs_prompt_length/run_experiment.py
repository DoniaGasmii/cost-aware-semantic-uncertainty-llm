"""
Run Experiment 1.A: Speedup vs Prompt Length

This script:
1. Loads prompts of different lengths
2. Runs EAB generation with metrics tracking
3. Runs naive generation with same sample count
4. Computes efficiency metrics
5. Saves results incrementally
"""

import sys
import json
import yaml
import torch
from pathlib import Path
from tqdm import tqdm
from typing import Dict, Any, List

# Add parent directories to path
experiment_dir = Path(__file__).parent
analysis_dir = experiment_dir.parent.parent
eab_dir = analysis_dir.parent
sys.path.insert(0, str(analysis_dir))
sys.path.insert(0, str(eab_dir))

# Import EAB (both implementations)
from eab.core import EntropyAdaptiveBranching as EAB_Original
from eab.core_cow import EntropyAdaptiveBranching as EAB_COW
from typing import Union

# Type alias for EAB instances
EABInstance = Union[EAB_Original, EAB_COW]

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


def load_prompts(length_dir: Path) -> List[Dict[str, Any]]:
    """Load prompts from directory."""
    prompts_file = length_dir / "prompts.json"
    with open(prompts_file, 'r') as f:
        data = json.load(f)
    return data['prompts']


def run_eab_generation(
    prompt_text: str,
    eab: EABInstance,
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
    # Each sample has 'length' which is prompt + generated
    total_tokens = sum(s.get('length', len(s['tokens'])) for s in samples)
    tracker.record_token_steps(total_tokens)

    # Extract branching info from samples (EAB stores it in each sample)
    all_branch_points = set()  # Use set to avoid duplicates
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
        tracker: MetricsTracker instance

    Returns:
        (samples, metrics_snapshot)
    """
    tracker.start()

    samples = []
    device = config['model']['device']

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
                pad_token_id=tokenizer.eos_token_id  # Prevent warnings
            )

        # Decode ONLY the generated part (exclude prompt)
        generated_ids = output_ids[0][prompt_ids.shape[1]:]  # Remove prompt tokens
        generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True)
        full_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)

        samples.append({
            'text': full_text,  # Full response (prompt + generation)
            'generated_only': generated_text,  # Only the generated part
            'tokens': output_ids[0].tolist(),
            'num_generated_tokens': len(generated_ids)
        })

        # Track token steps: each sample processes entire sequence (prompt + generated)
        total_length = len(output_ids[0])
        tracker.record_token_steps(total_length)

        # Update memory
        tracker.update_memory()

    tracker.record_samples(num_samples)
    metrics = tracker.stop()

    return samples, metrics


def save_generated_texts(
    prompt_id: str,
    prompt_text: str,
    eab_samples: List[Dict],
    naive_samples: List[Dict],
    output_dir: Path
):
    """Save generated texts in human-readable format for inspection."""
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / f"{prompt_id}_generations.txt"

    with open(output_file, 'w') as f:
        f.write("=" * 80 + "\n")
        f.write(f"GENERATED RESPONSES FOR: {prompt_id}\n")
        f.write("=" * 80 + "\n\n")

        f.write("PROMPT:\n")
        f.write("-" * 80 + "\n")
        f.write(prompt_text + "\n")
        f.write("-" * 80 + "\n\n")

        f.write("EAB GENERATIONS:\n")
        f.write("=" * 80 + "\n")
        for i, sample in enumerate(eab_samples, 1):
            f.write(f"\n[EAB Sample {i}]")
            if 'path_id' in sample:
                f.write(f" (path {sample['path_id']}, {sample.get('num_branches', 0)} branches)")
            f.write("\n")
            # EAB stores full text in 'text' field
            text = sample.get('text', '')
            # Remove prompt if present (EAB might include it)
            if text.startswith(prompt_text):
                text = text[len(prompt_text):].strip()
            f.write(text + "\n")
            f.write("-" * 40 + "\n")

        f.write("\n" + "=" * 80 + "\n")
        f.write("NAIVE GENERATIONS:\n")
        f.write("=" * 80 + "\n")
        for i, sample in enumerate(naive_samples, 1):
            f.write(f"\n[Naive Sample {i}]\n")
            f.write(sample.get('generated_only', sample.get('text', '')) + "\n")
            f.write("-" * 40 + "\n")


def run_single_experiment(
    prompt: Dict[str, Any],
    eab: EABInstance,
    model,
    tokenizer,
    config: Dict[str, Any],
    target_samples: int
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
        target_samples: Reference sample count (for info only, not enforced)

    Returns:
        Result dictionary with all metrics
    """
    prompt_text = prompt['text']
    prompt_length = prompt['actual_length']

    print(f"  Prompt: {prompt['id']} (length: {prompt_length})")
    print(f"    Reference target: {target_samples} samples")

    # Step 1: Run EAB FIRST with natural behavior
    print(f"    Running EAB (natural behavior)...")
    eab_tracker = MetricsTracker(device=config['model']['device'])
    eab_samples, eab_metrics = run_eab_generation(
        prompt_text, eab, config, eab_tracker
    )
    num_eab_samples = len(eab_samples)
    print(f"      ✓ Generated {num_eab_samples} samples")

    # Step 2: Run Naive with EXACT same count as EAB for fair comparison
    print(f"    Running Naive ({num_eab_samples} samples to match EAB)...")
    naive_tracker = MetricsTracker(device=config['model']['device'])
    naive_samples, naive_metrics = run_naive_generation(
        prompt_text, num_eab_samples, model, tokenizer, config, naive_tracker
    )
    print(f"      ✓ Generated {len(naive_samples)} samples")
    print(f"      ✓ Avg generated tokens: {sum(s['num_generated_tokens'] for s in naive_samples) / len(naive_samples):.1f}")

    # Save generated texts if configured
    if config['output'].get('save_generated_texts', False):
        texts_dir = experiment_dir / config['output'].get('texts_dir', 'results/generated_texts')
        save_generated_texts(prompt['id'], prompt_text, eab_samples, naive_samples, texts_dir)

    # Compute efficiency metrics
    efficiency = compute_efficiency_metrics(naive_metrics, eab_metrics)

    # Compute branching stats
    branching_stats = compute_branching_stats(eab_metrics)

    # Compile result
    result = {
        'prompt_id': prompt['id'],
        'prompt_length': prompt_length,
        'target_length': prompt['target_length'],
        'reference_target_samples': target_samples,  # Reference value (not enforced)
        'num_eab_samples': num_eab_samples,  # Actual EAB output (natural behavior)
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
    print("EXPERIMENT 1.A: SPEEDUP VS PROMPT LENGTH")
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
        prompt_lengths = config['debug']['prompt_lengths']
        prompts_per_length = config['debug']['prompts_per_length']
        target_samples = config['debug']['target_samples']
    else:
        print("   🚀 FULL EXPERIMENT MODE")
        prompt_lengths = config['prompt_lengths']
        prompts_per_length = config['prompts_per_length']
        target_samples = config['target_samples']

    print(f"   Prompt lengths: {prompt_lengths}")
    print(f"   Prompts per length: {prompts_per_length}")
    print(f"   Reference target samples: {target_samples} (EAB uses natural behavior)")
    print(f"   Device: {config['model']['device']}")

    # Check implementation mode
    impl_mode = config['eab'].get('implementation', 'cow')
    print(f"   Implementation: {impl_mode}")

    # Initialize model and EAB
    print("\n2. Initializing model and EAB...")
    device = config['model']['device']

    # Prepare model kwargs
    model_kwargs = {
        'model_name': config['model']['name'],
        'entropy_threshold': config['eab']['entropy_threshold'],
        'branch_factor': config['eab']['branch_factor'],
        'max_paths': config['eab']['max_paths'],
        'device': device
    }

    # Add FP16 for CUDA to reduce memory usage
    if device == 'cuda':
        model_kwargs['torch_dtype'] = torch.float16
        print(f"   Using FP16 precision to reduce memory usage")

    # Select implementation class
    if impl_mode == 'cow':
        EntropyAdaptiveBranching = EAB_COW
        print(f"   Using COW (Copy-on-Write) implementation")
    elif impl_mode == 'original':
        EntropyAdaptiveBranching = EAB_Original
        print(f"   Using Original (deep copy) implementation")
    elif impl_mode == 'both':
        print(f"   Will run with BOTH implementations (COW and Original)")
    else:
        print(f"   ⚠ Unknown implementation '{impl_mode}', defaulting to COW")
        EntropyAdaptiveBranching = EAB_COW

    # Initialize EAB
    if impl_mode != 'both':
        eab = EntropyAdaptiveBranching(**model_kwargs)
        print(f"   ✓ EAB initialized with threshold={config['eab']['entropy_threshold']}")

        # Get model and tokenizer for naive sampling
        model = eab.model  # Reuse EAB's model
        tokenizer = eab.tokenizer
        print(f"   ✓ Model: {config['model']['name']}")
    else:
        # For 'both' mode, initialize both implementations
        eab_cow = EAB_COW(**model_kwargs)
        print(f"   ✓ COW EAB initialized")

        eab_original = EAB_Original(**model_kwargs)
        print(f"   ✓ Original EAB initialized")

        # Get model and tokenizer for naive sampling (from COW instance)
        model = eab_cow.model
        tokenizer = eab_cow.tokenizer
        print(f"   ✓ Model: {config['model']['name']}")

        # For compatibility with single mode code below
        eab = None

    # Prepare results directory
    results_dir = experiment_dir / config['output']['results_dir']
    results_dir.mkdir(exist_ok=True, parents=True)
    results_file = results_dir / "raw_results.json"

    # Initialize results file
    save_json({
        'metadata': {
            'config': config,
            'total_experiments': len(prompt_lengths) * prompts_per_length
        },
        'results': []
    }, results_file)

    # Run experiments
    print("\n3. Running experiments...")
    print("-" * 70)

    all_results = []
    total_experiments = len(prompt_lengths) * prompts_per_length
    experiment_count = 0

    for length in prompt_lengths:
        print(f"\n📏 Prompt Length: {length} tokens")
        print("-" * 70)

        # Load prompts
        length_dir = experiment_dir / "prompts" / f"length_{length:03d}"
        prompts = load_prompts(length_dir)[:prompts_per_length]

        # Run on each prompt
        for prompt in prompts:
            experiment_count += 1
            print(f"\n[{experiment_count}/{total_experiments}]")

            try:
                if impl_mode == 'both':
                    # Run with both implementations
                    print(f"  Running with COW implementation...")
                    result_cow = run_single_experiment(
                        prompt, eab_cow, model, tokenizer, config, target_samples
                    )
                    result_cow['implementation'] = 'cow'
                    all_results.append(result_cow)

                    print(f"\n  Running with Original implementation...")
                    result_orig = run_single_experiment(
                        prompt, eab_original, model, tokenizer, config, target_samples
                    )
                    result_orig['implementation'] = 'original'
                    all_results.append(result_orig)

                    # Save incrementally
                    if config['output']['save_intermediate']:
                        append_result(result_cow, results_file)
                        append_result(result_orig, results_file)
                else:
                    # Run with single implementation
                    result = run_single_experiment(
                        prompt, eab, model, tokenizer, config, target_samples
                    )
                    result['implementation'] = impl_mode
                    all_results.append(result)

                    # Save incrementally
                    if config['output']['save_intermediate']:
                        append_result(result, results_file)

            except Exception as e:
                print(f"    ✗ Error: {e}")
                import traceback
                traceback.print_exc()
                continue

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
