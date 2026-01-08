"""
Run Experiment 1.A.4: Speedup vs Domain

This script:
1. Loads domain-specific prompts (factual QA, creative, code)
2. Runs EAB generation with metrics tracking
3. Runs naive generation with same sample count as EAB
4. Computes efficiency metrics by domain
5. Analyzes correlation between entropy and speedup
6. Saves results incrementally
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


def load_domain_prompts(domain_config: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Load prompts for a specific domain."""
    prompts_file = experiment_dir / domain_config['prompts_file']
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
    domain_name: str
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
        domain_name: Name of the domain being tested

    Returns:
        Result dictionary with all metrics
    """
    prompt_text = prompt['text']
    prompt_length = prompt.get('actual_length', len(tokenizer.encode(prompt_text)))

    print(f"  Prompt: {prompt['id']} (length: {prompt_length})")

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

    # Compute efficiency metrics
    efficiency = compute_efficiency_metrics(naive_metrics, eab_metrics)

    # Compute branching stats
    branching_stats = compute_branching_stats(eab_metrics)

    # Compile result
    result = {
        'prompt_id': prompt['id'],
        'domain': domain_name,
        'prompt_length': prompt_length,
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
    print(f"      → Branches: {branching_stats.get('total_branches', 0)}")

    return result


def main():
    """Main experiment runner."""
    print("=" * 70)
    print("EXPERIMENT 1.A.4: SPEEDUP VS DOMAIN")
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
        domains = config['debug']['domains']
        prompts_per_domain = config['debug']['prompts_per_domain']
    else:
        print("   🚀 FULL EXPERIMENT MODE")
        domains = config['domains']
        prompts_per_domain = config['prompts_per_domain']

    print(f"   Domains to test: {[d['name'] for d in domains]}")
    print(f"   Prompts per domain: {prompts_per_domain}")
    print(f"   Target prompt length: ~{config['prompt_length']} tokens")
    print(f"   Device: {config['model']['device']}")

    # Initialize model and EAB (reuse for all domains)
    print("\n2. Initializing model and EAB...")
    device = config['model']['device']

    eab = EntropyAdaptiveBranching(
        model_name=config['model']['name'],
        entropy_threshold=config['eab']['entropy_threshold'],
        branch_factor=config['eab']['branch_factor'],
        max_paths=config['eab']['max_paths'],
        device=device
    )
    print(f"   ✓ EAB initialized with threshold={config['eab']['entropy_threshold']}")

    # Get model and tokenizer for naive sampling
    model = eab.model
    tokenizer = eab.tokenizer
    print(f"   ✓ Model: {config['model']['name']}")

    # Prepare results directory
    results_dir = experiment_dir / config['output']['results_dir']
    results_dir.mkdir(exist_ok=True, parents=True)
    results_file = results_dir / "raw_results.json"

    # Initialize results file
    save_json({
        'metadata': {
            'config': config,
            'total_experiments': len(domains) * prompts_per_domain
        },
        'results': []
    }, results_file)

    # Run experiments
    print("\n3. Running experiments...")
    print("-" * 70)

    all_results = []
    total_experiments = len(domains) * prompts_per_domain
    experiment_count = 0

    for domain_cfg in domains:
        domain_name = domain_cfg['name']
        domain_desc = domain_cfg['description']

        print(f"\n📂 Domain: {domain_name} - {domain_desc}")
        print("-" * 70)

        # Load domain-specific prompts
        print(f"   Loading prompts for {domain_name}...")
        prompts = load_domain_prompts(domain_cfg)[:prompts_per_domain]
        print(f"   ✓ Loaded {len(prompts)} prompts")

        # Run on each prompt
        for prompt in prompts:
            experiment_count += 1
            print(f"\n[{experiment_count}/{total_experiments}]")

            try:
                result = run_single_experiment(
                    prompt, eab, model, tokenizer, config, domain_name
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

    # Quick summary by domain
    print("\n" + "=" * 70)
    print("EXPERIMENT COMPLETE - SUMMARY BY DOMAIN")
    print("=" * 70)

    for domain_cfg in domains:
        domain_name = domain_cfg['name']
        domain_results = [r for r in all_results if r['domain'] == domain_name]
        if domain_results:
            avg_speedup = sum(r['efficiency']['speedup_token_steps'] for r in domain_results) / len(domain_results)
            avg_branches = sum(r['branching_stats'].get('total_branches', 0) for r in domain_results) / len(domain_results)
            print(f"{domain_name:15s}: {len(domain_results):2d} prompts, "
                  f"Avg speedup: {avg_speedup:.2f}×, Avg branches: {avg_branches:.1f}")

    print(f"\nResults saved to: {results_file}")
    print("\nNext steps:")
    print("  1. Run: python analyze_results.py")
    print("  2. Run: python plot_results.py")
    print("=" * 70)


if __name__ == "__main__":
    main()
