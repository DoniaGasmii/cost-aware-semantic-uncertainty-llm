"""
Run Experiment 1.C.4: Temperature × Threshold Interaction

Variables: temperature × entropy_threshold (2D grid)
Fixed: branch_factor, max_paths, prompt_length
"""

import sys
import json
import yaml
import torch
from pathlib import Path
from typing import Dict, Any, List

experiment_dir = Path(__file__).parent
analysis_dir = experiment_dir.parent.parent
eab_dir = analysis_dir.parent
sys.path.insert(0, str(analysis_dir))
sys.path.insert(0, str(eab_dir))

from eab.core_cow import EntropyAdaptiveBranching
from utils.metrics import MetricsTracker, compute_efficiency_metrics, compute_branching_stats
from utils.data_utils import save_json, append_result


def load_config() -> Dict[str, Any]:
    config_path = experiment_dir / "config.yaml"
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def load_prompts(prompts_dir: Path) -> List[Dict[str, Any]]:
    prompts_file = prompts_dir / "prompts.json"
    with open(prompts_file, 'r') as f:
        data = json.load(f)
    return data['prompts']


def compute_quality_metrics(samples: List[Dict]) -> Dict[str, float]:
    if not samples:
        return {}

    all_tokens = []
    for sample in samples:
        tokens = sample.get('tokens', [])
        all_tokens.extend(tokens)

    unique_ratio = len(set(all_tokens)) / len(all_tokens) if all_tokens else 0

    return {
        'unique_tokens_ratio': unique_ratio,
        'num_unique_tokens': len(set(all_tokens)),
        'total_tokens': len(all_tokens)
    }


def run_eab_generation(
    prompt_text: str,
    eab: EntropyAdaptiveBranching,
    max_new_tokens: int,
    temperature: float,
    tracker: MetricsTracker
) -> tuple:
    tracker.start()

    samples = eab.generate(
        prompt=prompt_text,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        use_chat_template=True
    )

    num_samples = len(samples)
    tracker.record_samples(num_samples)

    total_tokens = sum(s.get('length', len(s['tokens'])) for s in samples)
    tracker.record_token_steps(total_tokens)

    all_branch_points = set()
    for sample in samples:
        all_branch_points.update(sample.get('branch_points', []))

    for bp in sorted(all_branch_points):
        tracker.record_branch(bp)

    tracker.record_final_paths(num_samples)
    tracker.update_memory()

    metrics = tracker.stop()

    # Extract entropy stats
    all_entropies = []
    for sample in samples:
        if 'entropy_history' in sample:
            all_entropies.extend(sample['entropy_history'])

    entropy_stats = {}
    if all_entropies:
        import numpy as np
        entropy_stats = {
            'avg_entropy': float(np.mean(all_entropies)),
            'max_entropy': float(np.max(all_entropies)),
            'entropy_std': float(np.std(all_entropies)),
        }

    return samples, metrics, entropy_stats


def run_naive_generation(
    prompt_text: str,
    num_samples: int,
    max_new_tokens: int,
    temperature: float,
    model,
    tokenizer,
    device: str,
    tracker: MetricsTracker
) -> tuple:
    tracker.start()

    samples = []
    prompt_ids = tokenizer.encode(prompt_text, return_tensors='pt').to(device)

    for _ in range(num_samples):
        with torch.no_grad():
            output_ids = model.generate(
                prompt_ids,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                do_sample=True,
                top_p=1.0,
                pad_token_id=tokenizer.eos_token_id
            )

        generated_ids = output_ids[0][prompt_ids.shape[1]:]
        generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True)
        full_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)

        samples.append({
            'text': full_text,
            'generated_only': generated_text,
            'tokens': output_ids[0].tolist(),
            'num_generated_tokens': len(generated_ids)
        })

        tracker.record_token_steps(len(output_ids[0]))
        tracker.update_memory()

    tracker.record_samples(num_samples)
    metrics = tracker.stop()

    return samples, metrics


def run_single_experiment(
    prompt: Dict[str, Any],
    temperature: float,
    entropy_threshold: float,
    model,
    tokenizer,
    config: Dict[str, Any]
) -> Dict[str, Any]:
    """Run experiment with specific (temperature, entropy_threshold) combination."""
    prompt_text = prompt['text']

    print(f"  Prompt: {prompt['id']} | Temp: {temperature}, Threshold: {entropy_threshold}")

    # Initialize EAB with specific parameters
    device = config['model']['device']
    eab = EntropyAdaptiveBranching(
        model_name=config['model']['name'],
        entropy_threshold=entropy_threshold,  # VARIABLE 1
        branch_factor=config['eab']['branch_factor'],
        max_paths=config['eab']['max_paths'],
        device=device,
        torch_dtype=torch.float16 if device == 'cuda' else None
    )

    # Run EAB with specific temperature
    print(f"    Running EAB...")
    eab_tracker = MetricsTracker(device=device)
    eab_samples, eab_metrics, entropy_stats = run_eab_generation(
        prompt_text, eab, config['generation']['max_new_tokens'], temperature, eab_tracker
    )
    num_eab_samples = len(eab_samples)
    print(f"      ✓ Generated {num_eab_samples} samples")

    # Run Naive with same temperature
    print(f"    Running Naive ({num_eab_samples} samples)...")
    naive_tracker = MetricsTracker(device=device)
    naive_samples, naive_metrics = run_naive_generation(
        prompt_text, num_eab_samples, config['generation']['max_new_tokens'],
        temperature, model, tokenizer, device, naive_tracker
    )
    print(f"      ✓ Generated {len(naive_samples)} samples")

    # Compute metrics
    efficiency = compute_efficiency_metrics(naive_metrics, eab_metrics)
    branching_stats = compute_branching_stats(eab_metrics)

    eab_quality = compute_quality_metrics(eab_samples)
    naive_quality = compute_quality_metrics(naive_samples)

    result = {
        'prompt_id': prompt['id'],
        'temperature': temperature,
        'entropy_threshold': entropy_threshold,
        'prompt_length': prompt['actual_length'],
        'num_eab_samples': num_eab_samples,
        'num_naive_samples': num_eab_samples,
        'eab_metrics': eab_metrics.to_dict(),
        'naive_metrics': naive_metrics.to_dict(),
        'efficiency': efficiency,
        'branching_stats': branching_stats,
        'entropy_stats': entropy_stats,
        'eab_quality': eab_quality,
        'naive_quality': naive_quality,
    }

    print(f"      → Speedup: {efficiency['speedup_token_steps']:.2f}×")

    # Cleanup
    del eab
    torch.cuda.empty_cache() if torch.cuda.is_available() else None

    return result


def main():
    print("=" * 70)
    print("EXPERIMENT 1.C.4: TEMPERATURE × THRESHOLD INTERACTION")
    print("=" * 70)

    print("\n1. Loading configuration...")
    config = load_config()

    torch.manual_seed(config['seed'])
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config['seed'])

    if config['debug']['enabled']:
        print("   🔧 DEBUG MODE")
        temperatures = config['debug']['temperatures']
        entropy_thresholds = config['debug']['entropy_thresholds']
        prompts_per_combination = config['debug']['prompts_per_combination']
    else:
        print("   🚀 FULL EXPERIMENT MODE")
        temperatures = config['temperatures']
        entropy_thresholds = config['entropy_thresholds']
        prompts_per_combination = config['prompts_per_combination']

    print(f"   Temperatures: {temperatures}")
    print(f"   Entropy thresholds: {entropy_thresholds}")
    print(f"   Prompts per combination: {prompts_per_combination}")
    print(f"   Total combinations: {len(temperatures) * len(entropy_thresholds)}")
    print(f"   Device: {config['model']['device']}")

    print("\n2. Loading shared model...")
    from transformers import AutoModelForCausalLM, AutoTokenizer
    device = config['model']['device']

    model = AutoModelForCausalLM.from_pretrained(
        config['model']['name'],
        torch_dtype=torch.float16 if device == 'cuda' else None,
        device_map=device
    )
    tokenizer = AutoTokenizer.from_pretrained(config['model']['name'])
    print(f"   ✓ Model: {config['model']['name']}")

    results_dir = experiment_dir / config['output']['results_dir']
    results_dir.mkdir(exist_ok=True, parents=True)
    results_file = results_dir / "raw_results.json"

    total_combinations = len(temperatures) * len(entropy_thresholds)
    save_json({
        'metadata': {
            'config': config,
            'total_experiments': total_combinations * prompts_per_combination
        },
        'results': []
    }, results_file)

    print("\n3. Loading prompts...")
    prompts_dir = experiment_dir / "prompts" / f"length_{config['prompt_length']:03d}"
    prompts = load_prompts(prompts_dir)[:prompts_per_combination]
    print(f"   ✓ Loaded {len(prompts)} prompts")

    print("\n4. Running experiments (2D grid)...")
    print("-" * 70)

    all_results = []
    total_experiments = len(temperatures) * len(entropy_thresholds) * len(prompts)
    experiment_count = 0

    for temp in temperatures:
        for threshold in entropy_thresholds:
            print(f"\n🎯 Temperature: {temp}, Threshold: {threshold}")
            print("-" * 70)

            for prompt in prompts:
                experiment_count += 1
                print(f"\n[{experiment_count}/{total_experiments}]")

                try:
                    result = run_single_experiment(
                        prompt, temp, threshold, model, tokenizer, config
                    )
                    all_results.append(result)

                    if config['output']['save_intermediate']:
                        append_result(result, results_file)

                except Exception as e:
                    print(f"    ✗ Error: {e}")
                    import traceback
                    traceback.print_exc()
                    continue

    print("\n" + "=" * 70)
    print("5. Saving final results...")
    save_json({
        'metadata': {
            'config': config,
            'total_experiments': len(all_results),
            'completed_successfully': len(all_results),
        },
        'results': all_results
    }, results_file)
    print(f"   ✓ Results saved to {results_file}")

    print("\n" + "=" * 70)
    print("EXPERIMENT COMPLETE")
    print("=" * 70)
    print(f"Total experiments: {len(all_results)}")
    print(f"Grid size: {len(temperatures)}×{len(entropy_thresholds)} = {total_combinations} combinations")
    print("\nNext steps:")
    print("  1. Run: python analyze_results.py")
    print("  2. Run: python plot_results.py (will generate 2D heatmaps)")
    print("=" * 70)


if __name__ == "__main__":
    main()
