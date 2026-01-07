#!/usr/bin/env python3
"""
Interactive Demo for Entropy-Adaptive Branching (EAB)

Visualize how EAB works: entropy spikes, branching decisions, sample trees,
and resource comparisons with naive sampling.

Usage:
    # Interactive mode (prompts for inputs)
    python interactive_demo.py

    # Command-line mode
    python interactive_demo.py \
        --prompt "The capital of France is" \
        --threshold 0.3 \
        --branch-factor 3 \
        --max-tokens 20 \
        --max-paths 20 \
        --temperature 0.8 \
        --save-plots
"""

import sys
import os
import time
import argparse
import tracemalloc
import traceback
from pathlib import Path

# Add parent directory to path to import eab
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from utils import (
    plot_entropy_vs_tokens,
    plot_sample_tree,
    plot_resource_comparison,
    save_samples_to_file
)

try:
    from eab.core import EntropyAdaptiveBranching as EAB_Original
    from eab.core_cow import EntropyAdaptiveBranching as EAB_COW
except ImportError:
    print("Error: Could not import EntropyAdaptiveBranching")
    print("Make sure you're running from the demos/ directory")
    sys.exit(1)


def get_user_input():
    """Interactive mode: prompt user for all parameters."""
    print("\n" + "="*60)
    print("  EAB Interactive Demo - Verify Branching Behavior")
    print("="*60 + "\n")

    # Model selection
    print("Select a model:")
    print("  1. Llama-3.2-1B-Instruct (1B params, fastest)")
    print("  2. Llama-3.2-3B-Instruct (3B params)")
    print("  3. Qwen2.5-1.5B-Instruct (1.5B params, fast)")
    print("  4. Qwen2.5-3B-Instruct (3B params, default)")
    print("  5. Qwen2.5-7B-Instruct (7B params, high quality)")
    print("  6. Custom (enter model name manually)")
    print("\nChoice (1-6, default: 4):")

    model_choice = input("> ").strip()

    model_map = {
        '1': 'meta-llama/Llama-3.2-1B-Instruct',
        '2': 'meta-llama/Llama-3.2-3B-Instruct',
        '3': 'Qwen/Qwen2.5-1.5B-Instruct',
        '4': 'Qwen/Qwen2.5-3B-Instruct',
        '5': 'Qwen/Qwen2.5-7B-Instruct',
    }

    if model_choice == '6':
        print("\nEnter custom model name (HuggingFace format):")
        model_name = input("> ").strip()
        if not model_name:
            model_name = 'Qwen/Qwen2.5-3B-Instruct'
            print(f"No model entered, using default: {model_name}")
    elif model_choice in model_map:
        model_name = model_map[model_choice]
        print(f"Selected: {model_name}")
    else:
        model_name = 'Qwen/Qwen2.5-3B-Instruct'
        print(f"Invalid choice, using default: {model_name}")

    # Check if model requires HuggingFace token (e.g., Llama models)
    hf_token = None
    if 'llama' in model_name.lower() or 'meta-llama' in model_name.lower():
        print("\n⚠ This model requires a HuggingFace token (gated model).")
        print("You need to:")
        print("  1. Accept the license at https://huggingface.co/" + model_name)
        print("  2. Get your token from https://huggingface.co/settings/tokens")
        print("\nEnter your HuggingFace token (or press Enter to skip):")
        hf_token = input("> ").strip()
        if not hf_token:
            print("⚠ Warning: No token provided. Model loading may fail for gated models.")
        else:
            print("✓ Token provided")

    # Get prompt
    print("\nEnter your prompt (or press Enter for default):")
    prompt = input("> ").strip()
    if not prompt:
        prompt = "The capital of France is"
        print(f"Using default: '{prompt}'")

    # Get threshold
    print("\nEntropy threshold (0.0-1.0, default: 0.055):")
    threshold_str = input("> ").strip()
    threshold = float(threshold_str) if threshold_str else 0.055

    # Get branch factor
    print("\nBranch factor (how many paths to create, default: 3):")
    branch_str = input("> ").strip()
    branch_factor = int(branch_str) if branch_str else 3

    # Get max tokens
    print("\nMax new tokens to generate (default: 20):")
    tokens_str = input("> ").strip()
    max_tokens = int(tokens_str) if tokens_str else 20

    # Get max paths
    print("\nMax total paths (default: 20):")
    paths_str = input("> ").strip()
    max_paths = int(paths_str) if paths_str else 20

    # Get temperature
    print("\nTemperature (0.0-2.0, default: 0.8):")
    temp_str = input("> ").strip()
    temperature = float(temp_str) if temp_str else 0.8

    # Device selection
    print("\nDevice (cpu/cuda, default: cuda):")
    device_str = input("> ").strip().lower()
    device = device_str if device_str in ['cpu', 'cuda'] else 'cuda'
    if device == 'cuda':
        print(f"Selected: CUDA (GPU) with FP16 precision")
    else:
        print(f"Selected: CPU")

    # Save plots?
    print("\nSave plots to demo_results/? (y/n, default: y):")
    save_str = input("> ").strip().lower()
    save_plots = save_str != 'n'

    # COW cache?
    print("\nUse Copy-on-Write cache for memory efficiency? (y/n, default: y):")
    print("  (COW cache reduces memory usage by 60-70% during branching)")
    cow_str = input("> ").strip().lower()
    use_cow = cow_str != 'n'

    print("\n" + "-"*60)

    return {
        'model': model_name,
        'device': device,
        'prompt': prompt,
        'threshold': threshold,
        'branch_factor': branch_factor,
        'max_tokens': max_tokens,
        'max_paths': max_paths,
        'temperature': temperature,
        'save_plots': save_plots,
        'hf_token': hf_token,
        'use_cow': use_cow
    }


def run_naive_sampling(model, tokenizer, prompt, num_samples, max_tokens, temperature, device):
    """Run naive sampling for comparison."""
    print(f"\n[Naive] Generating {num_samples} samples independently...")

    # Start tracking - use GPU memory if CUDA, otherwise Python memory
    if device == 'cuda':
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        mem_before = torch.cuda.memory_allocated()
    else:
        tracemalloc.start()

    start_time = time.time()
    total_tokens = 0

    samples = []

    # Use chat template for fair comparison
    messages = [{"role": "user", "content": prompt}]
    formatted_prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    prompt_ids = tokenizer.encode(formatted_prompt, return_tensors='pt').to(device)

    for i in range(num_samples):
        output_ids = model.generate(
            prompt_ids,
            max_new_tokens=max_tokens,
            temperature=temperature,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id
        )

        # Count only generated tokens
        generated_ids = output_ids[0][prompt_ids.shape[1]:]
        total_tokens += len(generated_ids)

        full_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
        generated_only = tokenizer.decode(generated_ids, skip_special_tokens=True)

        samples.append({
            'text': full_text,
            'generated_only': generated_only,
            'num_tokens': len(generated_ids)
        })

    # Stop tracking
    end_time = time.time()

    if device == 'cuda':
        mem_after = torch.cuda.memory_allocated()
        peak = torch.cuda.max_memory_allocated()
        current = mem_after - mem_before
    else:
        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()

    metrics = {
        'wall_time': end_time - start_time,
        'total_tokens': total_tokens,
        'peak_memory_mb': peak / 1024 / 1024,
        'num_samples': num_samples
    }

    print(f"  ✓ Generated {num_samples} samples")
    print(f"  ✓ Total tokens: {total_tokens}")
    print(f"  ✓ Time: {metrics['wall_time']:.2f}s")

    return samples, metrics


def display_summary(eab_samples, eab_metrics, naive_samples, naive_metrics, params):
    """Display summary statistics."""
    print("\n" + "="*60)
    print("  SUMMARY")
    print("="*60)

    print(f"\nPrompt: '{params['prompt']}'")
    print(f"Threshold: {params['threshold']}")
    print(f"Branch Factor: {params['branch_factor']}")
    print(f"Max Tokens: {params['max_tokens']}")
    print(f"Implementation: {'COW (Copy-on-Write)' if params.get('use_cow', True) else 'Original (Deep Copy)'}")

    print(f"\n--- EAB Results ---")
    print(f"Samples generated: {len(eab_samples)}")
    print(f"Total branches: {eab_metrics['total_branches']}")
    print(f"Branch positions: {eab_metrics['branch_positions'][:10]}{'...' if len(eab_metrics['branch_positions']) > 10 else ''}")
    print(f"Total tokens: {eab_metrics['total_tokens']}")
    print(f"Wall time: {eab_metrics['wall_time']:.2f}s")
    print(f"Peak memory: {eab_metrics['peak_memory_mb']:.1f} MB")

    # Display entropy statistics
    if 'entropy_history' in eab_metrics and eab_metrics['entropy_history']:
        entropy_stats = eab_metrics['entropy_history']['statistics']
        print(f"\nEntropy Statistics:")
        print(f"  Mean entropy: {entropy_stats['mean_entropy']:.3f}")
        print(f"  Max entropy: {entropy_stats['max_entropy']:.3f}")
        print(f"  Min entropy: {entropy_stats['min_entropy']:.3f}")
        print(f"  Branch rate: {entropy_stats['branch_rate']:.1%}")

    print(f"\n--- Naive Results ---")
    print(f"Samples generated: {len(naive_samples)}")
    print(f"Total tokens: {naive_metrics['total_tokens']}")
    print(f"Wall time: {naive_metrics['wall_time']:.2f}s")
    print(f"Peak memory: {naive_metrics['peak_memory_mb']:.1f} MB")

    print(f"\n--- Comparison ---")
    if naive_metrics['wall_time'] > 0:
        speedup = naive_metrics['wall_time'] / eab_metrics['wall_time']
        token_ratio = naive_metrics['total_tokens'] / max(eab_metrics['total_tokens'], 1)
        memory_ratio = naive_metrics['peak_memory_mb'] / max(eab_metrics['peak_memory_mb'], 1)

        print(f"Speedup: {speedup:.2f}×")
        print(f"Token reduction: {token_ratio:.2f}×")
        print(f"Memory reduction: {memory_ratio:.2f}×")

    print("\n" + "="*60)


def display_branching_info(samples, threshold):
    """Display information about when and why branching occurred."""
    print("\n--- Branching Analysis ---")

    # Collect all branch points
    all_branch_points = set()
    for sample in samples:
        branch_points = sample.get('branch_points', [])
        all_branch_points.update(branch_points)

    if not all_branch_points:
        print("  No branching occurred (entropy never exceeded threshold)")
        print(f"  Threshold was: {threshold}")
        print("  Try:")
        print("    - Lower threshold (e.g., 0.2)")
        print("    - More uncertain prompt")
        print("    - Higher temperature")
        return

    print(f"  Branching occurred at {len(all_branch_points)} positions:")
    for pos in sorted(all_branch_points)[:10]:
        # Find samples that branched at this position
        samples_at_pos = [s for s in samples if pos in s.get('branch_points', [])]
        print(f"    Position {pos}: {len(samples_at_pos)} paths branched")

    if len(all_branch_points) > 10:
        print(f"    ... and {len(all_branch_points) - 10} more positions")


def main():
    parser = argparse.ArgumentParser(description='Interactive EAB Demo')
    parser.add_argument('--prompt', type=str, help='Input prompt')
    parser.add_argument('--threshold', type=float, default=0.4, help='Entropy threshold')
    parser.add_argument('--branch-factor', type=int, default=3, help='Branch factor')
    parser.add_argument('--max-tokens', type=int, default=20, help='Max new tokens')
    parser.add_argument('--max-paths', type=int, default=20, help='Max paths')
    parser.add_argument('--temperature', type=float, default=0.8, help='Temperature')
    parser.add_argument('--save-plots', action='store_true', help='Save plots to disk')
    parser.add_argument('--model', type=str, default='Qwen/Qwen2.5-3B-Instruct', help='Model name (default: Qwen/Qwen2.5-3B-Instruct)')
    parser.add_argument('--device', type=str, default='cuda', help='Device (cpu/cuda, default: cuda)')
    parser.add_argument('--hf-token', type=str, default=None, help='HuggingFace token for gated models')
    parser.add_argument('--use-cow', action='store_true', default=True, help='Use Copy-on-Write cache for memory efficiency (default: True)')
    parser.add_argument('--no-cow', dest='use_cow', action='store_false', help='Disable COW cache (use original deep copy)')

    args = parser.parse_args()

    # Get parameters (interactive or command-line)
    if args.prompt is None:
        # Interactive mode (model and device selected interactively)
        params = get_user_input()
    else:
        # Command-line mode
        params = {
            'prompt': args.prompt,
            'threshold': args.threshold,
            'branch_factor': args.branch_factor,
            'max_tokens': args.max_tokens,
            'max_paths': args.max_paths,
            'temperature': args.temperature,
            'save_plots': args.save_plots,
            'model': args.model,
            'device': args.device,
            'hf_token': args.hf_token,
            'use_cow': args.use_cow
        }

    print("\n[Setup] Loading model and tokenizer...")

    # Select implementation based on use_cow flag
    if params.get('use_cow', True):
        EntropyAdaptiveBranching = EAB_COW
        impl_name = "COW (Copy-on-Write cache)"
        print(f"  Using COW implementation for memory efficiency")
    else:
        EntropyAdaptiveBranching = EAB_Original
        impl_name = "Original (deep copy cache)"
        print(f"  Using original implementation")

    # Initialize EAB
    try:
        # Prepare model loading kwargs
        model_kwargs = {
            'model_name': params['model'],
            'entropy_threshold': params['threshold'],
            'branch_factor': params['branch_factor'],
            'max_paths': params['max_paths'],
            'device': params['device']
        }

        # Add FP16 for CUDA
        if params['device'] == 'cuda':
            model_kwargs['torch_dtype'] = torch.float16
            print(f"  Using FP16 precision to reduce memory usage")

        # Add HF token if provided (for gated models like Llama)
        if params.get('hf_token'):
            # Token will be used via huggingface_hub login
            from huggingface_hub import login
            login(token=params['hf_token'])
            print(f"  ✓ Logged in to HuggingFace")

        eab = EntropyAdaptiveBranching(**model_kwargs)
        print(f"  ✓ Loaded {params['model']} ({impl_name}) on {params['device']}")
    except Exception as e:
        print(f"  ✗ Error loading EAB: {e}")
        traceback.print_exc()
        return

    # Also load model/tokenizer for naive comparison
    print("\n[Setup] Loading model for naive comparison...")
    try:
        load_kwargs = {}
        if params['device'] == 'cuda':
            load_kwargs['torch_dtype'] = torch.float16

        model = AutoModelForCausalLM.from_pretrained(params['model'], **load_kwargs).to(params['device'])
        tokenizer = AutoTokenizer.from_pretrained(params['model'])
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        print(f"  ✓ Loaded model for naive sampling")
    except Exception as e:
        print(f"  ✗ Error loading naive model: {e}")
        print(f"  Continuing without naive comparison...")
        model = None
        tokenizer = None

    # Run EAB generation
    print(f"\n[EAB] Generating with threshold={params['threshold']}...")

    # Measure GPU memory if using CUDA, otherwise Python memory
    if params['device'] == 'cuda':
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        mem_before_eab = torch.cuda.memory_allocated()
    else:
        tracemalloc.start()

    start_time = time.time()

    try:
        eab_samples = eab.generate(
            prompt=params['prompt'],
            max_new_tokens=params['max_tokens'],
            temperature=params['temperature'],
            use_chat_template=True  # Use chat template for coherent results
        )
    except Exception as e:
        print(f"  ✗ Error during EAB generation: {e}")
        traceback.print_exc()
        return

    end_time = time.time()

    if params['device'] == 'cuda':
        mem_after_eab = torch.cuda.memory_allocated()
        peak = torch.cuda.max_memory_allocated()
        current = mem_after_eab - mem_before_eab
    else:
        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()

    # Extract entropy history from EAB
    entropy_data = eab.get_entropy_history()

    # Extract metrics from EAB samples
    all_branch_points = set()
    total_tokens = 0

    for sample in eab_samples:
        branch_points = sample.get('branch_points', [])
        all_branch_points.update(branch_points)
        total_tokens += sample.get('length', len(sample.get('tokens', [])))

    eab_metrics = {
        'wall_time': end_time - start_time,
        'total_tokens': total_tokens,
        'peak_memory_mb': peak / 1024 / 1024,
        'total_branches': len(all_branch_points),
        'branch_positions': sorted(all_branch_points),
        'num_samples': len(eab_samples),
        'entropy_history': entropy_data  # Store full entropy data
    }

    print(f"  ✓ Generated {len(eab_samples)} samples")
    print(f"  ✓ Total branches: {len(all_branch_points)}")
    print(f"  ✓ Time: {eab_metrics['wall_time']:.2f}s")

    # Run naive sampling with SAME sample count for fair comparison
    if model is not None and tokenizer is not None:
        naive_samples, naive_metrics = run_naive_sampling(
            model, tokenizer, params['prompt'],
            num_samples=len(eab_samples),
            max_tokens=params['max_tokens'],
            temperature=params['temperature'],
            device=params['device']
        )
    else:
        print("\n[Naive] Skipping naive comparison (model not loaded)")
        naive_samples = []
        naive_metrics = {'peak_memory_mb': 0, 'wall_time': 0, 'tokens_per_sec': 0}

    # Display results
    display_summary(eab_samples, eab_metrics, naive_samples, naive_metrics, params)
    display_branching_info(eab_samples, params['threshold'])

    # Create visualizations
    print("\n[Visualization] Generating plots...")

    results_dir = Path(__file__).parent / 'demo_results'
    results_dir.mkdir(exist_ok=True)

    try:
        # 1. Entropy vs Tokens plot
        print("  - Entropy vs Tokens...")
        plot_entropy_vs_tokens(
            eab_samples,
            params['threshold'],
            entropy_data=entropy_data,
            save_path=results_dir / 'entropy_vs_tokens.png' if params['save_plots'] else None
        )

        # 2. Sample Tree
        print("  - Sample Tree...")
        plot_sample_tree(
            eab_samples,
            save_path=results_dir / 'sample_tree.png' if params['save_plots'] else None
        )

        # 3. Resource Comparison
        print("  - Resource Comparison...")
        plot_resource_comparison(
            eab_metrics,
            naive_metrics,
            save_path=results_dir / 'resource_comparison.png' if params['save_plots'] else None
        )

        # 4. Save sample texts (both EAB and Naive)
        print("  - Saving sample texts...")
        save_samples_to_file(
            eab_samples,
            results_dir / 'all_samples.txt',
            params['prompt'],
            naive_samples=naive_samples
        )

        # 5. Save entropy data for later analysis
        print("  - Saving entropy data...")
        import json
        with open(results_dir / 'entropy_data.json', 'w') as f:
            # Convert numpy arrays to lists if needed
            entropy_save = {
                'positions': list(entropy_data['positions']),
                'entropies': list(entropy_data['entropies']),
                'branched': list(entropy_data['branched']),
                'statistics': entropy_data['statistics'],
                'threshold': params['threshold'],
                'prompt': params['prompt']
            }
            json.dump(entropy_save, f, indent=2)

        if params['save_plots']:
            print(f"\n  ✓ All outputs saved to: {results_dir}/")
        else:
            print("\n  ✓ Plots displayed (not saved)")

        # Show plots if not saving
        if not params['save_plots']:
            plt.show()

    except Exception as e:
        print(f"  ✗ Error creating visualizations: {e}")
        traceback.print_exc()

    print("\n" + "="*60)
    print("  Demo Complete!")
    print("="*60 + "\n")


if __name__ == '__main__':
    main()
