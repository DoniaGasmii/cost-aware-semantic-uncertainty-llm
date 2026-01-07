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

    # 3-way comparison mode (Naive vs COW EAB vs Original EAB)
    python interactive_demo.py \
        --prompt "Name one important skill students should develop today." \
        --compare-all \
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

    # Comparison mode?
    print("\nRun 3-way comparison? (Naive vs COW EAB vs Original EAB) (y/n, default: n):")
    print("  This will run all three approaches and compare resource costs")
    compare_str = input("> ").strip().lower()
    compare_all = compare_str == 'y'

    use_cow = True  # Default
    if not compare_all:
        # COW cache choice only if not doing full comparison
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
        'use_cow': use_cow,
        'compare_all': compare_all
    }


def run_naive_sampling(model, tokenizer, prompt, num_samples, max_tokens, temperature, device):
    """Run naive sampling for comparison."""
    print(f"\n[Naive] Generating {num_samples} samples independently...")

    # Start tracking - measure generation overhead only (excluding model weights)
    if device == 'cuda':
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        mem_before = torch.cuda.memory_allocated()
        peak_before = mem_before  # Baseline (model weights)
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
        peak_total = torch.cuda.max_memory_allocated()
        # Generation overhead = delta from baseline (excludes model weights)
        generation_overhead = mem_after - mem_before
        peak_overhead = peak_total - peak_before
    else:
        generation_overhead, peak_overhead = tracemalloc.get_traced_memory()
        tracemalloc.stop()

    metrics = {
        'wall_time': end_time - start_time,
        'total_tokens': total_tokens,
        'peak_memory_mb': peak_overhead / 1024 / 1024,  # Only generation overhead
        'num_samples': num_samples
    }

    print(f"  ✓ Generated {num_samples} samples")
    print(f"  ✓ Total tokens: {total_tokens}")
    print(f"  ✓ Time: {metrics['wall_time']:.2f}s")

    return samples, metrics


def run_eab_generation(eab_instance, params, impl_name="EAB"):
    """Run EAB generation and collect metrics."""
    print(f"\n[{impl_name}] Generating with threshold={params['threshold']}...")

    # Measure generation overhead (excluding model weights)
    if params['device'] == 'cuda':
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        mem_before = torch.cuda.memory_allocated()
        peak_before = mem_before  # Baseline (model weights)
    else:
        tracemalloc.start()

    start_time = time.time()

    try:
        samples = eab_instance.generate(
            prompt=params['prompt'],
            max_new_tokens=params['max_tokens'],
            temperature=params['temperature'],
            use_chat_template=True  # Use chat template for coherent results
        )
    except Exception as e:
        print(f"  ✗ Error during {impl_name} generation: {e}")
        traceback.print_exc()
        return None, None

    end_time = time.time()

    if params['device'] == 'cuda':
        mem_after = torch.cuda.memory_allocated()
        peak_total = torch.cuda.max_memory_allocated()
        # Generation overhead = delta from baseline (excludes model weights)
        generation_overhead = mem_after - mem_before
        peak_overhead = peak_total - peak_before
    else:
        generation_overhead, peak_overhead = tracemalloc.get_traced_memory()
        tracemalloc.stop()

    # Extract entropy history
    entropy_data = eab_instance.get_entropy_history()

    # Extract metrics
    all_branch_points = set()
    total_tokens = 0

    for sample in samples:
        branch_points = sample.get('branch_points', [])
        all_branch_points.update(branch_points)
        total_tokens += sample.get('length', len(sample.get('tokens', [])))

    metrics = {
        'wall_time': end_time - start_time,
        'total_tokens': total_tokens,
        'peak_memory_mb': peak_overhead / 1024 / 1024,  # Only generation overhead
        'total_branches': len(all_branch_points),
        'branch_positions': sorted(all_branch_points),
        'num_samples': len(samples),
        'entropy_history': entropy_data  # Store full entropy data
    }

    print(f"  ✓ Generated {len(samples)} samples")
    print(f"  ✓ Total branches: {len(all_branch_points)}")
    print(f"  ✓ Time: {metrics['wall_time']:.2f}s")
    print(f"  ✓ Memory overhead: {metrics['peak_memory_mb']:.1f} MB")

    return samples, metrics


def display_three_way_comparison(naive_metrics, cow_metrics, original_metrics, params):
    """Display 3-way comparison table."""
    print("\n" + "="*80)
    print("  3-WAY RESOURCE COMPARISON")
    print("="*80)

    print(f"\nPrompt: '{params['prompt']}'")
    print(f"Threshold: {params['threshold']}, Branch Factor: {params['branch_factor']}, Max Tokens: {params['max_tokens']}")

    # Header
    print("\n" + "-"*80)
    print(f"{'Metric':<30} {'Naive':<15} {'COW EAB':<15} {'Original EAB':<15}")
    print("-"*80)

    # Samples
    print(f"{'Samples generated':<30} {naive_metrics['num_samples']:<15} {cow_metrics['num_samples']:<15} {original_metrics['num_samples']:<15}")

    # Time
    print(f"{'Wall time (s)':<30} {naive_metrics['wall_time']:<15.2f} {cow_metrics['wall_time']:<15.2f} {original_metrics['wall_time']:<15.2f}")

    # Memory
    print(f"{'Memory overhead (MB)':<30} {naive_metrics['peak_memory_mb']:<15.1f} {cow_metrics['peak_memory_mb']:<15.1f} {original_metrics['peak_memory_mb']:<15.1f}")

    # Tokens
    print(f"{'Total tokens':<30} {naive_metrics['total_tokens']:<15} {cow_metrics['total_tokens']:<15} {original_metrics['total_tokens']:<15}")

    # Branches (N/A for naive)
    print(f"{'Total branches':<30} {'N/A':<15} {cow_metrics.get('total_branches', 'N/A'):<15} {original_metrics.get('total_branches', 'N/A'):<15}")

    print("-"*80)

    # Speedup analysis (compared to naive)
    print("\n--- Speedup vs Naive ---")
    cow_speedup = naive_metrics['wall_time'] / max(cow_metrics['wall_time'], 0.001)
    orig_speedup = naive_metrics['wall_time'] / max(original_metrics['wall_time'], 0.001)
    print(f"  COW EAB:      {cow_speedup:.2f}×")
    print(f"  Original EAB: {orig_speedup:.2f}×")

    # Memory analysis
    print("\n--- Memory Overhead Comparison ---")
    print(f"  Naive:        {naive_metrics['peak_memory_mb']:.1f} MB (baseline)")
    print(f"  COW EAB:      {cow_metrics['peak_memory_mb']:.1f} MB ({cow_metrics['peak_memory_mb']/naive_metrics['peak_memory_mb']:.2f}× naive)")
    print(f"  Original EAB: {original_metrics['peak_memory_mb']:.1f} MB ({original_metrics['peak_memory_mb']/naive_metrics['peak_memory_mb']:.2f}× naive)")

    # COW savings vs Original
    cow_savings = (original_metrics['peak_memory_mb'] - cow_metrics['peak_memory_mb']) / original_metrics['peak_memory_mb'] * 100
    print(f"\n--- COW Memory Savings vs Original ---")
    print(f"  COW reduces memory by {cow_savings:.1f}% compared to deep copy")

    print("\n" + "="*80)


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
    print(f"Generation overhead: {eab_metrics['peak_memory_mb']:.1f} MB (excl. model weights)")

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
    print(f"Generation overhead: {naive_metrics['peak_memory_mb']:.1f} MB (excl. model weights)")

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
    parser.add_argument('--compare-all', action='store_true', help='Run 3-way comparison (Naive vs COW EAB vs Original EAB)')

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
            'use_cow': args.use_cow,
            'compare_all': args.compare_all
        }

    print("\n[Setup] Loading model and tokenizer...")

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

    # Initialize EAB implementations based on mode
    eab_cow = None
    eab_original = None
    eab = None

    if params.get('compare_all', False):
        # 3-way comparison mode: load both implementations
        print(f"  Loading both COW and Original implementations for comparison...")

        try:
            eab_cow = EAB_COW(**model_kwargs)
            print(f"  ✓ Loaded COW implementation")
        except Exception as e:
            print(f"  ✗ Error loading COW EAB: {e}")
            traceback.print_exc()
            return

        try:
            eab_original = EAB_Original(**model_kwargs)
            print(f"  ✓ Loaded Original implementation")
        except Exception as e:
            print(f"  ✗ Error loading Original EAB: {e}")
            traceback.print_exc()
            return
    else:
        # Single mode: load selected implementation
        if params.get('use_cow', True):
            EntropyAdaptiveBranching = EAB_COW
            impl_name = "COW (Copy-on-Write cache)"
            print(f"  Using COW implementation for memory efficiency")
        else:
            EntropyAdaptiveBranching = EAB_Original
            impl_name = "Original (deep copy cache)"
            print(f"  Using original implementation")

        try:
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

    # Generation phase
    if params.get('compare_all', False):
        # ====== 3-WAY COMPARISON MODE ======
        print("\n" + "="*80)
        print("  RUNNING 3-WAY COMPARISON")
        print("="*80)

        # Run COW EAB
        cow_samples, cow_metrics = run_eab_generation(eab_cow, params, impl_name="COW EAB")
        if cow_samples is None:
            return

        # Run Original EAB
        original_samples, original_metrics = run_eab_generation(eab_original, params, impl_name="Original EAB")
        if original_samples is None:
            return

        # Run Naive sampling (use same count as COW)
        if model is not None and tokenizer is not None:
            naive_samples, naive_metrics = run_naive_sampling(
                model, tokenizer, params['prompt'],
                num_samples=len(cow_samples),
                max_tokens=params['max_tokens'],
                temperature=params['temperature'],
                device=params['device']
            )
        else:
            print("\n[Naive] Skipping naive comparison (model not loaded)")
            naive_samples = []
            naive_metrics = {'peak_memory_mb': 0.001, 'wall_time': 0.001, 'total_tokens': 0, 'num_samples': 0}

        # Display 3-way comparison
        display_three_way_comparison(naive_metrics, cow_metrics, original_metrics, params)

        # Use COW samples/metrics for visualization
        eab_samples = cow_samples
        eab_metrics = cow_metrics
        entropy_data = cow_metrics.get('entropy_history')

    else:
        # ====== SINGLE MODE ======
        # Run EAB generation
        eab_samples, eab_metrics = run_eab_generation(eab, params, impl_name="EAB")
        if eab_samples is None:
            return

        entropy_data = eab_metrics.get('entropy_history')

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
            naive_metrics = {'peak_memory_mb': 0.001, 'wall_time': 0.001, 'total_tokens': 0, 'num_samples': 0}

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
