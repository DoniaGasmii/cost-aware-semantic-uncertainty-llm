"""
Experiment 1.B.1: EAB Branching vs Human Ambiguity

Tests whether EAB's branching behavior correlates with human-perceived ambiguity.
"""

import sys
import json
import yaml
import csv
import torch
import numpy as np
from pathlib import Path
from tqdm import tqdm
from typing import Dict, Any, List
from datetime import datetime

# Resolve paths
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[3]

# Add necessary paths
sys.path.insert(0, str(PROJECT_ROOT / "entropy-adaptive-branching"))
sys.path.insert(0, str(PROJECT_ROOT))

from eab.core import EntropyAdaptiveBranching


def load_config() -> Dict[str, Any]:
    """Load experiment configuration."""
    config_path = SCRIPT_DIR / "config.yaml"
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def load_prompts_with_ratings() -> List[Dict[str, Any]]:
    """Load prompts and their human ambiguity ratings."""
    prompts_file = SCRIPT_DIR / "prompts_with_ratings.csv"
    prompts = []

    with open(prompts_file, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            prompts.append({
                'prompt': row['prompt'].strip('"'),
                'human_ambiguity_score': float(row['human_ambiguity_score'])
            })

    return prompts


def setup_eab(config: Dict[str, Any]) -> EntropyAdaptiveBranching:
    """Initialize EAB generator."""
    device = config['model']['device']
    dtype = torch.float16 if config['model']['torch_dtype'] == 'float16' else torch.float32

    eab = EntropyAdaptiveBranching(
        model_name=config['model']['name'],
        device=device,
        entropy_threshold=config['eab']['entropy_threshold'],
        branch_factor=3,  # Default branching factor
        max_paths=config['generation']['num_samples'],
        torch_dtype=dtype,
        use_cow=True
    )

    return eab


def run_eab_on_prompt(prompt: str, eab: EntropyAdaptiveBranching, config: Dict[str, Any]) -> Dict[str, Any]:
    """Run EAB on a single prompt and collect branching metrics."""

    # Generate with EAB
    results = eab.generate(
        prompt=prompt,
        max_new_tokens=config['generation']['max_new_tokens'],
        temperature=config['generation']['temperature'],
        top_p=config['generation']['top_p'],
        return_metadata=True,
        show_progress=False,  # Disable progress for each prompt
        use_chat_template=True  # For instruct models
    )

    # Get entropy history (branching information)
    entropy_history = eab.get_entropy_history()

    # Extract samples
    samples = [r['text'] for r in results]

    # Count total branches
    branching_decisions = entropy_history.get('branched', [])
    num_branches = sum(branching_decisions)

    # Get branch positions (token indices where branching occurred)
    branch_positions = [i for i, branched in enumerate(branching_decisions) if branched]

    # Get entropy values at branch points
    all_entropies = entropy_history.get('entropies', [])
    branch_entropies = [all_entropies[i] for i in branch_positions if i < len(all_entropies)]

    # Calculate average entropy at branches
    avg_branch_entropy = np.mean(branch_entropies) if branch_entropies else 0.0

    # Calculate branching frequency (branches per token)
    total_tokens = len(branching_decisions)
    branching_frequency = num_branches / total_tokens if total_tokens > 0 else 0.0

    # Did it branch at all?
    did_branch = num_branches > 0

    # Calculate average entropy overall
    avg_entropy = np.mean(all_entropies) if all_entropies else 0.0

    return {
        'prompt': prompt,
        'samples': samples,
        'num_branches': num_branches,
        'branch_positions': branch_positions,
        'branch_entropies': branch_entropies,
        'avg_branch_entropy': avg_branch_entropy,
        'avg_entropy': avg_entropy,
        'branching_frequency': branching_frequency,
        'did_branch': did_branch,
        'total_tokens': total_tokens,
        'num_samples': len(samples)
    }


def save_json(data, path: Path):
    """Save data to JSON file."""
    with open(path, 'w') as f:
        json.dump(data, f, indent=2, default=str)


def main():
    print("=" * 70)
    print("EXPERIMENT 1.B.1: EAB vs HUMAN AMBIGUITY CORRELATION")
    print("=" * 70)

    # Load configuration
    config = load_config()

    # Set random seeds
    torch.manual_seed(config['seed'])
    np.random.seed(config['seed'])
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config['seed'])

    # Load prompts with ratings
    prompts_data = load_prompts_with_ratings()
    print(f"\nLoaded {len(prompts_data)} prompts with human ambiguity ratings")

    # Setup EAB generator
    eab = setup_eab(config)

    # Create results directory
    results_dir = SCRIPT_DIR / config['output']['results_dir']
    results_dir.mkdir(exist_ok=True, parents=True)

    # Run EAB on each prompt
    print("\n" + "=" * 70)
    print("RUNNING EAB ON PROMPTS")
    print("=" * 70)

    all_results = []

    for i, prompt_data in enumerate(tqdm(prompts_data, desc="Processing prompts")):
        try:
            eab_result = run_eab_on_prompt(
                prompt=prompt_data['prompt'],
                eab=eab,
                config=config
            )

            # Combine with human rating
            result = {
                'prompt_id': i,
                'human_ambiguity_score': prompt_data['human_ambiguity_score'],
                **eab_result
            }

            all_results.append(result)

            # Save intermediate results
            if config['output']['save_intermediate']:
                save_json(
                    {'results': all_results},
                    results_dir / "raw_results_intermediate.json"
                )

        except Exception as e:
            print(f"\nError on prompt {i}: {e}")
            import traceback
            traceback.print_exc()
            continue

    # Save final results
    final_output = {
        'metadata': {
            'config': config,
            'timestamp': datetime.now().isoformat(),
            'num_prompts': len(all_results)
        },
        'results': all_results
    }

    save_json(final_output, results_dir / "raw_results.json")

    # Print summary
    print("\n" + "=" * 70)
    print("EXPERIMENT COMPLETE")
    print("=" * 70)
    print(f"Total prompts processed: {len(all_results)}")
    print(f"Prompts with branching: {sum(1 for r in all_results if r['did_branch'])}")
    print(f"Average branches per prompt: {np.mean([r['num_branches'] for r in all_results]):.2f}")
    print(f"Results saved to: {results_dir / 'raw_results.json'}")


if __name__ == "__main__":
    main()
