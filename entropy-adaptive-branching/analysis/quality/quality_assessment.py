"""
Quality Assessment for EAB vs Naive Generation.

Computes diversity and quality metrics for generated text samples.
"""

import json
import csv
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Tuple, Any
from collections import Counter
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from nltk.tokenize import word_tokenize
import nltk

# Ensure punkt and punkt_tab are available (for different NLTK versions)
try:
    nltk.data.find('tokenizers/punkt_tab')
except LookupError:
    nltk.download('punkt_tab', quiet=True)

try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', quiet=True)
    
quality_dir = Path(__file__).parent


# =============================================================================
# METRICS COMPUTATION
# =============================================================================

def compute_self_bleu(generations: List[str], n: int = 2) -> float:
    """
    Compute Self-BLEU: average pairwise BLEU-n score between generations.

    Lower Self-BLEU indicates higher diversity (less repetition).

    Args:
        generations: List of generated text samples
        n: N-gram order (2, 3, or 4)

    Returns:
        Average Self-BLEU score (0-1)
    """
    if len(generations) < 2:
        return 0.0

    # Tokenize all generations
    tokenized = [word_tokenize(text.lower()) for text in generations]

    # Set n-gram weights
    if n == 2:
        weights = (0.5, 0.5, 0, 0)
    elif n == 3:
        weights = (0.33, 0.33, 0.33, 0)
    elif n == 4:
        weights = (0.25, 0.25, 0.25, 0.25)
    else:
        raise ValueError(f"Invalid n-gram order: {n}. Must be 2, 3, or 4.")

    smoothing = SmoothingFunction().method1

    # Compute pairwise BLEU scores
    bleu_scores = []
    for i, hypothesis in enumerate(tokenized):
        # Compare with all other generations
        other_generations = tokenized[:i] + tokenized[i+1:]

        # Compute BLEU with each reference
        for reference in other_generations:
            score = sentence_bleu(
                [reference],
                hypothesis,
                weights=weights,
                smoothing_function=smoothing
            )
            bleu_scores.append(score)

    return np.mean(bleu_scores) if bleu_scores else 0.0


def compute_distinct_n(generations: List[str], n: int = 2) -> float:
    """
    Compute Distinct-n: ratio of unique n-grams to total n-grams.

    Higher Distinct-n indicates higher diversity.

    Args:
        generations: List of generated text samples
        n: N-gram order (2, 3, or 4)

    Returns:
        Distinct-n score (0-1)
    """
    # Tokenize and combine all generations
    all_tokens = []
    for text in generations:
        tokens = word_tokenize(text.lower())
        all_tokens.extend(tokens)

    # Extract n-grams
    ngrams = []
    for i in range(len(all_tokens) - n + 1):
        ngram = tuple(all_tokens[i:i+n])
        ngrams.append(ngram)

    if not ngrams:
        return 0.0

    # Compute ratio
    unique_ngrams = len(set(ngrams))
    total_ngrams = len(ngrams)

    return unique_ngrams / total_ngrams


def compute_prompt_metrics(
    prompt: str,
    eab_generations: List[str],
    naive_generations: List[str]
) -> Dict[str, Any]:
    """
    Compute all quality metrics for a single prompt.

    Args:
        prompt: The input prompt
        eab_generations: List of EAB generations
        naive_generations: List of naive generations

    Returns:
        Dictionary of metrics
    """
    metrics = {'prompt': prompt}

    # EAB metrics
    metrics['eab_count'] = len(eab_generations)
    metrics['eab_self_bleu_2'] = compute_self_bleu(eab_generations, n=2)
    metrics['eab_self_bleu_3'] = compute_self_bleu(eab_generations, n=3)
    metrics['eab_distinct_2'] = compute_distinct_n(eab_generations, n=2)
    metrics['eab_distinct_3'] = compute_distinct_n(eab_generations, n=3)
    metrics['eab_distinct_4'] = compute_distinct_n(eab_generations, n=4)
    metrics['eab_avg_length'] = np.mean([len(text.split()) for text in eab_generations])

    # Naive metrics
    metrics['naive_count'] = len(naive_generations)
    metrics['naive_self_bleu_2'] = compute_self_bleu(naive_generations, n=2)
    metrics['naive_self_bleu_3'] = compute_self_bleu(naive_generations, n=3)
    metrics['naive_distinct_2'] = compute_distinct_n(naive_generations, n=2)
    metrics['naive_distinct_3'] = compute_distinct_n(naive_generations, n=3)
    metrics['naive_distinct_4'] = compute_distinct_n(naive_generations, n=4)
    metrics['naive_avg_length'] = np.mean([len(text.split()) for text in naive_generations])

    return metrics


# =============================================================================
# PROMPT SELECTION FOR HUMAN EVALUATION
# =============================================================================

def is_short_and_open_ended(prompt: str, max_words: int = 15) -> bool:
    """
    Check if prompt is short and open-ended.

    Args:
        prompt: The prompt text
        max_words: Maximum word count

    Returns:
        True if prompt meets criteria
    """
    # Check length
    word_count = len(prompt.split())
    if word_count > max_words:
        return False

    # Check for open-ended patterns
    open_ended_patterns = [
        'the best', 'how can', 'what are', 'why should',
        'describe', 'explain', 'tell me', 'what would',
        'imagine', 'consider', 'think about'
    ]

    prompt_lower = prompt.lower()
    return any(pattern in prompt_lower for pattern in open_ended_patterns)


def select_human_eval_prompts(
    eab_data: Dict[str, List[str]],
    naive_data: Dict[str, List[str]],
    target_count: int = 5,
    target_gen_count: int = 5
) -> Dict[str, Dict[str, Any]]:
    """
    Select prompts for human evaluation.

    Criteria:
    - Both methods have exactly target_gen_count generations (or closest)
    - Prompts are short (< 15 words) and open-ended

    Args:
        eab_data: EAB samples {prompt: [generations]}
        naive_data: Naive samples {prompt: [generations]}
        target_count: Number of prompts to select
        target_gen_count: Target number of generations per method

    Returns:
        Selected prompts with samples
    """
    candidates = []

    for prompt in eab_data:
        if prompt not in naive_data:
            continue

        eab_gens = eab_data[prompt]
        naive_gens = naive_data[prompt]

        # Check if short and open-ended
        if not is_short_and_open_ended(prompt):
            continue

        # Compute distance from target count
        eab_dist = abs(len(eab_gens) - target_gen_count)
        naive_dist = abs(len(naive_gens) - target_gen_count)
        total_dist = eab_dist + naive_dist

        candidates.append({
            'prompt': prompt,
            'eab_gens': eab_gens,
            'naive_gens': naive_gens,
            'distance': total_dist,
            'eab_count': len(eab_gens),
            'naive_count': len(naive_gens)
        })

    # Sort by distance (prefer closer to target_gen_count)
    candidates.sort(key=lambda x: x['distance'])

    # Select top candidates
    selected = {}
    for i, candidate in enumerate(candidates[:target_count]):
        prompt_id = f"prompt_{i+1}"

        # Truncate/pad to exactly target_gen_count if needed
        eab_selected = candidate['eab_gens'][:target_gen_count]
        naive_selected = candidate['naive_gens'][:target_gen_count]

        selected[prompt_id] = {
            'prompt': candidate['prompt'],
            'eab': eab_selected,
            'naive': naive_selected
        }

    return selected


# =============================================================================
# DEMO EXAMPLE SELECTION
# =============================================================================

def select_demo_example(
    human_eval_prompts: Dict[str, Dict[str, Any]],
    all_metrics: List[Dict[str, Any]]
) -> Tuple[str, Dict[str, Any]]:
    """
    Select one clear demo prompt that shows meaningful diversity.

    Criteria:
    - EAB has higher distinct-n than naive
    - EAB has lower or similar Self-BLEU than naive
    - Average generation length is reasonable (20-100 words)

    Args:
        human_eval_prompts: Selected human eval prompts
        all_metrics: List of all computed metrics

    Returns:
        (prompt_id, prompt_data)
    """
    # Create metrics lookup
    metrics_lookup = {m['prompt']: m for m in all_metrics}

    best_prompt_id = None
    best_score = -float('inf')

    for prompt_id, data in human_eval_prompts.items():
        prompt = data['prompt']

        if prompt not in metrics_lookup:
            continue

        metrics = metrics_lookup[prompt]

        # Check length constraint
        avg_length = (metrics['eab_avg_length'] + metrics['naive_avg_length']) / 2
        if avg_length < 20 or avg_length > 100:
            continue

        # Compute diversity advantage score
        distinct_advantage = (
            (metrics['eab_distinct_2'] - metrics['naive_distinct_2']) +
            (metrics['eab_distinct_3'] - metrics['naive_distinct_3']) +
            (metrics['eab_distinct_4'] - metrics['naive_distinct_4'])
        ) / 3

        self_bleu_advantage = (
            metrics['naive_self_bleu_2'] - metrics['eab_self_bleu_2']
        )

        # Combined score (higher is better for EAB)
        score = distinct_advantage + 0.5 * self_bleu_advantage

        if score > best_score:
            best_score = score
            best_prompt_id = prompt_id

    if best_prompt_id is None:
        # Fallback to first prompt
        best_prompt_id = list(human_eval_prompts.keys())[0]

    return best_prompt_id, human_eval_prompts[best_prompt_id]


# =============================================================================
# PLOTTING
# =============================================================================

def set_publication_style():
    """Set matplotlib style for publication-quality figures."""
    plt.rcParams.update({
        'font.size': 12,
        'axes.labelsize': 14,
        'axes.titlesize': 14,
        'xtick.labelsize': 11,
        'ytick.labelsize': 11,
        'legend.fontsize': 11,
        'figure.titlesize': 16,
        'figure.dpi': 100,
        'savefig.dpi': 300,
        'savefig.bbox': 'tight',
    })


def plot_self_bleu_comparison(
    metrics: List[Dict[str, Any]],
    output_path: Path
):
    """
    Create boxplot comparing Self-BLEU scores.

    Args:
        metrics: List of prompt metrics
        output_path: Path to save figure
    """
    set_publication_style()

    # Extract Self-BLEU-2 scores
    eab_scores = [m['eab_self_bleu_2'] for m in metrics]
    naive_scores = [m['naive_self_bleu_2'] for m in metrics]

    # Prepare data
    data = {
        'Method': ['EAB'] * len(eab_scores) + ['Naive'] * len(naive_scores),
        'Self-BLEU-2': eab_scores + naive_scores
    }

    # Plot
    fig, ax = plt.subplots(figsize=(8, 6))

    positions = [1, 2]
    bp = ax.boxplot(
        [eab_scores, naive_scores],
        positions=positions,
        widths=0.6,
        patch_artist=True,
        labels=['EAB', 'Naive']
    )

    # Customize colors
    colors = ['#06A77D', '#E63946']
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)

    # Add grid and labels
    ax.set_ylabel('Self-BLEU-2 Score', fontweight='bold')
    ax.set_title('Diversity Comparison: Self-BLEU-2\n(Lower is More Diverse)', fontweight='bold', pad=20)
    ax.grid(True, alpha=0.3, axis='y', linestyle=':')

    # Add median values as text
    for i, (pos, scores) in enumerate(zip(positions, [eab_scores, naive_scores])):
        median = np.median(scores)
        ax.text(pos, median, f'{median:.3f}', ha='center', va='bottom', fontweight='bold')

    plt.tight_layout()
    plt.savefig(output_path, bbox_inches='tight', dpi=300)
    plt.close()

    print(f"✓ Saved Self-BLEU plot to {output_path}")


def plot_distinct_n_comparison(
    metrics: List[Dict[str, Any]],
    output_path: Path
):
    """
    Create bar plot comparing Distinct-n scores.

    Args:
        metrics: List of prompt metrics
        output_path: Path to save figure
    """
    set_publication_style()

    # Compute averages
    avg_metrics = {
        'EAB': {
            'Distinct-2': np.mean([m['eab_distinct_2'] for m in metrics]),
            'Distinct-3': np.mean([m['eab_distinct_3'] for m in metrics]),
            'Distinct-4': np.mean([m['eab_distinct_4'] for m in metrics]),
        },
        'Naive': {
            'Distinct-2': np.mean([m['naive_distinct_2'] for m in metrics]),
            'Distinct-3': np.mean([m['naive_distinct_3'] for m in metrics]),
            'Distinct-4': np.mean([m['naive_distinct_4'] for m in metrics]),
        }
    }

    # Prepare data
    n_grams = ['Distinct-2', 'Distinct-3', 'Distinct-4']
    eab_values = [avg_metrics['EAB'][n] for n in n_grams]
    naive_values = [avg_metrics['Naive'][n] for n in n_grams]

    # Plot
    fig, ax = plt.subplots(figsize=(10, 6))

    x = np.arange(len(n_grams))
    width = 0.35

    bars1 = ax.bar(x - width/2, eab_values, width, label='EAB', color='#06A77D', alpha=0.8)
    bars2 = ax.bar(x + width/2, naive_values, width, label='Naive', color='#E63946', alpha=0.8)

    # Add value labels on bars
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width()/2., height,
                f'{height:.3f}',
                ha='center', va='bottom', fontsize=10, fontweight='bold'
            )

    # Labels
    ax.set_ylabel('Distinct-n Score', fontweight='bold')
    ax.set_title('Lexical Diversity: Distinct-n Metrics\n(Higher is More Diverse)', fontweight='bold', pad=20)
    ax.set_xticks(x)
    ax.set_xticklabels(n_grams)
    ax.legend(loc='best', framealpha=0.9)
    ax.grid(True, alpha=0.3, axis='y', linestyle=':')
    ax.set_ylim(bottom=0, top=max(max(eab_values), max(naive_values)) * 1.15)

    plt.tight_layout()
    plt.savefig(output_path, bbox_inches='tight', dpi=300)
    plt.close()

    print(f"✓ Saved Distinct-n plot to {output_path}")


# =============================================================================
# MAIN WORKFLOW
# =============================================================================

def main():
    """Main quality assessment workflow."""
    print("=" * 70)
    print("QUALITY ASSESSMENT: EAB vs NAIVE GENERATION")
    print("=" * 70)

    # -------------------------------------------------------------------------
    # 1. Load data
    # -------------------------------------------------------------------------
    print("\n1. Loading generated samples...")

    eab_file = quality_dir / "eab_samples.json"
    naive_file = quality_dir / "naive_samples.json"

    if not eab_file.exists() or not naive_file.exists():
        print(f"\n❌ ERROR: Sample files not found!")
        print(f"   Expected files:")
        print(f"   - {eab_file}")
        print(f"   - {naive_file}")
        print(f"\n   Please create these files with format:")
        print(f"   {{\"prompt_text\": [\"generation1\", \"generation2\", ...], ...}}")
        return

    with open(eab_file, 'r') as f:
        eab_data = json.load(f)

    with open(naive_file, 'r') as f:
        naive_data = json.load(f)

    # Find common prompts
    common_prompts = set(eab_data.keys()) & set(naive_data.keys())

    print(f"   ✓ Loaded EAB samples: {len(eab_data)} prompts")
    print(f"   ✓ Loaded Naive samples: {len(naive_data)} prompts")
    print(f"   ✓ Common prompts: {len(common_prompts)}")

    if len(common_prompts) == 0:
        print("\n❌ ERROR: No common prompts found between EAB and Naive samples!")
        return

    # -------------------------------------------------------------------------
    # 2. Compute metrics
    # -------------------------------------------------------------------------
    print("\n2. Computing diversity and quality metrics...")

    all_metrics = []
    for prompt in common_prompts:
        metrics = compute_prompt_metrics(
            prompt,
            eab_data[prompt],
            naive_data[prompt]
        )
        all_metrics.append(metrics)

    print(f"   ✓ Computed metrics for {len(all_metrics)} prompts")

    # Save to CSV
    csv_file = quality_dir / "metrics.csv"
    fieldnames = [
        'prompt',
        'eab_count', 'eab_self_bleu_2', 'eab_self_bleu_3',
        'eab_distinct_2', 'eab_distinct_3', 'eab_distinct_4',
        'eab_avg_length',
        'naive_count', 'naive_self_bleu_2', 'naive_self_bleu_3',
        'naive_distinct_2', 'naive_distinct_3', 'naive_distinct_4',
        'naive_avg_length'
    ]

    with open(csv_file, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_metrics)

    print(f"   ✓ Saved metrics to {csv_file}")

    # -------------------------------------------------------------------------
    # 3. Select prompts for human evaluation
    # -------------------------------------------------------------------------
    print("\n3. Selecting prompts for human evaluation...")

    human_eval_prompts = select_human_eval_prompts(eab_data, naive_data)

    print(f"   ✓ Selected {len(human_eval_prompts)} prompts")

    # Save to JSON
    human_eval_file = quality_dir / "human_eval_prompts.json"
    with open(human_eval_file, 'w') as f:
        json.dump(human_eval_prompts, f, indent=2)

    print(f"   ✓ Saved to {human_eval_file}")

    # -------------------------------------------------------------------------
    # 4. Select demo example
    # -------------------------------------------------------------------------
    print("\n4. Selecting demo example...")

    if human_eval_prompts:
        demo_id, demo_data = select_demo_example(human_eval_prompts, all_metrics)

        demo_file = quality_dir / "demo_example.txt"
        with open(demo_file, 'w') as f:
            f.write("=" * 80 + "\n")
            f.write("DEMO EXAMPLE: EAB vs NAIVE GENERATION\n")
            f.write("=" * 80 + "\n\n")
            f.write(f"Prompt: {demo_data['prompt']}\n\n")
            f.write("-" * 80 + "\n")
            f.write("EAB GENERATIONS:\n")
            f.write("-" * 80 + "\n\n")
            for i, text in enumerate(demo_data['eab'], 1):
                f.write(f"[{i}] {text}\n\n")
            f.write("-" * 80 + "\n")
            f.write("NAIVE GENERATIONS:\n")
            f.write("-" * 80 + "\n\n")
            for i, text in enumerate(demo_data['naive'], 1):
                f.write(f"[{i}] {text}\n\n")
            f.write("=" * 80 + "\n")

        print(f"   ✓ Saved demo to {demo_file}")
    else:
        print("   ⚠ No suitable prompts found for demo")

    # -------------------------------------------------------------------------
    # 5. Generate plots
    # -------------------------------------------------------------------------
    print("\n5. Generating plots...")

    plot_self_bleu_comparison(
        all_metrics,
        quality_dir / "self_bleu_comparison.png"
    )

    plot_distinct_n_comparison(
        all_metrics,
        quality_dir / "distinct_n_comparison.png"
    )

    # -------------------------------------------------------------------------
    # Summary statistics
    # -------------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("SUMMARY STATISTICS")
    print("=" * 70)

    print("\nSelf-BLEU-2 (Lower is More Diverse):")
    eab_self_bleu = [m['eab_self_bleu_2'] for m in all_metrics]
    naive_self_bleu = [m['naive_self_bleu_2'] for m in all_metrics]
    print(f"  EAB:   {np.mean(eab_self_bleu):.4f} ± {np.std(eab_self_bleu):.4f}")
    print(f"  Naive: {np.mean(naive_self_bleu):.4f} ± {np.std(naive_self_bleu):.4f}")

    print("\nDistinct-2 (Higher is More Diverse):")
    eab_distinct_2 = [m['eab_distinct_2'] for m in all_metrics]
    naive_distinct_2 = [m['naive_distinct_2'] for m in all_metrics]
    print(f"  EAB:   {np.mean(eab_distinct_2):.4f} ± {np.std(eab_distinct_2):.4f}")
    print(f"  Naive: {np.mean(naive_distinct_2):.4f} ± {np.std(naive_distinct_2):.4f}")

    print("\nDistinct-3 (Higher is More Diverse):")
    eab_distinct_3 = [m['eab_distinct_3'] for m in all_metrics]
    naive_distinct_3 = [m['naive_distinct_3'] for m in all_metrics]
    print(f"  EAB:   {np.mean(eab_distinct_3):.4f} ± {np.std(eab_distinct_3):.4f}")
    print(f"  Naive: {np.mean(naive_distinct_3):.4f} ± {np.std(naive_distinct_3):.4f}")

    print("\nDistinct-4 (Higher is More Diverse):")
    eab_distinct_4 = [m['eab_distinct_4'] for m in all_metrics]
    naive_distinct_4 = [m['naive_distinct_4'] for m in all_metrics]
    print(f"  EAB:   {np.mean(eab_distinct_4):.4f} ± {np.std(eab_distinct_4):.4f}")
    print(f"  Naive: {np.mean(naive_distinct_4):.4f} ± {np.std(naive_distinct_4):.4f}")

    print("\nAverage Generation Length (words):")
    eab_length = [m['eab_avg_length'] for m in all_metrics]
    naive_length = [m['naive_avg_length'] for m in all_metrics]
    print(f"  EAB:   {np.mean(eab_length):.1f} ± {np.std(eab_length):.1f}")
    print(f"  Naive: {np.mean(naive_length):.1f} ± {np.std(naive_length):.1f}")

    print("\n" + "=" * 70)
    print("ANALYSIS COMPLETE")
    print("=" * 70)
    print("\nGenerated files:")
    print(f"  • {csv_file.name}")
    print(f"  • {human_eval_file.name}")
    print(f"  • {demo_file.name}")
    print(f"  • self_bleu_comparison.png")
    print(f"  • distinct_n_comparison.png")
    print("=" * 70)


if __name__ == "__main__":
    main()
