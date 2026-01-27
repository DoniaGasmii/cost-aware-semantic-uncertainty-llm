"""
Compute diversity and quality metrics for EAB vs Naive sampling.

Clean, concise implementation comparing sample quality.
"""

import json
import numpy as np
from pathlib import Path
from typing import List, Dict, Any
from collections import Counter
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from tqdm import tqdm

experiment_dir = Path(__file__).parent


def tokenize(text: str) -> List[str]:
    """Simple tokenization."""
    return text.lower().split()


def compute_self_bleu(samples: List[str], n: int = 2) -> float:
    """
    Compute Self-BLEU score.

    For each sample, compute BLEU with all other samples as references.
    Lower = more diverse (samples are less similar to each other).
    """
    if len(samples) < 2:
        return 0.0

    tokenized_samples = [tokenize(s) for s in samples]
    smoothing = SmoothingFunction().method1

    scores = []
    for i, candidate in enumerate(tokenized_samples):
        references = [s for j, s in enumerate(tokenized_samples) if j != i]
        if references:
            score = sentence_bleu(
                references,
                candidate,
                weights=[1/n]*n + [0]*(4-n),
                smoothing_function=smoothing
            )
            scores.append(score)

    return np.mean(scores)


def compute_distinct_n(samples: List[str], n: int = 2) -> float:
    """
    Compute Distinct-n: ratio of unique n-grams to total n-grams.

    Higher = more diverse (more unique phrases).
    """
    all_ngrams = []

    for sample in samples:
        tokens = tokenize(sample)
        ngrams = [tuple(tokens[i:i+n]) for i in range(len(tokens)-n+1)]
        all_ngrams.extend(ngrams)

    if not all_ngrams:
        return 0.0

    unique_ngrams = len(set(all_ngrams))
    total_ngrams = len(all_ngrams)

    return unique_ngrams / total_ngrams


def compute_lexical_diversity(samples: List[str]) -> float:
    """
    Compute lexical diversity: ratio of unique tokens to total tokens.

    Higher = more diverse vocabulary.
    """
    all_tokens = []
    for sample in samples:
        all_tokens.extend(tokenize(sample))

    if not all_tokens:
        return 0.0

    return len(set(all_tokens)) / len(all_tokens)


def compute_metrics_for_samples(samples: List[str]) -> Dict[str, float]:
    """Compute all diversity metrics for a set of samples."""
    return {
        'self_bleu_1': compute_self_bleu(samples, n=1),
        'self_bleu_2': compute_self_bleu(samples, n=2),
        'self_bleu_3': compute_self_bleu(samples, n=3),
        'distinct_1': compute_distinct_n(samples, n=1),
        'distinct_2': compute_distinct_n(samples, n=2),
        'distinct_3': compute_distinct_n(samples, n=3),
        'lexical_diversity': compute_lexical_diversity(samples),
        'num_samples': len(samples),
        'avg_length': np.mean([len(tokenize(s)) for s in samples])
    }


def load_eab_samples(results_path: Path) -> List[List[str]]:
    """Load EAB-generated samples from exp results."""
    with open(results_path) as f:
        data = json.load(f)

    samples_per_prompt = []
    for result in data['results']:
        if 'generated_samples' in result:
            samples_per_prompt.append(result['generated_samples'])

    return samples_per_prompt


def load_naive_samples(results_path: Path) -> List[List[str]]:
    """Load naive-generated samples from exp results."""
    with open(results_path) as f:
        data = json.load(f)

    samples_per_prompt = []
    for result in data['results']:
        if 'generated_samples' in result:
            samples_per_prompt.append(result['generated_samples'])

    return samples_per_prompt


def main():
    """Main analysis."""
    print("="*70)
    print("COMPUTING DIVERSITY METRICS: EAB vs NAIVE")
    print("="*70)

    # Configuration
    # Update these paths to your actual data
    eab_results = experiment_dir / "../exp_2a_1_se_auroc_triviaqa/results_eab/raw_results_eab.json"
    naive_results = experiment_dir / "../exp_2a_1_se_auroc_triviaqa/results/raw_results.json"

    output_file = experiment_dir / "results" / "diversity_metrics.json"
    output_file.parent.mkdir(exist_ok=True)

    # Load samples
    print("\n1. Loading samples...")
    eab_samples = load_eab_samples(eab_results)
    naive_samples = load_naive_samples(naive_results)

    print(f"   EAB: {len(eab_samples)} prompts")
    print(f"   Naive: {len(naive_samples)} prompts")

    # Compute metrics for EAB
    print("\n2. Computing metrics for EAB samples...")
    eab_metrics_per_prompt = []
    for samples in tqdm(eab_samples, desc="EAB"):
        eab_metrics_per_prompt.append(compute_metrics_for_samples(samples))

    # Compute metrics for Naive
    print("\n3. Computing metrics for Naive samples...")
    naive_metrics_per_prompt = []
    for samples in tqdm(naive_samples, desc="Naive"):
        naive_metrics_per_prompt.append(compute_metrics_for_samples(samples))

    # Aggregate statistics
    print("\n4. Aggregating statistics...")

    def aggregate_metrics(metrics_list):
        """Compute mean and std across prompts."""
        keys = metrics_list[0].keys()
        aggregated = {}
        for key in keys:
            values = [m[key] for m in metrics_list]
            aggregated[key] = {
                'mean': float(np.mean(values)),
                'std': float(np.std(values)),
                'min': float(np.min(values)),
                'max': float(np.max(values))
            }
        return aggregated

    eab_aggregated = aggregate_metrics(eab_metrics_per_prompt)
    naive_aggregated = aggregate_metrics(naive_metrics_per_prompt)

    # Compute ratios
    print("\n5. Computing comparison ratios...")
    comparison = {}
    for key in eab_aggregated.keys():
        if key not in ['num_samples', 'avg_length']:
            eab_val = eab_aggregated[key]['mean']
            naive_val = naive_aggregated[key]['mean']
            ratio = naive_val / eab_val if eab_val > 0 else 0
            comparison[key] = {
                'eab': eab_val,
                'naive': naive_val,
                'ratio': ratio,  # How many times more diverse is naive
                'interpretation': 'higher_is_more_diverse' if 'distinct' in key or 'lexical' in key else 'lower_is_more_diverse'
            }

    # Save results
    print("\n6. Saving results...")
    results = {
        'eab_aggregated': eab_aggregated,
        'naive_aggregated': naive_aggregated,
        'comparison': comparison,
        'summary': {
            'n_prompts_eab': len(eab_samples),
            'n_prompts_naive': len(naive_samples)
        }
    }

    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"   Saved to: {output_file}")

    # Print summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)

    print("\n📊 Key Metrics:")
    print(f"\nSelf-BLEU-2 (lower = more diverse):")
    print(f"  EAB:   {eab_aggregated['self_bleu_2']['mean']:.3f} ± {eab_aggregated['self_bleu_2']['std']:.3f}")
    print(f"  Naive: {naive_aggregated['self_bleu_2']['mean']:.3f} ± {naive_aggregated['self_bleu_2']['std']:.3f}")
    print(f"  → Naive is {comparison['self_bleu_2']['ratio']:.1f}× more diverse")

    print(f"\nDistinct-2 (higher = more diverse):")
    print(f"  EAB:   {eab_aggregated['distinct_2']['mean']:.3f} ± {eab_aggregated['distinct_2']['std']:.3f}")
    print(f"  Naive: {naive_aggregated['distinct_2']['mean']:.3f} ± {naive_aggregated['distinct_2']['std']:.3f}")
    print(f"  → Naive is {comparison['distinct_2']['ratio']:.1f}× more diverse")

    print(f"\nLexical Diversity (higher = more diverse):")
    print(f"  EAB:   {eab_aggregated['lexical_diversity']['mean']:.3f} ± {eab_aggregated['lexical_diversity']['std']:.3f}")
    print(f"  Naive: {naive_aggregated['lexical_diversity']['mean']:.3f} ± {naive_aggregated['lexical_diversity']['std']:.3f}")
    print(f"  → Naive is {comparison['lexical_diversity']['ratio']:.1f}× more diverse")

    print("\n" + "="*70)
    print("COMPLETE")
    print("="*70)
    print("\nNext step: python plot_diversity.py")


if __name__ == "__main__":
    main()
