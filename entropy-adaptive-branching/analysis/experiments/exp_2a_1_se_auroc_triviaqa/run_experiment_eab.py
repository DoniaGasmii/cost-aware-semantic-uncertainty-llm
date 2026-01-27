"""
Experiment 2.A.1: Semantic Entropy AUROC on TriviaQA (with EAB)

This script evaluates the semantic entropy pipeline (Layer 2) by:
1. Loading TriviaQA validation questions
2. Generating multiple samples per question using EAB (Entropy-Adaptive Branching)
3. Computing semantic entropy for each question's responses
4. Evaluating correctness using RougeL against ground truth
5. Computing AUROC to measure how well SE predicts incorrectness

This version uses EAB generation instead of naive sampling.
"""

import os
import sys
import json
import yaml
import numpy as np
import time
import argparse
from pathlib import Path
from tqdm import tqdm
from typing import Dict, Any, List, Optional
from datetime import datetime

# Add parent directories to path
experiment_dir = Path(__file__).parent
analysis_dir = experiment_dir.parent.parent
eab_dir = analysis_dir.parent
project_root = eab_dir.parent
sys.path.insert(0, str(analysis_dir))
sys.path.insert(0, str(eab_dir))
sys.path.insert(0, str(project_root))

# Import dependencies
from datasets import load_dataset
from rouge_score import rouge_scorer

# Import EAB
from eab import EntropyAdaptiveBranching

# Import semantic entropy estimator
sys.path.insert(0, str(project_root / "semantic-entropy-based-uncertainty"))
from semantic_entropy.estimator import SemanticUncertaintyEstimator


class CostTracker:
    """Track computational costs: time, memory, tokens."""

    def __init__(self, device: str = 'cpu'):
        self.device = device
        self.reset()

    def reset(self):
        self.start_time = None
        self.total_generation_time = 0.0
        self.total_se_time = 0.0
        self.total_tokens_generated = 0
        self.question_times = []
        self.total_branches = 0

    def start(self):
        self.start_time = time.time()

    def record_generation(self, elapsed: float, num_tokens: int, num_branches: int = 0):
        self.total_generation_time += elapsed
        self.total_tokens_generated += num_tokens
        self.total_branches += num_branches

    def record_se(self, elapsed: float):
        self.total_se_time += elapsed

    def record_question(self, elapsed: float):
        self.question_times.append(elapsed)

    def get_stats(self) -> Dict[str, Any]:
        total_time = time.time() - self.start_time if self.start_time else 0
        return {
            'total_time_seconds': total_time,
            'total_generation_time': self.total_generation_time,
            'total_se_time': self.total_se_time,
            'total_tokens_generated': self.total_tokens_generated,
            'total_branches': self.total_branches,
            'avg_time_per_question': np.mean(self.question_times) if self.question_times else 0,
            'tokens_per_second': self.total_tokens_generated / self.total_generation_time if self.total_generation_time > 0 else 0,
        }


def load_config(config_path: Optional[str] = None) -> Dict[str, Any]:
    """Load experiment configuration."""
    if config_path is None:
        config_path = experiment_dir / "config_eab.yaml"
    else:
        config_path = Path(config_path)
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def load_triviaqa(config: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Load TriviaQA dataset.

    Returns list of dicts with 'question' and 'answers' keys.
    """
    print("Loading TriviaQA dataset...")

    # Determine number of questions
    if config['debug']['enabled']:
        num_questions = config['debug']['num_questions']
    else:
        num_questions = config['dataset']['num_questions']

    # Load dataset
    dataset = load_dataset(
        config['dataset']['name'],
        config['dataset']['subset'],
        split=config['dataset']['split']
    )

    # Shuffle with seed and select subset
    dataset = dataset.shuffle(seed=config['dataset']['seed'])
    dataset = dataset.select(range(min(num_questions, len(dataset))))

    # Extract questions and answers
    questions = []
    for item in dataset:
        # TriviaQA has multiple valid answers
        answers = item['answer']['aliases'] + [item['answer']['value']]
        answers = list(set(a.strip() for a in answers if a.strip()))

        questions.append({
            'question': item['question'].strip(),
            'answers': answers,
            'question_id': item['question_id']
        })

    print(f"Loaded {len(questions)} questions")
    return questions


def setup_eab(config: Dict[str, Any], cpu_only: bool = False) -> EntropyAdaptiveBranching:
    """Initialize EAB system."""
    print(f"Loading EAB with model: {config['model']['name']}...")

    device = 'cpu' if cpu_only else config['model']['device']

    # Convert dtype string to None for CPU (torch types not needed for string param)
    torch_dtype_str = config['model'].get('torch_dtype', 'float32')

    # Import torch here to handle dtype
    import torch
    if torch_dtype_str == 'float16' and device == 'cuda':
        torch_dtype = torch.float16
    else:
        torch_dtype = torch.float32  # Use float32 for CPU

    eab = EntropyAdaptiveBranching(
        model_name=config['model']['name'],
        device=device,
        entropy_threshold=config['eab']['entropy_threshold'],
        branch_factor=config['eab']['branch_factor'],
        max_paths=config['eab']['max_paths'],
        torch_dtype=torch_dtype,
        use_cow=config['eab'].get('use_cow', True)
    )

    print(f"EAB loaded on {device}")
    return eab


def generate_samples_with_eab(
    question: str,
    eab: EntropyAdaptiveBranching,
    config: Dict[str, Any]
) -> tuple:
    """
    Generate multiple samples for a question using EAB.

    Returns tuple of (samples, generation_time, total_tokens, num_branches, entropy_stats).
    """
    # Determine generation parameters
    if config['debug']['enabled']:
        max_new_tokens = config['debug'].get('max_new_tokens', 50)
    else:
        max_new_tokens = config['generation']['max_new_tokens']

    start_time = time.time()

    # Generate using EAB
    results = eab.generate(
        prompt=f"Answer this question briefly: {question}",
        max_new_tokens=max_new_tokens,
        temperature=config['generation']['temperature'],
        top_p=config['generation'].get('top_p'),
        return_metadata=True,
        show_progress=False,
        use_chat_template=True
    )

    generation_time = time.time() - start_time

    # Extract samples (just the generated text, not the full output)
    samples = [r['text'] for r in results]

    # Count tokens (sum across all paths)
    # Note: metadata fields are at top level, not nested under 'metadata'
    total_tokens = sum(r['length'] for r in results)

    # Get entropy statistics
    entropy_stats = eab.get_entropy_history()['statistics']
    num_branches = entropy_stats.get('num_branches', 0)

    return samples, generation_time, total_tokens, num_branches, entropy_stats


def compute_correctness(
    generated: str,
    ground_truth_answers: List[str],
    scorer: rouge_scorer.RougeScorer,
    threshold: float = 0.3
) -> Dict[str, Any]:
    """
    Check if generated answer is correct using RougeL.

    Returns dict with is_correct, rouge_l_score, and matched_answer.
    """
    best_score = 0.0
    best_answer = None

    for answer in ground_truth_answers:
        try:
            scores = scorer.score(answer.lower(), generated.lower())
            rouge_l = scores['rougeL'].fmeasure

            if rouge_l > best_score:
                best_score = rouge_l
                best_answer = answer
        except Exception:
            continue

    return {
        'is_correct': best_score >= threshold,
        'rouge_l_score': best_score,
        'matched_answer': best_answer
    }


def run_single_question(
    question_data: Dict[str, Any],
    eab: EntropyAdaptiveBranching,
    se_estimator: SemanticUncertaintyEstimator,
    scorer: rouge_scorer.RougeScorer,
    config: Dict[str, Any],
    cost_tracker: CostTracker
) -> Dict[str, Any]:
    """
    Run full pipeline on a single question.

    Returns dict with question, samples, correctness, SE metrics, and timing.
    """
    question_start = time.time()
    question = question_data['question']
    answers = question_data['answers']

    # Step 1: Generate samples using EAB (with timing)
    samples, gen_time, num_tokens, num_branches, entropy_stats = generate_samples_with_eab(
        question, eab, config
    )
    cost_tracker.record_generation(gen_time, num_tokens, num_branches)

    # Step 2: Compute semantic entropy (with timing)
    se_start = time.time()
    se_result = se_estimator.compute(samples, return_details=False)
    se_time = time.time() - se_start
    cost_tracker.record_se(se_time)

    # Step 3: Determine correctness for each sample
    correctness_threshold = config['correctness']['threshold']
    sample_correctness = []

    for sample in samples:
        corr = compute_correctness(sample, answers, scorer, correctness_threshold)
        sample_correctness.append(corr)

    # Aggregate correctness
    num_correct = sum(1 for c in sample_correctness if c['is_correct'])
    any_correct = num_correct > 0
    majority_correct = num_correct > len(samples) / 2

    # Best sample correctness (highest RougeL)
    best_sample_idx = max(range(len(sample_correctness)),
                          key=lambda i: sample_correctness[i]['rouge_l_score'])
    best_correct = sample_correctness[best_sample_idx]['is_correct']
    best_rouge = sample_correctness[best_sample_idx]['rouge_l_score']

    # Record question time
    question_time = time.time() - question_start
    cost_tracker.record_question(question_time)

    return {
        'question_id': question_data['question_id'],
        'question': question,
        'ground_truth_answers': answers,
        'generated_samples': samples,
        'num_samples': len(samples),

        # Semantic entropy metrics
        'se_entropy': se_result['entropy'],
        'se_normalized_entropy': se_result['normalized_entropy'],
        'se_uncertainty_score': se_result['uncertainty_score'],
        'se_n_clusters': se_result['n_clusters'],

        # EAB metrics
        'eab_num_branches': num_branches,
        'eab_mean_entropy': entropy_stats['mean_entropy'],
        'eab_branch_rate': entropy_stats['branch_rate'],

        # Correctness metrics
        'sample_correctness': sample_correctness,
        'num_correct_samples': num_correct,
        'any_correct': any_correct,
        'majority_correct': majority_correct,
        'best_sample_correct': best_correct,
        'best_rouge_l': best_rouge,

        # Cost metrics (per question)
        'generation_time': gen_time,
        'se_time': se_time,
        'total_time': question_time,
        'num_tokens_generated': num_tokens,
    }


def save_json(data: Any, path: Path):
    """Save data to JSON file."""
    with open(path, 'w') as f:
        json.dump(data, f, indent=2, default=str)


def main():
    """Main experiment runner."""
    parser = argparse.ArgumentParser(description="Run Experiment 2.A.1 with EAB")
    parser.add_argument('--config', type=str, default=None, help='Path to config file')
    parser.add_argument('--cpu-only', action='store_true', help='Force CPU-only mode (no GPU)')
    args = parser.parse_args()

    # Force CPU mode if requested (must be done before importing torch)
    if args.cpu_only:
        os.environ['CUDA_VISIBLE_DEVICES'] = ''
        print("NOTE: Running in CPU-only mode")

    print("=" * 70)
    print("EXPERIMENT 2.A.1: SEMANTIC ENTROPY AUROC ON TRIVIAQA (with EAB)")
    print("=" * 70)

    # Load configuration
    print("\n1. Loading configuration...")
    config = load_config(args.config)

    if config['debug']['enabled']:
        print("   [DEBUG MODE ENABLED]")

    # Set random seeds
    import torch
    torch.manual_seed(config['seed'])
    np.random.seed(config['seed'])

    # Load dataset
    print("\n2. Loading TriviaQA dataset...")
    questions = load_triviaqa(config)

    # Setup EAB
    print("\n3. Setting up EAB generation system...")
    eab = setup_eab(config, cpu_only=args.cpu_only)

    # Setup semantic entropy estimator
    print("\n4. Setting up semantic entropy estimator...")
    encoder_map = {"mpnet": "all-mpnet-base-v2", "minilm": "all-MiniLM-L6-v2"}
    encoder_name = encoder_map.get(
        config['semantic_entropy']['encoder_model'],
        config['semantic_entropy']['encoder_model']
    )

    se_device = 'cpu' if args.cpu_only else config['model']['device']
    se_estimator = SemanticUncertaintyEstimator(
        encoder_model=encoder_name,
        distance_threshold=config['semantic_entropy']['default_threshold'],
        linkage=config['semantic_entropy']['linkage'],
        device=se_device
    )
    print(f"   Encoder: {encoder_name}")
    print(f"   Distance threshold: {config['semantic_entropy']['default_threshold']}")

    # Setup RougeL scorer
    scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)

    # Setup cost tracker
    cost_tracker = CostTracker(device=se_device)

    # Prepare results directory
    results_dir = experiment_dir / config['output']['results_dir']
    results_dir.mkdir(exist_ok=True, parents=True)

    # Run experiment
    print("\n5. Running experiment...")
    print("-" * 70)

    all_results = []
    cost_tracker.start()

    for i, question_data in enumerate(tqdm(questions, desc="Processing questions")):
        try:
            result = run_single_question(
                question_data, eab, se_estimator, scorer, config, cost_tracker
            )
            all_results.append(result)

            # Save intermediate results
            if config['output']['save_intermediate'] and (i + 1) % 10 == 0:
                save_json({
                    'metadata': {'config': config, 'num_processed': len(all_results)},
                    'results': all_results
                }, results_dir / "raw_results_eab_intermediate.json")

        except Exception as e:
            print(f"\n   Error on question {i}: {e}")
            import traceback
            traceback.print_exc()
            continue

    # Get final cost stats
    cost_stats = cost_tracker.get_stats()

    # Save final results
    print("\n6. Saving results...")
    final_output = {
        'metadata': {
            'config': config,
            'timestamp': datetime.now().isoformat(),
            'num_questions': len(all_results),
            'generation_method': 'eab',  # Using EAB instead of naive sampling
            'cpu_only': args.cpu_only,
        },
        'cost_stats': cost_stats,
        'results': all_results
    }
    save_json(final_output, results_dir / "raw_results_eab.json")

    # Quick summary
    print("\n" + "=" * 70)
    print("EXPERIMENT COMPLETE")
    print("=" * 70)

    # Compute quick stats
    se_scores = [r['se_uncertainty_score'] for r in all_results]
    correctness = [r['best_sample_correct'] for r in all_results]
    eab_branches = [r['eab_num_branches'] for r in all_results]

    print(f"\nQuick Summary:")
    print(f"  Total questions: {len(all_results)}")
    print(f"  Accuracy (best sample): {np.mean(correctness):.1%}")
    print(f"  Avg SE uncertainty: {np.mean(se_scores):.3f}")
    print(f"  Avg SE clusters: {np.mean([r['se_n_clusters'] for r in all_results]):.1f}")
    print(f"  Avg EAB branches: {np.mean(eab_branches):.1f}")

    print(f"\nCost Summary:")
    print(f"  Total time: {cost_stats['total_time_seconds']:.1f}s")
    print(f"  Generation time: {cost_stats['total_generation_time']:.1f}s")
    print(f"  SE computation time: {cost_stats['total_se_time']:.1f}s")
    print(f"  Total tokens generated: {cost_stats['total_tokens_generated']}")
    print(f"  Total branches: {cost_stats['total_branches']}")
    print(f"  Tokens/second: {cost_stats['tokens_per_second']:.1f}")

    print(f"\nResults saved to: {results_dir / 'raw_results_eab.json'}")
    print("\nNext steps:")
    print("  1. Run: python analyze_results.py")
    print("  2. Compare with naive sampling results")
    print("=" * 70)


if __name__ == "__main__":
    main()
