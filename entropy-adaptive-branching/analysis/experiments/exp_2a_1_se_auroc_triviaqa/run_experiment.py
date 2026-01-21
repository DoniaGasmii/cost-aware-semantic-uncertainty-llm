"""
Experiment 2.A.1: Semantic Entropy AUROC on TriviaQA

This script evaluates the semantic entropy pipeline (Layer 2) by:
1. Loading TriviaQA validation questions
2. Generating multiple samples per question using naive temperature sampling
3. Computing semantic entropy for each question's responses
4. Evaluating correctness using RougeL against ground truth
5. Computing AUROC to measure how well SE predicts incorrectness

This is an ablation study to validate SE before integrating with EAB.
"""

import sys
import json
import yaml
import torch
import numpy as np
import time
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
from transformers import AutoModelForCausalLM, AutoTokenizer
from rouge_score import rouge_scorer

# Import semantic entropy estimator
sys.path.insert(0, str(project_root / "semantic-entropy-based-uncertainty"))
from semantic_entropy.estimator import SemanticUncertaintyEstimator


class CostTracker:
    """Track computational costs: time, memory, tokens."""

    def __init__(self, device: str = 'cuda'):
        self.device = device
        self.reset()

    def reset(self):
        self.start_time = None
        self.total_generation_time = 0.0
        self.total_se_time = 0.0
        self.total_tokens_generated = 0
        self.peak_memory_mb = 0.0
        self.question_times = []

    def start(self):
        self.start_time = time.time()
        if self.device == 'cuda' and torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()

    def record_generation(self, elapsed: float, num_tokens: int):
        self.total_generation_time += elapsed
        self.total_tokens_generated += num_tokens

    def record_se(self, elapsed: float):
        self.total_se_time += elapsed

    def record_question(self, elapsed: float):
        self.question_times.append(elapsed)

    def update_peak_memory(self):
        if self.device == 'cuda' and torch.cuda.is_available():
            torch.cuda.synchronize()
            peak_bytes = torch.cuda.max_memory_allocated()
            self.peak_memory_mb = max(self.peak_memory_mb, peak_bytes / (1024 * 1024))

    def get_stats(self) -> Dict[str, Any]:
        total_time = time.time() - self.start_time if self.start_time else 0
        return {
            'total_time_seconds': total_time,
            'total_generation_time': self.total_generation_time,
            'total_se_time': self.total_se_time,
            'total_tokens_generated': self.total_tokens_generated,
            'peak_memory_mb': self.peak_memory_mb,
            'avg_time_per_question': np.mean(self.question_times) if self.question_times else 0,
            'tokens_per_second': self.total_tokens_generated / self.total_generation_time if self.total_generation_time > 0 else 0,
        }


def load_config() -> Dict[str, Any]:
    """Load experiment configuration."""
    config_path = experiment_dir / "config.yaml"
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
        answers = list(set(answers))  # Remove duplicates

        questions.append({
            'question': item['question'],
            'answers': answers,  # List of valid answers
            'question_id': item['question_id']
        })

    print(f"Loaded {len(questions)} questions")
    return questions


def setup_model(config: Dict[str, Any]):
    """Initialize language model and tokenizer."""
    print(f"Loading model: {config['model']['name']}...")

    device = config['model']['device']
    dtype = torch.float16 if config['model']['torch_dtype'] == 'float16' else torch.float32

    tokenizer = AutoTokenizer.from_pretrained(config['model']['name'])
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        config['model']['name'],
        torch_dtype=dtype
    ).to(device)
    model.eval()

    print(f"Model loaded on {device}")
    return model, tokenizer


def generate_samples(
    question: str,
    model,
    tokenizer,
    config: Dict[str, Any]
) -> tuple:
    """
    Generate multiple samples for a question using naive temperature sampling.

    Returns tuple of (samples, generation_time, total_tokens).
    """
    device = config['model']['device']

    # Determine number of samples
    if config['debug']['enabled']:
        num_samples = config['debug']['num_samples']
    else:
        num_samples = config['generation']['num_samples']

    # Format prompt with chat template
    messages = [{"role": "user", "content": f"Answer this question briefly: {question}"}]
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    input_ids = tokenizer.encode(prompt, return_tensors='pt').to(device)

    samples = []
    total_tokens = 0
    start_time = time.time()

    for _ in range(num_samples):
        with torch.no_grad():
            output_ids = model.generate(
                input_ids,
                max_new_tokens=config['generation']['max_new_tokens'],
                temperature=config['generation']['temperature'],
                top_p=config['generation']['top_p'],
                do_sample=config['generation']['do_sample'],
                pad_token_id=tokenizer.eos_token_id
            )

        # Decode only generated part (NOT the prompt)
        generated_ids = output_ids[0][input_ids.shape[1]:]
        generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True)
        samples.append(generated_text.strip())
        total_tokens += len(generated_ids)

    generation_time = time.time() - start_time
    return samples, generation_time, total_tokens


def compute_correctness(
    generated: str,
    ground_truth_answers: List[str],
    scorer: rouge_scorer.RougeScorer,
    threshold: float = 0.3
) -> Dict[str, Any]:
    """
    Check if generated answer is correct using RougeL.

    Returns dict with is_correct, best_score, and matched_answer.
    """
    best_score = 0.0
    best_answer = None

    for answer in ground_truth_answers:
        scores = scorer.score(answer.lower(), generated.lower())
        rouge_l = scores['rougeL'].fmeasure

        if rouge_l > best_score:
            best_score = rouge_l
            best_answer = answer

    return {
        'is_correct': best_score >= threshold,
        'rouge_l_score': best_score,
        'matched_answer': best_answer
    }


def run_single_question(
    question_data: Dict[str, Any],
    model,
    tokenizer,
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

    # Step 1: Generate samples (with timing)
    samples, gen_time, num_tokens = generate_samples(question, model, tokenizer, config)
    cost_tracker.record_generation(gen_time, num_tokens)
    cost_tracker.update_peak_memory()

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

    # Aggregate correctness: use majority vote or any-correct
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
        'se_cluster_labels': se_result['cluster_labels'],

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
    print("=" * 70)
    print("EXPERIMENT 2.A.1: SEMANTIC ENTROPY AUROC ON TRIVIAQA")
    print("=" * 70)

    # Load configuration
    print("\n1. Loading configuration...")
    config = load_config()

    if config['debug']['enabled']:
        print("   [DEBUG MODE ENABLED]")

    # Set random seeds
    torch.manual_seed(config['seed'])
    np.random.seed(config['seed'])
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config['seed'])

    # Load dataset
    print("\n2. Loading TriviaQA dataset...")
    questions = load_triviaqa(config)

    # Setup model
    print("\n3. Setting up language model...")
    model, tokenizer = setup_model(config)

    # Setup semantic entropy estimator
    print("\n4. Setting up semantic entropy estimator...")
    se_estimator = SemanticUncertaintyEstimator(
        encoder_model=config['semantic_entropy']['encoder_model'],
        distance_threshold=config['semantic_entropy']['default_threshold'],
        linkage=config['semantic_entropy']['linkage'],
        device=config['model']['device']
    )
    print(f"   Encoder: {config['semantic_entropy']['encoder_model']}")
    print(f"   Distance threshold: {config['semantic_entropy']['default_threshold']}")

    # Setup RougeL scorer
    scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)

    # Setup cost tracker
    cost_tracker = CostTracker(device=config['model']['device'])

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
                question_data, model, tokenizer, se_estimator, scorer, config, cost_tracker
            )
            all_results.append(result)

            # Save intermediate results
            if config['output']['save_intermediate'] and (i + 1) % 10 == 0:
                save_json({
                    'metadata': {'config': config, 'num_processed': len(all_results)},
                    'results': all_results
                }, results_dir / "raw_results_intermediate.json")

        except Exception as e:
            print(f"\n   Error on question {i}: {e}")
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
            'generation_method': 'naive_sampling',  # For comparison with EAB later
        },
        'cost_stats': cost_stats,
        'results': all_results
    }
    save_json(final_output, results_dir / "raw_results.json")

    # Quick summary
    print("\n" + "=" * 70)
    print("EXPERIMENT COMPLETE")
    print("=" * 70)

    # Compute quick stats
    se_scores = [r['se_uncertainty_score'] for r in all_results]
    correctness = [r['any_correct'] for r in all_results]

    print(f"\nQuick Summary:")
    print(f"  Total questions: {len(all_results)}")
    print(f"  Accuracy (any correct): {sum(correctness)/len(correctness):.1%}")
    print(f"  Avg SE uncertainty: {np.mean(se_scores):.3f}")
    print(f"  Avg clusters: {np.mean([r['se_n_clusters'] for r in all_results]):.1f}")

    print(f"\nCost Summary:")
    print(f"  Total time: {cost_stats['total_time_seconds']:.1f}s")
    print(f"  Generation time: {cost_stats['total_generation_time']:.1f}s")
    print(f"  SE computation time: {cost_stats['total_se_time']:.1f}s")
    print(f"  Total tokens generated: {cost_stats['total_tokens_generated']}")
    print(f"  Tokens/second: {cost_stats['tokens_per_second']:.1f}")
    print(f"  Peak GPU memory: {cost_stats['peak_memory_mb']:.1f} MB")

    print(f"\nResults saved to: {results_dir / 'raw_results.json'}")
    print("\nNext steps:")
    print("  1. Run: python analyze_results.py")
    print("  2. Run: python plot_results.py")
    print("=" * 70)


if __name__ == "__main__":
    main()
