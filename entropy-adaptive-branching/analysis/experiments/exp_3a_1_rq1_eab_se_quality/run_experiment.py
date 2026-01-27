"""
RQ1: EAB+SE Quality Evaluation

Generates samples using EAB (with strict branching threshold) and evaluates
SE quality to validate that EAB-generated diversity maintains SE effectiveness.
"""

import sys
import json
import yaml
import torch
import numpy as np
import time
import argparse
from pathlib import Path
from tqdm import tqdm
from typing import Dict, Any, List
from datetime import datetime

# Resolve paths
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[3]

# Add necessary paths
sys.path.insert(0, str(PROJECT_ROOT / "entropy-adaptive-branching"))
sys.path.insert(0, str(PROJECT_ROOT / "semantic-entropy-based-uncertainty"))
sys.path.insert(0, str(PROJECT_ROOT))

# Import dependencies
from datasets import load_dataset
from rouge_score import rouge_scorer

# Import EAB
from eab import EntropyAdaptiveBranching

# Import semantic entropy estimator
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
        self.total_branches = 0
        self.peak_memory_mb = 0.0
        self.question_times = []

    def start(self):
        self.start_time = time.time()
        if self.device == 'cuda' and torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()

    def record_generation(self, elapsed: float, num_tokens: int, num_branches: int = 0):
        self.total_generation_time += elapsed
        self.total_tokens_generated += num_tokens
        self.total_branches += num_branches

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
            'total_branches': self.total_branches,
            'peak_memory_mb': self.peak_memory_mb,
            'avg_time_per_question': np.mean(self.question_times) if self.question_times else 0,
            'tokens_per_second': self.total_tokens_generated / self.total_generation_time if self.total_generation_time > 0 else 0,
        }


def load_config(config_path: str = None) -> Dict[str, Any]:
    if config_path is None:
        config_path = SCRIPT_DIR / "config.yaml"
    else:
        config_path = Path(config_path)
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def load_triviaqa(config: Dict[str, Any]) -> List[Dict[str, Any]]:
    print("Loading TriviaQA dataset...")
    num_questions = config['dataset']['num_questions']
    dataset = load_dataset(
        config['dataset']['name'],
        config['dataset']['subset'],
        split=config['dataset']['split']
    )
    dataset = dataset.shuffle(seed=config['seed'])
    dataset = dataset.select(range(min(num_questions, len(dataset))))

    questions = []
    for item in dataset:
        answers = item['answer']['aliases'] + [item['answer']['value']]
        answers = list(set(a.strip() for a in answers if a.strip()))
        questions.append({
            'question': item['question'].strip(),
            'answers': answers,
            'question_id': item['question_id']
        })
    print(f"Loaded {len(questions)} questions")
    return questions


def setup_eab(config: Dict[str, Any]):
    """Initialize EAB (which loads the model internally)."""
    dtype = torch.float16 if config['model']['torch_dtype'] == 'float16' else torch.float32

    # Initialize EAB - it loads the model internally
    eab = EntropyAdaptiveBranching(
        model_name=config['model']['name'],
        device=config['model']['device'],
        torch_dtype=dtype,
        entropy_threshold=config['eab']['entropy_threshold'],
        branch_factor=config['eab']['branch_factor'],
        max_paths=config['eab']['max_concurrent_paths'],
        use_cow=True  # Use copy-on-write for memory efficiency
    )

    print(f"\n✓ EAB initialized")
    print(f"  - Entropy threshold: {config['eab']['entropy_threshold']}")
    print(f"  - Branch factor: {config['eab']['branch_factor']}")
    print(f"  - Max paths: {config['eab']['max_concurrent_paths']}")
    return eab


def generate_samples_with_eab(question: str, eab: EntropyAdaptiveBranching, config: Dict[str, Any]) -> tuple:
    """Generate samples using EAB."""
    messages = [{"role": "user", "content": f"Answer briefly: {question}"}]
    prompt = eab.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    # Generate with EAB
    start_time = time.time()
    result = eab.generate(
        prompt=prompt,
        max_new_tokens=config['eab']['max_generation_length'],
        temperature=config['eab']['temperature'],
        top_p=config['eab']['top_p']
    )
    elapsed = time.time() - start_time

    # Extract samples from result (EAB returns list of dicts with 'text' key)
    samples = [r['text'] for r in result]

    # Get statistics
    stats = eab.get_entropy_history()['statistics']
    num_branches = stats['num_branches']
    mean_entropy = stats['mean_entropy']
    branch_rate = stats['branch_rate']

    total_tokens = sum(len(eab.tokenizer.encode(s)) for s in samples)

    return samples, elapsed, total_tokens, num_branches, mean_entropy, branch_rate


def compute_semantic_entropy(samples: List[str], estimator: SemanticUncertaintyEstimator) -> Dict[str, Any]:
    """Compute semantic entropy metrics."""
    se_results = estimator.compute(samples)
    return {
        'se_entropy': se_results['entropy'],
        'se_normalized_entropy': se_results['normalized_entropy'],
        'se_uncertainty_score': se_results['uncertainty_score'],
        'se_n_clusters': se_results['n_clusters'],
        'se_cluster_labels': se_results['cluster_labels']
    }


def evaluate_correctness(samples: List[str], ground_truth: List[str], config: Dict[str, Any]) -> Dict[str, Any]:
    """Evaluate sample correctness using RougeL."""
    scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=False)
    threshold = config['correctness']['threshold']

    correctness = []
    for sample in samples:
        sample_scores = []
        for gt in ground_truth:
            score = scorer.score(gt.lower(), sample.lower())['rougeL'].fmeasure
            sample_scores.append(score)
        best_score = max(sample_scores) if sample_scores else 0.0
        correctness.append({
            'is_correct': best_score >= threshold,
            'rouge_l_score': best_score,
            'matched_answer': ground_truth[sample_scores.index(best_score)] if sample_scores else None
        })

    num_correct = sum(1 for c in correctness if c['is_correct'])
    best_rouge = max([c['rouge_l_score'] for c in correctness]) if correctness else 0.0

    return {
        'sample_correctness': correctness,
        'num_correct_samples': num_correct,
        'any_correct': num_correct > 0,
        'majority_correct': num_correct > len(samples) / 2,
        'best_sample_correct': best_rouge >= threshold,
        'best_rouge_l': best_rouge
    }


def run_experiment(config: Dict[str, Any]):
    """Main experiment loop."""
    # Setup
    questions = load_triviaqa(config)
    eab = setup_eab(config)

    # Initialize SE estimator
    se_estimator = SemanticUncertaintyEstimator(
        encoder_model=config['semantic_entropy']['encoder_model'],
        distance_threshold=config['semantic_entropy']['default_threshold'],
        linkage=config['semantic_entropy']['linkage']
    )

    # Cost tracking
    cost_tracker = CostTracker(device=config['model']['device'])
    cost_tracker.start()

    # Run experiment
    results = []
    print(f"\nProcessing {len(questions)} questions...")

    for q in tqdm(questions, desc="Running RQ1 Experiment"):
        q_start = time.time()

        # Generate samples with EAB
        samples, gen_time, num_tokens, num_branches, mean_entropy, branch_rate = generate_samples_with_eab(
            q['question'], eab, config
        )

        # Compute semantic entropy
        se_start = time.time()
        se_metrics = compute_semantic_entropy(samples, se_estimator)
        se_time = time.time() - se_start

        # Evaluate correctness
        correctness = evaluate_correctness(samples, q['answers'], config)

        # Record
        cost_tracker.record_generation(gen_time, num_tokens, num_branches)
        cost_tracker.record_se(se_time)
        cost_tracker.record_question(time.time() - q_start)
        cost_tracker.update_peak_memory()

        result = {
            'question_id': q['question_id'],
            'question': q['question'],
            'ground_truth_answers': q['answers'],
            'generated_samples': samples,
            'num_samples': len(samples),
            **se_metrics,
            'eab_num_branches': num_branches,
            'eab_mean_entropy': mean_entropy,
            'eab_branch_rate': branch_rate,
            **correctness,
            'generation_time': gen_time,
            'se_time': se_time,
            'total_time': time.time() - q_start,
            'num_tokens_generated': num_tokens
        }
        results.append(result)

    return results, cost_tracker.get_stats()


def save_results(results: List[Dict[str, Any]], cost_stats: Dict[str, Any], config: Dict[str, Any]):
    """Save experiment results."""
    output_dir = SCRIPT_DIR / config['output']['results_dir']
    output_dir.mkdir(parents=True, exist_ok=True)

    output_data = {
        'metadata': {
            'experiment': config['experiment']['name'],
            'description': config['experiment']['description'],
            'timestamp': datetime.now().isoformat(),
            'num_questions': len(results),
            'generation_method': 'eab',
            'eab_entropy_threshold': config['eab']['entropy_threshold'],
            'se_distance_threshold': config['semantic_entropy']['default_threshold']
        },
        'cost_stats': cost_stats,
        'results': results
    }

    output_file = output_dir / "raw_results.json"
    with open(output_file, 'w') as f:
        json.dump(output_data, f, indent=2)

    print(f"\n✓ Results saved to: {output_file}")


def main():
    parser = argparse.ArgumentParser(description="RQ1: EAB+SE Quality Evaluation")
    parser.add_argument('--config', type=str, default=None, help='Path to config file')
    args = parser.parse_args()

    print("=" * 70)
    print("RQ1: VALIDATING EAB MAINTAINS SE QUALITY")
    print("=" * 70)

    config = load_config(args.config)

    # Debug mode
    if config.get('debug', {}).get('enabled', False):
        print("\n⚠️  DEBUG MODE ENABLED")
        config['dataset']['num_questions'] = config['debug']['num_questions']

    results, cost_stats = run_experiment(config)
    save_results(results, cost_stats, config)

    print("\n" + "=" * 70)
    print("EXPERIMENT COMPLETE")
    print("=" * 70)
    print(f"Questions processed: {len(results)}")
    print(f"Total time: {cost_stats['total_time_seconds']:.1f}s")
    print(f"Avg samples per question: {np.mean([r['num_samples'] for r in results]):.1f}")
    print(f"Avg branches per question: {cost_stats['total_branches'] / len(results):.1f}")


if __name__ == "__main__":
    main()
