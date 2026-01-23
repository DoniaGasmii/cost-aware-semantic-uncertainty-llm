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
from typing import Dict, Any, List
from datetime import datetime

# Resolve paths relative to this script
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[3]  # repo root

# Add necessary paths
sys.path.insert(0, str(PROJECT_ROOT / "semantic-entropy-based-uncertainty"))
sys.path.insert(0, str(PROJECT_ROOT))

# Import dependencies
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from rouge_score import rouge_scorer

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
    config_path = SCRIPT_DIR / "config.yaml"
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


def setup_model(config: Dict[str, Any]):
    print(f"Loading model: {config['model']['name']}...")
    device = config['model']['device']
    dtype = torch.float16 if config['model']['torch_dtype'] == 'float16' else torch.float32
    tokenizer = AutoTokenizer.from_pretrained(config['model']['name'])
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        config['model']['name'],
        torch_dtype=dtype,
        device_map="auto",
        low_cpu_mem_usage=True
    )
    model.eval()
    print(f"Model loaded on {device}")
    return model, tokenizer


def generate_samples(question: str, model, tokenizer, config: Dict[str, Any]) -> tuple:
    device = config['model']['device']
    num_samples = config['generation']['num_samples']
    messages = [{"role": "user", "content": f"Answer briefly: {question}"}]
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
                pad_token_id=tokenizer.eos_token_id,
                use_cache=True
            )
        generated_ids = output_ids[0][input_ids.shape[1]:]
        generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
        samples.append(generated_text)
        total_tokens += len(generated_ids)
    generation_time = time.time() - start_time
    return samples, generation_time, total_tokens


def compute_correctness(generated: str, ground_truth_answers: List[str], scorer, threshold: float = 0.3) -> Dict[str, Any]:
    best_score = 0.0
    for answer in ground_truth_answers:
        try:
            scores = scorer.score(answer.lower(), generated.lower())
            rouge_l = scores['rougeL'].fmeasure
            if rouge_l > best_score:
                best_score = rouge_l
        except Exception:
            continue
    return {'is_correct': best_score >= threshold, 'rouge_l_score': best_score}


def run_single_question(question_data, model, tokenizer, se_estimator, scorer, config, cost_tracker):
    question_start = time.time()
    question = question_data['question']
    answers = question_data['answers']
    samples, gen_time, num_tokens = generate_samples(question, model, tokenizer, config)
    cost_tracker.record_generation(gen_time, num_tokens)
    cost_tracker.update_peak_memory()
    se_start = time.time()
    se_result = se_estimator.compute(samples, return_details=False)
    se_time = time.time() - se_start
    cost_tracker.record_se(se_time)
    sample_correctness = [
        compute_correctness(s, answers, scorer, config['correctness']['threshold'])
        for s in samples
    ]
    num_correct = sum(1 for c in sample_correctness if c['is_correct'])
    any_correct = num_correct > 0
    best_idx = max(range(len(sample_correctness)), key=lambda i: sample_correctness[i]['rouge_l_score'])
    best_correct = sample_correctness[best_idx]['is_correct']
    question_time = time.time() - question_start
    cost_tracker.record_question(question_time)
    return {
        'question_id': question_data['question_id'],
        'question': question,
        'generated_samples': samples,
        'se_entropy': se_result['entropy'],
        'se_normalized_entropy': se_result['normalized_entropy'],
        'se_uncertainty_score': se_result['uncertainty_score'],
        'se_n_clusters': se_result['n_clusters'],
        'any_correct': any_correct,
        'majority_correct': num_correct > len(samples) / 2,
        'best_sample_correct': best_correct,
        'best_rouge_l': sample_correctness[best_idx]['rouge_l_score'],
        'num_samples': len(samples),
        'generation_time': gen_time,
        'num_tokens_generated': num_tokens,
    }


def save_json(data, path: Path):
    with open(path, 'w') as f:
        json.dump(data, f, indent=2, default=str)


def main():
    print("=" * 70)
    print("EXPERIMENT 2.A.1: SEMANTIC ENTROPY AUROC ON TRIVIAQA")
    print("=" * 70)
    config = load_config()
    if config['debug']['enabled']:
        print("   [DEBUG MODE ENABLED]")
    torch.manual_seed(config['seed'])
    np.random.seed(config['seed'])
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config['seed'])
    questions = load_triviaqa(config)
    model, tokenizer = setup_model(config)
    encoder_map = {"mpnet": "all-mpnet-base-v2", "minilm": "all-MiniLM-L6-v2"}
    encoder_name = encoder_map.get(config['semantic_entropy']['encoder_model'], config['semantic_entropy']['encoder_model'])
    se_estimator = SemanticUncertaintyEstimator(
        encoder_model=encoder_name,
        distance_threshold=config['semantic_entropy']['default_threshold'],
        linkage=config['semantic_entropy']['linkage'],
        device=config['model']['device']
    )
    scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
    cost_tracker = CostTracker(device=config['model']['device'])
    results_dir = SCRIPT_DIR / config['output']['results_dir']
    results_dir.mkdir(exist_ok=True, parents=True)
    print("\n5. Running experiment...")
    all_results = []
    cost_tracker.start()
    for i, q in enumerate(tqdm(questions, desc="Processing")):
        try:
            result = run_single_question(q, model, tokenizer, se_estimator, scorer, config, cost_tracker)
            all_results.append(result)
            if config['output']['save_intermediate'] and (i + 1) % 50 == 0:
                save_json({'metadata': {'config': config, 'processed': len(all_results)}, 'results': all_results},
                          results_dir / "raw_results_intermediate.json")
        except Exception as e:
            print(f"\nError on question {i}: {e}")
            continue
    cost_stats = cost_tracker.get_stats()
    final_output = {
        'metadata': {
            'config': config,
            'timestamp': datetime.now().isoformat(),
            'num_questions': len(all_results),
            'generation_method': 'naive_sampling'
        },
        'cost_stats': cost_stats,
        'results': all_results
    }
    save_json(final_output, results_dir / "raw_results.json")
    se_scores = [r['se_uncertainty_score'] for r in all_results]
    correctness = [r['best_sample_correct'] for r in all_results]
    print("\n" + "=" * 70)
    print("EXPERIMENT COMPLETE")
    print("=" * 70)
    print(f"Total questions: {len(all_results)}")
    print(f"Accuracy (best sample): {np.mean(correctness):.1%}")
    print(f"Avg SE uncertainty: {np.mean(se_scores):.3f}")
    print(f"Peak GPU memory: {cost_stats['peak_memory_mb']:.1f} MB")
    print(f"Results saved to: {results_dir / 'raw_results.json'}")


if __name__ == "__main__":
    main()