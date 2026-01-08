#!/usr/bin/env python3
"""
Quick Test: 5-prompt sanity check for Semantic Entropy

Verifies that the semantic entropy estimator is working correctly
by testing on 5 prompts with expected entropy levels.

Usage:
    python quick_test.py [--embedder mpnet|sentence-t5|minilm] [--n-samples 10]

Expected runtime: < 30 seconds
"""

import argparse
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from semantic_entropy.estimator import SemanticUncertaintyEstimator
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import numpy as np


# Test prompts with expected uncertainty levels
QUICK_PROMPTS = [
    {
        "prompt": "What is the capital of France?",
        "type": "factual",
        "expected_entropy": "LOW",
        "description": "Simple factual question with one clear answer"
    },
    {
        "prompt": "What is the best programming language?",
        "type": "opinion",
        "expected_entropy": "MEDIUM-HIGH",
        "description": "Subjective question with multiple valid opinions"
    },
    {
        "prompt": "Once upon a time in a magical forest",
        "type": "creative",
        "expected_entropy": "HIGH",
        "description": "Creative continuation with many possibilities"
    },
    {
        "prompt": "Explain quantum computing in simple terms.",
        "type": "technical",
        "expected_entropy": "MEDIUM",
        "description": "Technical explanation with some variation"
    },
    {
        "prompt": "What is 2 + 2?",
        "type": "trivial",
        "expected_entropy": "VERY LOW",
        "description": "Trivial math with single answer"
    }
]


class SampleGenerator:
    """Generate samples using Qwen2.5-3B-Instruct with chat template."""

    def __init__(self, model_name: str = "Qwen/Qwen2.5-3B-Instruct", device: str = "cuda"):
        """
        Initialize generator with Qwen model.

        Args:
            model_name: HuggingFace model name
            device: Device to run on ('cuda' or 'cpu')
        """
        print(f"Loading model: {model_name}...")
        self.device = device if torch.cuda.is_available() else "cpu"

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
            device_map="auto" if self.device == "cuda" else None
        )

        if self.device == "cpu":
            self.model = self.model.to(self.device)

        self.model.eval()
        print(f"✓ Model loaded on {self.device}\n")

    def generate_samples(
        self,
        prompt: str,
        n_samples: int = 10,
        max_new_tokens: int = 50,
        temperature_range: tuple = (0.7, 1.3)
    ) -> list:
        """
        Generate multiple samples for a given prompt with varied temperatures.

        Args:
            prompt: User prompt
            n_samples: Number of samples to generate
            max_new_tokens: Maximum tokens to generate
            temperature_range: (min, max) temperature range for diversity

        Returns:
            List of generated text samples (assistant responses only)
        """
        # Format with chat template
        messages = [{"role": "user", "content": prompt}]
        chat_prompt = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )

        # Tokenize
        inputs = self.tokenizer(chat_prompt, return_tensors="pt").to(self.device)
        input_length = inputs['input_ids'].shape[1]

        # Generate temperature values across range for diversity
        temperatures = np.linspace(temperature_range[0], temperature_range[1], n_samples)

        samples = []

        with torch.no_grad():
            for i in range(n_samples):
                # Use different temperature for each sample
                temp = float(temperatures[i])

                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    temperature=temp,
                    do_sample=True,
                    top_p=0.9,
                    pad_token_id=self.tokenizer.eos_token_id
                )

                # Extract only the generated tokens (excluding prompt)
                generated_tokens = outputs[0][input_length:]

                # Decode only the assistant's response
                response = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)
                samples.append(response.strip())

        return samples


def run_quick_test(embedder: str = "sentence-t5", n_samples: int = 10):
    """
    Run quick test on 5 prompts.

    Args:
        embedder: Embedder to use ('mpnet', 'sentence-t5', 'minilm')
        n_samples: Number of samples per prompt
    """
    print("=" * 70)
    print("SEMANTIC ENTROPY - QUICK TEST")
    print("=" * 70)
    print(f"\nConfiguration:")
    print(f"  Embedder: {embedder}")
    print(f"  Samples per prompt: {n_samples}")
    print(f"  Test prompts: {len(QUICK_PROMPTS)}")
    print("\n" + "=" * 70 + "\n")

    # Initialize components
    generator = SampleGenerator()
    estimator = SemanticUncertaintyEstimator(encoder_model=embedder)

    results = []

    # Test each prompt
    for i, prompt_data in enumerate(QUICK_PROMPTS, 1):
        prompt = prompt_data["prompt"]
        expected_level = prompt_data["expected_entropy"]
        prompt_type = prompt_data["type"]

        print(f"[{i}/{len(QUICK_PROMPTS)}] Testing: {prompt_type.upper()}")
        print(f"  Prompt: \"{prompt}\"")
        print(f"  Expected entropy: {expected_level}")

        # Generate samples
        print(f"  Generating {n_samples} samples...")
        samples = generator.generate_samples(prompt, n_samples=n_samples)

        # Compute semantic entropy
        result = estimator.compute(samples, return_details=True)

        # Display results
        print(f"  ✓ Generated {len(samples)} samples")
        print(f"  ✓ Found {result['n_clusters']} semantic clusters")
        print(f"  ✓ Normalized entropy: {result['normalized_entropy']:.3f}")
        print(f"  ✓ Uncertainty score: {result['uncertainty_score']:.3f}")

        # Interpretation
        interpretation = estimator.interpret_uncertainty(result['uncertainty_score'])
        print(f"  → {interpretation}")

        # Show cluster distribution
        cluster_probs = result['cluster_probs']
        print(f"  → Cluster distribution: {[f'{p:.2f}' for p in cluster_probs]}")

        print()

        results.append({
            'prompt': prompt,
            'type': prompt_type,
            'expected': expected_level,
            'entropy': result['normalized_entropy'],
            'uncertainty': result['uncertainty_score'],
            'n_clusters': result['n_clusters'],
            'interpretation': interpretation
        })

    # Summary table
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"{'Type':<12} {'Expected':<15} {'Entropy':<10} {'Clusters':<10} {'Status'}")
    print("-" * 70)

    for r in results:
        # Determine if result matches expectation
        if r['expected'] == 'VERY LOW':
            status = "✓ PASS" if r['entropy'] < 0.2 else "✗ FAIL"
        elif r['expected'] == 'LOW':
            status = "✓ PASS" if r['entropy'] < 0.3 else "✗ FAIL"
        elif r['expected'] == 'MEDIUM':
            status = "✓ PASS" if 0.3 <= r['entropy'] <= 0.7 else "✗ FAIL"
        elif r['expected'] == 'MEDIUM-HIGH':
            status = "✓ PASS" if r['entropy'] >= 0.5 else "✗ FAIL"
        elif r['expected'] == 'HIGH':
            status = "✓ PASS" if r['entropy'] >= 0.7 else "✗ FAIL"
        else:
            status = "?"

        print(f"{r['type']:<12} {r['expected']:<15} {r['entropy']:<10.3f} {r['n_clusters']:<10} {status}")

    print("=" * 70)
    print("\n✓ Quick test complete!")


def main():
    parser = argparse.ArgumentParser(description="Quick test for semantic entropy")
    parser.add_argument(
        '--embedder',
        type=str,
        default='sentence-t5',
        choices=['mpnet', 'sentence-t5', 'minilm'],
        help='Embedder model to use'
    )
    parser.add_argument(
        '--n-samples',
        type=int,
        default=10,
        help='Number of samples per prompt'
    )

    args = parser.parse_args()

    run_quick_test(embedder=args.embedder, n_samples=args.n_samples)


if __name__ == "__main__":
    main()
