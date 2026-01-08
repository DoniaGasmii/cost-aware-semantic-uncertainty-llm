#!/usr/bin/env python3
"""
Interactive Demo: Semantic Entropy Clustering Debugger

Features:
- Enter custom prompts or select presets
- Generate samples with Qwen2.5-3B-Instruct
- Visualize clustering (2D, 3D, heatmap)
- Interactive threshold tuning
- Export analysis and plots

Usage:
    python interactive_demo.py [--embedder sentence-t5|mpnet|minilm]
"""

import argparse
import sys
from pathlib import Path
from datetime import datetime

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from semantic_entropy.estimator import SemanticUncertaintyEstimator
from clustering_visualizer import ClusteringVisualizer, create_visualization_suite
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import json


# Preset prompts for quick testing
PRESET_PROMPTS = {
    "1": {"prompt": "What is the capital of France?", "type": "factual"},
    "2": {"prompt": "What is the best programming language?", "type": "opinion"},
    "3": {"prompt": "Once upon a time in a magical forest", "type": "creative"},
    "4": {"prompt": "Explain quantum computing", "type": "technical"},
    "5": {"prompt": "What are the main causes of climate change?", "type": "factual-complex"},
}


class SampleGenerator:
    """Generate samples using Qwen2.5-3B-Instruct."""

    def __init__(self, model_name: str = "Qwen/Qwen2.5-3B-Instruct", device: str = "cuda"):
        print(f"\nLoading model: {model_name}...")
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
        print(f"✓ Model loaded on {self.device}")

    def generate_samples(self, prompt: str, n_samples: int = 10, max_new_tokens: int = 50,
                        temperature_range: tuple = (0.7, 1.3)) -> list:
        """Generate multiple samples for a prompt with varied temperatures."""
        import numpy as np

        messages = [{"role": "user", "content": prompt}]
        chat_prompt = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        inputs = self.tokenizer(chat_prompt, return_tensors="pt").to(self.device)
        input_length = inputs['input_ids'].shape[1]

        # Generate temperature values across range for diversity
        temperatures = np.linspace(temperature_range[0], temperature_range[1], n_samples)

        samples = []
        print(f"\nGenerating {n_samples} samples (temp range: {temperature_range[0]:.1f}-{temperature_range[1]:.1f})", end="", flush=True)

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

                generated_tokens = outputs[0][input_length:]
                response = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)
                samples.append(response.strip())

                if (i + 1) % 5 == 0:
                    print(".", end="", flush=True)

        print(" Done!")
        return samples


class InteractiveDemo:
    """Interactive clustering demo."""

    def __init__(self, embedder: str = "sentence-t5"):
        self.generator = SampleGenerator()
        self.estimator = SemanticUncertaintyEstimator(encoder_model=embedder)
        self.visualizer = ClusteringVisualizer()
        self.output_dir = Path("demo_results")
        self.output_dir.mkdir(exist_ok=True)

        print(f"\n✓ Semantic Entropy Interactive Demo initialized")
        print(f"  Embedder: {embedder}")
        print(f"  Output directory: {self.output_dir}\n")

    def get_prompt(self) -> str:
        """Get prompt from user (preset or custom)."""
        print("=" * 70)
        print("SELECT PROMPT")
        print("=" * 70)
        print("\nPresets:")
        for key, data in PRESET_PROMPTS.items():
            print(f"  {key}. {data['prompt']} [{data['type']}]")
        print("  c. Enter custom prompt")

        choice = input("\nYour choice: ").strip().lower()

        if choice in PRESET_PROMPTS:
            return PRESET_PROMPTS[choice]['prompt']
        else:
            return input("\nEnter your custom prompt: ").strip()

    def display_clusters(self, result: dict, samples: list):
        """Display cluster information."""
        print("\n" + "=" * 70)
        print("CLUSTER ANALYSIS")
        print("=" * 70)

        cluster_analysis = result['cluster_analysis']
        cluster_probs = result['cluster_probs']

        for cluster_id, info in cluster_analysis['representatives'].items():
            print(f"\nCluster {cluster_id} ({info['size']} samples, {cluster_probs[cluster_id]:.1%}):")
            print(f"  Representative: \"{info['text'][:80]}...\"")

            # Show all samples in this cluster
            cluster_samples = [samples[i] for i, label in enumerate(result['cluster_labels'])
                             if label == cluster_id]
            print(f"  Samples:")
            for i, sample in enumerate(cluster_samples[:3], 1):  # Show first 3
                print(f"    {i}. \"{sample[:60]}...\"")
            if len(cluster_samples) > 3:
                print(f"    ... and {len(cluster_samples) - 3} more")

    def display_metrics(self, result: dict):
        """Display entropy metrics."""
        print("\n" + "=" * 70)
        print("ENTROPY METRICS")
        print("=" * 70)
        print(f"  Raw entropy:        {result['entropy']:.3f}")
        print(f"  Normalized entropy: {result['normalized_entropy']:.3f}")
        print(f"  Uncertainty score:  {result['uncertainty_score']:.3f}")
        print(f"  Number of clusters: {result['n_clusters']}")

        interpretation = self.estimator.interpret_uncertainty(result['uncertainty_score'])
        print(f"\n  → {interpretation}")

    def generate_visualizations(self, result: dict, samples: list, prompt: str):
        """Generate all visualizations."""
        print("\n" + "=" * 70)
        print("GENERATING VISUALIZATIONS")
        print("=" * 70)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        prefix = f"demo_{timestamp}"

        embeddings = result['embeddings']
        labels = result['cluster_labels']
        similarity_matrix = result['similarity_matrix']

        # Create full visualization suite
        create_visualization_suite(
            embeddings=embeddings,
            labels=labels,
            texts=samples,
            similarity_matrix=similarity_matrix,
            threshold=self.estimator.clusterer.distance_threshold,
            output_dir=self.output_dir,
            prefix=prefix
        )

        # Save analysis text
        analysis_path = self.output_dir / f"{prefix}_analysis.txt"
        with open(analysis_path, 'w') as f:
            f.write("SEMANTIC ENTROPY ANALYSIS\n")
            f.write("=" * 70 + "\n\n")
            f.write(f"Prompt: {prompt}\n")
            f.write(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Embedder: {self.estimator.encoder_model_name}\n")
            f.write(f"Threshold: {self.estimator.clusterer.distance_threshold}\n\n")
            f.write(f"Metrics:\n")
            f.write(f"  Raw entropy: {result['entropy']:.3f}\n")
            f.write(f"  Normalized entropy: {result['normalized_entropy']:.3f}\n")
            f.write(f"  Uncertainty score: {result['uncertainty_score']:.3f}\n")
            f.write(f"  Number of clusters: {result['n_clusters']}\n\n")

            f.write("Clusters:\n")
            for cluster_id, info in result['cluster_analysis']['representatives'].items():
                f.write(f"\nCluster {cluster_id} ({info['size']} samples):\n")
                f.write(f"  Representative: {info['text']}\n")
                cluster_samples = [samples[i] for i, label in enumerate(result['cluster_labels'])
                                 if label == cluster_id]
                for i, sample in enumerate(cluster_samples, 1):
                    f.write(f"  {i}. {sample}\n")

        print(f"✓ Analysis saved to: {analysis_path}")

        # Save results JSON
        results_path = self.output_dir / f"{prefix}_results.json"
        results_data = {
            'prompt': prompt,
            'timestamp': datetime.now().isoformat(),
            'embedder': self.estimator.encoder_model_name,
            'threshold': self.estimator.clusterer.distance_threshold,
            'entropy': result['entropy'],
            'normalized_entropy': result['normalized_entropy'],
            'uncertainty_score': result['uncertainty_score'],
            'n_clusters': result['n_clusters'],
            'cluster_probs': result['cluster_probs'],
            'samples': samples
        }

        with open(results_path, 'w') as f:
            json.dump(results_data, f, indent=2)

        print(f"✓ Results saved to: {results_path}")

    def tune_threshold(self, samples: list):
        """Interactive threshold tuning."""
        print("\n" + "=" * 70)
        print("THRESHOLD TUNING")
        print("=" * 70)

        thresholds = [0.10, 0.15, 0.20, 0.25, 0.30]

        print(f"\n{'Threshold':<12} {'Clusters':<10} {'Entropy':<12} {'Norm Entropy':<15} {'Max Cluster %'}")
        print("-" * 70)

        for threshold in thresholds:
            self.estimator.clusterer.distance_threshold = threshold
            result = self.estimator.compute(samples)

            max_cluster_pct = max(result['cluster_probs']) * 100

            print(f"{threshold:<12.2f} {result['n_clusters']:<10} "
                  f"{result['entropy']:<12.3f} {result['normalized_entropy']:<15.3f} "
                  f"{max_cluster_pct:.1f}%")

        # Ask user to select threshold
        new_threshold = input("\nEnter new threshold (or press Enter to keep current): ").strip()
        if new_threshold:
            try:
                threshold_val = float(new_threshold)
                self.estimator.clusterer.distance_threshold = threshold_val
                print(f"✓ Threshold updated to {threshold_val}")
            except ValueError:
                print("Invalid threshold, keeping current value")

    def run(self):
        """Run interactive demo loop."""
        print("\n" + "=" * 70)
        print("SEMANTIC ENTROPY - INTERACTIVE CLUSTERING DEBUGGER")
        print("=" * 70)

        while True:
            # Get prompt
            prompt = self.get_prompt()

            # Get parameters
            n_samples = int(input("\nNumber of samples to generate (default: 10): ").strip() or "10")
            max_tokens = int(input("Max tokens per sample (default: 50): ").strip() or "50")

            # Generate samples
            samples = self.generator.generate_samples(prompt, n_samples=n_samples, max_new_tokens=max_tokens)

            # Compute semantic entropy
            print("\n✓ Computing semantic entropy...")
            result = self.estimator.compute(samples, return_details=True)

            # Display results
            self.display_metrics(result)
            self.display_clusters(result, samples)

            # Generate visualizations
            generate_viz = input("\nGenerate visualizations? (y/n, default: y): ").strip().lower()
            if generate_viz != 'n':
                self.generate_visualizations(result, samples, prompt)

            # Threshold tuning
            tune = input("\nTune clustering threshold? (y/n, default: n): ").strip().lower()
            if tune == 'y':
                self.tune_threshold(samples)

                # Recompute with new threshold
                print("\nRecomputing with new threshold...")
                result = self.estimator.compute(samples, return_details=True)
                self.display_metrics(result)
                self.display_clusters(result, samples)

            # Continue?
            again = input("\nAnalyze another prompt? (y/n, default: n): ").strip().lower()
            if again != 'y':
                break

        print("\n✓ Demo session complete!")
        print(f"  Results saved in: {self.output_dir}")


def main():
    parser = argparse.ArgumentParser(description="Interactive semantic entropy demo")
    parser.add_argument(
        '--embedder',
        type=str,
        default='sentence-t5',
        choices=['mpnet', 'sentence-t5', 'minilm'],
        help='Embedder model to use'
    )

    args = parser.parse_args()

    demo = InteractiveDemo(embedder=args.embedder)
    demo.run()


if __name__ == "__main__":
    main()
