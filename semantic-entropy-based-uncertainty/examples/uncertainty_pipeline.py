#!/usr/bin/env python3
"""
Experiment Pipeline with GPT-2 + Uncertainty Quantification via Semantic Clustering

Features:
- Multiple generations per prompt
- Semantic clustering with 0.4 threshold (permissive branching)
- Uncertainty measured by cluster diversity
"""

import json
import os
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, asdict
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime
import argparse

# Import transformers for GPT-2
try:
    from transformers import GPT2LMHeadModel, GPT2Tokenizer
    import torch
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    print("⚠️  transformers not installed. Install with: pip install transformers torch")

# Import sentence transformers for semantic clustering
try:
    from sentence_transformers import SentenceTransformer
    from sklearn.cluster import AgglomerativeClustering
    from sklearn.metrics.pairwise import cosine_similarity
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False
    print("⚠️  sentence-transformers not installed. Install with: pip install sentence-transformers scikit-learn")


@dataclass
class PromptBatch:
    """Container for a batch of prompts"""
    name: str
    prompts: List[Dict[str, Any]]
    metadata: Optional[Dict[str, Any]] = None


@dataclass
class GenerationResult:
    """Container for multiple generations from a single prompt"""
    batch_name: str
    prompt_id: str
    prompt: str
    responses: List[str]
    embeddings: Optional[np.ndarray]
    num_clusters: int
    uncertainty_score: float
    mean_score: float
    metadata: Optional[Dict[str, Any]] = None


class GPT2Generator:
    """Wrapper for GPT-2 model with uncertainty quantification"""
    
    def __init__(self, model_name="gpt2-xl", device=None):
        """
        Initialize GPT-2 model
        
        Available models:
        - gpt2 (117M params)
        - gpt2-medium (345M params)
        - gpt2-large (774M params)
        - gpt2-xl (1.5B params) ← Recommended
        """
        if not TRANSFORMERS_AVAILABLE:
            raise ImportError("transformers not installed")
        
        print(f"Loading {model_name}... (this may take a minute)")
        
        # Auto-detect device
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        
        # Load model and tokenizer
        self.tokenizer = GPT2Tokenizer.from_pretrained(model_name)
        self.model = GPT2LMHeadModel.from_pretrained(model_name)
        self.model.to(self.device)
        self.model.eval()
        
        print(f"✓ Loaded {model_name} on {self.device}")
    
    def generate_multiple(
        self, 
        prompt: str, 
        n_generations: int = 5,
        max_length: int = 150, 
        temperature: float = 0.8
    ) -> List[str]:
        """
        Generate multiple diverse responses for uncertainty quantification
        
        Args:
            prompt: Input prompt
            n_generations: Number of responses to generate
            max_length: Maximum length of generated text
            temperature: Higher = more diversity (0.8 recommended for branching)
        
        Returns:
            List of generated responses
        """
        responses = []
        inputs = self.tokenizer.encode(prompt, return_tensors="pt").to(self.device)
        
        for i in range(n_generations):
            with torch.no_grad():
                outputs = self.model.generate(
                    inputs,
                    max_length=max_length,
                    temperature=temperature,
                    do_sample=True,
                    top_p=0.9,
                    top_k=50,
                    pad_token_id=self.tokenizer.eos_token_id,
                    # Ensure diversity
                    num_return_sequences=1
                )
            
            # Decode
            generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Extract only the generated part (remove prompt)
            response = generated_text[len(prompt):].strip()
            responses.append(response)
        
        return responses


class SemanticClusterer:
    """Semantic clustering for measuring response diversity"""
    
    def __init__(self, model_name="all-MiniLM-L6-v2"):
        """
        Initialize sentence transformer for semantic embeddings
        
        Args:
            model_name: SentenceTransformer model
                - all-MiniLM-L6-v2: Fast, good quality (default)
                - all-mpnet-base-v2: Higher quality, slower
        """
        if not SENTENCE_TRANSFORMERS_AVAILABLE:
            raise ImportError("sentence-transformers not installed")
        
        print(f"Loading semantic embedding model {model_name}...")
        self.model = SentenceTransformer(model_name)
        print("✓ Loaded embedding model")
    
    def cluster_responses(
        self, 
        responses: List[str], 
        distance_threshold: float = 0.4
    ) -> Tuple[int, np.ndarray]:
        """
        Cluster responses semantically using hierarchical clustering
        
        Args:
            responses: List of text responses
            distance_threshold: Clustering threshold (0.4 = permissive, allows branching)
                - Lower = more clusters (more permissive)
                - Higher = fewer clusters (more strict)
        
        Returns:
            (num_clusters, embeddings)
        """
        # Generate embeddings
        embeddings = self.model.encode(responses)
        
        # Compute similarity matrix
        similarity_matrix = cosine_similarity(embeddings)
        
        # Convert to distance (1 - similarity)
        distance_matrix = 1 - similarity_matrix
        
        # Hierarchical clustering with distance threshold
        clustering = AgglomerativeClustering(
            n_clusters=None,
            distance_threshold=distance_threshold,
            metric='precomputed',
            linkage='average'
        )
        
        clustering.fit(distance_matrix)
        
        num_clusters = len(set(clustering.labels_))
        
        return num_clusters, embeddings
    
    def compute_uncertainty(
        self, 
        responses: List[str], 
        num_clusters: int
    ) -> float:
        """
        Compute uncertainty score based on cluster diversity
        
        High uncertainty = many different clusters
        Low uncertainty = few clusters (similar responses)
        
        Returns:
            Uncertainty score between 0 and 1
        """
        n = len(responses)
        
        # Normalize by maximum possible clusters
        uncertainty = num_clusters / n
        
        return uncertainty


class ExperimentPipeline:
    """Main pipeline for running experiments with uncertainty quantification"""
    
    def __init__(
        self, 
        output_dir: str = "./experiment_results", 
        model_name: str = "gpt2-xl",
        n_generations: int = 5,
        clustering_threshold: float = 0.4
    ):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.results_file = self.output_dir / "results.jsonl"
        self.n_generations = n_generations
        self.clustering_threshold = clustering_threshold
        
        # Initialize GPT-2
        if TRANSFORMERS_AVAILABLE:
            self.generator = GPT2Generator(model_name=model_name)
        else:
            self.generator = None
            print("⚠️  Running in dummy mode (no model loaded)")
        
        # Initialize semantic clusterer
        if SENTENCE_TRANSFORMERS_AVAILABLE:
            self.clusterer = SemanticClusterer()
        else:
            self.clusterer = None
            print("⚠️  Semantic clustering disabled")
        
    def load_batches(self, batch_files: List[str]) -> List[PromptBatch]:
        """Load prompt batches from JSON files"""
        batches = []
        for file_path in batch_files:
            with open(file_path, 'r') as f:
                data = json.load(f)
                batch = PromptBatch(
                    name=data.get('name', Path(file_path).stem),
                    prompts=data.get('prompts', []),
                    metadata=data.get('metadata', {})
                )
                batches.append(batch)
                print(f"Loaded batch '{batch.name}' with {len(batch.prompts)} prompts")
        return batches
    
    def generate_responses(self, prompt: str) -> List[str]:
        """Generate multiple responses using GPT-2"""
        if self.generator:
            return self.generator.generate_multiple(
                prompt, 
                n_generations=self.n_generations
            )
        else:
            # Fallback to dummy responses
            return [f"Dummy response {i} to: {prompt[:50]}..." for i in range(self.n_generations)]
    
    def analyze_uncertainty(self, responses: List[str]) -> Tuple[int, float]:
        """
        Analyze uncertainty via semantic clustering
        
        Returns:
            (num_clusters, uncertainty_score)
        """
        if self.clusterer and len(responses) > 1:
            num_clusters, embeddings = self.clusterer.cluster_responses(
                responses, 
                distance_threshold=self.clustering_threshold
            )
            uncertainty = self.clusterer.compute_uncertainty(responses, num_clusters)
            return num_clusters, uncertainty
        else:
            # Fallback: assume all responses are different
            return len(responses), 1.0
    
    def score_response(self, prompt: str, response: str, metadata: Optional[Dict] = None) -> float:
        """
        Score a single response based on your criteria.
        
        This is a PLACEHOLDER - replace with your actual scoring logic!
        """
        difficulty = metadata.get('difficulty', 'medium') if metadata else 'medium'
        
        # Dummy scoring based on difficulty
        if difficulty == 'easy':
            mean, std = 0.8, 0.1
        elif difficulty == 'medium':
            mean, std = 0.6, 0.15
        else:  # hard
            mean, std = 0.4, 0.15
        
        return np.clip(np.random.normal(mean, std), 0, 1)
    
    def process_batch(self, batch: PromptBatch) -> List[GenerationResult]:
        """Process all prompts in a batch with uncertainty quantification"""
        results = []
        
        print(f"\nProcessing batch: {batch.name}")
        print(f"  Generating {self.n_generations} responses per prompt")
        print(f"  Clustering threshold: {self.clustering_threshold}")
        
        for i, prompt_data in enumerate(batch.prompts):
            prompt_text = prompt_data.get('prompt', prompt_data.get('text', ''))
            prompt_id = prompt_data.get('id', f"{batch.name}_{i}")
            
            print(f"\n  [{i+1}/{len(batch.prompts)}] Processing: {prompt_id}")
            
            # Generate multiple responses
            responses = self.generate_responses(prompt_text)
            print(f"    Generated {len(responses)} responses")
            
            # Analyze uncertainty via clustering
            num_clusters, uncertainty_score = self.analyze_uncertainty(responses)
            print(f"    Clusters: {num_clusters}/{len(responses)} | Uncertainty: {uncertainty_score:.3f}")
            
            # Score each response
            scores = [
                self.score_response(prompt_text, resp, metadata=prompt_data.get('metadata'))
                for resp in responses
            ]
            mean_score = np.mean(scores)
            
            # Store result
            result = GenerationResult(
                batch_name=batch.name,
                prompt_id=prompt_id,
                prompt=prompt_text,
                responses=responses,
                embeddings=None,  # Don't save embeddings to JSON
                num_clusters=num_clusters,
                uncertainty_score=uncertainty_score,
                mean_score=mean_score,
                metadata=prompt_data.get('metadata')
            )
            results.append(result)
            
            # Save incrementally
            self._save_result(result)
        
        return results
    
    def _save_result(self, result: GenerationResult):
        """Save a single result to JSONL file"""
        # Convert to dict but exclude embeddings
        result_dict = asdict(result)
        result_dict['embeddings'] = None
        
        with open(self.results_file, 'a') as f:
            json.dump(result_dict, f)
            f.write('\n')
    
    def run_experiment(self, batch_files: List[str]) -> Dict[str, List[GenerationResult]]:
        """Run the complete experiment pipeline"""
        print("=" * 80)
        print("UNCERTAINTY QUANTIFICATION EXPERIMENT")
        print("=" * 80)
        print(f"Generations per prompt: {self.n_generations}")
        print(f"Clustering threshold: {self.clustering_threshold} (0.4 = permissive branching)")
        print("=" * 80)
        
        # Load batches
        batches = self.load_batches(batch_files)
        
        # Process each batch
        all_results = {}
        for batch in batches:
            results = self.process_batch(batch)
            all_results[batch.name] = results
        
        print("\n" + "=" * 80)
        print("Experiment Complete!")
        print("=" * 80)
        
        return all_results
    
    def compute_statistics(self, results: Dict[str, List[GenerationResult]]) -> Dict[str, Dict]:
        """Compute statistics for each batch"""
        stats = {}
        
        for batch_name, generation_results in results.items():
            uncertainties = [r.uncertainty_score for r in generation_results]
            mean_scores = [r.mean_score for r in generation_results]
            num_clusters_list = [r.num_clusters for r in generation_results]
            
            stats[batch_name] = {
                'count': len(generation_results),
                'mean_uncertainty': np.mean(uncertainties),
                'std_uncertainty': np.std(uncertainties),
                'mean_score': np.mean(mean_scores),
                'std_score': np.std(mean_scores),
                'mean_clusters': np.mean(num_clusters_list),
                'max_clusters': np.max(num_clusters_list),
            }
        
        return stats
    
    def plot_distributions(self, results: Dict[str, List[GenerationResult]], 
                          save_path: Optional[str] = None):
        """Plot uncertainty distributions for each batch"""
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        
        colors = plt.cm.Set2(np.linspace(0, 1, len(results)))
        
        # Plot 1: Uncertainty distributions
        ax = axes[0]
        for idx, (batch_name, generation_results) in enumerate(results.items()):
            uncertainties = [r.uncertainty_score for r in generation_results]
            
            mean = np.mean(uncertainties)
            std = np.std(uncertainties)
            
            # Create smooth curve
            x = np.linspace(0, 1, 1000)
            y = (1 / (std * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x - mean) / std) ** 2)
            
            ax.plot(x, y, linewidth=2.5, label=batch_name, color=colors[idx], alpha=0.8)
            ax.fill_between(x, y, alpha=0.2, color=colors[idx])
            ax.axvline(mean, color=colors[idx], linestyle='--', linewidth=1.5, alpha=0.7)
            
            ax.text(mean, ax.get_ylim()[1] * 0.9 - (idx * 0.15 * ax.get_ylim()[1]), 
                   f'μ = {mean:.2f}, σ = {std:.2f}',
                   ha='center', fontsize=10,
                   bbox=dict(boxstyle='round', facecolor=colors[idx], alpha=0.3))
        
        ax.set_xlabel('Uncertainty Score', fontsize=12, fontweight='bold')
        ax.set_ylabel('Density', fontsize=12, fontweight='bold')
        ax.set_title('Uncertainty Distributions Across Batches', fontsize=14, fontweight='bold')
        ax.legend(loc='upper right', fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.set_xlim(0, 1)
        
        # Plot 2: Mean quality scores
        ax = axes[1]
        for idx, (batch_name, generation_results) in enumerate(results.items()):
            mean_scores = [r.mean_score for r in generation_results]
            
            mean = np.mean(mean_scores)
            std = np.std(mean_scores)
            
            x = np.linspace(0, 1, 1000)
            y = (1 / (std * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x - mean) / std) ** 2)
            
            ax.plot(x, y, linewidth=2.5, label=batch_name, color=colors[idx], alpha=0.8)
            ax.fill_between(x, y, alpha=0.2, color=colors[idx])
            ax.axvline(mean, color=colors[idx], linestyle='--', linewidth=1.5, alpha=0.7)
            
            ax.text(mean, ax.get_ylim()[1] * 0.9 - (idx * 0.15 * ax.get_ylim()[1]), 
                   f'μ = {mean:.2f}, σ = {std:.2f}',
                   ha='center', fontsize=10,
                   bbox=dict(boxstyle='round', facecolor=colors[idx], alpha=0.3))
        
        ax.set_xlabel('Mean Quality Score', fontsize=12, fontweight='bold')
        ax.set_ylabel('Density', fontsize=12, fontweight='bold')
        ax.set_title('Quality Score Distributions', fontsize=14, fontweight='bold')
        ax.legend(loc='upper right', fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.set_xlim(0, 1)
        
        plt.tight_layout()
        
        if save_path is None:
            save_path = self.output_dir / 'uncertainty_distributions.png'
        
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"\nPlot saved to: {save_path}")
        plt.close()
    
    def generate_report(self, results: Dict[str, List[GenerationResult]]):
        """Generate a text report with statistics"""
        stats = self.compute_statistics(results)
        
        report_path = self.output_dir / 'report.txt'
        
        with open(report_path, 'w') as f:
            f.write("=" * 80 + "\n")
            f.write("UNCERTAINTY QUANTIFICATION REPORT\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Generations per prompt: {self.n_generations}\n")
            f.write(f"Clustering threshold: {self.clustering_threshold}\n")
            f.write("=" * 80 + "\n\n")
            
            for batch_name, batch_stats in stats.items():
                f.write(f"Batch: {batch_name}\n")
                f.write("-" * 40 + "\n")
                f.write(f"  Sample size:         {batch_stats['count']}\n")
                f.write(f"  Mean uncertainty:    {batch_stats['mean_uncertainty']:.4f}\n")
                f.write(f"  Std uncertainty:     {batch_stats['std_uncertainty']:.4f}\n")
                f.write(f"  Mean quality score:  {batch_stats['mean_score']:.4f}\n")
                f.write(f"  Std quality score:   {batch_stats['std_score']:.4f}\n")
                f.write(f"  Avg clusters:        {batch_stats['mean_clusters']:.2f} / {self.n_generations}\n")
                f.write(f"  Max clusters:        {batch_stats['max_clusters']} / {self.n_generations}\n")
                f.write("\n")
        
        print(f"\nReport saved to: {report_path}")
        
        # Also print to console
        with open(report_path, 'r') as f:
            print(f.read())


def main():
    parser = argparse.ArgumentParser(description='Run uncertainty quantification experiment')
    parser.add_argument('--batch-files', nargs='+', help='JSON files containing prompt batches')
    parser.add_argument('--output-dir', default='./uncertainty_experiment', help='Output directory')
    parser.add_argument('--model', default='gpt2-xl', 
                       choices=['gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl'],
                       help='GPT-2 model size')
    parser.add_argument('--n-generations', type=int, default=5,
                       help='Number of generations per prompt')
    parser.add_argument('--clustering-threshold', type=float, default=0.4,
                       help='Clustering threshold (0.4 = permissive, allows branching)')
    
    args = parser.parse_args()
    
    if not args.batch_files:
        print("Error: No batch files specified. Use --batch-files")
        print("\nExample:")
        print("  python uncertainty_pipeline.py \\")
        print("      --batch-files factual_questions.json subjective_questions.json \\")
        print("      --n-generations 5 \\")
        print("      --clustering-threshold 0.4")
        return
    
    # Initialize pipeline
    pipeline = ExperimentPipeline(
        output_dir=args.output_dir, 
        model_name=args.model,
        n_generations=args.n_generations,
        clustering_threshold=args.clustering_threshold
    )
    
    # Run experiment
    results = pipeline.run_experiment(args.batch_files)
    
    # Generate visualizations and reports
    pipeline.plot_distributions(results)
    pipeline.generate_report(results)
    
    print(f"\n✅ All results saved to: {pipeline.output_dir}")


if __name__ == "__main__":
    main()