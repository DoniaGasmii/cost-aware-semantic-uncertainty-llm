#!/usr/bin/env python3
"""
Experiment Pipeline with GPT-2-XL
"""

import json
import os
from pathlib import Path
from typing import List, Dict, Any, Optional
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


@dataclass
class PromptBatch:
    """Container for a batch of prompts"""
    name: str
    prompts: List[Dict[str, Any]]
    metadata: Optional[Dict[str, Any]] = None


@dataclass
class ScoredResponse:
    """Container for a scored response"""
    batch_name: str
    prompt_id: str
    prompt: str
    response: str
    score: float
    metadata: Optional[Dict[str, Any]] = None


class GPT2Generator:
    """Wrapper for GPT-2 model"""
    
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
    
    def generate(self, prompt: str, max_length: int = 150, temperature: float = 0.7) -> str:
        """Generate text from GPT-2"""
        # Encode prompt
        inputs = self.tokenizer.encode(prompt, return_tensors="pt").to(self.device)
        
        # Generate
        with torch.no_grad():
            outputs = self.model.generate(
                inputs,
                max_length=max_length,
                temperature=temperature,
                do_sample=True,
                top_p=0.9,
                pad_token_id=self.tokenizer.eos_token_id
            )
        
        # Decode
        generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Extract only the generated part (remove prompt)
        response = generated_text[len(prompt):].strip()
        
        return response


class ExperimentPipeline:
    """Main pipeline for running experiments"""
    
    def __init__(self, output_dir: str = "./experiment_results", model_name: str = "gpt2-xl"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.results_file = self.output_dir / "results.jsonl"
        
        # Initialize GPT-2
        if TRANSFORMERS_AVAILABLE:
            self.generator = GPT2Generator(model_name=model_name)
        else:
            self.generator = None
            print("⚠️  Running in dummy mode (no model loaded)")
        
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
    
    def generate_response(self, prompt: str) -> str:
        """Generate a response using GPT-2"""
        if self.generator:
            return self.generator.generate(prompt)
        else:
            # Fallback to dummy response
            return f"Dummy response to: {prompt[:50]}..."
    
    def score_response(self, prompt: str, response: str, metadata: Optional[Dict] = None) -> float:
        """
        Score a response based on your criteria.
        
        This is a PLACEHOLDER - replace with your actual scoring logic:
        - LLM-as-judge
        - Rule-based evaluation
        - Human evaluation
        - Automated metrics
        """
        difficulty = metadata.get('difficulty', 'medium') if metadata else 'medium'
        
        # Dummy scoring based on difficulty
        # TODO: Replace this with actual evaluation logic
        if difficulty == 'easy':
            mean, std = 0.8, 0.1
        elif difficulty == 'medium':
            mean, std = 0.6, 0.15
        else:  # hard
            mean, std = 0.4, 0.15
        
        return np.clip(np.random.normal(mean, std), 0, 1)
    
    def process_batch(self, batch: PromptBatch) -> List[ScoredResponse]:
        """Process all prompts in a batch"""
        scored_responses = []
        
        print(f"\nProcessing batch: {batch.name}")
        for i, prompt_data in enumerate(batch.prompts):
            prompt_text = prompt_data.get('prompt', prompt_data.get('text', ''))
            prompt_id = prompt_data.get('id', f"{batch.name}_{i}")
            
            print(f"  [{i+1}/{len(batch.prompts)}] Processing prompt: {prompt_id}")
            
            # Generate response
            response = self.generate_response(prompt_text)
            
            # Score response
            score = self.score_response(
                prompt_text, 
                response, 
                metadata=prompt_data.get('metadata')
            )
            
            # Store result
            scored = ScoredResponse(
                batch_name=batch.name,
                prompt_id=prompt_id,
                prompt=prompt_text,
                response=response,
                score=score,
                metadata=prompt_data.get('metadata')
            )
            scored_responses.append(scored)
            
            # Save incrementally
            self._save_result(scored)
        
        return scored_responses
    
    def _save_result(self, result: ScoredResponse):
        """Save a single result to JSONL file"""
        with open(self.results_file, 'a') as f:
            json.dump(asdict(result), f)
            f.write('\n')
    
    def run_experiment(self, batch_files: List[str]) -> Dict[str, List[ScoredResponse]]:
        """Run the complete experiment pipeline"""
        print("=" * 80)
        print("Starting Experiment Pipeline")
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
    
    def compute_statistics(self, results: Dict[str, List[ScoredResponse]]) -> Dict[str, Dict]:
        """Compute statistics for each batch"""
        stats = {}
        
        for batch_name, responses in results.items():
            scores = [r.score for r in responses]
            
            stats[batch_name] = {
                'count': len(scores),
                'mean': np.mean(scores),
                'std': np.std(scores),
                'median': np.median(scores),
                'min': np.min(scores),
                'max': np.max(scores),
                'q25': np.percentile(scores, 25),
                'q75': np.percentile(scores, 75),
            }
        
        return stats
    
    def plot_distributions(self, results: Dict[str, List[ScoredResponse]], 
                          save_path: Optional[str] = None):
        """Plot score distributions for each batch"""
        fig, ax = plt.subplots(figsize=(12, 6))
        
        colors = plt.cm.Set2(np.linspace(0, 1, len(results)))
        
        for idx, (batch_name, responses) in enumerate(results.items()):
            scores = [r.score for r in responses]
            
            # Compute statistics
            mean = np.mean(scores)
            std = np.std(scores)
            
            # Create smooth curve
            x = np.linspace(0, 1, 1000)
            y = (1 / (std * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x - mean) / std) ** 2)
            
            # Plot distribution curve
            ax.plot(x, y, linewidth=2.5, label=batch_name, color=colors[idx], alpha=0.8)
            
            # Fill under curve
            ax.fill_between(x, y, alpha=0.2, color=colors[idx])
            
            # Add mean line
            ax.axvline(mean, color=colors[idx], linestyle='--', linewidth=1.5, alpha=0.7)
            
            # Add annotation
            ax.text(mean, ax.get_ylim()[1] * 0.9, 
                   f'μ = {mean:.2f}, σ = {std:.2f}',
                   ha='center', va='top', fontsize=10,
                   bbox=dict(boxstyle='round', facecolor=colors[idx], alpha=0.3))
        
        ax.set_xlabel('Score', fontsize=12)
        ax.set_ylabel('Density', fontsize=12)
        ax.set_title('Score Distributions Across Batches', fontsize=14, fontweight='bold')
        ax.legend(loc='upper right', fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.set_xlim(0, 1)
        
        plt.tight_layout()
        
        if save_path is None:
            save_path = self.output_dir / 'distribution_plot.png'
        
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"\nPlot saved to: {save_path}")
        plt.close()
    
    def generate_report(self, results: Dict[str, List[ScoredResponse]]):
        """Generate a text report with statistics"""
        stats = self.compute_statistics(results)
        
        report_path = self.output_dir / 'report.txt'
        
        with open(report_path, 'w') as f:
            f.write("=" * 80 + "\n")
            f.write("EXPERIMENT REPORT\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("=" * 80 + "\n\n")
            
            for batch_name, batch_stats in stats.items():
                f.write(f"Batch: {batch_name}\n")
                f.write("-" * 40 + "\n")
                f.write(f"  Sample size: {batch_stats['count']}\n")
                f.write(f"  Mean score:  {batch_stats['mean']:.4f}\n")
                f.write(f"  Std dev:     {batch_stats['std']:.4f}\n")
                f.write(f"  Median:      {batch_stats['median']:.4f}\n")
                f.write(f"  Range:       [{batch_stats['min']:.4f}, {batch_stats['max']:.4f}]\n")
                f.write(f"  IQR:         [{batch_stats['q25']:.4f}, {batch_stats['q75']:.4f}]\n")
                f.write("\n")
        
        print(f"\nReport saved to: {report_path}")
        
        # Also print to console
        with open(report_path, 'r') as f:
            print(f.read())


def main():
    parser = argparse.ArgumentParser(description='Run experiment pipeline with GPT-2')
    parser.add_argument('--batch-files', nargs='+', help='JSON files containing prompt batches')
    parser.add_argument('--output-dir', default='./gpt2_experiment', help='Output directory')
    parser.add_argument('--model', default='gpt2-xl', 
                       choices=['gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl'],
                       help='GPT-2 model size')
    
    args = parser.parse_args()
    
    if not args.batch_files:
        print("Error: No batch files specified. Use --batch-files")
        print("\nExample:")
        print("  python gpt2_pipeline.py --batch-files factual_questions.json subjective_questions.json")
        return
    
    # Initialize pipeline
    pipeline = ExperimentPipeline(output_dir=args.output_dir, model_name=args.model)
    
    # Run experiment
    results = pipeline.run_experiment(args.batch_files)
    
    # Generate visualizations and reports
    pipeline.plot_distributions(results)
    pipeline.generate_report(results)
    
    print(f"\n✅ All results saved to: {pipeline.output_dir}")


if __name__ == "__main__":
    main()