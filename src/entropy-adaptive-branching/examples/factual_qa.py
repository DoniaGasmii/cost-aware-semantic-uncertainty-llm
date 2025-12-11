"""
Factual Question Answering with Entropy-Adaptive Branching.

This example demonstrates using EAB for factual QA, where we expect
low branching rates due to high model confidence.
"""

from eab import EntropyAdaptiveBranching
from eab.utils import compute_diversity_metrics, save_results
import json


def main():
    print("=" * 70)
    print("Factual Question Answering with Entropy-Adaptive Branching")
    print("=" * 70)
    
    # Initialize EAB with conservative settings for factual QA
    eab = EntropyAdaptiveBranching(
        model_name="gpt2",
        entropy_threshold=0.6,  # Higher threshold = less branching
        branch_factor=2,  # Fewer branches when they do occur
        max_paths=10
    )
    
    # Factual questions
    questions = [
        "The capital of France is",
        "Water freezes at",
        "The largest planet in our solar system is",
        "The speed of light is approximately",
        "The currency of Japan is the",
        "The Great Wall of China was built in",
        "The human body has",
        "The Eiffel Tower is located in",
    ]
    
    all_results = []
    
    for i, question in enumerate(questions, 1):
        print(f"\n{'='*70}")
        print(f"Question {i}/{len(questions)}: {question}")
        print('='*70)
        
        # Generate answers
        results = eab.generate(
            prompt=question,
            max_new_tokens=15,
            temperature=0.7,
            seed=42 + i,
            show_progress=False
        )
        
        # Display top 3 answers
        print(f"\nTop 3 answers:")
        for j, result in enumerate(results[:3], 1):
            answer = result['text'].strip()
            prob = result['probability']
            print(f"  {j}. {answer} (p={prob:.4f})")
        
        # Branching stats
        stats = eab.get_entropy_history()['statistics']
        print(f"\nBranching statistics:")
        print(f"  Total paths: {len(results)}")
        print(f"  Branch rate: {stats['branch_rate']:.1%}")
        print(f"  Avg entropy: {stats['mean_entropy']:.3f}")
        
        # Compute diversity
        texts = [r['text'] for r in results]
        diversity = compute_diversity_metrics(texts)
        print(f"\nDiversity metrics:")
        print(f"  Unique answers: {diversity['num_unique']}/{len(texts)}")
        print(f"  Unique ratio: {diversity['unique_ratio']:.1%}")
        
        # Store results
        all_results.append({
            'question': question,
            'answers': results[:5],  # Top 5
            'stats': stats,
            'diversity': diversity
        })
    
    # Summary
    print(f"\n{'='*70}")
    print("Summary Across All Questions")
    print('='*70)
    
    avg_branch_rate = sum(r['stats']['branch_rate'] for r in all_results) / len(all_results)
    avg_entropy = sum(r['stats']['mean_entropy'] for r in all_results) / len(all_results)
    avg_unique_ratio = sum(r['diversity']['unique_ratio'] for r in all_results) / len(all_results)
    
    print(f"\nAverage branch rate: {avg_branch_rate:.1%}")
    print(f"Average entropy: {avg_entropy:.3f}")
    print(f"Average unique ratio: {avg_unique_ratio:.1%}")
    
    print("\nObservations for factual QA:")
    print("  ✓ Low branch rate (model is confident)")
    print("  ✓ Low entropy (concentrated probability mass)")
    print("  ✓ High unique ratio may indicate some uncertainty")
    print("  ✓ Efficient: most computation shared across samples")
    
    # Save results
    output_file = "factual_qa_results.json"
    save_results(all_results, output_file)
    print(f"\nResults saved to: {output_file}")


if __name__ == "__main__":
    main()