"""Example: Semantic uncertainty for factual question answering."""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from semantic_entropy import SemanticUncertaintyEstimator
import argparse


def analyze_qa_pair(question: str, generations: list, estimator):
    """Analyze uncertainty for a Q&A pair."""
    print(f"\nQuestion: {question}")
    print("-" * 60)
    
    result = estimator.compute(generations, return_details=True)
    
    print(f"Number of generations: {len(generations)}")
    print(f"Semantic clusters found: {result['n_clusters']}")
    print(f"Normalized uncertainty: {result['normalized_entropy']:.4f}")
    print(f"Uncertainty score: {result['uncertainty_score']:.4f}")
    print(f"Interpretation: {estimator.interpret_uncertainty(result['uncertainty_score'])}")
    
    # Show all generations per cluster
    print("\nClusters (grouped by meaning):")
    
    # Group generations by cluster
    cluster_labels = result['cluster_labels']
    cluster_probs = result['cluster_probs']
    
    clusters = {}
    for i, label in enumerate(cluster_labels):
        if label not in clusters:
            clusters[label] = []
        clusters[label].append(generations[i])
    
    # Sort clusters by probability (most common first)
    sorted_clusters = sorted(clusters.items(), 
                            key=lambda x: cluster_probs[x[0]], 
                            reverse=True)
    
    for cluster_id, texts in sorted_clusters:
        prob = cluster_probs[cluster_id]
        print(f"\n  Cluster {cluster_id} [{prob:.1%}] - {len(texts)} generation(s):")
        for text in texts:
            print(f"    • {text}")
    
    print()


def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(
        description='Analyze semantic uncertainty in factual QA',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python factual_qa.py                    # Use default threshold (0.15)
  python factual_qa.py --threshold 0.25   # Use threshold 0.25 (more lenient)
  python factual_qa.py -t 0.30            # Short form
        """
    )
    parser.add_argument(
        '--threshold', '-t',
        type=float,
        default=0.15,
        help='Distance threshold for clustering (default: 0.15). '
             'Lower = stricter (more clusters), Higher = more lenient (fewer clusters). '
             'Recommended range: 0.10-0.35'
    )
    
    args = parser.parse_args()
    
    # Validate threshold
    if args.threshold < 0.0 or args.threshold > 1.0:
        print("Error: Threshold must be between 0.0 and 1.0")
        sys.exit(1)
    
    print("=" * 60)
    print(f"SEMANTIC UNCERTAINTY ANALYSIS")
    print("=" * 60)
    print(f"Distance Threshold: {args.threshold}")
    print(f"Cosine Similarity Threshold: {1 - args.threshold:.2f}")
    print("=" * 60)
    
    estimator = SemanticUncertaintyEstimator(
        encoder_model="all-mpnet-base-v2",
        distance_threshold=args.threshold
    )
    
    # Test 1: Unambiguous factual question
    analyze_qa_pair(
        question="What is the capital of France?",
        generations=[
            "The capital of France is Paris.",
            "Paris",
            "It's Paris.",
            "Paris is the capital.",
            "The answer is Paris.",
        ],
        estimator=estimator
    )
    
    # Test 2: Question with some disagreement
    analyze_qa_pair(
        question="Who invented the telephone?",
        generations=[
            "Alexander Graham Bell invented the telephone.",
            "Alexander Graham Bell",
            "Bell invented it.",
            "Antonio Meucci invented the telephone first.",
            "Meucci had the original design.",
            "It was Alexander Graham Bell.",
        ],
        estimator=estimator
    )
    
    # Test 3: Highly ambiguous question
    analyze_qa_pair(
        question="What is the meaning of life?",
        generations=[
            "The meaning of life is to find happiness.",
            "Life has no inherent meaning.",
            "The meaning is whatever you make it.",
            "To love and be loved.",
            "42, according to Douglas Adams.",
            "To serve God and others.",
            "There's no single answer to this question.",
        ],
        estimator=estimator
    )
    
    # Test 4: Factually incorrect but semantically consistent
    analyze_qa_pair(
        question="What is the largest planet in our solar system?",
        generations=[
            "Jupiter is the largest planet.",
            "The largest planet is Jupiter.",
            "Jupiter",
            "It's Jupiter.",
            "Jupiter is the biggest.",
        ],
        estimator=estimator
    )


if __name__ == "__main__":
    main()