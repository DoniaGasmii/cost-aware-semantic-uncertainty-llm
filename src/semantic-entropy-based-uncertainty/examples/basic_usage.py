"""Basic usage example for semantic uncertainty estimation."""

import sys
from pathlib import Path
import argparse

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from semantic_entropy import SemanticUncertaintyEstimator


def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(
        description='Basic semantic uncertainty examples',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python basic_usage.py                    # Use default threshold (0.15)
  python basic_usage.py --threshold 0.25   # Use threshold 0.25 (more lenient)
  python basic_usage.py -t 0.30            # Short form
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
    print(f"Distance Threshold: {args.threshold} (Similarity: {1 - args.threshold:.2f})")
    print("=" * 60)
    print()
    
    # Initialize the estimator
    estimator = SemanticUncertaintyEstimator(
        encoder_model="all-mpnet-base-v2",
        distance_threshold=args.threshold
    )
    
    # Example 1: Low uncertainty (consistent answers)
    print("=" * 60)
    print("Example 1: Low Uncertainty (Factual Question)")
    print("=" * 60)
    
    generations_low = [
        "The capital of France is Paris.",
        "Paris is the capital of France.",
        "It's Paris.",
        "The answer is Paris.",
    ]
    
    result = estimator.compute(generations_low, return_details=True)
    
    print(f"Generations: {len(generations_low)}")
    print(f"Semantic Entropy: {result['entropy']:.4f}")
    print(f"Normalized Entropy: {result['normalized_entropy']:.4f}")
    print(f"Number of Clusters: {result['n_clusters']}")
    print(f"Cluster Distribution: {result['cluster_probs']}")
    print(f"Interpretation: {estimator.interpret_uncertainty(result['normalized_entropy'])}")
    
    # Show all generations per cluster
    print("\nClusters (grouped by meaning):")
    cluster_labels = result['cluster_labels']
    cluster_probs = result['cluster_probs']
    
    clusters = {}
    for i, label in enumerate(cluster_labels):
        if label not in clusters:
            clusters[label] = []
        clusters[label].append(generations_low[i])
    
    # Sort by probability
    sorted_clusters = sorted(clusters.items(), 
                            key=lambda x: cluster_probs[x[0]], 
                            reverse=True)
    
    for cluster_id, texts in sorted_clusters:
        prob = cluster_probs[cluster_id]
        print(f"\n  Cluster {cluster_id} [{prob:.1%}] - {len(texts)} generation(s):")
        for text in texts:
            print(f"    • {text}")
    print()
    
    # Example 2: High uncertainty (ambiguous question)
    print("=" * 60)
    print("Example 2: High Uncertainty (Ambiguous Question)")
    print("=" * 60)
    
    generations_high = [
        "Python is the best programming language.",
        "JavaScript is the best for web development.",
        "Java is the most reliable choice.",
        "C++ offers the best performance.",
        "It depends on your use case.",
        "There's no single best language.",
    ]
    
    
    result = estimator.compute(generations_high, return_details=True)
    
    print(f"Generations: {len(generations_high)}")
    print(f"Semantic Entropy: {result['entropy']:.4f}")
    print(f"Normalized Entropy: {result['normalized_entropy']:.4f}")
    print(f"Number of Clusters: {result['n_clusters']}")
    print(f"Cluster Distribution: {result['cluster_probs']}")
    print(f"Interpretation: {estimator.interpret_uncertainty(result['normalized_entropy'])}")
    
    # Show all generations per cluster
    print("\nClusters (grouped by meaning):")
    cluster_labels = result['cluster_labels']
    cluster_probs = result['cluster_probs']
    
    clusters = {}
    for i, label in enumerate(cluster_labels):
        if label not in clusters:
            clusters[label] = []
        clusters[label].append(generations_high[i])
    
    # Sort by probability
    sorted_clusters = sorted(clusters.items(), 
                            key=lambda x: cluster_probs[x[0]], 
                            reverse=True)
    
    for cluster_id, texts in sorted_clusters:
        prob = cluster_probs[cluster_id]
        print(f"\n  Cluster {cluster_id} [{prob:.1%}] - {len(texts)} generation(s):")
        for text in texts:
            print(f"    • {text}")
    print()


if __name__ == "__main__":
    main()