"""Basic usage example for semantic uncertainty estimation."""

from semantic_entropy import SemanticUncertaintyEstimator


def main():
    # Initialize the estimator
    estimator = SemanticUncertaintyEstimator(
        encoder_model="all-mpnet-base-v2",
        distance_threshold=0.15  # Cosine similarity threshold of 0.85
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
    
    result = estimator.compute(generations_low)
    
    print(f"Generations: {len(generations_low)}")
    print(f"Semantic Entropy: {result['entropy']:.4f}")
    print(f"Normalized Entropy: {result['normalized_entropy']:.4f}")
    print(f"Number of Clusters: {result['n_clusters']}")
    print(f"Cluster Distribution: {result['cluster_probs']}")
    print(f"Interpretation: {estimator.interpret_uncertainty(result['normalized_entropy'])}")
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
    
    result = estimator.compute(generations_high)
    
    print(f"Generations: {len(generations_high)}")
    print(f"Semantic Entropy: {result['entropy']:.4f}")
    print(f"Normalized Entropy: {result['normalized_entropy']:.4f}")
    print(f"Number of Clusters: {result['n_clusters']}")
    print(f"Cluster Distribution: {result['cluster_probs']}")
    print(f"Interpretation: {estimator.interpret_uncertainty(result['normalized_entropy'])}")
    print()
    
    # Example 3: Get cluster representatives
    print("=" * 60)
    print("Example 3: Cluster Representatives")
    print("=" * 60)
    
    representatives = estimator.get_cluster_representatives(generations_high)
    
    for cluster_id, info in representatives.items():
        print(f"\nCluster {cluster_id}:")
        print(f"  Representative: {info['text']}")
        print(f"  Size: {info['size']} samples")
        print(f"  Probability: {info['probability']:.3f}")
    print()


if __name__ == "__main__":
    main()