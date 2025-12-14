"""Example: Integration with Entropy-Adaptive Branching (EAB)."""

from semantic_entropy import SemanticUncertaintyEstimator

# NOTE: This example assumes you have the EAB package installed
# from the entropy-adaptive-branching directory
# Uncomment the following line when EAB is available:
# from eab import EntropyAdaptiveBranching


def main():
    """
    Demonstrate full pipeline:
    1. Generate diverse samples with EAB (efficient)
    2. Compute semantic uncertainty
    3. Use uncertainty for decision-making
    """
    
    # Initialize semantic uncertainty estimator
    estimator = SemanticUncertaintyEstimator(
        encoder_model="all-mpnet-base-v2",
        distance_threshold=0.15,
        use_weighted_probs=True  # Use EAB log-probs for weighting
    )
    
    print("=" * 60)
    print("Entropy-Adaptive Branching + Semantic Uncertainty")
    print("=" * 60)
    print()
    
    # ============================================================
    # PLACEHOLDER: When EAB is available, use this code:
    # ============================================================
    # eab = EntropyAdaptiveBranching(
    #     model_name="gpt2",
    #     entropy_threshold=0.4,
    #     branch_factor=3,
    #     max_paths=20
    # )
    # 
    # prompt = "Question: What is the capital of France? Answer:"
    # results = eab.generate(
    #     prompt=prompt,
    #     max_new_tokens=50,
    #     temperature=0.8
    # )
    # 
    # # Extract texts and log-probabilities
    # texts = [r['text'] for r in results]
    # log_probs = [r['log_prob'] for r in results]
    # ============================================================
    
    # For now, simulate EAB output
    print("Note: Using simulated EAB output for demonstration")
    print("Install EAB package to use real entropy-adaptive generation")
    print()
    
    # Simulated generations with log-probabilities
    texts = [
        "The capital of France is Paris.",
        "Paris is the capital of France.",
        "It's Paris.",
        "The capital is Lyon.",  # Wrong answer (low probability)
    ]
    
    # Simulated log-probabilities (lower = less likely)
    log_probs = [-2.5, -2.3, -2.7, -8.5]  # Lyon answer has very low prob
    
    print(f"Generated {len(texts)} samples via EAB")
    print()
    
    # Compute semantic uncertainty
    result = estimator.compute(
        texts=texts,
        log_probs=log_probs,
        return_details=True
    )
    
    print(f"Semantic Entropy: {result['entropy']:.4f}")
    print(f"Normalized Entropy: {result['normalized_entropy']:.4f}")
    print(f"Number of Clusters: {result['n_clusters']}")
    print(f"Interpretation: {estimator.interpret_uncertainty(result['normalized_entropy'])}")
    print()
    
    # Show cluster representatives with probabilities
    print("Cluster Analysis:")
    representatives = estimator.get_cluster_representatives(texts, log_probs)
    
    for cluster_id, info in sorted(representatives.items(), 
                                   key=lambda x: x[1]['probability'], 
                                   reverse=True):
        print(f"\nCluster {cluster_id} (probability: {info['probability']:.2%}):")
        print(f"  Representative: '{info['text']}'")
        print(f"  Size: {info['size']} samples")
    
    print()
    print("=" * 60)
    
    # Decision-making based on uncertainty
    threshold = 0.5
    if result['normalized_entropy'] < threshold:
        print("✓ CONFIDENT: Model is certain about this answer")
        print("  → Use the most probable cluster's answer")
    else:
        print("⚠ UNCERTAIN: Model has mixed interpretations")
        print("  → May want to ask for clarification or use ensemble")
    
    print("=" * 60)


if __name__ == "__main__":
    main()