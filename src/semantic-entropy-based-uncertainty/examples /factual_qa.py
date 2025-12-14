"""Example: Semantic uncertainty for factual question answering."""

from semantic_entropy import SemanticUncertaintyEstimator


def analyze_qa_pair(question: str, generations: list, estimator):
    """Analyze uncertainty for a Q&A pair."""
    print(f"\nQuestion: {question}")
    print("-" * 60)
    
    result = estimator.compute(generations, return_details=True)
    
    print(f"Number of generations: {len(generations)}")
    print(f"Semantic clusters found: {result['n_clusters']}")
    print(f"Normalized uncertainty: {result['normalized_entropy']:.4f}")
    print(f"Interpretation: {estimator.interpret_uncertainty(result['normalized_entropy'])}")
    
    # Show cluster representatives
    print("\nCluster Representatives:")
    for cluster_id, info in result['cluster_analysis']['representatives'].items():
        prob = result['cluster_probs'][cluster_id]
        print(f"  [{prob:.2%}] {info['text']}")
    
    print()


def main():
    estimator = SemanticUncertaintyEstimator(
        encoder_model="all-mpnet-base-v2",
        distance_threshold=0.15
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