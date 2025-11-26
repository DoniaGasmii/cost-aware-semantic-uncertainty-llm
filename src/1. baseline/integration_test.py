# test_integration.py
from baseline_exact_replication import ExactPaperReplication

def test_exact_replication():
    """Test that we can replicate paper's method"""
    
    estimator = ExactPaperReplication(
        model_name="gpt2",
        entailment_model="deberta"
    )
    
    question = "What is the capital of France?"
    prompt = f"Question: {question}\nAnswer:"
    
    result = estimator.estimate_uncertainty(
        prompt=prompt,
        question=question,
        n_samples=10,
        temperature=1.0
    )
    
    print(f"✓ Semantic entropy: {result.semantic_entropy:.3f}")
    print(f"✓ Clusters: {result.n_clusters}")
    print(f"✓ Semantic IDs: {result.semantic_ids}")
    
    # Expected: Low entropy (most responses say "Paris")
    assert result.semantic_entropy < 1.0, "Expected low entropy for factual question"
    assert result.n_clusters <= 3, "Expected few clusters"
    
    print("\n✅ Integration test passed!")

if __name__ == "__main__":
    test_exact_replication()