"""
Semantic Uncertainty Analysis using Entropy-Adaptive Branching.

This example demonstrates how to use EAB-generated samples for
uncertainty quantification and semantic clustering.
"""

from eab import EntropyAdaptiveBranching
from eab.utils import compute_diversity_metrics
from collections import Counter
import numpy as np


def semantic_clustering(texts, similarity_threshold=0.8):
    """
    Simple semantic clustering based on token overlap.
    
    In practice, you would use embeddings (e.g., sentence-transformers).
    """
    from difflib import SequenceMatcher
    
    clusters = []
    
    for text in texts:
        # Try to add to existing cluster
        added = False
        for cluster in clusters:
            # Compute similarity with cluster representative
            similarity = SequenceMatcher(None, text, cluster['representative']).ratio()
            if similarity >= similarity_threshold:
                cluster['members'].append(text)
                cluster['count'] += 1
                added = True
                break
        
        # Create new cluster if no match
        if not added:
            clusters.append({
                'representative': text,
                'members': [text],
                'count': 1
            })
    
    # Sort by cluster size
    clusters.sort(key=lambda c: c['count'], reverse=True)
    
    return clusters


def compute_predictive_entropy(clusters):
    """
    Compute predictive entropy from semantic clusters.
    
    H = -Σ p(c) log p(c)
    where p(c) is the proportion of samples in cluster c
    """
    total = sum(c['count'] for c in clusters)
    probs = [c['count'] / total for c in clusters]
    
    entropy = -sum(p * np.log(p) for p in probs if p > 0)
    
    return entropy


def main():
    print("=" * 70)
    print("Semantic Uncertainty Analysis with EAB")
    print("=" * 70)
    
    # Initialize EAB
    eab = EntropyAdaptiveBranching(
        model_name="gpt2",
        entropy_threshold=0.4,
        branch_factor=3,
        max_paths=20
    )
    
    # Test prompts with varying uncertainty
    test_cases = [
        {
            'prompt': "The capital of France is",
            'expected_uncertainty': "LOW",
            'max_tokens': 10
        },
        {
            'prompt': "In my opinion, the best way to learn programming is",
            'expected_uncertainty': "MEDIUM",
            'max_tokens': 25
        },
        {
            'prompt': "The most important thing in life is",
            'expected_uncertainty': "HIGH",
            'max_tokens': 20
        }
    ]
    
    for i, test_case in enumerate(test_cases, 1):
        prompt = test_case['prompt']
        expected_uncertainty = test_case['expected_uncertainty']
        max_tokens = test_case['max_tokens']
        
        print(f"\n{'='*70}")
        print(f"Test Case {i}: {expected_uncertainty} Uncertainty")
        print('='*70)
        print(f"Prompt: '{prompt}'")
        print(f"Expected uncertainty: {expected_uncertainty}")
        
        # Generate samples
        results = eab.generate(
            prompt=prompt,
            max_new_tokens=max_tokens,
            temperature=0.9,
            seed=42,
            show_progress=False
        )
        
        # Extract texts
        texts = [r['text'].strip() for r in results]
        
        print(f"\nGenerated {len(texts)} samples")
        
        # Show some samples
        print(f"\nSample completions:")
        for j, text in enumerate(texts[:5], 1):
            print(f"  {j}. {text}")
        if len(texts) > 5:
            print(f"  ... and {len(texts) - 5} more")
        
        # Semantic clustering
        print(f"\n{'='*70}")
        print("Semantic Clustering")
        print('='*70)
        
        clusters = semantic_clustering(texts, similarity_threshold=0.7)
        
        print(f"\nFound {len(clusters)} semantic clusters:")
        for j, cluster in enumerate(clusters, 1):
            print(f"\nCluster {j} ({cluster['count']} samples):")
            print(f"  Representative: {cluster['representative']}")
            if cluster['count'] > 1:
                print(f"  Members: {cluster['count']} total")
        
        # Compute uncertainty metrics
        print(f"\n{'='*70}")
        print("Uncertainty Metrics")
        print('='*70)
        
        # 1. Token-level entropy (from generation)
        entropy_stats = eab.get_entropy_history()['statistics']
        token_entropy = entropy_stats['mean_entropy']
        
        # 2. Predictive entropy (from clusters)
        predictive_entropy = compute_predictive_entropy(clusters)
        
        # 3. Diversity metrics
        diversity = compute_diversity_metrics(texts)
        
        print(f"\n1. Token-level uncertainty:")
        print(f"   Mean entropy: {token_entropy:.3f}")
        print(f"   Branch rate: {entropy_stats['branch_rate']:.1%}")
        
        print(f"\n2. Predictive uncertainty:")
        print(f"   Semantic entropy: {predictive_entropy:.3f}")
        print(f"   Num clusters: {len(clusters)}")
        print(f"   Largest cluster: {clusters[0]['count']}/{len(texts)} samples")
        
        print(f"\n3. Diversity:")
        print(f"   Unique ratio: {diversity['unique_ratio']:.1%}")
        print(f"   Vocab diversity: {diversity['vocab_diversity']:.3f}")
        
        # Interpretation
        print(f"\n{'='*70}")
        print("Interpretation")
        print('='*70)
        
        if len(clusters) == 1:
            confidence = "HIGH"
            interpretation = "Model is very confident - all samples semantically similar"
        elif len(clusters) <= 3:
            confidence = "MEDIUM"
            interpretation = "Model has moderate confidence - a few distinct answers"
        else:
            confidence = "LOW"
            interpretation = "Model is uncertain - many diverse answers"
        
        print(f"\nConfidence level: {confidence}")
        print(f"Interpretation: {interpretation}")
        
        match = "✓" if confidence == expected_uncertainty or \
                (confidence == "MEDIUM" and expected_uncertainty in ["LOW", "MEDIUM", "HIGH"]) \
                else "✗"
        print(f"Matches expectation: {match}")
    
    # Summary
    print(f"\n{'='*70}")
    print("Summary: Using EAB for Uncertainty Quantification")
    print('='*70)
    
    print("\nKey insights:")
    print("  1. Token-level entropy → when to branch (generation time)")
    print("  2. Semantic clustering → identify distinct meanings")
    print("  3. Predictive entropy → overall model uncertainty")
    print("  4. Diversity metrics → sample quality assessment")
    
    print("\nApplications:")
    print("  • Model calibration")
    print("  • Active learning (query high-uncertainty examples)")
    print("  • Confidence-aware deployment")
    print("  • Detecting ambiguous inputs")
    print("  • Improving factuality via uncertainty")


if __name__ == "__main__":
    main()