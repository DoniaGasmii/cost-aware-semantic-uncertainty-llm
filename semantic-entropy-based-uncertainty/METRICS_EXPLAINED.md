# Understanding Uncertainty Metrics

## The Problem with Normalized Entropy

When you first ran the examples, you noticed something strange:

```
Example 1: 2 clusters → Normalized Entropy = 1.0
Example 2: 6 clusters → Normalized Entropy = 1.0
```

Both show maximum uncertainty even though Example 2 clearly has more semantic diversity!

## Why This Happens

**Normalized Entropy** is defined as:
```
normalized_entropy = entropy / log(n_clusters)
```

This normalizes entropy to [0, 1] **relative to the number of clusters**.

### The Issue

For **uniform distributions** (all clusters equally likely):
- 2 clusters: [0.5, 0.5] → H = log(2) → normalized = log(2)/log(2) = **1.0**
- 6 clusters: [0.167, 0.167, ...] → H = log(6) → normalized = log(6)/log(6) = **1.0**

Both get **1.0** because they're both maximally uncertain **given their cluster count**.

But intuitively, 6 different meanings is more uncertain than 2 different meanings!

## Our Solution: Uncertainty Score

We introduced a **combined uncertainty score** that considers:

1. **Distribution uniformity** (normalized entropy)
2. **Number of distinct meanings** (cluster diversity)

### Formula

```python
uncertainty_score = 0.6 × normalized_entropy + 0.4 × cluster_diversity

where:
  normalized_entropy = entropy / log(n_clusters)
  cluster_diversity = (n_clusters - 1) / (n_samples - 1)
```

### Why This Works

**Component 1: Normalized Entropy (60% weight)**
- Measures how uniform the cluster distribution is
- [0.25, 0.25, 0.25, 0.25] → High (uniform = uncertain)
- [0.7, 0.2, 0.1] → Lower (skewed = one dominant meaning)

**Component 2: Cluster Diversity (40% weight)**
- Measures how many distinct clusters exist relative to sample size
- 1 cluster out of 4 samples → 0.0 (no diversity)
- 4 clusters out of 4 samples → 1.0 (maximum diversity)
- 3 clusters out of 6 samples → 0.4 (moderate diversity)

### Example Comparison

**Scenario 1: Factual question (4 samples)**
```
Threshold 0.3: 2 clusters [0.5, 0.5]
  - Normalized Entropy: 1.0 (maximally uniform given 2 clusters)
  - Cluster Diversity: (2-1)/(4-1) = 0.33
  - Uncertainty Score: 0.6×1.0 + 0.4×0.33 = 0.73

Threshold 0.4: 1 cluster [1.0]
  - Normalized Entropy: 0.0 (single cluster)
  - Cluster Diversity: 0.0
  - Uncertainty Score: 0.0 ✓ LOW UNCERTAINTY
```

**Scenario 2: Ambiguous question (6 samples)**
```
Threshold 0.3: 6 clusters [0.167, 0.167, 0.167, 0.167, 0.167, 0.167]
  - Normalized Entropy: 1.0 (maximally uniform)
  - Cluster Diversity: (6-1)/(6-1) = 1.0 (all different!)
  - Uncertainty Score: 0.6×1.0 + 0.4×1.0 = 1.0 ✓ HIGH UNCERTAINTY

Threshold 0.4: 5 clusters
  - Normalized Entropy: 0.97
  - Cluster Diversity: (5-1)/(6-1) = 0.80
  - Uncertainty Score: 0.6×0.97 + 0.4×0.80 = 0.90 ✓ STILL HIGH
```

## When to Use Which Metric

### Raw Entropy
**Use for**: Direct comparison when cluster count is the same

**Example**: Comparing two models on the same dataset
```
Model A: 3 clusters, entropy = 0.8
Model B: 3 clusters, entropy = 1.1
→ Model B is more uncertain (higher entropy)
```

### Normalized Entropy
**Use for**: Understanding distribution shape

**Example**: Is the distribution uniform or skewed?
```
Distribution [0.5, 0.5] → norm_ent = 1.0 (maximally uncertain for 2 clusters)
Distribution [0.9, 0.1] → norm_ent = 0.47 (skewed, dominant answer)
```

### Uncertainty Score (RECOMMENDED)
**Use for**: Overall uncertainty assessment

**Example**: Should I trust this model's output?
```
Score < 0.3: High confidence, safe to use ✓
Score 0.3-0.7: Moderate uncertainty, review recommended
Score > 0.7: High uncertainty, needs human verification ⚠
```

## Practical Decision Making

```python
result = estimator.compute(generations)

# For production systems
if result['uncertainty_score'] < 0.3:
    # High confidence - use the answer
    return get_most_common_cluster(result)
elif result['uncertainty_score'] < 0.7:
    # Moderate - flag for review
    return answer, confidence="medium", needs_review=True
else:
    # High uncertainty - don't use
    return None, error="Model too uncertain"

# For research/analysis
print(f"Clusters: {result['n_clusters']}")          # How many meanings?
print(f"Normalized: {result['normalized_entropy']}") # How uniform?
print(f"Overall: {result['uncertainty_score']}")     # Combined assessment
```

## Comparison Table

| Metric | Range | What it measures | Best for |
|--------|-------|------------------|----------|
| **Entropy** | [0, ∞) | Absolute uncertainty | Same cluster count comparisons |
| **Normalized Entropy** | [0, 1] | Distribution uniformity | Understanding cluster balance |
| **Uncertainty Score** | [0, 1] | Combined assessment | **Production decisions** ✓ |
| **Number of Clusters** | [1, N] | Semantic diversity | Quick sanity check |

## Real Example from Your Output

```
# Threshold 0.3
Example 1: 2 clusters [0.5, 0.5]
  Normalized Entropy: 1.0  ← Misleading! Seems maximally uncertain
  Uncertainty Score: 0.73  ← More accurate: moderate-high
  
Example 2: 6 clusters [0.167, 0.167, ...]
  Normalized Entropy: 1.0  ← Same as Example 1!
  Uncertainty Score: 1.0   ← Correctly shows this is more uncertain

# Threshold 0.4
Example 1: 1 cluster [1.0]
  Normalized Entropy: 0.0  ← Correct: no uncertainty
  Uncertainty Score: 0.0   ← Agrees: high confidence ✓
```

## Key Takeaway

**Use `uncertainty_score` for decision-making**, but keep `normalized_entropy` and `n_clusters` for detailed analysis.

The uncertainty score gives you a single number that balances both:
- How uncertain the distribution is (normalized entropy)
- How many distinct meanings exist (cluster diversity)

This makes it much better for real-world use cases! 🎯