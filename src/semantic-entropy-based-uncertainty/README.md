# Semantic Entropy-Based Uncertainty Quantification

Measure **semantic diversity** in LLM generations to quantify model uncertainty about its answer.

## Core Idea

Traditional uncertainty methods count token-level variance. We measure **meaning-level disagreement**:

```
Question: "What's the capital of France?"

Generation 1: "The capital is Paris."
Generation 2: "Paris is the capital."  
Generation 3: "It's Paris."
Generation 4: "The capital is Lyon."

→ Cluster 1: {Gen 1, 2, 3} - "Paris" (75%)
→ Cluster 2: {Gen 4} - "Lyon" (25%)
→ Semantic Entropy: 0.56 (moderate uncertainty)
```

**High entropy** → Many different meanings → Model is uncertain  
**Low entropy** → Similar meanings → Model is confident

---

## Algorithm

```
Input: N text generations (from EAB or any sampling method)

1. Encode with sentence transformer
   └─► Embeddings capture semantic meaning, not surface form
   
2. Cluster adaptively (no pre-set k)
   └─► Agglomerative clustering with distance threshold
   └─► Merges generations with similar meaning
   
3. Compute cluster distribution
   └─► p_i = fraction of generations in cluster i
   
4. Calculate semantic entropy
   └─► H = -Σ p_i log(p_i)
   └─► Normalized: H_norm = H / log(k) ∈ [0, 1]

Output: Uncertainty score + cluster analysis
```

---

## Installation

```bash
# Clone the repository
cd semantic-entropy-based-uncertainty

# Install dependencies
pip install -r requirements.txt

# Or install as package
pip install -e .
```

**Dependencies**:
- `sentence-transformers` - Semantic encoding
- `scikit-learn` - Clustering
- `numpy` - Numerical operations
- `torch` - PyTorch backend

---

## Quick Start

```python
from semantic_entropy import SemanticUncertaintyEstimator

# Initialize
estimator = SemanticUncertaintyEstimator(
    encoder_model="all-mpnet-base-v2",
    distance_threshold=0.15  # Cosine similarity threshold of 0.85
)

# Compute uncertainty
generations = [
    "The capital is Paris.",
    "Paris is the capital of France.",
    "It's Paris.",
    "The capital is Lyon."
]

result = estimator.compute(generations)

print(f"Semantic Entropy: {result['entropy']:.3f}")
print(f"Normalized (0-1): {result['normalized_entropy']:.3f}")
print(f"Distinct Meanings: {result['n_clusters']}")
print(f"Cluster Distribution: {result['cluster_probs']}")
```

**Output**:
```
Semantic Entropy: 0.562
Normalized (0-1): 0.811
Distinct Meanings: 2
Cluster Distribution: [0.75, 0.25]
```

---

## Interpretation Guide

| Entropy | Interpretation | Example |
|---------|----------------|---------|
| < 0.3 | **High confidence** | All generations say the same thing |
| 0.3-0.7 | **Moderate uncertainty** | A few competing meanings |
| > 0.7 | **High uncertainty** | Many different answers |

**Normalized entropy** (0-1 scale):
- **0.0**: Perfect agreement (1 cluster)
- **0.5**: Moderate diversity (e.g., 3 clusters with [0.5, 0.3, 0.2] distribution)
- **1.0**: Maximum diversity (all clusters equally likely)

---

## Integration with EAB

```python
from eab import EntropyAdaptiveBranching
from semantic_entropy import SemanticUncertaintyEstimator

# Generate diverse samples efficiently
eab = EntropyAdaptiveBranching(model_name="gpt2")
generations = eab.generate(
    prompt="Question: What is the capital of France? Answer:",
    max_new_tokens=50
)

# Extract text and probabilities
texts = [g['text'] for g in generations]
log_probs = [g['log_prob'] for g in generations]

# Compute semantic uncertainty
estimator = SemanticUncertaintyEstimator(
    use_weighted_probs=True  # Use EAB probabilities
)
uncertainty = estimator.compute(texts, log_probs=log_probs)

print(f"Model uncertainty: {uncertainty['normalized_entropy']:.3f}")
```

---

## Advanced Usage

### Get Cluster Representatives

```python
representatives = estimator.get_cluster_representatives(texts)

for cluster_id, info in representatives.items():
    print(f"Cluster {cluster_id}:")
    print(f"  Representative: {info['text']}")
    print(f"  Size: {info['size']} samples")
    print(f"  Probability: {info['probability']:.3f}")
```

### Detailed Analysis

```python
result = estimator.compute(texts, return_details=True)

# Access cluster analysis
analysis = result['cluster_analysis']
print(f"Cluster cohesion: {analysis['cohesion']}")

# Access similarity matrix
sim_matrix = result['similarity_matrix']
print(f"Pairwise similarities:\n{sim_matrix}")
```

### Generate Text Report

```python
report = estimator.compute_with_report(texts)
print(report)
```

Output:
```
============================================================
SEMANTIC UNCERTAINTY REPORT
============================================================
Semantic Entropy:      0.5623
Normalized Entropy:    0.8109
Number of Clusters:    2
Cluster Distribution:  [0.75, 0.25]

Interpretation: HIGH UNCERTAINTY - Many different interpretations
============================================================
```

---

## Hyperparameters

### `encoder_model`
Sentence transformer for semantic encoding:
- `all-mpnet-base-v2`: **Recommended** (best quality, 768-dim)
- `all-MiniLM-L6-v2`: Faster, smaller (384-dim)
- `paraphrase-multilingual-mpnet-base-v2`: For non-English

### `distance_threshold`
Controls clustering granularity (1 - cosine_similarity):

| Value | Similarity | Behavior |
|-------|------------|----------|
| 0.10 | 0.90 | Strict - more clusters |
| **0.15** | **0.85** | **Balanced (default)** |
| 0.25 | 0.75 | Lenient - fewer clusters |

**Tuning**: Start at 0.15, inspect clusters manually, adjust if needed.

### `linkage`
How to measure cluster-to-cluster distance:
- `average`: **Default** - average pairwise similarity
- `complete`: More conservative (furthest pair)
- `single`: More aggressive (closest pair)

### `use_weighted_probs`
- `False` (default): Uniform weighting (count-based)
- `True`: Weight clusters by sample probabilities (requires `log_probs`)

---

## Implementation Details

### Why Agglomerative Clustering?

**Alternatives considered**:
- ❌ K-means: Requires knowing k ahead of time
- ❌ DBSCAN: May mark valid answers as "noise"
- ✅ **Agglomerative**: Adaptive k, no noise labels, deterministic

**Key advantage**: Distance threshold has clear semantic meaning  
("Merge if cosine similarity > 0.85")

### Why Sentence Transformers?

- ✅ Trained specifically for semantic similarity
- ✅ Representation models (not generative) - no circularity
- ✅ Captures paraphrases: "Paris is the capital" ≈ "The capital is Paris"
- ✅ Fast inference (no autoregressive decoding)

### Edge Cases

**What if all generations are identical?**
- Result: 1 cluster, entropy = 0 (perfect confidence) ✓

**What if all generations are completely different?**
- Result: N clusters, maximum entropy ✓

**What about factually different but syntactically similar?**
- "The capital is Paris" vs "The capital is Lyon"
- Good sentence transformers encode *semantic content*, not just syntax
- These should cluster separately ✓

---

## Project Structure

```
semantic-entropy-based-uncertainty/
├── README.md                     # This file
├── requirements.txt              # Dependencies
├── setup.py                      # Package installation
├── semantic_entropy/
│   ├── __init__.py              # Package exports
│   ├── estimator.py             # Main SemanticUncertaintyEstimator
│   ├── clustering.py            # SemanticClusterer class
│   └── utils.py                 # Helper functions
├── examples/
│   ├── basic_usage.py           # Simple examples
│   ├── factual_qa.py            # Question answering demo
│   └── integration_with_eab.py  # EAB integration
└── tests/
    ├── test_estimator.py        # Estimator tests
    └── test_clustering.py       # Clustering tests
```

---

## Running Examples

```bash
# Basic usage
python examples/basic_usage.py

# Factual QA analysis
python examples/factual_qa.py

# Integration with EAB (requires EAB package)
python examples/integration_with_eab.py
```

---

## Running Tests

```bash
# Install pytest
pip install pytest

# Run all tests
pytest tests/ -v

# Run specific test file
pytest tests/test_estimator.py -v
```

---

## Use Cases

### 1. Uncertainty Quantification
Measure how confident a model is in its answer:
```python
uncertainty = estimator.compute(generations)
if uncertainty['normalized_entropy'] < 0.3:
    print("High confidence - safe to use")
else:
    print("Low confidence - needs human review")
```

### 2. Model Calibration
Detect when models are uncertain vs certain:
```python
# Collect uncertainty scores on validation set
uncertainties = []
for prompt in validation_set:
    generations = generate_samples(prompt)
    result = estimator.compute(generations)
    uncertainties.append(result['normalized_entropy'])

# Analyze calibration
threshold = find_optimal_threshold(uncertainties, labels)
```

### 3. Active Learning
Select informative samples for annotation:
```python
# Generate predictions with uncertainty
for sample in unlabeled_data:
    generations = model.generate(sample, n=10)
    uncertainty = estimator.compute(generations)
    
    if uncertainty['normalized_entropy'] > threshold:
        # High uncertainty - add to annotation queue
        annotation_queue.append(sample)
```

### 4. Hallucination Detection
Identify when models generate inconsistent responses:
```python
# For factual questions, high entropy suggests hallucination
result = estimator.compute(generations)
if result['n_clusters'] > 3:
    print("Warning: Multiple conflicting answers detected")
```

---

## Limitations & Future Work

### Current Limitations
1. **Threshold sensitivity**: Distance threshold affects cluster count
   - Mitigation: Use validation set to tune threshold per domain
2. **Encoder bias**: Sentence transformers may have domain-specific biases
   - Mitigation: Fine-tune encoder on domain-specific data
3. **Computational cost**: Clustering is O(N²) for N samples
   - Mitigation: Use smaller encoder model for large N

### Future Enhancements
- [ ] Adaptive threshold selection based on prompt type
- [ ] Multi-lingual support with appropriate encoders
- [ ] Hierarchical cluster analysis for finer-grained uncertainty
- [ ] Probability-weighted clustering (using EAB log-probs)
- [ ] GPU acceleration for large-scale batch processing
- [ ] Integration with uncertainty calibration methods

---

## References

- **Sentence-Transformers**: Reimers & Gurevych, 2019. [Paper](https://arxiv.org/abs/1908.10084)
- **Semantic Entropy**: Kuhn et al., 2023. [Paper](https://arxiv.org/abs/2302.09664)
- **Agglomerative Clustering**: Sklearn Documentation. [Link](https://scikit-learn.org/stable/modules/clustering.html#hierarchical-clustering)

---
