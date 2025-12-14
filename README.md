# Cost-Aware Semantic Uncertainty for LLMs

Efficiently estimate semantic uncertainty in LLM responses by combining **adaptive, cost-aware decoding** with **semantic diversity measurement**.

---

## **Goal**

Quantify **semantic uncertainty** of an LLM's responses to a prompt by measuring how **semantically diverse** multiple completions are, without incurring the high cost of generating dozens of fully independent samples.

---

## **Project Structure**

```
cost-aware-semantic-uncertainty-llm/
├── entropy-adaptive-branching/        # Efficient multi-sample generation
│   ├── eab/                          # Core EAB implementation
│   ├── examples/                     # Usage examples
│   ├── tests/                        # Unit tests
│   └── README.md                     # EAB documentation
│
├── semantic-entropy-based-uncertainty/ # Semantic uncertainty quantification
│   ├── semantic_entropy/             # Core uncertainty estimation
│   ├── examples/                     # Usage examples
│   ├── tests/                        # Unit tests
│   ├── README.md                     # Semantic entropy documentation
│   ├── METRICS_EXPLAINED.md          # Understanding uncertainty metrics
│   └── THRESHOLD_TUNING.md           # Clustering threshold guide
│
└── README.md                         # This file
```

---

## **Core Components**

### 1. **Entropy-Adaptive Branching (EAB)** - Cost-Efficient Generation

Instead of *N* full forward passes, generate diverse samples efficiently:

- **Share the KV cache** of the prompt (and shared prefix) across all candidates
- **Adaptive branching**:
  - At each token position, compute **next-token entropy** from current logits
  - If entropy > threshold → **fork** into multiple continuations
  - If entropy ≤ threshold → continue with a single path
- Creates a **decoding tree** where branching only occurs in "uncertain" regions
- **All samples share computation** until they semantically diverge

**Result**: Total cost ≈ cost of generating a few full sequences, not *N*.

📁 **See**: `entropy-adaptive-branching/README.md` for full details

### 2. **Semantic Entropy** - Uncertainty Quantification

Measure semantic diversity of generated samples:

- **Encode** completions with sentence transformer (`all-mpnet-base-v2`)
- **Cluster** embeddings using agglomerative clustering (adaptive *k*)
- **Compute uncertainty** from cluster distribution:
  - Raw entropy: \( H = -\sum_{i=1}^k p_i \log p_i \)
  - Uncertainty score: Combines entropy + cluster diversity

**Result**: Single uncertainty metric indicating model confidence.

📁 **See**: `semantic-entropy-based-uncertainty/README.md` for full details

---

## **Complete Pipeline**

```python
from eab import EntropyAdaptiveBranching
from semantic_entropy import SemanticUncertaintyEstimator

# Step 1: Generate diverse samples efficiently
eab = EntropyAdaptiveBranching(
    model_name="gpt2",
    entropy_threshold=0.4,
    branch_factor=3
)

generations = eab.generate(
    prompt="Question: What is the capital of France? Answer:",
    max_new_tokens=50
)

# Step 2: Quantify semantic uncertainty
estimator = SemanticUncertaintyEstimator(
    distance_threshold=0.25,
    use_weighted_probs=True  # Use EAB log-probs
)

texts = [g['text'] for g in generations]
log_probs = [g['log_prob'] for g in generations]

result = estimator.compute(texts, log_probs)

# Step 3: Make decision based on uncertainty
if result['uncertainty_score'] < 0.3:
    print("✓ High confidence - safe to use")
elif result['uncertainty_score'] < 0.7:
    print("⚠ Moderate uncertainty - review recommended")
else:
    print("❌ High uncertainty - needs verification")
```

---

## **Why This Works**

### Efficiency: Entropy-Adaptive Branching
- **KV-cache reuse** drastically reduces FLOPs, memory, and latency
- **Adaptive**: Spends compute only where model is uncertain
- **Typical speedup**: 2-5x vs. naive multi-sampling

### Accuracy: Semantic Entropy
- Captures **meaning-level** disagreement (not just token-level variance)
- **Agglomerative clustering** automatically finds number of distinct meanings
- **Uncertainty score** accounts for both distribution uniformity and cluster count

### Combination
- EAB generates diverse samples cheaply
- Semantic entropy quantifies uncertainty accurately
- Together: Fast + meaningful uncertainty estimates

---

## **Installation**

### Quick Start (Both Components)

```bash
# Clone the repository
git clone https://github.com/DoniaGasmii/cost-aware-semantic-uncertainty-llm.git
cd cost-aware-semantic-uncertainty-llm

# Install EAB
cd entropy-adaptive-branching
pip install -r requirements.txt
pip install -e .

# Install Semantic Entropy
cd ../semantic-entropy-based-uncertainty
pip install -r requirements.txt
pip install -e .
```

### Individual Components

```bash
# EAB only
cd entropy-adaptive-branching
pip install -r requirements.txt
pip install -e .

# Semantic Entropy only
cd semantic-entropy-based-uncertainty
pip install -r requirements.txt
pip install -e .
```

---

## **Quick Examples**

### EAB: Generate Diverse Samples

```python
from eab import EntropyAdaptiveBranching

eab = EntropyAdaptiveBranching(model_name="gpt2")
results = eab.generate(
    prompt="The capital of France is",
    max_new_tokens=20
)

for i, result in enumerate(results):
    print(f"{i+1}. {result['text']}")
```

### Semantic Entropy: Measure Uncertainty

```python
from semantic_entropy import SemanticUncertaintyEstimator

estimator = SemanticUncertaintyEstimator()
generations = [
    "The capital is Paris.",
    "Paris is the capital.",
    "It's Paris.",
]

result = estimator.compute(generations)
print(f"Uncertainty: {result['uncertainty_score']:.3f}")
print(f"Clusters: {result['n_clusters']}")
```

---

## **Implementation Details**

### Entropy-Adaptive Branching
- **Branching signal**: Token-level entropy from next-token distribution
- **Cache management**: Deep-copy KV states at branch points
- **Path tracking**: Maintains cumulative log-probabilities per path
- **Pruning**: Limits active paths via `max_paths` parameter

### Semantic Entropy
- **Sentence encoder**: Fine-tuned for semantic similarity (not just embeddings)
- **Clustering method**: Agglomerative with distance threshold (adaptive *k*)
  - No need to specify number of clusters
  - Distance threshold: `1 - cosine_similarity_threshold`
  - Default: 0.25 (merge if similarity > 0.75)
- **Metrics**:
  - **Raw entropy**: Absolute uncertainty measure
  - **Normalized entropy**: Distribution uniformity [0, 1]
  - **Uncertainty score** (recommended): Combined measure accounting for both entropy and cluster count

---

## **Use Cases**

### 1. **Uncertainty Quantification**
Determine when model is confident vs. uncertain:
```python
if uncertainty_score < 0.3:
    # Use answer with confidence
elif uncertainty_score < 0.7:
    # Flag for human review
else:
    # Reject - too uncertain
```

### 2. **Model Calibration**
Identify when models are overconfident or underconfident on validation sets.

### 3. **Active Learning**
Select most informative samples for annotation:
```python
for sample in unlabeled_data:
    uncertainty = estimate_uncertainty(sample)
    if uncertainty > threshold:
        annotation_queue.append(sample)
```

### 4. **Hallucination Detection**
High semantic diversity suggests model is "making things up":
```python
if n_clusters > 3 and uncertainty_score > 0.7:
    print("Warning: Possible hallucination detected")
```

### 5. **Self-Consistency Reasoning**
Generate multiple reasoning paths, cluster semantically similar answers:
```python
# Generate via EAB
reasoning_paths = eab.generate(prompt, max_new_tokens=100)

# Cluster by semantic similarity
result = estimator.compute([p['text'] for p in reasoning_paths])

# Use most common cluster as answer
most_common_cluster = max(result['cluster_probs'])
```

---

## **Key Hyperparameters**

### EAB Parameters
| Parameter | Default | Description |
|-----------|---------|-------------|
| `entropy_threshold` | 0.4 | Branch when token entropy > this |
| `branch_factor` | 3 | Number of paths per branch |
| `max_paths` | 20 | Maximum concurrent paths |

**Tuning**: Higher threshold → less branching → cheaper but less diverse

### Semantic Entropy Parameters
| Parameter | Default | Description |
|-----------|---------|-------------|
| `distance_threshold` | 0.15 | Clustering distance (1 - similarity) |
| `encoder_model` | `all-mpnet-base-v2` | Sentence transformer |
| `use_weighted_probs` | False | Weight clusters by sample probability |

**Tuning**: See `semantic-entropy-based-uncertainty/THRESHOLD_TUNING.md`

---

## **Benchmarks & Performance**

### EAB Speedup (vs. Naive Sampling)
- **Short prompts** (10-20 tokens): 1.5-2x speedup
- **Medium prompts** (50-100 tokens): 3-4x speedup
- **Long prompts** (200+ tokens): 5-8x speedup

Speedup depends on:
- Prompt length (longer = more shared computation)
- Model uncertainty (higher = more branching)
- Number of samples needed

### Semantic Entropy Accuracy
- **Correctly identifies** factual questions (low uncertainty)
- **Correctly identifies** ambiguous questions (high uncertainty)
- **Robust to** paraphrasing (same meaning → same cluster)
- **Sensitive to** factual differences (different facts → different clusters)

---

## **Testing**

```bash
# Test EAB
cd entropy-adaptive-branching
pytest tests/ -v

# Test Semantic Entropy
cd ../semantic-entropy-based-uncertainty
pytest tests/ -v
```

---

## **Documentation**

### EAB Documentation
- `entropy-adaptive-branching/README.md` - Complete guide
- `entropy-adaptive-branching/QUICK_REFERENCES.md` - API reference

### Semantic Entropy Documentation
- `semantic-entropy-based-uncertainty/README.md` - Complete guide
- `semantic-entropy-based-uncertainty/METRICS_EXPLAINED.md` - Understanding metrics
- `semantic-entropy-based-uncertainty/THRESHOLD_TUNING.md` - Clustering guide

---

## **References**

- **Semantic Entropy**: Kuhn et al., 2023. [Paper](https://arxiv.org/abs/2302.09664)
- **Sentence Transformers**: Reimers & Gurevych, 2019. [Paper](https://arxiv.org/abs/1908.10084)
- **Speculative Decoding**: Leviathan et al., 2023. [Paper](https://arxiv.org/abs/2211.17192)

---
