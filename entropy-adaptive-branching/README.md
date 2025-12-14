# Entropy-Adaptive Branching (EAB) for Efficient Multi-Sample Generation

## Idea Overview

We implement an efficient alternative to naive multi-sample generation by:

- Encode the **prompt once** and cache its KV states.  
- Generate tokens **autoregressively**, sharing all computation **until a branching decision**.  
- **Only when entropy is high**, fork into multiple paths; each copying the cache up to that point.  
- All shared tokens (prompt + generated prefix) are computed **exactly once**, no matter how many samples diverge later.

### Core Idea: Entropy-Adaptive Branching with KV-Cache Reuse

```
Step 1: Encode prompt ONCE
    Prompt → [Transformer] → KV-cache (store this!)

Step 2: Generate token-by-token, branching when uncertain
    Position 0: Check entropy → Low → Continue with 1 path
    Position 1: Check entropy → High → Branch into 3 paths
    Position 2: Check entropy → Low → Continue each path separately
    ...
    
Step 3: All paths reuse the same prompt cache
```

### Visual Example

**Prompt** (12 tokens):  
`"Question: What is the capital of France? Answer:"`

Generation proceeds confidently for 2 tokens:
- Token 13: `"The"`  
- Token 14: `"capital"`

At token 15, entropy is high → **branch into 3 paths**:

```
Prompt: "Question: What is the capital of France? Answer:"
        └─► "The" ──► "capital" ──► (high entropy → BRANCH)
                                      │
                                      ├─► "is Paris."                (6 more tokens)
                                      │
                                      ├─► "of France is Paris."      (6 more tokens)
                                      │
                                      └─► "city is Paris."           (6 more tokens)
```

- **Shared tokens**: 12 (prompt) + 2 (generated) = **14 tokens → computed once**  
- **Divergent tokens**: 3 branches × 6 tokens = **18 tokens → computed separately**

### Cost Comparison

- **Adaptive**: 14 + 18 = **32 token-steps**  
- **Naive** (3 samples): 3 × (12 + 2 + 6) = **60 token-steps**  

→ **47% fewer token-steps**, same 3 diverse completions.  
Savings grow with longer prompts or later branching.

---

## Installation

```bash
# Clone the repository
git clone https://github.com/DoniaGasmii/cost-aware-semantic-uncertainty-llm.git
cd cost-aware-semantic-uncertainty-llm/entropy-adaptive-branching

# Install dependencies
pip install -r requirements.txt
```

## Quick Start

```python
from eab import EntropyAdaptiveBranching
import torch

# Initialize the branching system
eab = EntropyAdaptiveBranching(
    model_name="gpt2",
    entropy_threshold=0.4,
    branch_factor=3,
    max_paths=20
)

# Generate multiple samples efficiently
prompt = "Question: What is the capital of France? Answer:"
results = eab.generate(
    prompt=prompt,
    max_new_tokens=50,
    temperature=0.8
)

# Access generated samples
for i, result in enumerate(results):
    print(f"Sample {i+1}: {result['text']}")
    print(f"Log probability: {result['log_prob']:.4f}")
    print(f"Branch points: {result['branch_points']}")
    print()
```

## Project Structure

```
entropy-adaptive-branching/
├── README.md                  # This file
├── requirements.txt           # Python dependencies
├── setup.py                   # Package installation
├── eab/
│   ├── __init__.py           # Package initialization
│   ├── core.py               # Main EAB implementation
│   ├── cache.py              # KV-cache management
│   ├── entropy.py            # Entropy computation
│   ├── path.py               # Generation path tracking
│   └── utils.py              # Utility functions
├── examples/
│   ├── basic_usage.py        # Simple example
│   ├── factual_qa.py         # Factual question answering
│   ├── creative_generation.py # Creative text generation
│   └── uncertainty_analysis.py # Semantic uncertainty demo
├── tests/
│   ├── test_core.py          # Core functionality tests
│   ├── test_cache.py         # Cache management tests
│   ├── test_entropy.py       # Entropy calculation tests
│   └── test_integration.py   # End-to-end tests
└── notebooks/
    ├── tutorial.ipynb        # Interactive tutorial
    └── analysis.ipynb        # Performance analysis
```

## Algorithm Details

### 1. Entropy as Branching Signal

**Why entropy?**

Entropy measures uncertainty in the next-token distribution:

```math
H(p) = - \sum_{i=1}^{V} p(x_i) \log p(x_i)
```

where: V = vocabulary size

#### Interpretation:
- **High entropy**: Probability mass spread across many tokens  
  → Model is uncertain → **Action: Branch** to explore multiple options  
- **Low entropy**: Probability mass concentrated on few tokens  
  → Model is confident → **Action: Continue with single path** (saves compute)

### Normalization

Raw entropy depends on vocabulary size. We normalize:

```python
normalized_entropy = H(p) / log(vocab_size)
# Now in range [0, 1]
```

This allows us to set a universal threshold (e.g., 0.4) regardless of model vocab size.

### 2. KV-Cache Mechanics

**What is KV-cache?**

In transformer generation:
- Each layer computes Key and Value matrices from all previous tokens
- Standard generation recomputes these every step (wasteful!)
- KV-cache stores them, only computing for new tokens

**How we use it**:

```python
# Initial forward pass with prompt
outputs = model(input_ids, use_cache=True)
past_key_values = outputs.past_key_values  # Store KV cache

# Subsequent generation reuses cache
outputs = model(
    input_ids=next_token,
    past_key_values=past_key_values,
    use_cache=True
)
```

**Branching with cache**:

```python
# When branching, deep copy the cache for each path
for _ in range(branch_factor):
    new_cache = deep_copy_cache(past_key_values)
    new_path = GenerationPath(tokens, log_prob, new_cache)
    paths.append(new_path)
```

### 3. Probability Tracking

Each path tracks its cumulative log-probability:

```python
class GenerationPath:
    tokens: List[int]          # Generated tokens
    log_prob: float            # Σ log P(token_i | context)
    cache: Tuple              # KV-cache state
```

**Why log-probabilities?**
- Numerical stability (avoids underflow)
- Easy accumulation: `log P(A,B) = log P(A) + log P(B)`
- Convert back: `P(path) = exp(log_prob)`

(Required for our downstream uncertainty quantification pipeline; optional for basic sampling.)

---

## Computational Complexity

### FLOPs Analysis

**Naive Sampling** (N samples):
```
FLOPs = N × (L_prompt + L_gen) × d_model × n_layers × O(attention)
      ≈ N × L_total × 12d²  (for standard transformer)
```

**Adaptive Branching**:
```
FLOPs = (L_prompt + L_shared) × 12d²              # Shared computation
      + Σ(branch_k × L_divergent_k) × 12d²        # Per-branch computation
```

**Speedup Factor**:
```
Speedup = (N × L_total) / (L_shared + Σ(branch_k × L_divergent_k))
```

For the France capital example:
```
Speedup = (3 × 20) / (14 + 18) = 60/32 = 1.875x
```

### Memory Analysis

**Naive Sampling**:
```
Memory = O(L_total × d_model × n_layers)  # Sequential, one at a time
```

**Adaptive Branching**:
```
Memory = O(n_active_paths × L_total × d_model × n_layers)
```

**Trade-off**:
- Naive: Low memory, high compute
- Adaptive: Higher memory (bounded by max_paths), lower compute

---

## Hyperparameter Guide

### `entropy_threshold` (0-1)

Controls branching aggressiveness:

| Value | Behavior | Use Case |
|-------|----------|----------|
| 0.2-0.3 | Very aggressive | Maximize diversity |
| 0.4-0.5 | Balanced | General purpose |
| 0.6-0.7 | Conservative | Efficiency-focused |

**Tuning tip**: 
```python
# For factual QA (less diversity needed)
threshold = 0.6  # Branch rarely, focus on confident answers

# For creative generation (more diversity wanted)
threshold = 0.3  # Branch often, explore many possibilities
```

### `branch_factor` (2-5)

How many branches per split:

| Value | Paths Generated | Diversity | Cost |
|-------|----------------|-----------|------|
| 2 | Fewer | Lower | Cheap |
| 3 | Moderate | Good | Balanced |
| 4-5 | More | Higher | Expensive |

**Recommendation**: Start with 3.

### `max_paths` (10-50)

Prevents exponential explosion:

```
Without max_paths:
Branch 3 ways × 5 times = 3^5 = 243 paths! 💥

With max_paths=20:
Prune low-probability paths, keep top-20
```

---

## Expected Behavior

### Factual Prompts (Low Uncertainty)

```
Prompt: "The capital of France is"

Expected branching:
- Position 0: Low entropy → "Paris" (confident)
- Position 1: Low entropy → "is" (confident)
- Total branches: 1-2
- Speedup: ~8-10x (barely any branching)
```

### Ambiguous Prompts (High Uncertainty)

```
Prompt: "The best programming language is"

Expected branching:
- Position 0: High entropy → Branch {"Python", "JavaScript", "Java"}
- Position 1: High entropy → Branch further
- Total branches: 8-15
- Speedup: ~3-5x (more branching, but shared prefix)
```

---

## Downstream Usage

The generated samples can be used directly for different applications. Our primary motivation is to use them for **semantic uncertainty** analysis.

Example workflow:
1. Generate N diverse samples with EAB
2. Cluster semantically similar completions
3. Compute uncertainty metrics based on cluster distribution
4. Use uncertainty scores for model calibration or active learning

---

## Limitations & Considerations

### 1. Not Always Faster

Adaptive branching helps most when:
- ✅ **Long prompts** (100+ tokens)
- ✅ **High branching** (model uncertain)
- ✅ **Many samples needed** (N ≥ 10)

### 2. Memory Trade-off

Naive: Sequential, low memory
Adaptive: Parallel, higher memory

**Mitigation**: Use `max_paths` to cap memory.

### 3. Path Imbalance

Some branches may be very unlikely:

```
Path 1: P = 0.5  (likely)
Path 2: P = 0.3  (likely)
Path 3: P = 0.001 (unlikely!)
```

**Solution**: Prune paths below probability threshold.

---

## Citation

If you use this code in your research, please cite:

```bibtex
@software{entropy_adaptive_branching,
  title={Entropy-Adaptive Branching for Efficient Multi-Sample Generation},
  author={Donia Gasmi},
  year={2025},
  url={https://github.com/DoniaGasmii/cost-aware-semantic-uncertainty-llm/entropy-adaptive-branching}
}
```
