# Threshold Tuning Guide

## Quick Reference

The `distance_threshold` parameter controls how similar two texts need to be to cluster together.

**Formula**: `distance_threshold = 1 - cosine_similarity_threshold`

| Threshold | Similarity | Behavior | Use Case |
|-----------|------------|----------|----------|
| 0.10 | 0.90 | Very strict - many clusters | When minor wording differences matter |
| **0.15** | **0.85** | **Strict (default)** | **Conservative clustering** |
| 0.20 | 0.80 | Moderate | Balance between precision and recall |
| **0.25** | **0.75** | **Balanced** | **Recommended for most cases** |
| 0.30 | 0.70 | Lenient | When you want to group paraphrases |
| 0.35 | 0.65 | Very lenient | Risk of over-clustering |

## How to Find Your Optimal Threshold

### Method 1: Use the Diagnostic Tool

```bash
python examples/tune_threshold.py
```

This will test multiple thresholds and show you which one correctly clusters your test cases.

### Method 2: Use Command-Line Arguments

```bash
# Try different thresholds
python examples/basic_usage.py --threshold 0.15
python examples/basic_usage.py --threshold 0.25
python examples/basic_usage.py --threshold 0.30

# Or short form
python examples/factual_qa.py -t 0.25
```

### Method 3: Manual Inspection

1. Run with default threshold (0.15)
2. Look at the cluster output
3. Ask yourself:
   - **Too many clusters?** → Increase threshold (e.g., 0.25)
   - **Too few clusters?** → Decrease threshold (e.g., 0.10)
   - **Just right?** → Keep current threshold

## Common Scenarios

### Scenario 1: Factual QA (e.g., "What is the capital of France?")

**Expected**: All variations of "Paris" should cluster together

```
Generations:
  - "The capital of France is Paris."
  - "Paris is the capital of France."
  - "It's Paris."
  - "The answer is Paris."

Desired: 1 cluster (all mean the same thing)
```

**If you get 3-4 clusters with threshold=0.15:**
- ❌ Too strict! 
- ✅ Try threshold=0.25 or 0.30

### Scenario 2: Ambiguous Questions (e.g., "What's the best programming language?")

**Expected**: Different opinions should cluster separately

```
Generations:
  - "Python is the best."
  - "JavaScript is the best."
  - "It depends on your use case."

Desired: 3 clusters (different opinions)
```

**If everything clusters into 1 with threshold=0.30:**
- ❌ Too lenient!
- ✅ Try threshold=0.15 or 0.20

### Scenario 3: Similar but Different Facts

**Expected**: Different cities should NOT cluster together

```
Generations:
  - "The capital is Paris."
  - "The capital is Lyon."
  - "The capital is Marseille."

Desired: 3 clusters (different answers)
```

**Good sentence transformers should handle this regardless of threshold!**

## Decision Tree

```
Start with threshold = 0.25 (recommended baseline)
  ↓
Run on your data
  ↓
Are semantically identical sentences clustering together?
  ├─ YES → Good! Are semantically different sentences in separate clusters?
  │         ├─ YES → ✓ Keep this threshold
  │         └─ NO → Decrease threshold (try 0.20 or 0.15)
  └─ NO → Increase threshold (try 0.30 or 0.35)
             ↓
         Repeat until satisfied
```

## Pro Tips

1. **Start with 0.25** - Works well for most use cases
2. **Use tune_threshold.py** - Automates the search
3. **Test on real data** - Don't just tune on synthetic examples
4. **Domain matters** - Technical text might need different thresholds than creative text
5. **When in doubt, go stricter** - Better to have more clusters than to incorrectly merge different meanings

## Examples with Different Thresholds

### Example: "Capital of France" Question

**Threshold = 0.15 (too strict)**
```
Cluster 0 [40.0%] - 2 generations:
  • The capital of France is Paris.
  • Paris is the capital of France.
Cluster 1 [20.0%] - 1 generation:
  • Paris
Cluster 2 [20.0%] - 1 generation:
  • It's Paris.
Cluster 3 [20.0%] - 1 generation:
  • The answer is Paris.

Result: 4 clusters (over-splitting!) ❌
```

**Threshold = 0.25 (just right)**
```
Cluster 0 [100.0%] - 4 generations:
  • The capital of France is Paris.
  • Paris is the capital of France.
  • Paris
  • It's Paris.
  • The answer is Paris.

Result: 1 cluster (correct!) ✓
```

**Threshold = 0.40 (too lenient)**
```
Cluster 0 [100.0%] - 6 generations:
  • The capital of France is Paris.
  • Paris is the capital of France.
  • It's Paris.
  • The capital of France is Lyon.  ← WRONG!
  • The capital is Marseille.      ← WRONG!
  • I don't know.                  ← WRONG!

Result: 1 cluster (under-splitting!) ❌
```

## Setting in Code

```python
# Conservative (strict)
estimator = SemanticUncertaintyEstimator(distance_threshold=0.15)

# Recommended (balanced)
estimator = SemanticUncertaintyEstimator(distance_threshold=0.25)

# Lenient (groups more aggressively)
estimator = SemanticUncertaintyEstimator(distance_threshold=0.30)
```