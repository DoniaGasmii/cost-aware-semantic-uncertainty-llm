# Entropy-Adaptive Branching - Quick Reference

## Installation

```bash
pip install -r requirements.txt
pip install -e .
```

## Basic Usage

```python
from eab import EntropyAdaptiveBranching

# Initialize
eab = EntropyAdaptiveBranching(
    model_name="gpt2",
    entropy_threshold=0.4,  # 0-1, lower = more branching
    branch_factor=3,        # 2-5, branches per split
    max_paths=20           # 10-50, max concurrent paths
)

# Generate
results = eab.generate(
    prompt="Your prompt here",
    max_new_tokens=50,
    temperature=0.8
)

# Access results
for r in results:
    print(r['text'])
    print(r['probability'])
    print(r['branch_points'])
```

## Key Parameters

### Initialization
| Parameter | Range | Default | Effect |
|-----------|-------|---------|--------|
| `entropy_threshold` | 0-1 | 0.4 | Lower = more branching |
| `branch_factor` | 2-5 | 3 | Branches per split |
| `max_paths` | 10-50 | 20 | Max concurrent paths |

### Generation
| Parameter | Range | Default | Effect |
|-----------|-------|---------|--------|
| `temperature` | 0.1-2.0 | 1.0 | Higher = more random |
| `top_k` | int or None | None | Keep top-k tokens |
| `top_p` | 0-1 or None | None | Nucleus sampling |
| `seed` | int or None | None | For reproducibility |

## Tuning Guide

### For Factual QA
```python
eab = EntropyAdaptiveBranching(
    entropy_threshold=0.6,  # Conservative
    branch_factor=2,        # Few branches
    max_paths=10
)
```

### For Creative Generation
```python
eab = EntropyAdaptiveBranching(
    entropy_threshold=0.3,  # Aggressive
    branch_factor=4,        # Many branches
    max_paths=20
)
```

### For General Purpose
```python
eab = EntropyAdaptiveBranching(
    entropy_threshold=0.4,  # Balanced
    branch_factor=3,        # Moderate
    max_paths=20
)
```

## Common Operations

### Get entropy history
```python
history = eab.get_entropy_history()
print(history['statistics'])
```

### Plot entropy
```python
eab.plot_entropy()
```

### Get cache statistics
```python
stats = eab.get_cache_statistics()
print(f"Memory used: {stats['peak_memory_mb']} MB")
```

### Update parameters
```python
eab.set_entropy_threshold(0.5)
eab.set_branch_factor(4)
eab.set_max_paths(30)
```

## Result Structure

Each result dictionary contains:
```python
{
    'text': str,              # Generated text
    'tokens': List[int],      # Token IDs
    'log_prob': float,        # Log probability
    'probability': float,     # Actual probability
    'path_id': int,           # Unique path ID
    'parent_id': int,         # Parent path ID
    'length': int,            # Number of tokens
    'branch_points': List[int], # Where branches occurred
    'num_branches': int       # Total branches
}
```

## Utilities

### Format results
```python
from eab.utils import format_results
formatted = format_results(paths, tokenizer)
```

### Compute statistics
```python
from eab.utils import compute_statistics
stats = compute_statistics(paths)
```

### Diversity metrics
```python
from eab.utils import compute_diversity_metrics
texts = [r['text'] for r in results]
diversity = compute_diversity_metrics(texts)
```

### Save/load results
```python
from eab.utils import save_results, load_results
save_results(results, 'output.json')
loaded = load_results('output.json')
```

### Set seed
```python
from eab.utils import set_seed
set_seed(42)
```

## Troubleshooting

### Out of memory
- Reduce `max_paths`
- Use smaller model
- Reduce `max_new_tokens`

### Too much branching
- Increase `entropy_threshold`
- Reduce `branch_factor`

### Not enough diversity
- Decrease `entropy_threshold`
- Increase `branch_factor`
- Increase `temperature`

### Slow generation
- Check if branching too aggressively
- Reduce `max_paths`
- Use GPU if available

## File Structure

```
entropy-adaptive-branching/
├── eab/              # Core implementation
├── examples/         # Usage examples
├── tests/           # Test suite
├── notebooks/       # Jupyter tutorials
├── README.md        # Full documentation
└── quickstart.py    # Quick test
```

## Example Commands

```bash
# Run quick test
python quickstart.py

# Run examples
python examples/basic_usage.py
python examples/factual_qa.py
python examples/creative_generation.py
python examples/uncertainty_analysis.py

# Run tests
pytest tests/ -v

# Start notebook
jupyter notebook notebooks/tutorial.ipynb
```

## Common Patterns

### Generate with high confidence
```python
results = eab.generate(
    prompt=prompt,
    temperature=0.7,  # Lower temperature
    entropy_threshold=0.6  # Higher threshold
)
```

### Generate with high diversity
```python
results = eab.generate(
    prompt=prompt,
    temperature=1.2,  # Higher temperature
    entropy_threshold=0.3  # Lower threshold
)
```

### Generate fixed number of samples
```python
eab.set_max_paths(10)  # Exactly 10 paths
results = eab.generate(prompt, max_new_tokens=50)
```

### Batch process prompts
```python
prompts = ["prompt1", "prompt2", "prompt3"]
all_results = []

for prompt in prompts:
    results = eab.generate(prompt, max_new_tokens=30)
    all_results.append(results)
```

## Performance Tips

1. **Reuse model**: Initialize once, generate many times
2. **Use GPU**: Significant speedup for larger models
3. **Tune threshold**: Match to your use case
4. **Batch prompts**: Process multiple prompts with same settings
5. **Monitor memory**: Use `get_cache_statistics()`

## Support

- Read: `README.md`
- Examples: `examples/`
- Tests: `tests/`
- Tutorial: `notebooks/tutorial.ipynb`
- Issues: GitHub issues (if public)

---

**Quick Start**: `python quickstart.py`

**Full Docs**: See `README.md`

**Tutorial**: Open `notebooks/tutorial.ipynb`