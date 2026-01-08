# Running Experiments with COW vs Original EAB

## Overview

The experiments now support tracking **both COW (Copy-on-Write)** and **Original (deep copy)** EAB implementations to compare memory efficiency.

## Configuration

In the experiment's `config.yaml`, set the `implementation` parameter:

```yaml
eab:
  entropy_threshold: 0.055
  branch_factor: 3
  max_paths: 20
  implementation: "cow"  # Options: "cow", "original", or "both"
```

### Options

1. **`"cow"`** (default, recommended)
   - Uses Copy-on-Write cache implementation
   - 40-60% less memory than deep copy
   - Best for memory-constrained environments

2. **`"original"`**
   - Uses deep copy cache implementation
   - Higher memory usage but well-tested
   - Use for baseline comparisons

3. **`"both"`**
   - Runs BOTH implementations on each prompt
   - Doubles experiment time but provides direct comparison
   - Results include `implementation` field: "cow" or "original"
   - Best for comprehensive memory analysis

## Running Experiments

### Single Implementation (Fast)

```bash
cd experiments/exp_1a_1_speedup_vs_prompt_length

# Run with COW only
python run_experiment.py

# Or run with Original only (edit config.yaml first to set implementation: "original")
python run_experiment.py
```

### Both Implementations (Comprehensive)

```bash
cd experiments/exp_1a_1_speedup_vs_prompt_length

# Edit config.yaml: set implementation: "both"
python run_experiment.py
```

## Result Structure

When running with `implementation: "both"`, each result includes an `implementation` field:

```json
{
  "prompt_id": "len50_prompt1",
  "implementation": "cow",
  "eab_metrics": {
    "memory_peak_mb": 45.2,
    ...
  },
  "naive_metrics": {
    "memory_peak_mb": 12.5,
    ...
  }
}
```

The results file will contain **two entries per prompt** - one for COW and one for Original.

## Analyzing Results

After running experiments with `"both"`, you can compare implementations:

```python
import json

# Load results
with open('results/raw_results.json') as f:
    data = json.load(f)

# Separate by implementation
cow_results = [r for r in data['results'] if r['implementation'] == 'cow']
orig_results = [r for r in data['results'] if r['implementation'] == 'original']

# Compare memory
for cow, orig in zip(cow_results, orig_results):
    assert cow['prompt_id'] == orig['prompt_id']

    cow_mem = cow['eab_metrics']['memory_peak_mb']
    orig_mem = orig['eab_metrics']['memory_peak_mb']
    savings = (orig_mem - cow_mem) / orig_mem * 100

    print(f"{cow['prompt_id']}: COW={cow_mem:.1f}MB, Orig={orig_mem:.1f}MB, Savings={savings:.1f}%")
```

## Expected Memory Results

With corrected memory tracking, you should see:

| Method | Memory Overhead (MB) | Notes |
|--------|---------------------|-------|
| **Naive** | 5-20 MB | Sequential generation, 1 path |
| **COW EAB** | 30-60 MB | Parallel paths with shared cache |
| **Original EAB** | 50-100 MB | Parallel paths with deep copy |

### Memory Comparisons

- **COW vs Original**: COW should use 40-60% less memory
- **EAB vs Naive**: Both EAB implementations use MORE memory than naive (they trade memory for speed)
- **Time speedup**: Both implementations should show 1.5-2.0× speedup over naive

## What Was Fixed

Previously, the experiments only tracked the original deep copy implementation. Now:

1. ✅ Both COW and Original implementations can be tested
2. ✅ Memory tracking uses `torch.cuda.max_memory_allocated()` for accuracy
3. ✅ Each result is labeled with its implementation type
4. ✅ "both" mode enables direct comparison on same prompts

## Recommendation

For **production experiments**, use `implementation: "cow"` (default) for best memory efficiency.

For **research/analysis**, use `implementation: "both"` to quantify the exact memory savings of COW.
