# Experiment 1.A.5: Speedup vs Generation Length

## Research Question
Does EAB efficiency increase with longer generations?

## Hypothesis
Longer generation = more branching opportunities = higher speedup

## Design

**Independent Variable**: `max_new_tokens` = [10, 30, 50, 100]

**Fixed**:
- Model: Qwen2.5-3B-Instruct (float16)
- Prompt length: 200 tokens
- Temperature: 0.8
- EAB threshold: 0.055, branch_factor: 3, max_paths: 20

**Sample Size**: 10 prompts per length = 40 runs total

## Expected Results
```
Generation Length | Expected Speedup
------------------|------------------
10 tokens         | 1.1-1.3×
30 tokens         | 1.5-2.0×
50 tokens         | 2.0-2.5×
100 tokens        | 2.5-3.5×
```

## Running

```bash
# Debug mode (2 lengths × 2 prompts)
python prompts/generate_prompts.py
python run_experiment.py

# Full experiment (set debug.enabled = false in config.yaml)
python run_experiment.py
python analyze_results.py
python plot_results.py
```

## Key Metrics
- Speedup factor (Naive/EAB cost)
- Branch frequency vs generation position
- Cost per sample vs generation length
