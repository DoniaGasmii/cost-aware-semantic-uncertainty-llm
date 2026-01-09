# Experiment 1.C.2: Max Paths Sensitivity

## Research Question
How does path budget affect cost-quality balance?

## Hypothesis
Higher `max_paths` = better quality but diminishing returns

## Design

**Independent Variable**: `max_paths` = [5, 10, 20, 50]

**Fixed**:
- Model: Qwen2.5-3B-Instruct (float16)
- Prompt length: 200 tokens, max_new_tokens: 30
- Temperature: 0.8
- EAB threshold: 0.055, branch_factor: 3

**Sample Size**: 10 prompts per value = 40 runs total

## Expected Results
- **Cost**: Near-linear increase with max_paths
- **Quality**: Logarithmic improvement (diminishing returns)
- **Optimal**: max_paths=20 likely sufficient

## Running

```bash
python prompts/generate_prompts.py
python run_experiment.py
python analyze_results.py
python plot_results.py
```

## Key Metrics
- Cost vs max_paths (with pruning statistics)
- Diversity vs max_paths (saturation curve)
- Sample count vs max_paths
