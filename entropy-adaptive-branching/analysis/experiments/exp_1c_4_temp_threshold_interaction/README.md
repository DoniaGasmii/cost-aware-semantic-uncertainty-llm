# Experiment 1.C.4: Temperature × Threshold Interaction

## Research Question
How do temperature and threshold interact to control branching behavior?

## Hypothesis
Optimal parameter combinations exist for different use cases (efficiency vs diversity)

## Design

**Independent Variables** (2D grid):
- `temperature` = [0.6, 0.8, 1.0, 1.2]
- `entropy_threshold` = [0.036, 0.055, 0.075, 0.100]
  - 0.036: Conservative (90th percentile high confidence)
  - 0.055: Balanced (pilot study recommendation)
  - 0.075: Moderate
  - 0.100: Aggressive

**Fixed**:
- Model: Qwen2.5-3B-Instruct (float16)
- Prompt length: 200 tokens, max_new_tokens: 30
- EAB branch_factor: 3, max_paths: 20

**Sample Size**: 5 prompts per combination = 4×4×5 = 80 runs total

## Expected Results
**Efficiency-first**: Low temp (0.6) + Conservative threshold (0.036)
**Balanced**: Medium temp (0.8) + Balanced threshold (0.055)
**Diversity-first**: High temp (1.2) + Aggressive threshold (0.100)

## Running

```bash
python prompts/generate_prompts.py
python run_experiment.py
python analyze_results.py
python plot_results.py
```

## Key Metrics
- 2D heatmap: Speedup vs (temp, threshold)
- 2D heatmap: Diversity vs (temp, threshold)
- Pareto frontier: Cost vs Quality
