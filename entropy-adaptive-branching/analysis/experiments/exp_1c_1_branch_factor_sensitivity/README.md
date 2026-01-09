# Experiment 1.C.1: Branch Factor Sensitivity

## Research Question
How does branching width affect cost-quality tradeoff?

## Hypothesis
Higher `branch_factor` = more diversity but higher cost

## Design

**Independent Variable**: `branch_factor` = [2, 3, 5, 7]

**Fixed**:
- Model: Qwen2.5-3B-Instruct (float16)
- Prompt length: 200 tokens, max_new_tokens: 30
- Temperature: 0.8
- EAB threshold: 0.055, max_paths: 20

**Sample Size**: 10 prompts per factor = 40 runs total

## Expected Results
- **Cost**: Linear increase with branch_factor
- **Diversity**: Increases up to a saturation point
- **Optimal**: branch_factor=3 likely best balance

## Running

```bash
python prompts/generate_prompts.py
python run_experiment.py
python analyze_results.py
python plot_results.py
```

## Key Metrics
- Cost per sample vs branch_factor
- Diversity score (Self-BLEU)
- Speedup factor vs branch_factor
