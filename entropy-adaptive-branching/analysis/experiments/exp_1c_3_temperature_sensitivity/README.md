# Experiment 1.C.3: Temperature Sensitivity

## Research Question
How does sampling temperature interact with EAB?

## Hypothesis
Higher temperature = higher entropy = more branching

**Key Insight**: Temperature affects BOTH naive baseline AND EAB entropy distribution

## Design

**Independent Variable**: `temperature` = [0.5, 0.7, 0.9, 1.1, 1.3]

**Fixed**:
- Model: Qwen2.5-3B-Instruct (float16)
- Prompt length: 200 tokens, max_new_tokens: 30
- EAB threshold: 0.055, branch_factor: 3, max_paths: 20

**Sample Size**: 10 prompts per temperature = 50 runs total

## Expected Results
- **Low temp (0.5)**: Low entropy → minimal branching → low speedup
- **High temp (1.3)**: High entropy → constant branching → high speedup
- **Optimal**: temperature ≈ 0.8-1.0 for balanced branching

## Running

```bash
python prompts/generate_prompts.py
python run_experiment.py
python analyze_results.py
python plot_results.py
```

## Key Metrics
- Average entropy vs temperature
- Branching rate vs temperature
- Speedup factor vs temperature
- Diversity vs temperature
