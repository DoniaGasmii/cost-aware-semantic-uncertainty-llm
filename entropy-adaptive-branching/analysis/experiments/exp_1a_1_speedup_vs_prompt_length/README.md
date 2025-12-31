# Experiment 1.A: Speedup vs Prompt Length

## Research Question

**RQ1.1**: How much does EAB reduce computational cost compared to naive sampling?

Specifically: Does EAB's efficiency increase with prompt length?

## Hypothesis

**H1**: EAB achieves linear or super-linear speedup growth with prompt length.

**Reasoning**: Longer prompts mean more shared computation before branching occurs. Since the prompt is encoded once and reused across all branches, longer prompts should lead to greater relative savings.

## Experimental Design

### Fixed Variables (Control)

| Parameter | Value | Reason |
|-----------|-------|--------|
| Model | GPT-2 (124M) | Fast for debugging, consistent behavior |
| Temperature | 0.8 | Standard for diverse sampling |
| Max new tokens | 50 | Keeps experiments fast |
| EAB threshold | 0.4 | Balanced branching |
| EAB branch factor | 3 | Standard branching |
| EAB max paths | 20 | Prevent exponential explosion |

### Independent Variable

**Prompt Length**: [50, 100, 200, 500] tokens

For each length, we test on **10 different prompts** for statistical robustness.

### Dependent Variables (Metrics)

**Cost Metrics**:
- Token-steps: Total forward passes × tokens processed (FLOPs proxy)
- Wall-clock time: Real execution time (seconds)
- Memory usage: Peak GPU/CPU memory (MB)
- Tokens per sample: Total tokens / number of samples

**Efficiency Metrics**:
- Speedup factor: Naive cost / EAB cost (higher is better)
- Cost ratio: EAB cost / Naive cost (lower is better)

**Branching Behavior**:
- Branch count: Total number of branches
- Average branch position: At which token did branching occur?
- Final path count: Number of samples generated

## Fair Comparison Protocol

To ensure fair comparison:

1. **Run EAB** with fixed parameters → generates N samples (varies by prompt)
2. **Run Naive** sampling N times → same sample count as EAB
3. **Compare costs** across all metrics

This ensures both methods produce the same number of diverse samples.

## Expected Results

**Prediction**: Speedup factor should increase with prompt length

```
Prompt Length | Expected Speedup
--------------|------------------
50 tokens     | 1.2-1.5×
100 tokens    | 1.5-2.0×
200 tokens    | 2.0-2.5×
500 tokens    | 2.5-3.5×
```

**Key Figure**: Line plot showing speedup vs prompt length with error bars

## Files

- `config.yaml`: All experimental parameters
- `prompts/generate_prompts.py`: Generate test prompts
- `run_experiment.py`: Main experiment runner
- `analyze_results.py`: Statistical analysis
- `plot_results.py`: Generate figures
- `results/raw_results.json`: All measurements
- `results/summary_stats.json`: Aggregated statistics
- `results/figures/`: Generated plots

## Running the Experiment

### Full Experiment

```bash
# Edit config.yaml: set debug.enabled = false

# Generate prompts
python prompts/generate_prompts.py

# Run full experiment (4 lengths × 10 prompts = 40 runs)
python run_experiment.py

# Analyze results
python analyze_results.py

# Generate plots
python plot_results.py
```

## Output Figures

1. **speedup_vs_length.png**: Multi-metric speedup vs prompt length
   - Lines: Token-steps, wall-clock time, memory efficiency
   - Error bars: ±1 std across 10 prompts

2. **cost_breakdown.png**: Stacked bar chart showing where computation is spent
   - Bars: Naive vs EAB for each prompt length
   - Segments: Prompt processing | Generation | Overhead

3. **branching_analysis.png**: EAB branching behavior
   - Subplot 1: Average branch position vs prompt length
   - Subplot 2: Branch frequency vs prompt length

## Statistical Analysis

For each metric, we compute:
- Mean ± standard deviation across 10 prompts
- Median and IQR (for skewed distributions)
- 95% confidence intervals
- Statistical significance (t-test comparing lengths)

## Success Criteria

Experiment is successful if:
- ✅ Speedup factor > 1.0 for all prompt lengths (EAB is faster)
- ✅ Speedup increases with prompt length (validates hypothesis)
- ✅ Standard deviation < 30% of mean (consistent results)
- ✅ Results align with theoretical predictions

## Notes

- All experiments use seed=42 for reproducibility
- Results are saved incrementally to avoid data loss
- Memory profiling requires psutil or GPUtil
