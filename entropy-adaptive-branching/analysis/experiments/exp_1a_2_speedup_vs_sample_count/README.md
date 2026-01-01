# Experiment 1.A.2: Speedup vs Sample Count

**Status**: 📋 Planned (not yet implemented)

**Parent**: Experiment 1.A - Efficiency Analysis (2 of 4)

---

## Research Question

**RQ1.1**: How does EAB's efficiency scale with the number of samples generated?

**Hypothesis**: EAB should show increasing speedup with more samples, as the shared prompt computation is amortized over more samples.

---

## Experimental Design

### Fixed Variables (Control)

| Parameter | Value | Reason |
|-----------|-------|--------|
| Model | GPT-2 (124M) | Consistent with exp_1a_1 |
| Prompt length | 200 tokens | Fixed, mid-range length |
| Temperature | 0.8 | Standard for diverse sampling |
| Max new tokens | 50 | Keeps experiments fast |
| EAB threshold | 0.4 | Balanced branching |
| EAB branch factor | 3 | Standard branching |
| Domain | Factual QA | Consistent behavior |

### Independent Variable

**Sample Count**: [5, 10, 20, 50] samples

For each sample count, test on **10 different prompts** for statistical robustness.

### Dependent Variables (Metrics)

Same as exp_1a_1:
- Token-steps (FLOPs proxy)
- Wall-clock time
- Memory usage
- Speedup factors
- Branching behavior

---

## Expected Results

**Prediction**: Speedup should increase with sample count

| Sample Count | Expected Speedup |
|--------------|------------------|
| 5 samples    | 1.2-1.5×         |
| 10 samples   | 1.5-2.0×         |
| 20 samples   | 2.0-2.5×         |
| 50 samples   | 2.5-3.0×         |

**Key Figure**: Speedup vs sample count (should show upward trend)

---

## Implementation Notes

- **Fair comparison**: Both EAB and Naive generate exactly N samples
- **Same prompts**: Use same 10 prompts (200 tokens) for all sample counts
- **Reuse utilities**: Use same metrics, plotting, and analysis code from exp_1a_1

---

## Files (To Be Created)

- `config.yaml`: Configuration with sample counts
- `prompts/generate_prompts.py`: Generate 200-token prompts
- `run_experiment.py`: Main runner
- `analyze_results.py`: Statistical analysis
- `plot_results.py`: Generate figures

---

*To implement: Adapt code from exp_1a_1, changing the independent variable from prompt_length to sample_count*
