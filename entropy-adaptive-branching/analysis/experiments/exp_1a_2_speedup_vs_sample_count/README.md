# Experiment 1.A.2: Speedup vs Sample Count

**Status**: ✅ Ready to Run (scripts created)

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
| Model | Qwen2.5-3B-Instruct | Consistent with exp_1a_1 |
| Prompt length | 200 tokens | Fixed, mid-range length |
| Temperature | 0.8 | Standard for diverse sampling |
| Max new tokens | 30 | Keeps experiments fast |
| EAB threshold | 0.055 | Tuned for Qwen models |
| EAB branch factor | 3 | Standard branching |
| Domain | General prompts | Consistent behavior |

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

## Fair Comparison Protocol

Following the same protocol as exp_1a_1:

1. **Run EAB first** with its natural behavior → generates N samples (varies by prompt)
2. **Run Naive N times** → match EAB's sample count
3. **Compare costs** fairly (same number of samples)

## Implementation Notes

- **Sample count control**: Varies `max_paths` [5, 10, 20, 50] to encourage different sample counts
- **Same prompts**: Use same 10 prompts (200 tokens) for all max_paths settings
- **Reuse utilities**: Use same metrics, plotting, and analysis code from exp_1a_1
- **Natural behavior**: EAB determines actual sample count based on entropy and branching

---

## Files

- ✅ `config.yaml`: Configuration with sample count targets (via max_paths)
- ✅ `run_experiment.py`: Main runner with max_paths variation
- ⏳ `prompts/generate_prompts.py`: Generate 200-token prompts
- ⏳ `analyze_results.py`: Statistical analysis (adapt from exp_1a_1)
- ⏳ `plot_results.py`: Generate figures (adapt from exp_1a_1)

## Running the Experiment

```bash
# Debug mode (2 sample counts × 2 prompts = 4 runs)
python run_experiment.py

# Full experiment (4 sample counts × 10 prompts = 40 runs)
# Edit config.yaml: set debug.enabled = false
python run_experiment.py
```
