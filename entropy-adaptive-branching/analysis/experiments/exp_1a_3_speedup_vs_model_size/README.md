# Experiment 1.A.3: Speedup vs Model Size

**Status**: 📋 Planned (not yet implemented)

**Parent**: Experiment 1.A - Efficiency Analysis (3 of 4)

---

## Research Question

**RQ1.3**: Does EAB provide greater speedup with larger models?

**Hypothesis**: Larger models should benefit more from EAB because there's more computation to share (each forward pass is more expensive).

---

## Experimental Design

### Fixed Variables (Control)

| Parameter | Value | Reason |
|-----------|-------|--------|
| Prompt length | 200 tokens | Fixed, mid-range |
| Sample count | 20 | Fixed for fair comparison |
| Temperature | 0.8 | Standard for diverse sampling |
| Max new tokens | 50 | Keeps experiments fast |
| EAB threshold | 0.4 | Balanced branching |
| EAB branch factor | 3 | Standard branching |
| Domain | Factual QA | Consistent behavior |

### Independent Variable

**Model Size**: [GPT-2 (124M), GPT-2-Medium (355M), GPT-2-Large (774M), GPT-2-XL (1.5B)]

*Note*: Larger models may require GPU or take significant time on CPU. Consider starting with smaller models.

For each model size, test on **10 different prompts** (200 tokens each).

### Dependent Variables (Metrics)

Same as exp_1a_1, plus:
- **FLOPs per forward pass** (theoretical, based on model architecture)
- **Memory per sample** (practical constraint for large models)

---

## Expected Results

**Prediction**: Speedup should increase with model size (more computation to save)

| Model Size | Parameters | Expected Speedup |
|------------|------------|------------------|
| GPT-2      | 124M       | 1.8-2.2×         |
| GPT-2-M    | 355M       | 2.0-2.5×         |
| GPT-2-L    | 774M       | 2.2-2.8×         |
| GPT-2-XL   | 1.5B       | 2.5-3.2×         |

**Key Figures**:
1. Speedup vs model parameters (log-log plot)
2. Absolute time savings vs model size

---

## Implementation Notes

- **Resource constraints**: Larger models require more memory
  - May need to reduce `max_paths` for GPT-2-XL
  - Consider batch_size=1 for all models
- **Consistent prompts**: Use same 10 prompts across all models
- **Fair comparison**: Same number of samples (N=20) for all models
- **FLOPs estimation**: Theoretical FLOPs = 2 × params × sequence_length

---

## Files (To Be Created)

- `config.yaml`: Configuration with model sizes
- `prompts/generate_prompts.py`: Generate 200-token prompts (reuse from 1a_2)
- `run_experiment.py`: Main runner (handles multiple models)
- `analyze_results.py`: Statistical analysis
- `plot_results.py`: Generate figures (log-scale for model size)

---

*To implement: Adapt code from exp_1a_1, changing independent variable from prompt_length to model_name*

**⚠️ Warning**: This experiment may take significantly longer and require more resources than 1a_1 and 1a_2.
