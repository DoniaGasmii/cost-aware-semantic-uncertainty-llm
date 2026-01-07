# Experiment 1.A.3: Speedup vs Model Size

**Status**: ✅ Ready to Run (scripts created)

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
| Temperature | 0.8 | Standard for diverse sampling |
| Max new tokens | 30 | Keeps experiments fast |
| EAB threshold | 0.055 | Tuned for Qwen models |
| EAB branch factor | 3 | Standard branching |
| EAB max paths | 20 | Control branching explosion |
| Domain | General prompts | Consistent behavior |

### Independent Variable

**Model Size**: Using modern instruct models

**Qwen2.5 (Default)**:
- Qwen/Qwen2.5-0.5B-Instruct (0.5B)
- Qwen/Qwen2.5-1.5B-Instruct (1.5B)
- Qwen/Qwen2.5-3B-Instruct (3B)
- Qwen/Qwen2.5-7B-Instruct (7B)

**Alternative - Llama 3**:
- meta-llama/Llama-3.2-1B-Instruct (1B)
- meta-llama/Llama-3.2-3B-Instruct (3B)
- meta-llama/Llama-3.1-8B-Instruct (8B)

*Note*: Larger models require GPU. Config currently uses Qwen2.5 series.

For each model size, test on **10 different prompts** (200 tokens each).

### Dependent Variables (Metrics)

Same as exp_1a_1, plus:
- **FLOPs per forward pass** (theoretical, based on model architecture)
- **Memory per sample** (practical constraint for large models)

---

## Expected Results

**Prediction**: Speedup should increase with model size (more computation to save)

| Model | Parameters | Expected Speedup |
|-------|------------|------------------|
| Qwen2.5-0.5B | 0.5B | 1.8-2.2× |
| Qwen2.5-1.5B | 1.5B | 2.0-2.5× |
| Qwen2.5-3B   | 3B   | 2.2-2.8× |
| Qwen2.5-7B   | 7B   | 2.5-3.2× |

**Key Figures**:
1. Speedup vs model parameters (log-log plot)
2. Absolute time savings vs model size

---

## Fair Comparison Protocol

Following the same protocol as exp_1a_1:

1. **Run EAB first** with its natural behavior → generates N samples (varies by prompt)
2. **Run Naive N times** → match EAB's sample count
3. **Compare costs** fairly (same number of samples)

## Implementation Notes

- **Resource constraints**: Larger models (7B) require GPU with sufficient VRAM
  - Use float16 to reduce memory usage
  - Models loaded sequentially (one at a time) with memory cleanup between runs
- **Consistent prompts**: Use same 10 prompts (200 tokens) across all models
- **Fair comparison**: EAB determines sample count, Naive matches it
- **FLOPs estimation**: Theoretical FLOPs ≈ 2 × params × sequence_length

---

## Files

- ✅ `config.yaml`: Configuration with Qwen2.5 model sizes
- ✅ `run_experiment.py`: Main runner (handles multiple models with memory cleanup)
- ⏳ `prompts/generate_prompts.py`: Generate 200-token prompts (can reuse from exp_1a_2)
- ⏳ `analyze_results.py`: Statistical analysis (adapt from exp_1a_1)
- ⏳ `plot_results.py`: Generate figures with log-scale for model size (adapt from exp_1a_1)

## Running the Experiment

```bash
# Debug mode (2 models × 2 prompts = 4 runs)
python run_experiment.py

# Full experiment (4 models × 10 prompts = 40 runs)
# Edit config.yaml: set debug.enabled = false
python run_experiment.py
```

**⚠️ Warning**: This experiment may take significantly longer than exp_1a_1/1a_2. The 7B model requires substantial GPU memory (~14GB VRAM with float16).
