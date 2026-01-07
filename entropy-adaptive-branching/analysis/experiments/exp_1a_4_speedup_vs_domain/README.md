# Experiment 1.A.4: Speedup vs Domain

**Status**: ✅ Ready to Run (scripts created)

**Parent**: Experiment 1.A - Efficiency Analysis (4 of 4)

---

## Research Question

**RQ1.3**: Does EAB's efficiency vary across different domains?

**Hypothesis**: EAB should provide greater speedup in domains with higher uncertainty (more branching opportunities), such as creative writing, compared to factual QA where the model may be more confident.

---

## Experimental Design

### Fixed Variables (Control)

| Parameter | Value | Reason |
|-----------|-------|--------|
| Model | Qwen2.5-3B-Instruct | Consistent with other exp_1a |
| Prompt length | ~200 tokens | Fixed, mid-range (approximately) |
| Temperature | 0.8 | Standard for diverse sampling |
| Max new tokens | 30 | Keeps experiments fast |
| EAB threshold | 0.055 | Tuned for Qwen models |
| EAB branch factor | 3 | Standard branching |
| EAB max paths | 20 | Control branching explosion |

### Independent Variable

**Domain**: [Factual QA, Creative Writing, Code Generation]

For each domain, test on **10 different prompts** (200 tokens each).

**Domain Characteristics**:
- **Factual QA**: Low entropy, deterministic answers (e.g., "What is the capital of France?")
- **Creative Writing**: High entropy, many valid continuations (e.g., "Once upon a time...")
- **Code Generation**: Medium entropy, structured but flexible (e.g., "Write a function to...")

### Dependent Variables (Metrics)

Same as exp_1a_1, with special emphasis on:
- **Branching frequency** per domain
- **Entropy distribution** per domain
- **Speedup correlation with average entropy**

---

## Expected Results

**Prediction**: Speedup should vary by domain based on uncertainty levels

| Domain | Uncertainty | Expected Speedup | Branching |
|--------|-------------|------------------|-----------|
| Factual QA | Low | 1.5-2.0× | Low (~5 branches) |
| Code Generation | Medium | 2.0-2.5× | Medium (~10 branches) |
| Creative Writing | High | 2.5-3.5× | High (~15 branches) |

**Key Insight**: Higher model uncertainty → more branching → greater shared computation → higher speedup

**Key Figures**:
1. Speedup by domain (bar chart with error bars)
2. Branching frequency vs speedup (scatter plot showing correlation)
3. Entropy distribution by domain (violin plots)

---

## Fair Comparison Protocol

Following the same protocol as exp_1a_1:

1. **Run EAB first** with its natural behavior → generates N samples (varies by prompt)
2. **Run Naive N times** → match EAB's sample count
3. **Compare costs** fairly (same number of samples)

## Implementation Notes

- **Domain-specific prompts**:
  - Factual QA: TriviaQA-style questions or knowledge queries
  - Creative: Story writing prompts or creative scenarios
  - Code: Programming tasks or algorithm challenges
- **Prompt length control**: All prompts ~200 tokens (may need padding/truncation)
- **Fair comparison**: EAB determines sample count, Naive matches it
- **Branching analysis**: Track where and how often EAB branches in each domain
- **Entropy tracking**: Measure average entropy per domain to validate hypothesis

---

## Files

- ✅ `config.yaml`: Configuration with domain specifications
- ✅ `run_experiment.py`: Main runner with domain-specific logic
- ⏳ `prompts/generate_prompts.py`: Generate domain-specific prompts
  - `prompts/factual_qa_prompts.json`: 10 factual QA prompts
  - `prompts/creative_prompts.json`: 10 creative writing prompts
  - `prompts/code_prompts.json`: 10 code generation prompts
- ⏳ `analyze_results.py`: Statistical analysis + domain comparison (adapt from exp_1a_1)
- ⏳ `plot_results.py`: Generate figures grouped by domain (adapt from exp_1a_1)

## Running the Experiment

```bash
# Debug mode (2 domains × 2 prompts = 4 runs)
python run_experiment.py

# Full experiment (3 domains × 10 prompts = 30 runs)
# Edit config.yaml: set debug.enabled = false
python run_experiment.py
```

---

## Additional Analysis

**Domain-Specific Insights**:
1. **Correlation analysis**: Entropy vs speedup (should be positive)
2. **Branching position analysis**: Where does branching occur in each domain?
3. **Quality check**: Do samples maintain domain-appropriate quality?

**Note**: This experiment provides practical insights into when EAB is most beneficial in real-world applications.
