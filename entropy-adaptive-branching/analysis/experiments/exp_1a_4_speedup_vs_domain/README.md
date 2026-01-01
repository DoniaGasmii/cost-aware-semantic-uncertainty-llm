# Experiment 1.A.4: Speedup vs Domain

**Status**: 📋 Planned (not yet implemented)

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
| Model | GPT-2 (124M) | Consistent with other exp_1a |
| Prompt length | 200 tokens | Fixed, mid-range |
| Sample count | 20 | Fixed for fair comparison |
| Temperature | 0.8 | Standard for diverse sampling |
| Max new tokens | 50 | Keeps experiments fast |
| EAB threshold | 0.4 | Balanced branching |
| EAB branch factor | 3 | Standard branching |

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

## Implementation Notes

- **Domain-specific prompts**:
  - Factual QA: Use TriviaQA or similar
  - Creative: Writing prompts from Reddit WritingPrompts
  - Code: Use HumanEval or LeetCode problem statements
- **Prompt length control**: All prompts ~200 tokens (may need padding/truncation)
- **Fair comparison**: Same sample count (N=20) across all domains
- **Branching analysis**: Track where and how often EAB branches in each domain

---

## Files (To Be Created)

- `config.yaml`: Configuration with domain specifications
- `prompts/generate_prompts.py`: Generate domain-specific prompts
  - `prompts/factual_qa/`: 10 factual QA prompts
  - `prompts/creative/`: 10 creative writing prompts
  - `prompts/code/`: 10 code generation prompts
- `run_experiment.py`: Main runner
- `analyze_results.py`: Statistical analysis + domain comparison
- `plot_results.py`: Generate figures (grouped by domain)

---

## Additional Analysis

**Domain-Specific Insights**:
1. **Correlation analysis**: Entropy vs speedup (should be positive)
2. **Branching position analysis**: Where does branching occur in each domain?
3. **Quality check**: Do samples maintain domain-appropriate quality?

---

*To implement: Adapt code from exp_1a_1, changing independent variable from prompt_length to domain*

**Note**: This experiment provides practical insights into when EAB is most beneficial in real-world applications.
