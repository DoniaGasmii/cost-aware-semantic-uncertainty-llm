# Quality Assessment Workflow

## Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                   QUALITY ASSESSMENT PIPELINE                   │
└─────────────────────────────────────────────────────────────────┘

INPUT FILES                    PROCESSING                    OUTPUTS
═══════════════════════════════════════════════════════════════════

eab_samples.json          ┌─────────────────┐
  {prompt: [20 gens]}  ──▶│  Load & Align   │
                          └────────┬────────┘
naive_samples.json              │
  {prompt: [20 gens]}  ──────────┘
                                  │
                          ┌───────▼────────┐
                          │ Compute Metrics│
                          │  • Self-BLEU   │         metrics.csv
                          │  • Distinct-n  │──────▶  (All metrics)
                          │  • Avg Length  │
                          └───────┬────────┘
                                  │
                    ┌─────────────┴────────────┐
                    │                          │
            ┌───────▼────────┐        ┌───────▼─────────┐
            │ Select Prompts │        │  Generate Plots │
            │ for Human Eval │        │                 │
            │  • Short       │        │  • Self-BLEU    │──▶ PNG files
            │  • Open-ended  │        │  • Distinct-n   │
            │  • ~5 gens ea. │        └─────────────────┘
            └───────┬────────┘
                    │
        ┌───────────┴──────────┐
        │                      │
  ┌─────▼──────┐      ┌───────▼────────┐
  │  Human     │      │  Demo Example  │
  │  Eval      │      │  • Best prompt │──▶ demo_example.txt
  │  Prompts   │──▶   │  • Shows       │
  └────────────┘      │    diversity   │
                      └────────────────┘
      human_eval_prompts.json

```

## Processing Steps

### 1. Data Loading & Validation
- Load both JSON files
- Find common prompts
- Validate format
- Report statistics

### 2. Metrics Computation

For each prompt, compute:

**Self-BLEU** (Lower = More Diverse)
```
For each generation:
  Compare with all other generations
  Compute BLEU-2, BLEU-3
Average all pairwise scores
```

**Distinct-n** (Higher = More Diverse)
```
Extract all n-grams (n=2,3,4)
Count unique vs total
Distinct-n = unique / total
```

### 3. Prompt Selection

**Human Evaluation Prompts**:
```
Filter criteria:
  ✓ Short (< 15 words)
  ✓ Open-ended question
  ✓ Has ~5 generations each method
  ✓ Diverse topics

Select top 5 by criteria match
```

**Demo Example**:
```
From human eval prompts, select best where:
  ✓ EAB has higher Distinct-n
  ✓ EAB has lower Self-BLEU
  ✓ Reasonable length (20-100 words)
```

### 4. Visualization

**Plot 1: Self-BLEU Comparison**
- Boxplot showing distribution
- EAB vs Naive side-by-side
- Median values annotated

**Plot 2: Distinct-n Comparison**
- Bar chart for n=2,3,4
- Direct comparison
- Values on bars

## Output Files

| File | Type | Purpose |
|------|------|---------|
| `metrics.csv` | CSV | All quantitative metrics for every prompt |
| `human_eval_prompts.json` | JSON | 5 prompts × 5 generations × 2 methods |
| `demo_example.txt` | Text | Human-readable demo showing diversity |
| `self_bleu_comparison.png` | Image | Boxplot comparing Self-BLEU scores |
| `distinct_n_comparison.png` | Image | Bar chart comparing Distinct-n |

## Typical Results Pattern

```
Expected if EAB works well:

Self-BLEU (Lower is Better)
  EAB:   0.25 ← Lower (more diverse)
  Naive: 0.35

Distinct-2 (Higher is Better)
  EAB:   0.78 ← Higher (more variety)
  Naive: 0.68

Distinct-3 (Higher is Better)
  EAB:   0.82 ← Higher
  Naive: 0.75
```

## Integration with Other Experiments

This quality assessment complements your efficiency experiments:

```
Efficiency Experiments (exp_1a_*, exp_1c_*)
    ↓
Shows: EAB is faster, uses less compute
    ↓
    ├─▶ BUT: Does quality suffer?
    │
Quality Assessment (this folder)
    ↓
Shows: EAB maintains/improves diversity
    ↓
Conclusion: EAB is both efficient AND high-quality
```

## Next Steps After Running

1. **Check metrics.csv**: Look for consistent patterns across prompts

2. **Review human_eval_prompts.json**: Use for manual quality assessment

3. **Share demo_example.txt**: Show clear example of EAB advantages

4. **Include plots in paper**: Publication-ready visualizations

5. **Statistical testing**: Run t-tests on metrics if needed:
   ```python
   from scipy import stats
   t_stat, p_value = stats.ttest_rel(eab_scores, naive_scores)
   ```
