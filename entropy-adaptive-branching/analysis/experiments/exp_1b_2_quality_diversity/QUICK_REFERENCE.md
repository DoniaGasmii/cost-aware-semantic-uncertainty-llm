# RQ3: Sample Quality & Diversity - Quick Reference

## TL;DR

**Finding**: EAB is 22-54% MORE diverse than naive sampling, contradicting the expected efficiency-diversity tradeoff.

---

## Key Numbers

| What | Value |
|------|-------|
| **Self-BLEU-2** | EAB: 0.729, Naive: 0.887 → **EAB 22% more diverse** |
| **Distinct-2** | EAB: 0.384, Naive: 0.292 → **EAB 32% more diverse** |
| **Lexical Diversity** | EAB: 0.317, Naive: 0.206 → **EAB 54% more diverse** |
| **Sample Count** | EAB: 14.6 avg (1-61 range), Naive: 10 fixed |
| **Dataset** | 250 EAB prompts, 200 Naive prompts (TriviaQA) |

---

## Files

### Reports
- [EXPERIMENT_REPORT.md](EXPERIMENT_REPORT.md) - Full detailed report
- [PAPER_SECTION.md](PAPER_SECTION.md) - Concise version for paper
- **This file** - Quick reference

### Scripts
- [compute_diversity_metrics.py](compute_diversity_metrics.py) - Compute metrics
- [plot_diversity.py](plot_diversity.py) - Generate bar charts and tables
- [visualize_samples.py](visualize_samples.py) - Show sample examples

### Results
- [results/diversity_metrics.json](results/diversity_metrics.json) - Raw metrics data

### Figures
- [results/figures/diversity_comparison_bars.png](results/figures/diversity_comparison_bars.png) - Main metric comparison
- [results/figures/diversity_ratios.png](results/figures/diversity_ratios.png) - Ratio visualization
- [results/figures/diversity_summary_table.png](results/figures/diversity_summary_table.png) - Summary table
- [results/figures/sample_comparison_example_1.png](results/figures/sample_comparison_example_1.png) - Sample examples (3 files total)

---

## Run Commands

```bash
# Regenerate everything
cd /localhome/gasmi/semester_project/cost-aware-semantic-uncertainty-llm/entropy-adaptive-branching/analysis/experiments/exp_1b_2_quality_diversity

# 1. Compute metrics
python3 compute_diversity_metrics.py

# 2. Generate plots
python3 plot_diversity.py

# 3. Visualize samples
python3 visualize_samples.py
```

---

## Interpretation Guide

### Self-BLEU (lower = more diverse)
- **What it measures**: Average similarity between all pairs of samples
- **Range**: 0 (completely different) to 1 (identical)
- **Result**: EAB = 0.729, Naive = 0.887
- **Meaning**: EAB samples are less similar to each other

### Distinct-n (higher = more diverse)
- **What it measures**: Ratio of unique n-grams to total n-grams
- **Range**: 0 (all n-grams repeated) to 1 (all n-grams unique)
- **Result**: EAB = 0.384, Naive = 0.292
- **Meaning**: EAB produces more unique phrases

### Lexical Diversity (higher = more diverse)
- **What it measures**: Ratio of unique words to total words
- **Range**: 0 to 1
- **Result**: EAB = 0.317, Naive = 0.206
- **Meaning**: EAB uses richer vocabulary

---

## Why is EAB More Diverse?

1. **Adaptive sampling**: Generates more samples for uncertain prompts
2. **Principled branching**: Diverges at high-entropy (ambiguous) points
3. **Structural exploration**: Branching > random temperature perturbations

---

## Paper Talking Points

### Hypothesis
> We hypothesized EAB would sacrifice diversity for efficiency due to shared prompt encoding.

### Result
> **Counter-intuitively, EAB exceeds naive sampling in diversity by 22-54%** across all metrics (Self-BLEU, Distinct-n, Lexical Diversity).

### Explanation
> EAB branches at high-entropy tokens, creating meaningful divergence at semantic ambiguity points. This structural exploration proves more effective than stochastic temperature sampling (T=0.7).

### Implication
> **No efficiency-diversity tradeoff exists**: EAB achieves 2-3× speedup (RQ1) AND higher sample quality, making efficiency gains "free" with respect to diversity.

---

## Next Steps

1. **Include in paper**: Use [PAPER_SECTION.md](PAPER_SECTION.md) as template
2. **Add figures**: Include diversity_comparison_bars.png and sample_comparison_example_1.png
3. **Link to RQ1**: Emphasize no tradeoff between efficiency and diversity
4. **Link to RQ5**: Predict higher diversity → better SE-AUROC (validate experimentally)

---

## Citation

```
@experiment{rq3_diversity_2024,
  title={Sample Quality and Diversity Analysis: EAB vs Naive Sampling},
  dataset={TriviaQA},
  finding={EAB is 22-54% more diverse than naive sampling},
  metrics={Self-BLEU, Distinct-n, Lexical Diversity},
  n_prompts={250 EAB, 200 Naive}
}
```
