# Experiment 1.B.2: Sample Quality and Diversity Comparison

## Research Question
**RQ3: Does EAB maintain sample quality and diversity compared to naive sampling?**

## Hypothesis
EAB samples are less diverse than naive temperature sampling due to shared prompt encoding and adaptive branching, but remain sufficiently diverse for uncertainty estimation tasks.

## Methodology

### Metrics
1. **Self-BLEU** (lower = more diverse): Measures similarity between generated samples
   - Self-BLEU-1, Self-BLEU-2, Self-BLEU-3

2. **Distinct-n** (higher = more diverse): Ratio of unique n-grams to total n-grams
   - Distinct-1, Distinct-2, Distinct-3

3. **Lexical Diversity** (higher = more diverse): Ratio of unique tokens to total tokens

### Data
- **EAB samples**: From exp_2a_1_se_auroc_triviaqa/results_eab
- **Naive samples**: From exp_2a_1_se_auroc_triviaqa/results

## Running the Experiment

```bash
# 1. Compute diversity metrics
python compute_diversity_metrics.py

# 2. Generate visualizations
python plot_diversity.py
```

## Expected Results

Based on preliminary analysis:
- **Self-BLEU**: EAB ~0.30, Naive ~0.05 (Naive is 6× more diverse)
- **Distinct-n**: EAB ~0.41, Naive ~0.89 (Naive is 2× more diverse)
- **Lexical Diversity**: EAB ~0.35, Naive ~0.75 (Naive is 2× more diverse)

## Interpretation

The lower diversity in EAB samples reflects the **efficiency-diversity tradeoff**:
- EAB shares prompt encoding → samples diverge less
- Adaptive branching → samples branch only at high-uncertainty points
- Naive sampling with varied temperatures → wider exploration space

**Critical question**: Does this reduced diversity hurt downstream performance (SE-AUROC)? Answer in RQ5.

## Outputs

- `results/diversity_metrics.json` - Raw metrics
- `results/figures/diversity_comparison_bars.png` - Bar chart comparison
- `results/figures/diversity_ratios.png` - Ratio visualization
- `results/figures/diversity_summary_table.png` - Summary table
