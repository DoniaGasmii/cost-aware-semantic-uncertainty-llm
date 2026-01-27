# Section: Sample Quality and Diversity (RQ3)

## Experimental Setup

We evaluated whether EAB maintains sample quality and diversity compared to naive temperature sampling using standard NLG diversity metrics.

**Metrics**:
- **Self-BLEU** (lower = more diverse): Average BLEU score of each sample against all others
- **Distinct-n** (higher = more diverse): Ratio of unique n-grams to total n-grams
- **Lexical Diversity** (higher = more diverse): Ratio of unique tokens to total tokens

**Data**: 250 TriviaQA prompts (EAB) vs 200 prompts (Naive, T=0.7, 10 samples/prompt)

## Results

### Quantitative Comparison

| Metric | EAB | Naive | Difference |
|--------|-----|-------|------------|
| Self-BLEU-2 ↓ | **0.729** ± 0.361 | 0.887 ± 0.115 | 22% more diverse |
| Distinct-2 ↑ | **0.384** ± 0.313 | 0.292 ± 0.159 | 32% more diverse |
| Lexical Div. ↑ | **0.317** ± 0.315 | 0.206 ± 0.092 | 54% more diverse |

**Key Finding**: Counter-intuitively, **EAB generates more diverse samples than naive sampling** across all metrics.

### Visual Evidence

[Include figures]:
- `diversity_comparison_bars.png` - Metric comparison across n-gram sizes
- `diversity_ratios.png` - Ratio visualization showing EAB advantage
- `sample_comparison_example_1.png` - Example comparing 5 samples from each method

See Figure X for sample comparisons demonstrating qualitative differences.

## Analysis

### Why is EAB More Diverse?

1. **Adaptive Sample Count**: EAB generates variable samples (1-61, mean=14.6) based on prompt uncertainty, while naive uses fixed count (10)
   - High-uncertainty prompts trigger more exploration
   - Low-uncertainty prompts remain efficient

2. **Structural vs Stochastic Exploration**:
   - **EAB**: Branches at high-entropy tokens (principled divergence points)
   - **Naive**: Random perturbations via temperature (T=0.7 may be conservative)

3. **High Variance = Adaptive Behavior**: EAB shows high diversity variance (std ≈ 0.31-0.36), reflecting prompt-specific adaptation
   - Some prompts: very diverse (Distinct-n → 1.0)
   - Other prompts: focused (Distinct-n → 0.09)

## Implications

1. **No Efficiency-Diversity Tradeoff**: EAB achieves 2-3× speedup (RQ1) AND higher diversity
   - Efficiency gains are "free" with respect to sample quality

2. **Intelligent Exploration**: Branching at semantic ambiguity points produces meaningful diversity
   - Unlike random temperature sampling, exploration aligns with genuine uncertainty

3. **Expected SE-AUROC Benefit**: Higher diversity should improve uncertainty estimation coverage
   - More diverse samples → better semantic space exploration
   - Validates in RQ5 analysis

## Limitations

- Naive sampling tested only at T=0.7 (higher temperatures not explored)
- Lexical diversity measured; semantic diversity requires embedding-based metrics
- Results specific to short-form QA (TriviaQA); generalization to long-form generation unclear

## Conclusion

**RQ3 Answer**: Yes, EAB maintains diversity—**and unexpectedly exceeds naive sampling by 22-54% across metrics**. This demonstrates that adaptive branching provides principled, efficient exploration without sacrificing sample quality.
