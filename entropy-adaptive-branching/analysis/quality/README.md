# Quality Assessment for EAB vs Naive Generation

This directory contains tools for assessing the quality and diversity of text generations from Entropy-Adaptive Branching (EAB) versus naive sampling.

## Overview

The quality assessment evaluates generated samples using:
- **Self-BLEU**: Measures intra-set similarity (lower = more diverse)
- **Distinct-n**: Measures lexical diversity (higher = more diverse)
- **Human evaluation prompts**: Selected examples for manual assessment
- **Demo examples**: Clear cases showing EAB advantages

## Input Format

Create two JSON files with your generated samples:

### `eab_samples.json`
```json
{
  "The best way to learn programming is": [
    "to practice every day with small projects...",
    "by building real applications that solve...",
    "through consistent practice and reading..."
  ],
  "How can we reduce climate change": [
    "by transitioning to renewable energy...",
    "through sustainable lifestyle changes...",
    "by implementing carbon taxes..."
  ]
}
```

### `naive_samples.json`
Same structure as above, with ~20 generations per prompt.

## Usage

1. **Prepare your data**:
   ```bash
   # Place your generated samples in:
   # - eab_samples.json
   # - naive_samples.json
   ```

2. **Run the assessment**:
   ```bash
   python quality_assessment.py
   ```

3. **Review outputs**:
   - `metrics.csv` - Quantitative metrics for all prompts
   - `human_eval_prompts.json` - 5 prompts selected for human evaluation
   - `demo_example.txt` - One clear demonstration example
   - `self_bleu_comparison.png` - Boxplot of Self-BLEU scores
   - `distinct_n_comparison.png` - Bar plot of Distinct-n metrics

## Metrics Explained

### Self-BLEU (Lower is Better)
- Measures average pairwise BLEU score between generations
- Lower scores indicate more diverse outputs
- Range: 0 (completely diverse) to 1 (identical)

### Distinct-n (Higher is Better)
- Ratio of unique n-grams to total n-grams
- Higher scores indicate more lexical variety
- Range: 0 (repetitive) to 1 (all unique)

### Human Evaluation Selection Criteria
- Short prompts (< 15 words)
- Open-ended questions
- Both methods have ~5 generations
- Diverse topics

## Dependencies

```bash
pip install nltk scikit-learn matplotlib seaborn numpy
```

The script will automatically download required NLTK data on first run.

## Example Output

```
======================================================================
QUALITY ASSESSMENT: EAB vs NAIVE GENERATION
======================================================================

1. Loading generated samples...
   ✓ Loaded EAB samples: 50 prompts
   ✓ Loaded Naive samples: 50 prompts
   ✓ Common prompts: 50

2. Computing diversity and quality metrics...
   ✓ Computed metrics for 50 prompts
   ✓ Saved metrics to metrics.csv

3. Selecting prompts for human evaluation...
   ✓ Selected 5 prompts
   ✓ Saved to human_eval_prompts.json

4. Selecting demo example...
   ✓ Saved demo to demo_example.txt

5. Generating plots...
   ✓ Saved Self-BLEU plot
   ✓ Saved Distinct-n plot

======================================================================
SUMMARY STATISTICS
======================================================================

Self-BLEU-2 (Lower is More Diverse):
  EAB:   0.2341 ± 0.0523
  Naive: 0.3124 ± 0.0612

Distinct-2 (Higher is More Diverse):
  EAB:   0.7823 ± 0.0421
  Naive: 0.7012 ± 0.0534
```

## Notes

- Ensure prompts are identical between EAB and Naive samples
- More generations per prompt (15-25) give more reliable metrics
- The script filters for suitable human evaluation prompts automatically
- Demo example is selected to highlight EAB's diversity advantage
