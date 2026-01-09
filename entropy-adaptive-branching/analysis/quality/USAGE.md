# Quick Start Guide

## Installation

```bash
cd entropy-adaptive-branching/analysis/quality
pip install -r requirements.txt
```

## Quick Test

Run with example data to verify everything works:

```bash
python test_with_examples.py
```

This will generate all outputs using the template data.

## Running with Your Data

1. **Prepare your samples**: Create or replace `eab_samples.json` and `naive_samples.json` with your actual generated samples.

2. **Run the assessment**:
   ```bash
   python quality_assessment.py
   ```

3. **Check the outputs**:
   - `metrics.csv` - All computed metrics
   - `human_eval_prompts.json` - Selected prompts for human evaluation
   - `demo_example.txt` - One demonstration example
   - `self_bleu_comparison.png` - Self-BLEU boxplot
   - `distinct_n_comparison.png` - Distinct-n bar chart

## File Structure

```
quality/
├── quality_assessment.py          # Main analysis script
├── test_with_examples.py          # Test script with example data
├── requirements.txt               # Python dependencies
├── README.md                      # Full documentation
├── USAGE.md                       # This quick start guide
├── eab_samples_template.json      # Example EAB samples
├── naive_samples_template.json    # Example Naive samples
│
└── [Generated outputs]
    ├── eab_samples.json           # Your EAB data (you provide)
    ├── naive_samples.json         # Your Naive data (you provide)
    ├── metrics.csv                # Computed metrics
    ├── human_eval_prompts.json    # Selected prompts
    ├── demo_example.txt           # Demo example
    ├── self_bleu_comparison.png   # Plot 1
    └── distinct_n_comparison.png  # Plot 2
```

## Expected Input Format

Both JSON files should have this structure:

```json
{
  "prompt text 1": [
    "generation 1",
    "generation 2",
    ...
    "generation 20"
  ],
  "prompt text 2": [
    "generation 1",
    ...
  ]
}
```

**Important**:
- Prompts must be identical between EAB and Naive files
- Aim for 15-25 generations per prompt for reliable metrics
- Keep prompt texts as dictionary keys

## Metrics Interpretation

| Metric | Range | Better Value | What it Measures |
|--------|-------|--------------|------------------|
| Self-BLEU-2/3 | 0-1 | Lower | Less similarity between outputs |
| Distinct-2/3/4 | 0-1 | Higher | More unique n-grams |

**Goal**: EAB should show lower Self-BLEU (more diverse) and higher Distinct-n (more lexical variety) than Naive, while maintaining quality.

## Troubleshooting

**Error: Sample files not found**
- Make sure `eab_samples.json` and `naive_samples.json` exist in the quality/ directory

**Error: No common prompts found**
- Check that prompt texts are exactly identical in both files (case-sensitive)

**NLTK download errors**
- Run: `python -c "import nltk; nltk.download('punkt')"`

**Import errors**
- Install dependencies: `pip install -r requirements.txt`
