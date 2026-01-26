# Experiment 1.B.1: EAB vs Human Ambiguity Correlation

## Overview

This experiment tests whether EAB's branching behavior correlates with human-perceived ambiguity.

**Research Question**: Does EAB branch more frequently on prompts that humans rate as ambiguous?

## Methodology

- **Dataset**: 15 diverse prompts rated by 80 humans on ambiguity (scale 1-3)
- **Model**: Qwen2.5-3B-Instruct
- **Method**: Run EAB on each prompt, track branching metrics
- **Analysis**: Compute Spearman correlation between human ratings and EAB metrics

## Running the Experiment

```bash
cd /localhome/gasmi/semester_project/cost-aware-semantic-uncertainty-llm/entropy-adaptive-branching/analysis/experiments/exp_1b_1_human_ambiguity_correlation

# Step 1: Run EAB on prompts
python3 run_experiment.py

# Step 2: Analyze correlations
python3 analyze_results.py
```

## Expected Outputs

### Results Files
- `results/raw_results.json` - Full EAB outputs and branching logs
- `results/correlations.json` - Correlation coefficients
- `results/correlation_table.tex` - LaTeX table for report

### Figures
- `results/figures/correlation_scatter.png` - Scatter plots (branches vs ambiguity)
- `results/figures/ambiguity_groups.png` - Branching by ambiguity category

## Metrics Tracked

1. **Number of branches**: Total times EAB branched during generation
2. **Branching frequency**: Branches per token
3. **Average branch entropy**: Mean entropy at branching decision points
4. **Did branch**: Binary indicator (yes/no)

## Interpretation

- **Positive correlation**: EAB branches more on ambiguous prompts → validates entropy as uncertainty signal
- **Weak/no correlation**: EAB may capture linguistic ambiguity differently than humans
- **Statistical significance**: p < 0.05 indicates reliable relationship
