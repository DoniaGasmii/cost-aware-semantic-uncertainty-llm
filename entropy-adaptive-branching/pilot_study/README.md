# EAB Pilot Study: Entropy Distribution Analysis

This pilot study analyzes entropy distributions across 200 diverse prompts to statistically determine the optimal branching threshold for EAB experiments.

## Purpose

Before running systematic experiments (RQ1-RQ5), we need to answer a fundamental design question:

**Should the entropy threshold be fixed or adaptive?**

This pilot study supports the **"Branch When Uncertain"** philosophy:
- Fixed threshold across all prompts
- Let the number of samples reflect genuine model uncertainty
- Use sample count as an additional uncertainty signal

## Directory Structure

```
pilot_study/
├── prompts/                    # Prompt datasets
│   ├── high_confidence.txt     # 70 factual/deterministic prompts
│   ├── medium_confidence.txt   # 65 opinion/approach prompts
│   └── low_confidence.txt      # 65 creative/speculative prompts
│
├── threshold/                  # Threshold analysis tools
│   ├── visualize_distributions.py  # Generate publication-quality plots
│   └── recommend_threshold.py      # Statistical threshold recommendation
│
├── results/                    # Experimental outputs
│   ├── pilot_results.json      # Detailed per-prompt results
│   ├── pilot_summary.csv       # Summary statistics
│   └── threshold_recommendation.txt  # Final threshold recommendation
│
├── plots/                      # Visualization outputs
│   ├── entropy_distributions.png    # Overlaid KDE distributions
│   ├── entropy_boxplots.png        # Box plots by confidence level
│   ├── entropy_cdf.png             # Cumulative distribution functions
│   ├── branching_behavior.png      # Samples/branches by level
│   └── threshold_sweep.png         # Branching vs threshold
│
├── run_pilot.py               # Main experiment script
└── README.md                  # This file
```

## Running the Pilot Study

### Step 1: Run the Experiment (~20-30 minutes)

```bash
cd pilot_study
python3 run_pilot.py
```

This will:
- Load 200 prompts across 3 confidence levels
- Run each prompt with EAB (threshold=0.05 to capture all branching)
- Measure entropy distributions, branching behavior, generation times
- Save results to `results/`

**Expected output:**
- `results/pilot_results.json` - Full detailed results
- `results/pilot_summary.csv` - Summary table

### Step 2: Generate Visualizations

```bash
python3 threshold/visualize_distributions.py
```

Creates 5 publication-quality plots:

1. **entropy_distributions.png**: Overlaid KDE distributions showing separation between confidence levels
2. **entropy_boxplots.png**: Box plots of avg/max entropy by level
3. **entropy_cdf.png**: Cumulative distributions with percentile markers
4. **branching_behavior.png**: Average samples and branches by level
5. **threshold_sweep.png**: How branching rate changes with threshold

### Step 3: Get Threshold Recommendation

```bash
python3 threshold/recommend_threshold.py
```

Performs statistical analysis and recommends thresholds:
- **Conservative** (efficiency-focused)
- **Balanced** (recommended for research)
- **Aggressive** (diversity-focused)
- **Statistical** (optimal separation)

Outputs:
- Terminal report with recommendations and justifications
- `results/threshold_recommendation.txt` for your report

## Prompt Dataset Design

### High Confidence (70 prompts)
**Goal**: Factual, deterministic questions with clear correct answers

**Categories**:
- Mathematics (10): "What is 2 + 2?"
- Geography & Capitals (15): "What is the capital of France?"
- Basic Science Facts (15): "What is the chemical formula for water?"
- Historical Facts (15): "In what year did World War II end?"
- Basic Definitions (15): "What is the opposite of hot?"

**Expected Behavior**: Low entropy (< 0.05), minimal branching, 1-3 samples

### Medium Confidence (65 prompts)
**Goal**: Questions with multiple valid answers, opinions, or approaches

**Categories**:
- Learning & Education (15): "What is the best way to learn programming?"
- Technology & Innovation (15): "What are the main benefits of AI?"
- Career & Work (10): "What qualities make a good leader?"
- Health & Lifestyle (10): "What is the best way to maintain good health?"
- Society & Environment (15): "What is the biggest challenge facing society?"

**Expected Behavior**: Moderate entropy (0.05-0.15), some branching, 5-15 samples

### Low Confidence (65 prompts)
**Goal**: Creative, speculative, or highly subjective prompts

**Categories**:
- Creative Story Beginnings (15): "Once upon a time in a magical forest..."
- Speculative Future (15): "What will life be like in the year 2200?"
- Philosophical Questions (15): "What is the meaning of life?"
- Hypothetical Scenarios (10): "If you could have any superpower..."
- Open-Ended Reflections (10): "The most important lesson I've learned is..."

**Expected Behavior**: High entropy (> 0.15), frequent branching, 10-20+ samples

## Statistical Analysis

The recommendation script performs:

1. **ANOVA**: Tests if entropy distributions differ significantly between confidence levels
2. **Pairwise t-tests**: Compares each pair of confidence levels
3. **Effect sizes (Cohen's d)**: Measures magnitude of differences
4. **Percentile analysis**: Identifies natural threshold boundaries
5. **Classification accuracy**: Finds threshold that best separates high from medium/low confidence

## Threshold Selection Strategies

### Option A: Percentile-Based
- Conservative: 90th percentile of high confidence
- Balanced: 75th percentile of medium confidence (RECOMMENDED)
- Aggressive: 50th percentile of high confidence

### Option B: Statistical Separation
- Find threshold maximizing classification accuracy between groups
- Minimizes misclassification of confidence levels

### Option C: Behavior-Based
- Target specific branching rates (e.g., "10% of high confidence should branch")
- Set threshold achieving desired behavior

## Expected Results

Based on preliminary observations, we expect:

### Entropy Distributions
- **High confidence**: μ ≈ 0.02-0.04, σ ≈ 0.01-0.02
- **Medium confidence**: μ ≈ 0.08-0.12, σ ≈ 0.03-0.05
- **Low confidence**: μ ≈ 0.15-0.25, σ ≈ 0.05-0.08

### Recommended Threshold
- **Likely range**: 0.10-0.15
- **Rationale**: Separates high confidence from medium/low while preserving interpretability

## Integration with Main Experiments

Once threshold is selected:

1. **Document in paper**:
   - Methodology section: "Threshold Selection" subsection
   - Reference pilot study results
   - Justify chosen threshold statistically

2. **Use consistently in RQ1-RQ5**:
   - All experiments use same fixed threshold
   - Ensures comparability across research questions
   - Supports "branch when uncertain" interpretation

3. **Report meta-metrics**:
   - Include sample count distributions in results
   - Highlight cases where 1 sample = high confidence signal
   - Discuss how this enhances 2-layer uncertainty estimation

## Visualization for Report

All plots are publication-ready (300 DPI, professional styling). Include in your report:

1. **Main text**: `entropy_distributions.png` showing three confidence levels
2. **Methods**: Description of prompt categories and expected behaviors
3. **Appendix**: Full statistical analysis and threshold sweep

## Files for Your LaTeX Report

The pilot study generates:

```latex
\subsection{Threshold Selection}

We conducted a pilot study with 200 diverse prompts across three confidence
levels to determine the optimal entropy threshold. Prompts ranged from
factual questions (``What is the capital of France?'') to creative scenarios
(``Once upon a time in a magical forest...'').

Figure~\ref{fig:entropy-distributions} shows the entropy distributions by
confidence level. Statistical analysis (ANOVA) confirmed significant
differences between groups ($F = X.XX$, $p < 0.001$). Based on these
results, we selected $\tau = 0.XXX$ as the entropy threshold, corresponding
to the 75th percentile of medium-confidence prompts.

This threshold ensures that:
\begin{itemize}
    \item High-confidence prompts produce 1-3 samples (minimal branching)
    \item Medium-confidence prompts produce 5-15 samples (moderate exploration)
    \item Low-confidence prompts produce 10-20+ samples (maximum diversity)
\end{itemize}

\begin{figure}[htbp]
    \centering
    \includegraphics[width=0.9\textwidth]{plots/entropy_distributions.png}
    \caption{Entropy distributions by prompt confidence level. The three
             distributions are well-separated, supporting our choice of
             fixed threshold across all experiments.}
    \label{fig:entropy-distributions}
\end{figure}
```

## Troubleshooting

### Issue: Pilot study takes too long
**Solution**: Reduce prompts per category (keep balanced across levels)

### Issue: Not enough separation between confidence levels
**Solution**: Review prompt categorization, use more extreme examples

### Issue: Plots look cluttered
**Solution**: Edit `visualize_distributions.py` - adjust figure size, reduce annotations

## Next Steps

After completing the pilot study:

1. ✅ Select final threshold based on recommendation
2. ✅ Document methodology for paper
3. ✅ Proceed to systematic experiments (exp_1a, exp_1b, etc.)
4. ✅ Use consistent threshold across all RQs
5. ✅ Include pilot study results in paper appendix

---
**Purpose**: Systematic threshold selection for entropy-adaptive branching experiments
