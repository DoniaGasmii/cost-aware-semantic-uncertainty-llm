# Pilot Study Quick Start Guide

## What is this?

A comprehensive pilot study to statistically determine the optimal entropy threshold for your EAB experiments.

## Why do this first?

Before running systematic experiments (RQ1-RQ5), you need to answer:
- **Should threshold be fixed or adaptive?**
- **What threshold value captures "genuine uncertainty"?**
- **How do different prompts behave?**

## What you'll get

1. **Statistical analysis** of 200 prompts across 3 confidence levels
2. **5 publication-quality plots** (300 DPI) for your report
3. **Threshold recommendation** with mathematical justification
4. **Documentation** ready to copy into your paper

## How to run (3 options)

### Option 1: Automated (Recommended)
```bash
./run_all.sh
```
Runs everything in sequence (~25-35 minutes total)

### Option 2: Step by Step
```bash
# Step 1: Run experiment (~20-30 min)
python3 run_pilot.py

# Step 2: Generate plots (~1 min)
python3 threshold/visualize_distributions.py

# Step 3: Get recommendation (~1 min)
python3 threshold/recommend_threshold.py
```

### Option 3: Quick Test (10 prompts only)
Edit `run_pilot.py` to load only first 10 prompts per level for testing.

## What gets created

```
results/
├── pilot_results.json              # Full detailed results
├── pilot_summary.csv               # Summary table
└── threshold_recommendation.txt    # Your threshold + justification

plots/
├── entropy_distributions.png       # Main plot for paper
├── entropy_boxplots.png           # Statistical summary
├── entropy_cdf.png                # Percentile analysis
├── branching_behavior.png         # Samples/branches by level
└── threshold_sweep.png            # Threshold sensitivity
```

## Dataset

- **70 high-confidence prompts**: Math, geography, basic facts
- **65 medium-confidence prompts**: Opinions, approaches, preferences
- **65 low-confidence prompts**: Creative, speculative, philosophical

Total: **200 diverse prompts**

## Expected Runtime

On NVIDIA A10 GPU with Qwen2.5-3B-Instruct + FP16:
- **Per prompt**: ~4-10 seconds
- **Total**: 20-30 minutes
- **Visualization**: 1-2 minutes
- **Analysis**: < 1 minute

## What to do with results

1. **Review plots** in `plots/` - especially `entropy_distributions.png`

2. **Read recommendation** in `results/threshold_recommendation.txt`

3. **Select threshold** (likely 0.10-0.15 recommended)

4. **Document in paper**:
   ```latex
   Based on a pilot study of 200 prompts (Section X.X), we selected
   entropy threshold τ = 0.XXX, corresponding to the 75th percentile
   of medium-confidence prompts. This ensures high-confidence prompts
   produce minimal samples while uncertain prompts branch extensively.
   ```

5. **Use consistently** in all experiments (RQ1-RQ5)

## Interpreting Results

### Good separation (expected):
- High confidence: μ ≈ 0.03, most prompts < 0.08
- Medium confidence: μ ≈ 0.10, prompts 0.08-0.15
- Low confidence: μ ≈ 0.18, most prompts > 0.12

→ **Choose threshold ≈ 0.12-0.15**

### Poor separation (unexpected):
- Overlapping distributions
- No clear threshold

→ **Re-categorize prompts or use adaptive threshold**

## Troubleshooting

**Q: Takes too long?**
A: Use smaller model (Qwen2.5-1.5B) or reduce prompts

**Q: Out of memory?**
A: Lower max_paths from 20 to 10, or use 8-bit quantization

**Q: Plots look weird?**
A: Check `results/pilot_summary.csv` for data issues

**Q: No clear recommendation?**
A: Review prompt categorization, may need more extreme examples

## Philosophy

This pilot study supports **"Branch When Uncertain"** design:
- Threshold is fixed across all prompts
- # of samples = uncertainty signal
- 1 sample = confident, many samples = uncertain
- Enables 2-layer uncertainty estimation

## Next Steps

After completing pilot study:

1. ✅ Select threshold (likely the "balanced" recommendation)
2. ✅ Add threshold justification to paper
3. ✅ Proceed to systematic experiments (exp_1a, etc.)
4. ✅ Use same threshold consistently in RQ1-RQ5
5. ✅ Include pilot plots in paper appendix

---

**Questions?** See full README.md for detailed documentation.

**Ready?** Run `./run_all.sh` to start!
