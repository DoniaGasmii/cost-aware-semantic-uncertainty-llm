# RQ1: EAB+SE Quality Evaluation

**Research Question:** Does EAB-generated sample diversity maintain SE quality compared to naive sampling?

## Experiment Design

This experiment validates that Entropy-Adaptive Branching (EAB) produces samples that are as effective (or more effective) for Semantic Entropy (SE) uncertainty estimation as naive temperature sampling.

### Key Parameters

**EAB Configuration:**
- Entropy threshold: `τ = 0.01` (strict, encourages branching)
- Branch factor: `k = 3`
- Max concurrent paths: `M = 20`
- Temperature: `T = 0.7`

**SE Configuration:**
- Encoder: MPNet
- Clustering threshold: **TUNED** (tested: 0.05, 0.10, 0.15, 0.20, 0.25)
- Linkage: average

**Dataset:**
- TriviaQA validation set
- 250 questions
- Correctness: RougeL ≥ 0.3

## Running the Experiment

### Full Threshold Tuning (Recommended)

```bash
cd /localhome/gasmi/semester_project/cost-aware-semantic-uncertainty-llm/entropy-adaptive-branching/analysis/experiments/exp_3a_1_rq1_eab_se_quality

python threshold_tuning.py
```

This will:
1. Run the full pipeline for each clustering threshold (0.05, 0.10, 0.15, 0.20, 0.25)
2. Generate plots for each threshold
3. Create a summary CSV comparing all thresholds
4. Identify the optimal threshold

**Expected runtime:** ~2-3 hours on GPU (5 thresholds × 250 questions)

### Single Threshold Run

```bash
# Run experiment
python run_experiment.py

# Analyze results
python analyze_results.py

# Generate plots
python plot_results.py
```

## Outputs

```
results/
├── threshold_0_05/          # δ = 0.05
│   ├── raw_results.json
│   ├── summary_stats.json
│   └── figures/
│       ├── roc_curve.png
│       ├── distribution_comparison.png
│       ├── cluster_accuracy.png
│       └── eab_generation_stats.png
├── threshold_0_10/          # δ = 0.10
│   └── ...
├── threshold_tuning_summary.csv
└── threshold_sensitivity.png
```

## Key Metrics

### Primary: AUROC
- Measures SE's ability to distinguish incorrect from correct answers
- Higher = better uncertainty estimation

### Secondary:
- Accuracy (any correct, majority correct, best correct)
- Average SE uncertainty score
- Average number of clusters
- Average samples per question (variable with EAB)

## Comparison to Baseline

To compare with naive sampling (SE-alone), refer to:
```
/localhome/gasmi/semester_project/cost-aware-semantic-uncertainty-llm/semantic-entropy-based-uncertainty/analysis/experiements/exp_2a_1_se_auroc_triviaqa/
```

**Expected Result:** AUROC with EAB ≈ AUROC without EAB (validates RQ1)

## Troubleshooting

**Out of memory:**
- Reduce `max_concurrent_paths` in config.yaml (try 15 or 10)
- Reduce `num_questions` for testing

**Slow generation:**
- EAB with τ=0.01 encourages branching, expect more samples per question
- Consider running on fewer questions first to validate setup
