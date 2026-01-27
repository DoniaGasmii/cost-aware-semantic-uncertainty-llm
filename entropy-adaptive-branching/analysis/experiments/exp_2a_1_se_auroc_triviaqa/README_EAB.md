# Experiment 2.A.1: SE AUROC with EAB

This directory contains two versions of the same experiment:

1. **[run_experiment.py](run_experiment.py)** - Uses **naive temperature sampling** (ablation study)
2. **[run_experiment_eab.py](run_experiment_eab.py)** - Uses **EAB generation** (full pipeline)

## Purpose

Both experiments evaluate how well Semantic Entropy (SE) predicts answer correctness on TriviaQA. The difference is the generation method:

- **Naive sampling**: Generates N independent samples using temperature sampling
- **EAB**: Generates samples using entropy-adaptive branching (more efficient, adaptive)

## Running the EAB Version

### Option 1: With GPU (if available)

Edit [config_eab.yaml](config_eab.yaml) and set:
```yaml
model:
  device: "cuda"
  torch_dtype: "float16"
```

Then run:
```bash
cd entropy-adaptive-branching/analysis/experiments/exp_2a_1_se_auroc_triviaqa
python run_experiment_eab.py
```

### Option 2: CPU-only (no GPU required)

The default config is already set for CPU. Just run:
```bash
cd entropy-adaptive-branching/analysis/experiments/exp_2a_1_se_auroc_triviaqa
python run_experiment_eab.py --cpu-only
```

The `--cpu-only` flag ensures no GPU is used even if available.

### Debug Mode

For quick testing (10 questions instead of 200):
```yaml
debug:
  enabled: true
  num_questions: 10
  max_new_tokens: 30
```

Then run the experiment as usual.

## Configuration

Key differences in [config_eab.yaml](config_eab.yaml):

```yaml
eab:
  entropy_threshold: 0.4   # When to branch (higher = less branching)
  branch_factor: 3         # How many branches to create
  max_paths: 10           # Maximum samples (similar to num_samples in naive)

model:
  device: "cpu"           # Use CPU (change to "cuda" for GPU)
  torch_dtype: "float32"  # Float32 for CPU (use "float16" for GPU)
```

## Output

Results are saved to:
- `results_eab/raw_results_eab.json` - Final results
- `results_eab/raw_results_eab_intermediate.json` - Intermediate checkpoints

## Comparison with Naive Sampling

After running both experiments, you can compare:

1. **Accuracy**: Do EAB and naive sampling achieve similar correctness?
2. **SE Quality**: Does SE computed on EAB samples predict correctness as well?
3. **Efficiency**: EAB should generate samples faster (shared computation)
4. **Sample Diversity**: Compare the number of clusters and entropy values

## Next Steps

1. Run the experiment: `python run_experiment_eab.py --cpu-only`
2. Analyze results: `python analyze_results.py` (adapt for EAB results)
3. Compare with naive sampling results from the original experiment
