# EAB Analysis & Experiments

This directory contains all experimental analysis code for evaluating Entropy-Adaptive Branching (EAB).

## Structure

```
analysis/
├── README.md                  # This file
├── utils/                     # Shared utilities
│   ├── metrics.py            # Metrics computation
│   ├── data_utils.py         # Save/load results
│   └── plotting.py           # Plotting functions
│
└── experiments/              # Individual experiments
    └── exp_1a_speedup_vs_prompt_length/
        ├── README.md         # Experiment documentation
        ├── config.yaml       # Configuration
        ├── prompts/          # Generated test prompts
        ├── results/          # Experimental results
        └── *.py             # Experiment scripts
```

## Experiments

### Experiment 1.A: Efficiency Analysis

**Goal**: Comprehensive evaluation of EAB's computational efficiency across different conditions

**Status**: 2/5 implemented

This experiment consists of 5 sub-experiments:

#### 1.A.1: Speedup vs Prompt Length ✅

**Research Question**: Does EAB efficiency increase with prompt length?

**Independent Variable**: Prompt length [50, 100, 200, 500 tokens]

**Fixed**: Model (GPT-2), Sample count (20), Domain (Factual QA)

**Status**: ✅ Implemented

**Location**: [`experiments/exp_1a_1_speedup_vs_prompt_length/`](experiments/exp_1a_1_speedup_vs_prompt_length/)

---

#### 1.A.2: Speedup vs Sample Count 📋

**Research Question**: How does EAB efficiency scale with number of samples?

**Independent Variable**: Sample count [5, 10, 20, 50]

**Fixed**: Model (GPT-2), Prompt length (200), Domain (Factual QA)

**Status**: 📋 Planned

**Location**: [`experiments/exp_1a_2_speedup_vs_sample_count/`](experiments/exp_1a_2_speedup_vs_sample_count/)

---

#### 1.A.3: Speedup vs Model Size 📋

**Research Question**: Does EAB provide greater speedup with larger models?

**Independent Variable**: Model size [GPT-2, GPT-2-Medium, GPT-2-Large, GPT-2-XL]

**Fixed**: Prompt length (200), Sample count (20), Domain (Factual QA)

**Status**: 📋 Planned

**Location**: [`experiments/exp_1a_3_speedup_vs_model_size/`](experiments/exp_1a_3_speedup_vs_model_size/)

---

#### 1.A.4: Speedup vs Domain 📋

**Research Question**: Does EAB efficiency vary across different domains?

**Independent Variable**: Domain [Factual QA, Creative Writing, Code Generation]

**Fixed**: Model (GPT-2), Prompt length (200), Sample count (20)

**Status**: 📋 Planned

**Location**: [`experiments/exp_1a_4_speedup_vs_domain/`](experiments/exp_1a_4_speedup_vs_domain/)

---

#### 1.A.5: Speedup vs Generation Length ✅

**Research Question**: Does EAB efficiency increase with longer generations?

**Independent Variable**: Generation length (max_new_tokens) [10, 30, 50, 100]

**Fixed**: Model (Qwen2.5-3B), Prompt length (200), Temperature (0.8)

**Status**: ✅ Implemented

**Location**: [`experiments/exp_1a_5_speedup_vs_generation_length/`](experiments/exp_1a_5_speedup_vs_generation_length/)

---

### Experiment 1.B: Quality Analysis (Planned)

**Research Question**: Does EAB maintain sample diversity despite sharing computation?

**Status**: 📋 Planned

---

### Experiment 1.C: Hyperparameter Sensitivity ✅

**Goal**: Analyze how EAB hyperparameters affect cost-quality tradeoff

**Status**: ✅ Implemented (4/4 sub-experiments)

This experiment analyzes the most impactful EAB hyperparameters:

#### 1.C.1: Branch Factor Sensitivity ✅

**Research Question**: How does branching width affect cost-quality tradeoff?

**Independent Variable**: branch_factor [2, 3, 5, 7]

**Fixed**: Model (Qwen2.5-3B), Prompt length (200), Threshold (0.055), max_paths (20)

**Status**: ✅ Implemented

**Location**: [`experiments/exp_1c_1_branch_factor_sensitivity/`](experiments/exp_1c_1_branch_factor_sensitivity/)

---

#### 1.C.2: Max Paths Sensitivity ✅

**Research Question**: How does path budget affect cost-quality balance?

**Independent Variable**: max_paths [5, 10, 20, 50]

**Fixed**: Model (Qwen2.5-3B), Prompt length (200), Threshold (0.055), branch_factor (3)

**Status**: ✅ Implemented

**Location**: [`experiments/exp_1c_2_max_paths_sensitivity/`](experiments/exp_1c_2_max_paths_sensitivity/)

---

#### 1.C.3: Temperature Sensitivity ✅

**Research Question**: How does sampling temperature interact with EAB?

**Independent Variable**: temperature [0.5, 0.7, 0.9, 1.1, 1.3]

**Fixed**: Model (Qwen2.5-3B), Prompt length (200), Threshold (0.055)

**Key Insight**: Temperature affects BOTH naive baseline AND EAB entropy distribution

**Status**: ✅ Implemented

**Location**: [`experiments/exp_1c_3_temperature_sensitivity/`](experiments/exp_1c_3_temperature_sensitivity/)

---

#### 1.C.4: Temperature × Threshold Interaction ✅

**Research Question**: How do temperature and threshold interact to control branching?

**Independent Variables**:
- temperature [0.6, 0.8, 1.0, 1.2]
- entropy_threshold [0.036, 0.055, 0.075, 0.100]

**Fixed**: Model (Qwen2.5-3B), Prompt length (200), branch_factor (3), max_paths (20)

**Design**: 2D grid search (4×4×5 prompts = 80 runs)

**Status**: ✅ Implemented

**Location**: [`experiments/exp_1c_4_temp_threshold_interaction/`](experiments/exp_1c_4_temp_threshold_interaction/)

---

## Shared Utilities

### Metrics (`utils/metrics.py`)

Track computational costs during generation:

```python
from utils.metrics import MetricsTracker

tracker = MetricsTracker(device="cuda")
tracker.start()

# ... run generation ...

tracker.record_token_steps(steps)
tracker.record_branch(position)
metrics = tracker.stop()

print(f"Token steps: {metrics.token_steps}")
print(f"Time: {metrics.wall_clock_time}s")
```

### Data Management (`utils/data_utils.py`)

Save and load experimental results:

```python
from utils.data_utils import save_results, load_results

# Save
save_results(results, output_dir="results")

# Load
results = load_results("results/raw_results.json")
```

### Plotting (`utils/plotting.py`)

Generate publication-quality figures:

```python
from utils.plotting import plot_speedup_vs_length

plot_speedup_vs_length(
    summary,
    output_path="figures/speedup.png"
)
```

---

## Running Experiments

### Quick Start (Debug Mode)

Test the full pipeline on a small scale:

```bash
cd experiments/exp_1a_speedup_vs_prompt_length

# 1. Edit config.yaml: set debug.enabled = true

# 2. Generate prompts
python prompts/generate_prompts.py

# 3. Run experiment (2 lengths × 2 prompts = 4 runs)
python run_experiment.py

# 4. Analyze
python analyze_results.py

# 5. Plot
python plot_results.py
```

### Full Experiment

```bash
# 1. Edit config.yaml: set debug.enabled = false

# 2. Generate prompts
python prompts/generate_prompts.py

# 3. Run full experiment (4 lengths × 10 prompts = 40 runs)
python run_experiment.py

# 4. Analyze
python analyze_results.py

# 5. Plot
python plot_results.py
```

---

## Adding New Experiments

To add a new experiment:

1. **Create experiment directory**:
   ```bash
   mkdir -p experiments/exp_XX_name/{prompts,results/figures}
   ```

2. **Create required files**:
   - `README.md`: Document research question, hypothesis, methodology
   - `config.yaml`: All experimental parameters
   - `run_experiment.py`: Main experiment runner
   - `analyze_results.py`: Statistical analysis
   - `plot_results.py`: Generate figures

3. **Use shared utilities**:
   ```python
   import sys
   from pathlib import Path
   sys.path.insert(0, str(Path(__file__).parent.parent.parent))

   from utils.metrics import MetricsTracker
   from utils.data_utils import save_results
   from utils.plotting import plot_speedup_vs_length
   ```

4. **Update this README** with experiment details

---

## Best Practices

### Reproducibility
- Always set random seeds in config
- Save full configuration with results
- Version control all code and configs

### Incremental Saving
- Save results after each prompt (don't lose data!)
- Use `append_result()` for incremental saves

### Debug Mode
- Always test with debug mode first
- Use small sample sizes to catch bugs early

### Documentation
- Document research questions clearly
- Include expected results and hypotheses
- Note any deviations from plan

---

## Dependencies

Required packages:
- `torch`
- `transformers`
- `numpy`
- `scipy`
- `matplotlib`
- `psutil` (for memory tracking)
- `tqdm` (for progress bars)
- `pyyaml`

Install:
```bash
pip install torch transformers numpy scipy matplotlib psutil tqdm pyyaml
```

---

## Contributing

When adding new experiments:
1. Follow the existing structure
2. Reuse shared utilities when possible
3. Document thoroughly
4. Test in debug mode first
5. Update this README

---

*Last updated: 2025-01-01*
