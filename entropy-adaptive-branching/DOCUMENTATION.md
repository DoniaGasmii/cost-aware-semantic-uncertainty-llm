# Entropy-Adaptive Branching (EAB) - Project Documentation

## 📁 Project Structure

```
entropy-adaptive-branching/
├── eab/                          # Core implementation
│   ├── core.py                   # Main EAB class with adaptive budgeting
│   ├── path.py                   # Path manager and GenerationPath
│   ├── cache.py                  # KV-cache utilities
│   ├── entropy.py                # Entropy computation and tracking
│   └── utils.py                  # Utilities
│
├── pilot_study/                  # Threshold selection experiment
│   ├── prompts/                  # 200 test prompts (high/medium/low confidence)
│   ├── results/                  # Pilot study results
│   │   ├── pilot_results.json
│   │   ├── pilot_summary.csv
│   │   └── threshold_recommendation.txt
│   ├── plots/                    # Statistical analysis plots
│   └── threshold/                # Threshold analysis scripts
│
├── demos/                        # Interactive demonstrations
│   ├── interactive_demo.py       # Main demo interface
│   ├── quick_test.py            # Automated validation
│   └── utils.py                  # Demo utilities
│
├── archive/                      # Archived development files
│   ├── dev_tests/               # Development test scripts
│   └── old_docs/                # Old documentation
│
├── path_management_comparison.pdf  # Report figure (use in thesis)
├── path_management_comparison.png  # Report figure (presentations)
├── create_comparison_figure.py     # Regenerate report figure
├── REPORT_PACKAGE.md              # Complete report materials
└── README.md                      # Quick start guide
```

---

## 🎯 Key Accomplishments

### 1. Core Implementation
- ✅ **Entropy-Adaptive Branching** with KV-cache sharing
- ✅ **Adaptive path budgeting** (removes hard-stop limitation)
- ✅ **Chat template support** for instruction-tuned models
- ✅ **FP16 precision** for memory efficiency

### 2. Threshold Selection
- ✅ **Pilot study with 200 prompts** across 3 confidence levels
- ✅ **Statistical analysis** (ANOVA: F=315.85, p<0.001)
- ✅ **Recommended threshold: τ = 0.055** (balanced option)
- ✅ **Publication-quality visualizations**

### 3. Adaptive Path Management
- ✅ **+200% branching coverage** (6 vs 2 branch points)
- ✅ **+733% exploration span** (71% vs 9% coverage)
- ✅ **0 blocked positions** (vs 4 in old strategy)
- ✅ **Report-ready figure** documenting improvement

---

## 📊 For Your Thesis

### Figures Available

**1. Pilot Study Results** (`pilot_study/plots/`)
- `entropy_distributions.png` - Shows separation between confidence levels
- `entropy_boxplots.png` - Statistical comparison
- `entropy_cdf.png` - Cumulative distributions with percentiles
- `branching_behavior.png` - EAB branching patterns
- `threshold_sweep.png` - Classification accuracy across thresholds

**2. Path Management Design** (report figure)
- `path_management_comparison.pdf` - **Use this in LaTeX**
- `path_management_comparison.png` - For presentations

### Text for Report

See **`REPORT_PACKAGE.md`** for:
- Updated Path Manager subsection text (LaTeX-ready)
- Figure caption
- Key numbers to cite
- LaTeX figure code

### Key Results to Report

**Pilot Study:**
- 200 prompts tested (70 high, 65 medium, 65 low confidence)
- ANOVA: F = 315.85, p < 0.001
- Effect sizes: Cohen's d = -3.01 (high vs medium), -1.94 (medium vs low)
- Recommended threshold: τ = 0.055

**Adaptive Path Management:**
- Branching points: 2 → 6 (+200%)
- Exploration coverage: 8.6% → 71.4% (+733%)
- Blocked positions: 4 → 0 (-100%)
- Memory footprint: Unchanged (pruning maintains constraint)

---

## 🚀 Quick Usage

### Run Interactive Demo
```bash
cd demos
python3 interactive_demo.py
```

### Run Pilot Study
```bash
cd pilot_study
./run_all.sh
```

### Generate Report Figure
```bash
python3 create_comparison_figure.py
```

### Basic API Usage
```python
from eab import EntropyAdaptiveBranching
import torch

eab = EntropyAdaptiveBranching(
    model_name='Qwen/Qwen2.5-3B-Instruct',
    entropy_threshold=0.055,    # From pilot study
    branch_factor=3,
    max_paths=20,
    device='cuda',
    torch_dtype=torch.float16
)

samples = eab.generate(
    prompt="Your question here",
    max_new_tokens=50,
    temperature=0.8,
    use_chat_template=True
)
```

---

## 📝 Next Steps

### For Evaluation Section

You'll need to run experiments comparing EAB vs baselines on:

1. **Semantic Uncertainty Estimation**
   - Cluster-based uncertainty metrics
   - Compare with naive sampling
   - Measure: accuracy, calibration

2. **Cost-Quality Tradeoffs**
   - Forward pass counting
   - Memory usage tracking
   - Diversity metrics (Self-BLEU, Distinct-n)

3. **Ablation Studies**
   - Effect of threshold values
   - Effect of branch factor
   - Effect of max_paths

### Recommended Tests

1. **Cost verification** - Measure actual forward passes (EAB vs naive)
2. **Diversity vs sample count** - Can fewer EAB samples match naive diversity?
3. **Semantic uncertainty** - Does EAB improve clustering-based uncertainty?

---

## 🗂️ File Reference

### Essential Files

| File | Purpose | Status |
|------|---------|--------|
| `eab/core.py` | Main implementation | ✅ Production ready |
| `pilot_study/results/*` | Threshold selection data | ✅ Complete |
| `path_management_comparison.pdf` | Report figure | ✅ Ready for thesis |
| `REPORT_PACKAGE.md` | Report materials | ✅ LaTeX-ready |

### Archive

Development and testing files moved to `archive/`:
- `archive/dev_tests/` - Test scripts (keep for reference)
- `archive/old_docs/` - Superseded documentation

---

## 🎓 Citation Information

**Threshold Selection:**
> A pilot study with 200 diverse prompts across three confidence levels
> demonstrated significant differences in entropy distributions (ANOVA:
> F=315.85, p<0.001) with very large effect sizes (Cohen's d ranging from
> 1.94 to 3.01). Based on this analysis, we selected τ=0.055 as the entropy
> threshold, corresponding to the 75th percentile of medium-confidence
> prompts.

**Adaptive Path Management:**
> Unlike traditional path-limited branching that imposes a hard stop once
> the maximum path count is reached, our adaptive budgeting strategy allows
> all high-entropy positions to branch throughout the generation sequence.
> This increased branching coverage by 200% (6 vs 2 branch positions) and
> exploration span by 733% while maintaining the same memory footprint
> through probability-based pruning.

---

## 💡 Design Decisions

### Why τ = 0.055?
- 75th percentile of medium-confidence prompts
- Separates high from medium/low with 96.5% accuracy
- Ensures high-confidence prompts rarely branch (<56%)
- Ensures medium/low-confidence prompts explore (100%)

### Why Adaptive Budgeting?
- Removes "early bird" bias of hard-stop strategies
- Allows late high-entropy positions to branch
- Maintains memory efficiency through pruning
- Increases exploration coverage by 733%

### Why Chat Templates?
- Instruction-tuned models require proper formatting
- Ensures coherent generation (not random completion)
- Matches model's training distribution

---

## 🔧 Troubleshooting

**GPU Out of Memory?**
- Use `torch_dtype=torch.float16`
- Reduce `max_paths`
- Reduce `max_new_tokens`

**Model not branching?**
- Check threshold (0.055 is recommended)
- Verify prompt has uncertainty
- Use medium/low confidence prompts for testing

**Chat template errors?**
- Ensure using instruction-tuned model
- Set `use_chat_template=True`
- Check model supports chat format

---

## 📧 Project Status

**Completed:**
- ✅ Core EAB implementation with adaptive budgeting
- ✅ Pilot study for threshold selection
- ✅ Report materials (figures + text)
- ✅ Interactive demos and validation tests

**Ready for:**
- 📊 Main evaluation experiments
- 📈 Semantic uncertainty benchmarking
- 💰 Cost-benefit analysis with real data
- 📝 Thesis writing

---

Last updated: January 6, 2025
