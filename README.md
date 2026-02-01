# Entropy-Adaptive Branching for Efficient Semantic Uncertainty Estimation

**Master's Semester Project** (12 ECTS) | EPFL | Fall 2024-Spring 2025

**Author**: Donia Gasmi

**Supervisors**: Dr. Geovani Rizk & Dr. Gauthier Voron (DCL Lab, EPFL)

**Professor**: Prof. Rachid Guerraoui

---

## Overview

This project addresses the computational challenge of uncertainty quantification in Large Language Models (LLMs). Existing methods generate multiple diverse samples to measure semantic disagreement, but scale linearly in cost, redundantly encoding the same prompt for each sample.

We introduce **Entropy-Adaptive Branching (EAB)**, a sampling algorithm that generates diverse outputs by branching only at high-entropy token positions. EAB encodes the prompt once and dynamically creates new generation paths when the model exhibits genuine uncertainty, achieving up to **11.93× speedup** for long prompts through KV-cache sharing.

**Key Results**:
- **Computational Efficiency**: 4.48× average token-step speedup over sequential sampling
- **Human Alignment**: Branching behavior strongly correlates with human ambiguity ratings (Spearman ρ = 0.837)
- **Sample Diversity**: Produces more lexically diverse outputs than naive sampling (lower Self-BLEU, higher Distinct-n)
- **Semantic Entropy Validation**: Distinguishes correct from incorrect answers with AUROC = 0.666, with 28.7 percentage point accuracy gap between unanimous (1 cluster) and fragmented (4+ clusters) responses

**Integration Challenge**: While both EAB and Semantic Entropy (SE) perform well independently, their naive combination degrades uncertainty estimation quality (AUROC drops from 0.666 to 0.61), revealing fundamental architectural constraints between token-level adaptive branching and clustering-based semantic analysis.

---

## Full Report

See [semester_project_report_dcl.pdf](semester_project_report_dcl.pdf) for complete methodology, experiments, and analysis.

---

## Quick Start

```bash
# Clone repository
git clone https://github.com/DoniaGasmii/cost-aware-semantic-uncertainty-llm.git
cd cost-aware-semantic-uncertainty-llm

# Install dependencies
pip install -r requirements.txt
```

```python
from eab import EntropyAdaptiveBranching

# Initialize EAB for efficient diverse sampling
eab = EntropyAdaptiveBranching(
    model_name="Qwen/Qwen2.5-3B-Instruct",
    entropy_threshold=0.055,
    branch_factor=3,
    max_paths=20
)

# Generate adaptive samples (branches only when uncertain)
prompt = "What is the capital of France?"
samples = eab.generate(prompt, max_new_tokens=50)

print(f"Generated {len(samples)} diverse samples")
```

---

## Project Structure

```
cost-aware-semantic-uncertainty-llm/
├── entropy-adaptive-branching/     # Layer 1: Efficient generation
├── semantic-entropy/               # Layer 2: Uncertainty measurement
├── integration/                    # Full pipeline & experiments
└── semester_project_report_dcl.pdf # Complete report
```

---

## Citation

```bibtex
@mastersthesis{gasmi2025eab,
  title={Entropy-Adaptive Branching for Efficient Semantic Uncertainty Estimation},
  author={Gasmi, Donia},
  year={2025},
  school={EPFL},
  type={Master's Semester Project}
}
```

---

## License

MIT License
