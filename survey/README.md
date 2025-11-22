# Survey: Response-Level Uncertainty Metrics for LLMs

A curated collection of state-of-the-art papers from top-tier conferences (NeurIPS, ICML, ICLR, ACL, EMNLP, NAACL) exploring response-level metrics that score LLM uncertainty.

## Table of Contents
- [Overview](#overview)
- [Methodology Categories](#methodology-categories)
- [Papers by Conference](#papers-by-conference)
- [Paper Reviews](#paper-reviews)
- [Resources](#resources)

---

## Overview

This survey focuses on methods that quantify uncertainty at the **response level** (entire generation) rather than token-level predictions. These methods are crucial for:
- Trustworthy AI systems that know when they don't know
- Detecting hallucinations and factual errors
- Improving calibration of model confidence
- Enabling safe deployment in high-stakes applications

---

## Methodology Categories

### 1. Verbalized Confidence
Models explicitly express uncertainty through natural language prompting.
- **Pros**: Black-box friendly, no model modification needed
- **Cons**: Tends to be overconfident, requires careful prompt engineering

### 2. Semantic Entropy & Clustering
Measures uncertainty by clustering semantically equivalent outputs.
- **Pros**: Accounts for linguistic variation, unsupervised
- **Cons**: Requires semantic similarity models, computational overhead

### 3. Self-Consistency
Samples multiple responses and measures agreement/consistency.
- **Pros**: Simple, effective for reasoning tasks
- **Cons**: Requires multiple generations, increased compute

### 4. Conformal Prediction
Provides statistical guarantees through prediction sets.
- **Pros**: Distribution-free, formal guarantees
- **Cons**: Requires calibration set, may produce large prediction sets

### 5. Token Probability Methods
Uses internal model probabilities (entropy, perplexity, MSP).
- **Pros**: Fine-grained, theoretically grounded
- **Cons**: Requires white-box access, not available for API models

### 6. Kernel & Similarity Methods
Uses semantic similarity kernels for fine-grained uncertainty.
- **Pros**: Captures pairwise dependencies, more nuanced
- **Cons**: Computationally intensive, requires embeddings

---

## Papers by Conference

### ICLR (International Conference on Learning Representations)

#### 1. Can LLMs Express Their Uncertainty? An Empirical Evaluation of Confidence Elicitation in LLMs
**Authors**: Miao Xiong et al.  
**Year**: ICLR 2024  
**Paper**: https://arxiv.org/abs/2306.13063  
**Code**: https://github.com/MiaoXiong2320/llm-uncertainty

**Key Contributions**:
- Systematic framework with three components: prompting strategies, sampling methods, aggregation techniques
- Benchmark on 5 LLMs (GPT-4, LLaMA 2) across multiple tasks
- Finding: LLMs are overconfident when verbalizing confidence

**Methods**: Verbalized confidence, self-consistency, aggregation strategies

**My Review**: 
*[...]*

---

#### 2. Semantic Uncertainty: Linguistic Invariances for Uncertainty Estimation in Natural Language Generation
**Authors**: Lorenz Kuhn et al.  
**Year**: ICLR 2023  
**Paper**: https://arxiv.org/abs/2302.09664  
**Code**: https://github.com/lorenzkuhn/semantic_uncertainty

**Key Contributions**:
- Introduces semantic entropy: entropy incorporating linguistic invariances
- Clusters semantically equivalent outputs using unsupervised methods
- Works with single model, no modifications needed

**Methods**: Semantic clustering, entropy calculation

**My Review**: 
*[...]*

---

#### 3. Self-Consistency Improves Chain of Thought Reasoning in Language Models
**Authors**: Xuezhi Wang et al.  
**Year**: ICLR 2023  
**Paper**: https://arxiv.org/abs/2203.11171  
**Code**: https://github.com/huggingface/transformers/tree/main/examples/research_projects/self-consistency

**Key Contributions**:
- Samples diverse reasoning paths and marginalizes to find consistent answer
- Significantly improves reasoning performance
- Simple yet effective decoding strategy

**Methods**: Self-consistency, sampling, majority voting

**My Review**: 
*[...]*

---

#### 4. Self-contradictory Hallucinations of Large Language Models: Evaluation, Detection and Mitigation
**Authors**: Various  
**Year**: ICLR 2024  
**Paper**: [Add link]  
**Code**: [Add link if available]

**Key Contributions**:
- Detects hallucinations through self-consistency checking
- Focuses on logical contradictions within model outputs

**Methods**: Self-consistency, contradiction detection

**My Review**: 
*[...]*

---

### NeurIPS (Neural Information Processing Systems)

#### 5. To Believe or Not to Believe Your LLM: Iterative Prompting for Estimating Epistemic Uncertainty
**Authors**: Various  
**Year**: NeurIPS 2024  
**Paper**: [Add link]  
**Code**: [Add link if available]

**Key Contributions**:
- Information-theoretic metric for epistemic uncertainty
- Iterative prompting based on previous responses
- Detects when only epistemic uncertainty is large

**Methods**: Iterative prompting, information theory

**My Review**: 
*[...]*

---

#### 6. Kernel Language Entropy: Fine-grained Uncertainty Quantification for LLMs from Semantic Similarities
**Authors**: Various  
**Year**: NeurIPS 2024  
**Paper**: [Add link]  
**Code**: [Add link if available]

**Key Contributions**:
- Defines positive semidefinite kernels for semantic similarities
- Uses von Neumann entropy for uncertainty quantification
- Considers pairwise semantic dependencies

**Methods**: Kernel methods, von Neumann entropy

**My Review**: 
*[...]*

---

#### 7. Benchmarking LLMs via Uncertainty Quantification
**Authors**: Various  
**Year**: NeurIPS 2024  
**Paper**: [Add link]  
**Code**: [Add link if available]

**Key Contributions**:
- Higher accuracy may show lower certainty
- Larger models may show greater uncertainty
- Instruction-finetuning increases uncertainty

**Methods**: Multiple UQ metrics, benchmarking

**My Review**: 
*[...]*

---

### ACL / EMNLP / NAACL (Computational Linguistics)

#### 8. Seeing is Believing, but How Much? A Comprehensive Analysis of Verbalized Calibration in Vision-Language Models
**Authors**: Various  
**Year**: EMNLP 2025  
**Paper**: [Add link]  
**Code**: [Add link if available]

**Key Contributions**:
- Evaluates verbalized confidence in VLMs
- VLMs show notable miscalibration across tasks
- Visual reasoning models have better calibration

**Methods**: Verbalized confidence, calibration metrics

**My Review**: 
*[...]*

---

#### 9. Strategies for Eliciting Calibrated Confidence Scores from Language Models Fine-Tuned with Human Feedback
**Authors**: Various  
**Year**: EMNLP 2023  
**Paper**: [Add link]  
**Code**: [Add link if available]

**Key Contributions**:
- Verbalized confidence better calibrated than conditional probabilities
- Temperature and prompting affect calibration
- Studies RLHF models specifically

**Methods**: Verbalized confidence, temperature scaling

**My Review**: 
*[...]*

---

#### 10. Uncertainty in Language Models: Assessment through Rank-Calibration
**Authors**: Various  
**Year**: EMNLP 2024  
**Paper**: [Add link]  
**Code**: [Add link if available]

**Key Contributions**:
- Novel Rank-Calibration framework
- Higher uncertainty should imply lower generation quality
- New assessment methodology

**Methods**: Rank-calibration, quality assessment

**My Review**: 
*[...]*

---

#### 11. Beyond Semantic Entropy: Boosting LLM Uncertainty Quantification with Pairwise Semantic Similarity
**Authors**: Various  
**Year**: ACL 2025 Findings  
**Paper**: [Add link]  
**Code**: [Add link if available]

**Key Contributions**:
- Black-box UQ method inspired by nearest neighbor entropy
- Considers intra-cluster and inter-cluster similarity
- Better for long one-sentence responses

**Methods**: Semantic similarity, nearest neighbor

**My Review**: 
*[...]*

---

#### 12. API Is Enough: Conformal Prediction for Large Language Models Without Logit-Access
**Authors**: Various  
**Year**: EMNLP 2024 Findings  
**Paper**: [Add link]  
**Code**: [Add link if available]

**Key Contributions**:
- Conformal prediction for API-only LLMs
- Uses sample frequency and semantic similarity
- No logit access required

**Methods**: Conformal prediction, semantic similarity

**My Review**: 
*[...]*

---

#### 13. ConU: Conformal Uncertainty in Large Language Models with Correctness Coverage Guarantees
**Authors**: Various  
**Year**: EMNLP 2024 Findings  
**Paper**: [Add link]  
**Code**: [Add link if available]

**Key Contributions**:
- Novel uncertainty measure based on self-consistency
- Conformal prediction with correctness guarantees
- Integrates uncertainty with conformity

**Methods**: Conformal prediction, self-consistency

**My Review**: 
*[...]*

---

### ICML (International Conference on Machine Learning)

#### 14. Linguistic Calibration of Long-Form Generations
**Authors**: Various  
**Year**: ICML 2024  
**Paper**: [Add link]  
**Code**: [Add link if available]

**Key Contributions**:
- Addresses calibration for long-form text
- Specific challenges for extended generations

**Methods**: Calibration techniques, long-form evaluation

**My Review**: 
*[...]*

---

### Cross-Venue / Other Notable Papers

#### 15. Benchmarking Uncertainty Quantification Methods for Large Language Models with LM-Polygraph
**Authors**: Various  
**Year**: TACL 2024  
**Paper**: [Add link]  
**Code**: https://github.com/IINemo/lm-polygraph

**Key Contributions**:
- Unified implementation of UQ baselines
- Large-scale consistent comparison
- Token-level and sequence-level methods

**Methods**: Multiple (MSP, entropy, perplexity, etc.)

**My Review**: 
*[...]*

---

#### 16. CritiCal: Can Critique Help LLM Uncertainty or Confidence Calibration?
**Authors**: Various  
**Year**: 2024 (arXiv)  
**Paper**: [Add link]  
**Code**: [Add link if available]

**Key Contributions**:
- Natural language critiques enhance verbalized confidence
- Studies what to critique and how to critique
- Self-critique and critique calibration training

**Methods**: Critique-based calibration

**My Review**: 
*[...]*

---

#### 17. Systematic Evaluation of Uncertainty Estimation Methods in Large Language Models
**Authors**: Various  
**Year**: 2024 (arXiv)  
**Paper**: [Add link]  
**Code**: [Add link if available]

**Key Contributions**:
- Comprehensive comparison of UQ methods
- Verbalized confidence tends to be overestimated
- Systematic evaluation framework

**Methods**: Multiple UQ methods

**My Review**: 
*[...]*

---

#### 18. Conformal Prediction for Natural Language Processing: A Survey
**Authors**: Various  
**Year**: TACL 2024  
**Paper**: [Add link]  
**Code**: [Add link if available]

**Key Contributions**:
- Comprehensive survey of conformal prediction
- Strong statistical guarantees
- Model-agnostic and distribution-free

**Methods**: Conformal prediction (survey)

**My Review**: 
*[...]*

---

## Resources

### Implementations & Tools
- **LM-Polygraph**: https://github.com/IINemo/lm-polygraph - Unified UQ framework
- **Semantic Uncertainty**: https://github.com/lorenzkuhn/semantic_uncertainty
- **LLM Uncertainty**: https://github.com/MiaoXiong2320/llm-uncertainty

### Related Surveys


### Datasets for Evaluation



---

## How Different Methods Compare

| Method | White-box? | Compute Cost | Calibration Quality | Best For |
|--------|-----------|--------------|---------------------|----------|
| Verbalized Confidence | No | Low | Moderate (overconfident) | API models, quick estimates |
| Semantic Entropy | No | Medium | Good | Factual QA, clustering-friendly |
| Self-Consistency | No | High | Good | Reasoning tasks |
| Conformal Prediction | No | Medium | Excellent (guaranteed) | Safety-critical apps |
| Token Probabilities | Yes | Low | Good | White-box models |
| Kernel Methods | No | High | Very Good | Fine-grained analysis |

---

## Open Questions & Future Directions

1. **Scalability**: How to make semantic methods more efficient?
2. **Long-form text**: Most methods tested on short responses
3. **Cross-task generalization**: Methods often task-specific
4. **Multimodal uncertainty**: Extending to vision-language models
5. **Theoretical foundations**: Connecting empirical methods to theory

---

## Citation

If you use this survey in your research, please cite:

```bibtex
@misc{yourname_survey_repo_2025,
  author       = {Donia Gasmi},
  title        = {Response-Level Uncertainty Metrics for LLMs},
  year         = {2025},
  howpublished = {\url{https://github.com/DoniaGasmii/llm-semantic-uncertainty}},
  note         = {Version 1.0, accessed 22 November 2025}
}

---

**Last Updated**: [22.11.2025]  
**Maintainer**: [Donia Gasmi]