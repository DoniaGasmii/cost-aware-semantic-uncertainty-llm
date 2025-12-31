# Cost-Aware Semantic Uncertainty for Large Language Models

**A comprehensive framework for efficient uncertainty quantification in LLMs through entropy-adaptive branching and semantic clustering.**

---

## Overview

This project addresses a fundamental challenge in LLM uncertainty quantification: **generating multiple diverse samples is computationally expensive**. We propose a two-component solution:

1. **Entropy-Adaptive Branching (EAB)**: Efficient multi-sample generation that branches only when the model is uncertain, reusing computation for shared token sequences.
2. **Semantic Entropy**: Meaning-level uncertainty quantification that clusters generations by semantic similarity rather than surface-form matching.

**Key Innovation**: By combining EAB with semantic entropy, we achieve high-quality uncertainty estimates at a fraction of the computational cost of naive multi-sample generation.

---

## System Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                     COST-AWARE SEMANTIC UNCERTAINTY             │
└─────────────────────────────────────────────────────────────────┘
                                  │
                ┌─────────────────┴─────────────────┐
                │                                   │
    ┌───────────▼──────────┐          ┌────────────▼─────────────┐
    │  Component 1: EAB    │          │ Component 2: Semantic    │
    │  (Generation)        │─────────▶│ Entropy (Measurement)    │
    │                      │          │                          │
    │ • KV-cache reuse     │          │ • Sentence embeddings    │
    │ • Entropy-based      │          │ • Adaptive clustering    │
    │   branching          │          │ • Uncertainty scores     │
    │ • ~50% cost savings  │          │                          │
    └──────────────────────┘          └──────────────────────────┘
                │                                   │
                └─────────────────┬─────────────────┘
                                  │
                      ┌───────────▼──────────────┐
                      │  Component 3:            │
                      │  Integration & Ablation  │
                      │                          │
                      │ • Full pipeline          │
                      │ • Empirical evaluation   │
                      │ • Ablation studies       │
                      └──────────────────────────┘
```

---

## Project Structure

```
cost-aware-semantic-uncertainty-llm/
│
├── README.md                                  # This file (main documentation)
├── requirements.txt                           # Top-level dependencies
│
├── entropy-adaptive-branching/                # Component 1: Efficient generation
│   ├── README.md                             # EAB-specific documentation
│   ├── eab/                                  # Core implementation
│   ├── examples/                             # Usage examples
│   └── tests/                                # Unit tests
│
├── semantic-entropy-based-uncertainty/        # Component 2: Uncertainty measurement
│   ├── README.md                             # Semantic entropy documentation
│   ├── semantic_entropy/                     # Core implementation
│   ├── examples/                             # Usage examples
│   └── tests/                                # Unit tests
│
└── integration/                               # Component 3: Full system
    ├── README.md                             # Integration documentation
    ├── pipeline/                             # Unified pipeline
    ├── experiments/                          # Ablation studies & evaluation
    ├── examples/                             # End-to-end demos
    └── results/                              # Experimental results
```

---

## Research Questions & Experimental Framework

### 1. Entropy-Adaptive Branching (EAB) - Efficiency Focus

#### Core Research Questions

1. **RQ1.1**: How much does EAB reduce computational cost compared to naive sampling?
   - **Hypothesis**: EAB should provide 40-60% cost reduction with long prompts
   - **Null hypothesis**: No significant difference in computational cost

2. **RQ1.2**: Does EAB maintain sample diversity/quality despite sharing computation?
   - **Hypothesis**: EAB samples are equally diverse (measured by Self-BLEU, embedding coverage)
   - **Concern**: Shared computation might reduce diversity

3. **RQ1.3**: What factors determine EAB's efficiency gains?
   - **Variables**: Prompt length, model uncertainty, domain, temperature
   - **Question**: When does EAB excel vs struggle?

4. **RQ1.4**: What's the cost-quality tradeoff across hyperparameters?
   - **Explore**: entropy_threshold × branch_factor × max_paths
   - **Find**: Pareto-optimal configurations

#### Experiments & Metrics

##### Experiment 1.A: Efficiency Analysis

**Setup**:
```python
Models: [GPT-2, GPT-2-Large, GPT-J-6B]
Prompt lengths: [50, 100, 200, 500 tokens]
Sample counts: [5, 10, 20, 50]
Domains: [Factual QA, Creative Writing, Code Generation]
```

**Metrics**:
- Total token-steps (FLOPs proxy)
- Wall-clock time (seconds)
- Memory usage (GB, peak)
- Tokens computed per sample
- Branching frequency (branches per generation)

**Expected Graphs**:
1. **Speedup vs Prompt Length**: Line plot showing speedup factor (y) vs prompt length (x)
   - *Hypothesis*: Linear or super-linear growth (longer prompts = more shared computation)
2. **Cost Breakdown**: Stacked bar chart comparing naive vs EAB
   - Bars: [Shared tokens | Divergent tokens]
3. **Branching Timeline**: Heatmap showing when branching occurs during generation
   - X-axis: Token position, Y-axis: Different prompts, Color: Branch frequency
4. **Scaling Curves**: Log-log plot of time/cost vs number of samples
   - *Expected*: Naive = linear, EAB = sublinear

##### Experiment 1.B: Quality Analysis

**Setup**:
```python
Generate 20 samples per prompt using:
- Naive sampling (temperature=0.8)
- EAB (entropy_threshold=0.4, branch_factor=3)

Compare on 100 diverse prompts
```

**Metrics**:
- **Sample Diversity**:
  - Self-BLEU (lower = more diverse)
  - Distinct n-grams ratio
- **Semantic Coverage**:
  - Embedding space volume (convex hull in t-SNE)
  - Pairwise cosine distance (mean, std)
- **Probability Mass Coverage**:
  - Entropy of sample distribution
  - Top-k probability mass captured

**Expected Graphs**:
1. **Diversity Comparison**: Box plots of Self-BLEU scores (Naive vs EAB)
   - *Hypothesis*: No significant difference
2. **Embedding Space Visualization**: Side-by-side t-SNE plots
   - Left: Naive samples, Right: EAB samples
   - Colors: Different generations from same prompt
3. **Coverage vs Cost**: Scatter plot with Pareto frontier
   - X-axis: Cost (token-steps), Y-axis: Coverage metric

##### Experiment 1.C: Hyperparameter Sensitivity

**Setup**:
```python
Grid search:
  entropy_threshold: [0.2, 0.3, 0.4, 0.5, 0.6, 0.7]
  branch_factor: [2, 3, 4, 5]
  max_paths: [10, 20, 50, unlimited]

Evaluate: Cost, diversity, quality
```

**Expected Graphs**:
1. **Threshold vs Branching Frequency**: Line plot showing how often branching occurs
2. **Pareto Frontier**: 2D plot of cost vs diversity
   - Each point = one hyperparameter configuration
   - Highlight dominated/non-dominated points
3. **Heatmap**: entropy_threshold (rows) × branch_factor (cols) → speedup (color)
4. **Sensitivity Analysis**: Tornado plot showing which parameter affects cost/quality most

---

### 2. Semantic Entropy - Uncertainty Quality Focus

#### Core Research Questions

1. **RQ2.1**: Is semantic entropy well-calibrated?
   - **Hypothesis**: High entropy → model is likely wrong (AUROC > 0.7)
   - **Calibration**: Expected calibration error (ECE) < 0.1

2. **RQ2.2**: Does semantic entropy outperform token-level uncertainty metrics?
   - **Baselines**: Token entropy, perplexity, self-consistency, variance
   - **Hypothesis**: Semantic entropy has better calibration and error detection

3. **RQ2.3**: Can semantic entropy detect hallucinations and factual errors?
   - **Hypothesis**: Hallucinated answers have higher semantic entropy
   - **Use case**: Hallucination detection (precision/recall)

4. **RQ2.4**: How robust is semantic entropy to hyperparameters?
   - **Concern**: Does distance_threshold significantly affect results?
   - **Desired**: Robust performance across reasonable threshold range

#### Experiments & Metrics

##### Experiment 2.A: Calibration Analysis

**Setup**:
```python
Datasets:
  - TriviaQA (factual, single answer)
  - NaturalQuestions (factual, short answer)
  - AmbigQA (ambiguous questions)

For each question:
  1. Generate N=20 samples
  2. Compute semantic entropy
  3. Check correctness against ground truth
```

**Metrics**:
- **Calibration**:
  - Expected Calibration Error (ECE)
  - Maximum Calibration Error (MCE)
  - Brier Score
- **Discrimination**:
  - AUROC for correctness prediction
  - AUPRC (precision-recall curve)
- **Correlation**:
  - Spearman correlation: entropy ↔ accuracy
  - Pearson correlation: entropy ↔ human confidence ratings (if available)

**Expected Graphs**:
1. **Calibration Curve**: Predicted uncertainty (x) vs actual error rate (y)
   - *Perfect calibration*: y = x line
   - Include confidence intervals
2. **Reliability Diagram**: Binned version of calibration curve
   - Bars: Predicted vs actual error rate per bin
3. **Entropy Distributions**: Overlapping histograms
   - Blue: Correct answers, Red: Incorrect answers
   - *Hypothesis*: Red should be right-shifted (higher entropy)
4. **ROC Curve**: True positive rate vs false positive rate
   - For detecting incorrect answers using entropy threshold
   - Report AUROC in legend

##### Experiment 2.B: Comparison with Baselines

**Setup**:
```python
Baselines:
  1. Token-level entropy (mean over sequence)
  2. Perplexity (exp of cross-entropy)
  3. Self-consistency (exact string match fraction)
  4. Variance in log-probabilities
  5. Longest common substring ratio

Compare all methods on same data
```

**Expected Graphs**:
1. **Calibration Comparison**: Side-by-side calibration curves for all methods
2. **ROC Comparison**: All methods on same ROC plot
   - Semantic entropy should dominate (higher AUROC)
3. **Metric Comparison Table**: ECE, AUROC, AUPRC for each method
4. **Confusion Matrix Heatmap**: When do different methods agree/disagree?
   - Rows/Cols: Methods, Color: Agreement rate

##### Experiment 2.C: Clustering Analysis

**Setup**:
```python
Research questions:
  - Are clusters semantically coherent?
  - Does distance threshold affect results?
  - Which encoder works best?

Ablations:
  distance_threshold: [0.10, 0.15, 0.20, 0.25]
  encoder: ['all-mpnet-base-v2', 'all-MiniLM-L6-v2',
            'paraphrase-multilingual-mpnet-base-v2']
```

**Metrics**:
- Silhouette score (cluster quality)
- Cluster coherence (manual inspection)
- Entropy sensitivity to threshold
- Inter-cluster distance (separation)

**Expected Graphs**:
1. **Cluster Coherence Visualization**: t-SNE with cluster labels and boundaries
   - Manually inspect: Do clusters make semantic sense?
2. **Threshold Sensitivity**: Entropy (y) vs distance_threshold (x)
   - *Desired*: Flat line (robust to threshold)
3. **Cluster Size Distribution**: Histogram of samples per cluster
4. **Silhouette Analysis**: Box plots of silhouette scores per threshold

##### Experiment 2.D: Error Type Analysis

**Setup**:
```python
Manually categorize errors into:
  1. Hallucinations (made-up facts)
  2. Contradictions (inconsistent multi-answer)
  3. Ambiguous questions (genuinely uncertain)
  4. Paraphrases (same meaning, different wording)
  5. Partial answers (incomplete but not wrong)

Analyze: Does entropy distinguish these?
```

**Expected Graphs**:
1. **Entropy by Error Type**: Box plots, one per category
   - *Hypothesis*: Hallucinations and contradictions have highest entropy
2. **Confusion Matrix**: True error type (rows) vs entropy bin (cols)
3. **Case Studies**: Show 3-5 examples from each category with entropy scores

---

### 3. Integration (EAB + Semantic Entropy) - Full System

#### Core Research Questions

1. **RQ3.1**: Does EAB+SE achieve comparable uncertainty quality to Naive+SE at lower cost?
   - **Primary hypothesis**: Quality maintained, cost reduced 40-60%
   - **Main ablation**: Naive vs EAB with same downstream analysis

2. **RQ3.2**: Do EAB's branching decisions align with semantic diversity?
   - **Hypothesis**: Model branches when meanings start to diverge
   - **Insight**: Does syntactic uncertainty (entropy) predict semantic uncertainty?

3. **RQ3.3**: What's the optimal cost-quality operating point?
   - **Explore**: Hyperparameter space of full pipeline
   - **Deliverable**: Recommended configurations for different use cases

4. **RQ3.4**: Can this enable practical selective prediction?
   - **Use case**: Only answer when confident (entropy < threshold)
   - **Metrics**: Coverage, accuracy on answered questions, cost efficiency

#### Experiments & Metrics

##### Experiment 3.A: Head-to-Head Ablation Study (PRIMARY EXPERIMENT)

**Setup**:
```python
Conditions:
  1. BASELINE: Naive sampling + Semantic Entropy
     - Generate N=20 samples independently
     - Temperature = 0.8

  2. PROPOSED: EAB + Semantic Entropy
     - Same final sample count (N=20)
     - entropy_threshold = 0.4
     - branch_factor = 3

Controlled variables:
  - Same prompts (500 from diverse domains)
  - Same semantic entropy parameters
  - Same evaluation metrics

Datasets:
  - TriviaQA (factual QA)
  - Creative Writing Prompts (open-ended)
  - HumanEval (code generation)
```

**Metrics**:

*Quality* (higher is better):
- Expected Calibration Error (ECE) ↓
- AUROC for correctness prediction ↑
- Correlation with ground truth ↑

*Cost* (lower is better):
- Total token-steps
- Wall-clock time
- Memory usage

*Efficiency* (derived):
- Quality per token-step
- Speedup factor at iso-quality

**Expected Graphs**:

1. **Cost-Quality Scatter Plot**:
   ```
   Y-axis: Quality metric (e.g., AUROC)
   X-axis: Cost metric (e.g., token-steps)

   Two points: Naive (top-right), EAB (top-left)
   Expected: EAB dominates (similar quality, lower cost)
   ```

2. **Pareto Frontier**:
   ```
   Include multiple configurations:
   - Naive: N ∈ {5, 10, 20, 50}
   - EAB: threshold ∈ {0.3, 0.4, 0.5}, N ∈ {5, 10, 20, 50}

   Plot convex hull of non-dominated points
   ```

3. **Side-by-Side Calibration Curves**:
   ```
   Left panel: Naive + SE
   Right panel: EAB + SE

   Both on same scale for comparison
   Report ECE in subplot titles
   ```

4. **Time-to-Quality**:
   ```
   Y-axis: Quality metric (e.g., AUROC)
   X-axis: Wall-clock time

   Show how quickly each method reaches target quality
   ```

5. **Per-Domain Breakdown**:
   ```
   Grouped bar chart:
   X-axis: Domain (Factual QA, Creative, Code)
   Y-axis: Speedup factor

   Shows where EAB excels most
   ```

##### Experiment 3.B: Branching-Diversity Alignment

**Research Question**: Do EAB's branching points (high token-level entropy) correspond to where semantic diversity emerges?

**Setup**:
```python
For each generation:
  1. Record where EAB branches (token positions)
  2. Compute semantic similarity at each position
     (how different are continuations from that point?)
  3. Correlate: branching frequency ↔ semantic divergence

Hypothesis:
  - High correlation → EAB branches when meanings diverge
  - This validates using token entropy as proxy for semantic uncertainty
```

**Metrics**:
- Correlation: branching frequency ↔ semantic divergence score
- Precision/Recall: Does branching predict divergence?
- Temporal alignment: How many tokens before divergence does branching occur?

**Expected Graphs**:

1. **Branching Heatmap**:
   ```
   X-axis: Token position in generation
   Y-axis: Different prompts (sorted by total branches)
   Color: Branching frequency at that position

   Overlay: Semantic divergence points (e.g., white crosses)

   Visual check: Do colors align with crosses?
   ```

2. **Case Study Visualizations**:
   ```
   Show 3-5 specific examples:

   Example 1: Perfect alignment
     - Branching tree diagram
     - Semantic clusters labeled
     - Highlight where branching occurred

   Example 2: Misalignment
     - Where branching occurred too early/late
     - Analysis of why
   ```

3. **Correlation Plot**:
   ```
   X-axis: Token-level entropy at position t
   Y-axis: Semantic similarity of continuations after t

   Each point = one (prompt, position) pair
   Fit regression line, report R²
   ```

4. **Alignment Score Distribution**:
   ```
   Histogram of alignment scores across all prompts

   Alignment = how often branching precedes semantic divergence
   ```

##### Experiment 3.C: Selective Prediction (Practical Application)

**Use Case**: Build a system that only answers when confident

**Setup**:
```python
Pipeline:
  1. Generate samples (Naive or EAB)
  2. Compute semantic entropy
  3. If entropy < threshold: Return answer
     Else: Abstain ("I don't know")

Sweep threshold to get coverage-accuracy curve
```

**Metrics**:
- **Coverage**: % of questions answered
- **Accuracy**: Accuracy on answered questions
- **Selective accuracy**: Accuracy gain vs answering everything
- **Cost per answered question**: Total cost / # answered
- **AUPRC**: Area under precision-recall curve

**Expected Graphs**:

1. **Coverage-Accuracy Curves**:
   ```
   X-axis: Coverage (% answered)
   Y-axis: Accuracy on answered

   Two curves: Naive, EAB
   Horizontal line: Baseline accuracy (answer everything)

   Higher is better (same coverage, higher accuracy)
   ```

2. **Cost-Coverage Tradeoff**:
   ```
   X-axis: Coverage (% answered)
   Y-axis: Total cost (token-steps)

   Two curves: Naive, EAB

   Shows: "To answer 80% of questions, EAB costs X, Naive costs Y"
   ```

3. **ROC-style Curve for Selective Prediction**:
   ```
   X-axis: False positive rate (incorrect answers given)
   Y-axis: True positive rate (correct answers given)

   Better methods dominate (top-left)
   ```

4. **Real-World Scenario Analysis**:
   ```
   Table or bar chart:

   Scenario: "Answer 80% with 95% accuracy"

   | Method | Cost | Time | Memory |
   |--------|------|------|--------|
   | Naive  |  X   |  Y   |   Z    |
   | EAB    |  X'  |  Y'  |   Z'   |
   ```

##### Experiment 3.D: Scaling Analysis

**Setup**:
```python
Questions:
  - Generalization across domains
  - Performance with different models
  - Scaling with dataset size

Datasets:
  - Factual QA: TriviaQA, NaturalQuestions (5k samples each)
  - Open-ended: WritingPrompts (1k samples)
  - Code: HumanEval (164 samples)
  - Medical: MedQA subset (1k samples)
  - Multi-lingual: XQUAD (if time permits)

Models:
  - GPT-2 (124M)
  - GPT-2-Large (774M)
  - GPT-J-6B (6B)
  - (Optional) Llama-7B if resources permit
```

**Expected Graphs**:

1. **Domain Generalization**:
   ```
   Heatmap:
   Rows: Domains
   Cols: Metrics (ECE, AUROC, Speedup)
   Color: Performance

   Shows: Where does EAB excel? Where does it struggle?
   ```

2. **Model Scaling**:
   ```
   X-axis: Model size (log scale)
   Y-axis: Speedup factor

   Question: Does EAB save more with larger models?
   Hypothesis: Yes (more computation to share)
   ```

3. **Cross-Domain Transfer**:
   ```
   Matrix plot:
   Rows: Training domain (threshold tuned on)
   Cols: Test domain
   Color: Performance drop

   Diagonal = perfect transfer
   Off-diagonal = generalization
   ```

---

## 4. Additional Analyses

### Failure Mode Analysis

**Questions**:
- When does EAB fail to save cost?
- When does semantic entropy fail to calibrate?
- Edge cases and limitations

**Setup**:
```python
Identify worst-case scenarios:
  - EAB: Already confident models (low branching)
  - SE: Encoder fails (e.g., code, math, multi-lingual)

Document failure patterns and mitigation strategies
```

**Expected Output**:
- Case studies of failures
- Quantification: "EAB saves <10% when X"
- Recommendations: "Use naive sampling when Y"

### Computational Cost Breakdown

**Detailed Analysis**:
```python
Profile each component:
  1. EAB overhead (entropy computation, cache copying)
  2. SE overhead (encoding, clustering)
  3. Actual generation time

Question: Where is time spent?
```

**Expected Graph**:
```
Stacked bar chart:
  Bars: Naive, EAB
  Segments: [Prompt encoding | Generation | Entropy comp | Clustering]

Shows: Where EAB saves time, where it adds overhead
```

### Qualitative Analysis

**Human Evaluation** (if resources permit):
```python
Sample 100 prompts
Show human raters:
  - Model uncertainty (semantic entropy score)
  - Model's answer

Ask: "Is the model's uncertainty appropriate?"

Metrics:
  - Human-model agreement
  - Precision/recall of uncertainty
```

---

## 5. Comprehensive Report Structure

### Proposed Organization

```
FINAL REPORT OUTLINE
====================

1. INTRODUCTION (3-4 pages)
   1.1 Motivation
       - Why uncertainty quantification matters
       - Why it's expensive (multi-sample generation)
   1.2 Problem Statement
       - Challenge: Cost vs quality tradeoff
       - Gap: Existing methods either fast OR high-quality
   1.3 Contributions
       - EAB: Efficient generation via adaptive branching
       - SE: Semantic-level uncertainty measurement
       - Integration: Full system with empirical validation
   1.4 Research Questions (list RQ1.1 - RQ3.4)

2. BACKGROUND & RELATED WORK (4-5 pages)
   2.1 Uncertainty Quantification in LLMs
       - Epistemic vs aleatoric uncertainty
       - Multi-sample methods
       - Single-forward methods
   2.2 Efficient LLM Inference
       - KV-cache optimization
       - Speculative decoding
       - Batching strategies
   2.3 Semantic Similarity & Clustering
       - Sentence transformers
       - Clustering methods for text
   2.4 Positioning
       - How our work differs/complements prior art

3. METHOD (6-8 pages)
   3.1 Entropy-Adaptive Branching (EAB)
       3.1.1 Algorithm
       3.1.2 KV-Cache Mechanics
       3.1.3 Computational Complexity Analysis
       3.1.4 Hyperparameters
   3.2 Semantic Entropy
       3.2.1 Algorithm
       3.2.2 Clustering Strategy
       3.2.3 Uncertainty Quantification
       3.2.4 Hyperparameters
   3.3 Integration
       3.3.1 Full Pipeline
       3.3.2 System Design
       3.3.3 Implementation Details

4. EXPERIMENTAL SETUP (3-4 pages)
   4.1 Datasets
       - TriviaQA, NaturalQuestions, etc.
       - Statistics and preprocessing
   4.2 Models
       - GPT-2, GPT-2-Large, GPT-J-6B
       - Configurations
   4.3 Baselines
       - Naive sampling
       - Token-level uncertainty methods
   4.4 Evaluation Metrics
       - Quality: ECE, AUROC, AUPRC
       - Efficiency: Token-steps, time, memory
   4.5 Hyperparameter Settings
       - How they were chosen (validation set)

5. RESULTS (10-15 pages)

   5.1 EAB Efficiency Analysis (RQ1.1 - RQ1.4)
       5.1.1 Computational Cost Reduction
             - Table 1: Speedup across configurations
             - Figure 1: Speedup vs prompt length
             - Figure 2: Scaling curves
       5.1.2 Quality Preservation
             - Table 2: Diversity metrics
             - Figure 3: Embedding space visualization
       5.1.3 Hyperparameter Sensitivity
             - Figure 4: Pareto frontier
             - Figure 5: Sensitivity heatmap

   5.2 Semantic Entropy Calibration (RQ2.1 - RQ2.4)
       5.2.1 Calibration Analysis
             - Table 3: Calibration metrics (ECE, MCE, Brier)
             - Figure 6: Calibration curves
             - Figure 7: Entropy distributions
       5.2.2 Comparison with Baselines
             - Table 4: Method comparison
             - Figure 8: ROC curves
       5.2.3 Clustering Quality
             - Figure 9: Cluster visualizations
             - Figure 10: Threshold sensitivity
       5.2.4 Error Type Analysis
             - Figure 11: Entropy by error type

   5.3 Integrated System Evaluation (RQ3.1 - RQ3.4)
       5.3.1 Main Ablation Study
             - Table 5: Naive vs EAB (primary results)
             - Figure 12: Cost-quality scatter
             - Figure 13: Side-by-side calibration
       5.3.2 Branching-Diversity Alignment
             - Figure 14: Branching heatmap
             - Figure 15: Correlation analysis
       5.3.3 Selective Prediction
             - Figure 16: Coverage-accuracy curves
             - Table 6: Real-world scenarios
       5.3.4 Domain Generalization
             - Figure 17: Per-domain performance
             - Table 7: Cross-domain transfer

6. ANALYSIS & DISCUSSION (4-5 pages)
   6.1 When Does EAB Excel?
       - Long prompts (>200 tokens)
       - Uncertain domains (creative writing)
       - Many samples needed (N>10)
   6.2 When Does EAB Struggle?
       - Very confident models
       - Short prompts
       - Single-sample scenarios
   6.3 Semantic Entropy Insights
       - Clustering quality
       - Encoder choice matters
       - Robustness to hyperparameters
   6.4 Cost-Quality Tradeoffs
       - Operating points
       - Recommendations per use case
   6.5 Limitations
       - Memory overhead (EAB)
       - Threshold sensitivity (SE)
       - Domain-specific tuning needed
   6.6 Practical Recommendations
       - When to use EAB vs naive
       - How to choose hyperparameters
       - Deployment considerations

7. CONCLUSION & FUTURE WORK (2 pages)
   7.1 Summary of Contributions
   7.2 Key Findings
       - RQ answers summarized
   7.3 Broader Impact
       - Enabling uncertainty-aware LLM systems
       - Cost reduction for practitioners
   7.4 Future Directions
       - Adaptive threshold selection
       - Multi-turn dialogue uncertainty
       - Other uncertainty metrics
       - Larger models (GPT-4, Llama-70B)
       - Production deployment studies

APPENDICES
   A. Additional Experimental Results
   B. Hyperparameter Tuning Details
   C. Dataset Statistics
   D. Qualitative Examples
   E. Code Availability & Reproducibility

REFERENCES
```

---

## 6. Key Tables for Report

### Table 1: Main Results (Ablation Study)

| Method      | Dataset    | Samples | Token-Steps | Time (s) | ECE ↓  | AUROC ↑ | Speedup |
|-------------|------------|---------|-------------|----------|--------|---------|---------|
| Naive       | TriviaQA   | 20      | 45,000      | 12.3     | 0.084  | 0.756   | 1.0×    |
| **EAB**     | TriviaQA   | 20      | **24,000**  | **6.8**  | 0.081  | 0.761   | **1.88×** |
| Naive       | Creative   | 20      | 52,000      | 14.1     | 0.102  | 0.712   | 1.0×    |
| **EAB**     | Creative   | 20      | **21,000**  | **5.9**  | 0.098  | 0.718   | **2.47×** |

*Bold = proposed method. Arrows indicate better direction.*

### Table 2: Baseline Comparison (Uncertainty Methods)

| Method              | ECE ↓  | AUROC ↑ | AUPRC ↑ | Calibration |
|---------------------|--------|---------|---------|-------------|
| Token Entropy       | 0.134  | 0.682   | 0.591   | Poor        |
| Perplexity          | 0.121  | 0.698   | 0.612   | Moderate    |
| Self-Consistency    | 0.095  | 0.731   | 0.687   | Good        |
| **Semantic Entropy**| **0.081** | **0.761** | **0.724** | **Excellent** |

### Table 3: Hyperparameter Recommendations

| Use Case              | Threshold | Branch Factor | Max Paths | Expected Speedup |
|-----------------------|-----------|---------------|-----------|------------------|
| Factual QA            | 0.5-0.6   | 2-3           | 20        | 1.5-2.0×         |
| Creative Generation   | 0.3-0.4   | 3-4           | 30        | 2.0-3.0×         |
| General Purpose       | 0.4       | 3             | 20        | 1.8-2.2×         |
| Efficiency-Focused    | 0.6-0.7   | 2             | 10        | 1.2-1.5×         |
| Diversity-Focused     | 0.2-0.3   | 4-5           | 50        | 2.5-3.5×         |

### Table 4: Domain Generalization

| Domain              | ECE (Naive) | ECE (EAB) | AUROC (Naive) | AUROC (EAB) | Speedup |
|---------------------|-------------|-----------|---------------|-------------|---------|
| Factual QA          | 0.084       | 0.081     | 0.756         | 0.761       | 1.88×   |
| Creative Writing    | 0.102       | 0.098     | 0.712         | 0.718       | 2.47×   |
| Code Generation     | 0.091       | 0.089     | 0.743         | 0.748       | 1.62×   |
| Medical QA          | 0.087       | 0.084     | 0.769         | 0.772       | 1.91×   |

---

## 7. Implementation Roadmap

### Phase 1: Component Development ✅
- [x] EAB implementation
- [x] Semantic entropy implementation
- [x] Basic testing

### Phase 2: Integration Setup 🔄
- [ ] Create integration folder structure
- [ ] Implement full pipeline
- [ ] Set up experiment tracking (e.g., Weights & Biases)
- [ ] Create unified configuration system

### Phase 3: Experimental Evaluation 📊
- [ ] **Experiment 1.A**: EAB efficiency analysis
- [ ] **Experiment 1.B**: EAB quality analysis
- [ ] **Experiment 1.C**: EAB hyperparameter sensitivity
- [ ] **Experiment 2.A**: Semantic entropy calibration
- [ ] **Experiment 2.B**: Baseline comparison
- [ ] **Experiment 2.C**: Clustering analysis
- [ ] **Experiment 2.D**: Error type analysis
- [ ] **Experiment 3.A**: Main ablation study (Naive vs EAB)
- [ ] **Experiment 3.B**: Branching-diversity alignment
- [ ] **Experiment 3.C**: Selective prediction
- [ ] **Experiment 3.D**: Scaling analysis

### Phase 4: Analysis & Visualization 📈
- [ ] Generate all figures (17 figures planned)
- [ ] Create all tables (7 tables planned)
- [ ] Statistical significance testing
- [ ] Failure mode analysis
- [ ] Qualitative examples

### Phase 5: Documentation & Report Writing 📝
- [ ] Complete technical report
- [ ] Code documentation
- [ ] Tutorial notebooks
- [ ] Reproducibility guide

---

## Quick Start

### Installation

```bash
# Clone repository
git clone https://github.com/DoniaGasmii/cost-aware-semantic-uncertainty-llm.git
cd cost-aware-semantic-uncertainty-llm

# Install all components
pip install -r requirements.txt

# Or install individually
pip install -e ./entropy-adaptive-branching
pip install -e ./semantic-entropy-based-uncertainty
pip install -e ./integration  # Coming soon
```

### Basic Usage

```python
from eab import EntropyAdaptiveBranching
from semantic_entropy import SemanticUncertaintyEstimator

# Initialize components
eab = EntropyAdaptiveBranching(
    model_name="gpt2",
    entropy_threshold=0.4,
    branch_factor=3
)

estimator = SemanticUncertaintyEstimator(
    encoder_model="all-mpnet-base-v2",
    distance_threshold=0.15
)

# Generate samples efficiently
prompt = "Question: What is the capital of France? Answer:"
samples = eab.generate(prompt, max_new_tokens=50)

# Measure uncertainty
texts = [s['text'] for s in samples]
uncertainty = estimator.compute(texts)

print(f"Semantic Entropy: {uncertainty['normalized_entropy']:.3f}")
print(f"Distinct Meanings: {uncertainty['n_clusters']}")

# Decision
if uncertainty['normalized_entropy'] < 0.3:
    print("High confidence - safe to use")
else:
    print("Low confidence - needs review")
```

---

## Citation

If you use this work in your research, please cite:

```bibtex
@software{cost_aware_semantic_uncertainty,
  title={Cost-Aware Semantic Uncertainty for Large Language Models},
  author={Gasmi, Donia},
  year={2025},
  url={https://github.com/DoniaGasmii/cost-aware-semantic-uncertainty-llm}
}
```

---

## License

MIT License - see individual component READMEs for details.

---

## Contact

- **Author**: Donia Gasmi
- **GitHub**: [@DoniaGasmii](https://github.com/DoniaGasmii)
- **Project**: Semester Project, 2025

---

## Acknowledgments

- Entropy-adaptive branching builds on transformer KV-cache optimization techniques
- Semantic entropy inspired by work on semantic uncertainty (Kuhn et al., 2023)
- Sentence embeddings powered by Sentence-Transformers library

---

*This is a living document. Research questions and experiments will be refined as the project progresses.*
