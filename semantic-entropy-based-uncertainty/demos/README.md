# Semantic Entropy - Interactive Demos

Debugging and visualization tools for semantic entropy analysis.

## Tools

### 1. Quick Test (`quick_test.py`)
**Purpose**: Fast sanity check (< 30 seconds)

**Usage**:
```bash
python quick_test.py [--embedder sentence-t5|mpnet|minilm] [--n-samples 10]
```

**What it does**:
- Tests 5 prompts with different expected entropy levels
- Verifies semantic entropy estimation is working correctly
- Shows expected vs actual results in a table

**Example**:
```bash
python quick_test.py --embedder sentence-t5 --n-samples 10
```

---

### 2. Interactive Demo (`interactive_demo.py`)
**Purpose**: Full clustering debugger with visualizations

**Usage**:
```bash
python interactive_demo.py [--embedder sentence-t5|mpnet|minilm]
```

**Features**:
- Enter custom prompts or select presets
- Generate samples with Qwen2.5-3B-Instruct
- Visualize clustering (2D PCA, t-SNE, 3D interactive, heatmap)
- Interactive threshold tuning
- Export analysis and plots

**Workflow**:
1. Select/enter prompt
2. Configure sample generation
3. View clustering analysis
4. Generate visualizations
5. Optionally tune threshold
6. Export results to `demo_results/`

**Outputs**:
- `demo_YYYYMMDD_HHMMSS_2d_pca.png` - 2D PCA projection
- `demo_YYYYMMDD_HHMMSS_2d_tsne.png` - 2D t-SNE projection
- `demo_YYYYMMDD_HHMMSS_3d_interactive.html` - Interactive 3D plot
- `demo_YYYYMMDD_HHMMSS_similarity_heatmap.png` - Similarity heatmap
- `demo_YYYYMMDD_HHMMSS_dendrogram.png` - Hierarchical dendrogram
- `demo_YYYYMMDD_HHMMSS_distribution.png` - Cluster size distribution
- `demo_YYYYMMDD_HHMMSS_analysis.txt` - Text analysis
- `demo_YYYYMMDD_HHMMSS_results.json` - JSON results

---

### 3. Clustering Visualizer (`clustering_visualizer.py`)
**Purpose**: Visualization library (used by other tools)

**Features**:
- `plot_2d_projection()` - PCA/t-SNE/UMAP projections with:
  - **All samples** shown as colored points
  - **Convex hull boundaries** around each cluster (shaded regions)
  - **Centroids** marked with stars (★)
  - **Sample indices** labeled (for datasets ≤30 samples)
  - **Representative text** annotated with arrows
- `plot_3d_interactive()` - Plotly 3D plots (rotatable, hover text)
- `plot_similarity_heatmap()` - Pairwise similarity matrix
- `plot_dendrogram()` - Hierarchical clustering tree
- `plot_cluster_distribution()` - Bar charts
- `create_visualization_suite()` - Generate all plots at once

**All plots**: 300 DPI, publication-quality

**Visualization Legend**:
- 🔵 **Circles**: Individual samples
- ⭐ **Stars**: Cluster centroids
- 🔷 **Shaded regions**: Cluster boundaries (convex hulls)
- 📝 **Text boxes**: Representative sample from each cluster

---

## Dependencies

**Core**:
- `sentence-transformers` - Embeddings
- `scikit-learn` - Clustering, PCA, t-SNE
- `transformers` - Qwen model
- `torch` - Model inference
- `matplotlib`, `seaborn` - Plotting

**Optional**:
- `plotly` - 3D interactive plots (install: `pip install plotly`)
- `umap-learn` - UMAP projections (install: `pip install umap-learn`)

---

## Quick Start

1. **Quick test** (verify everything works):
   ```bash
   python quick_test.py
   ```

2. **Interactive debugging** (explore clustering):
   ```bash
   python interactive_demo.py
   ```

3. **Custom analysis** (use as library):
   ```python
   from clustering_visualizer import ClusteringVisualizer
   from semantic_entropy.estimator import SemanticUncertaintyEstimator

   estimator = SemanticUncertaintyEstimator(encoder_model="sentence-t5")
   result = estimator.compute(samples, return_details=True)

   visualizer = ClusteringVisualizer()
   visualizer.plot_2d_projection(
       result['embeddings'],
       result['cluster_labels'],
       samples,
       method='tsne',
       save_path='clustering.png'
   )
   ```

---

## Tips

- **Embedder choice**: `sentence-t5` (best quality), `mpnet` (faster), `minilm` (fastest)
- **Threshold**: Lower = more clusters (stricter), Higher = fewer clusters (more lenient)
- **Sample count**: 10-20 samples usually sufficient for debugging
- **Max tokens**: 30-50 for short answers, 100+ for creative generation
- **Temperature diversity**: By default, samples are generated with **varied temperatures** (0.7-1.3) to ensure diversity. Each sample uses a different temperature from this range.

---

## Troubleshooting

**"Model not found"**: First run will download models (sentence-t5-xxl is ~3.7GB)

**"CUDA out of memory"**: Use CPU mode or reduce batch size

**"Plotly not installed"**: 3D plots are optional, install with `pip install plotly`

**"UMAP not available"**: UMAP is optional, use PCA or t-SNE instead
