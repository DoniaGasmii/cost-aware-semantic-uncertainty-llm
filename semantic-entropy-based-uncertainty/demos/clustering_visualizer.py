"""Clustering visualization toolkit for semantic entropy analysis."""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from pathlib import Path
from typing import List, Optional, Tuple
import warnings

# Optional dependencies
try:
    import umap
    HAS_UMAP = True
except ImportError:
    HAS_UMAP = False

try:
    import plotly.graph_objects as go
    import plotly.express as px
    HAS_PLOTLY = True
except ImportError:
    HAS_PLOTLY = False

warnings.filterwarnings('ignore')


class ClusteringVisualizer:
    """
    Visualization toolkit for semantic clustering analysis.

    Provides multiple visualization methods:
    - 2D projections (PCA, t-SNE, UMAP)
    - 3D interactive plots (Plotly)
    - Similarity heatmaps
    - Dendrograms

    All plots are publication-quality (300 DPI).
    """

    def __init__(self):
        """Initialize visualizer with publication-quality settings."""
        self.set_publication_style()

    def set_publication_style(self):
        """Set matplotlib rcParams for publication-quality figures."""
        plt.rcParams.update({
            'font.size': 12,
            'axes.labelsize': 14,
            'axes.titlesize': 16,
            'legend.fontsize': 11,
            'figure.figsize': (10, 8),
            'figure.dpi': 100,
            'savefig.dpi': 300,
            'savefig.bbox': 'tight',
            'font.family': 'sans-serif',
            'font.sans-serif': ['Arial', 'DejaVu Sans'],
        })

    def plot_2d_projection(
        self,
        embeddings: np.ndarray,
        labels,
        texts: List[str],
        method: str = "pca",
        save_path: Optional[Path] = None,
        title: Optional[str] = None
    ):
        """
        Create 2D projection with cluster coloring.

        Args:
            embeddings: Array of shape (n_samples, embedding_dim)
            labels: Cluster assignments for each sample (list or array)
            texts: Original text samples
            method: Dimensionality reduction method ('pca', 'tsne', or 'umap')
            save_path: Path to save figure (optional)
            title: Plot title (optional)
        """
        # Ensure labels is numpy array
        labels = np.array(labels)
        # Dimensionality reduction
        if method == "pca":
            reducer = PCA(n_components=2, random_state=42)
            coords_2d = reducer.fit_transform(embeddings)
            method_name = "PCA"
        elif method == "tsne":
            reducer = TSNE(n_components=2, random_state=42, perplexity=min(30, len(embeddings)-1))
            coords_2d = reducer.fit_transform(embeddings)
            method_name = "t-SNE"
        elif method == "umap":
            if not HAS_UMAP:
                raise ImportError("UMAP not installed. Install with: pip install umap-learn")
            reducer = umap.UMAP(n_components=2, random_state=42)
            coords_2d = reducer.fit_transform(embeddings)
            method_name = "UMAP"
        else:
            raise ValueError(f"Unknown method: {method}. Use 'pca', 'tsne', or 'umap'")

        # Create plot
        fig, ax = plt.subplots(figsize=(12, 10))

        n_clusters = len(np.unique(labels))
        colors = plt.cm.Set3(np.linspace(0, 1, n_clusters))

        # Plot each cluster
        for cluster_id in range(n_clusters):
            mask = labels == cluster_id
            cluster_count = mask.sum()

            ax.scatter(
                coords_2d[mask, 0],
                coords_2d[mask, 1],
                c=[colors[cluster_id]],
                label=f'Cluster {cluster_id} ({cluster_count} samples)',
                s=100,
                alpha=0.7,
                edgecolors='black',
                linewidths=1
            )

            # Add representative text annotation
            if cluster_count > 0:
                centroid = coords_2d[mask].mean(axis=0)
                representative_idx = np.where(mask)[0][0]
                representative_text = texts[representative_idx][:40] + "..."

                ax.annotate(
                    representative_text,
                    xy=centroid,
                    xytext=(10, 10),
                    textcoords='offset points',
                    bbox=dict(
                        boxstyle='round,pad=0.5',
                        facecolor=colors[cluster_id],
                        alpha=0.3,
                        edgecolor='black',
                        linewidth=0.5
                    ),
                    fontsize=9,
                    ha='left'
                )

        ax.set_xlabel(f'{method_name} Component 1', fontweight='bold')
        ax.set_ylabel(f'{method_name} Component 2', fontweight='bold')

        if title:
            ax.set_title(title, fontweight='bold', fontsize=16)
        else:
            ax.set_title(f'Semantic Clustering ({method_name} Projection)', fontweight='bold', fontsize=16)

        ax.legend(loc='best', framealpha=0.9)
        ax.grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✓ Saved: {save_path}")
        else:
            plt.show()

        plt.close()

    def plot_3d_interactive(
        self,
        embeddings: np.ndarray,
        labels,
        texts: List[str],
        save_path: Optional[Path] = None,
        title: Optional[str] = None
    ):
        """
        Create interactive 3D plot with plotly.

        Args:
            embeddings: Array of shape (n_samples, embedding_dim)
            labels: Cluster assignments (list or array)
            texts: Original text samples
            save_path: Path to save HTML file (optional)
            title: Plot title (optional)
        """
        if not HAS_PLOTLY:
            raise ImportError("Plotly not installed. Install with: pip install plotly")

        # Ensure labels is numpy array
        labels = np.array(labels)

        # PCA to 3D
        pca = PCA(n_components=3, random_state=42)
        coords_3d = pca.fit_transform(embeddings)

        # Create plotly figure
        fig = go.Figure()

        n_clusters = len(np.unique(labels))
        colors_hex = px.colors.qualitative.Set3[:n_clusters]

        for cluster_id in range(n_clusters):
            mask = labels == cluster_id
            cluster_count = mask.sum()

            # Get texts for this cluster
            cluster_texts = [texts[i] for i in np.where(mask)[0]]

            fig.add_trace(go.Scatter3d(
                x=coords_3d[mask, 0],
                y=coords_3d[mask, 1],
                z=coords_3d[mask, 2],
                mode='markers',
                name=f'Cluster {cluster_id} ({cluster_count})',
                marker=dict(
                    size=8,
                    color=colors_hex[cluster_id] if cluster_id < len(colors_hex) else '#808080',
                    line=dict(color='black', width=0.5)
                ),
                text=cluster_texts,
                hovertemplate='<b>%{text}</b><extra></extra>'
            ))

        if title is None:
            title = 'Interactive 3D Semantic Clustering'

        fig.update_layout(
            title=title,
            scene=dict(
                xaxis_title='PCA Component 1',
                yaxis_title='PCA Component 2',
                zaxis_title='PCA Component 3'
            ),
            width=1000,
            height=800,
            hovermode='closest'
        )

        if save_path:
            fig.write_html(str(save_path))
            print(f"✓ Saved: {save_path}")
        else:
            fig.show()

    def plot_similarity_heatmap(
        self,
        similarity_matrix: np.ndarray,
        labels,
        save_path: Optional[Path] = None,
        title: Optional[str] = None
    ):
        """
        Plot similarity heatmap with hierarchical ordering by cluster.

        Args:
            similarity_matrix: Pairwise similarity matrix (n_samples, n_samples)
            labels: Cluster assignments (list or array)
            save_path: Path to save figure (optional)
            title: Plot title (optional)
        """
        # Ensure labels is numpy array
        labels = np.array(labels)

        # Sort by cluster labels for better visualization
        sorted_indices = np.argsort(labels)
        sorted_similarity = similarity_matrix[sorted_indices][:, sorted_indices]
        sorted_labels = labels[sorted_indices]

        # Create figure
        fig, ax = plt.subplots(figsize=(12, 10))

        # Plot heatmap
        im = ax.imshow(sorted_similarity, cmap='RdYlGn', aspect='auto', vmin=0, vmax=1)

        # Add colorbar
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('Cosine Similarity', rotation=270, labelpad=20, fontweight='bold')

        # Add cluster boundaries
        n_clusters = len(np.unique(labels))
        cluster_boundaries = []
        for cluster_id in range(n_clusters):
            cluster_mask = sorted_labels == cluster_id
            cluster_positions = np.where(cluster_mask)[0]
            if len(cluster_positions) > 0:
                cluster_boundaries.append(cluster_positions[-1] + 0.5)

        for boundary in cluster_boundaries[:-1]:  # Don't draw last boundary
            ax.axhline(boundary, color='blue', linewidth=2, linestyle='--', alpha=0.7)
            ax.axvline(boundary, color='blue', linewidth=2, linestyle='--', alpha=0.7)

        if title:
            ax.set_title(title, fontweight='bold', fontsize=16)
        else:
            ax.set_title('Pairwise Similarity Heatmap (Ordered by Cluster)', fontweight='bold', fontsize=16)

        ax.set_xlabel('Sample Index (sorted by cluster)', fontweight='bold')
        ax.set_ylabel('Sample Index (sorted by cluster)', fontweight='bold')

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✓ Saved: {save_path}")
        else:
            plt.show()

        plt.close()

    def plot_dendrogram(
        self,
        embeddings: np.ndarray,
        threshold: float,
        save_path: Optional[Path] = None,
        title: Optional[str] = None
    ):
        """
        Plot hierarchical clustering dendrogram showing threshold cutoff.

        Args:
            embeddings: Array of shape (n_samples, embedding_dim)
            threshold: Distance threshold for clustering
            save_path: Path to save figure (optional)
            title: Plot title (optional)
        """
        from scipy.cluster.hierarchy import dendrogram, linkage
        from scipy.spatial.distance import pdist

        # Compute linkage
        distances = pdist(embeddings, metric='cosine')
        linkage_matrix = linkage(distances, method='average')

        # Create figure
        fig, ax = plt.subplots(figsize=(14, 8))

        # Plot dendrogram
        dendrogram(
            linkage_matrix,
            ax=ax,
            color_threshold=threshold,
            above_threshold_color='gray'
        )

        # Add threshold line
        ax.axhline(y=threshold, color='red', linewidth=2, linestyle='--',
                   label=f'Threshold = {threshold:.3f}')

        if title:
            ax.set_title(title, fontweight='bold', fontsize=16)
        else:
            ax.set_title('Hierarchical Clustering Dendrogram', fontweight='bold', fontsize=16)

        ax.set_xlabel('Sample Index', fontweight='bold')
        ax.set_ylabel('Distance', fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✓ Saved: {save_path}")
        else:
            plt.show()

        plt.close()

    def plot_cluster_distribution(
        self,
        labels,
        cluster_probs: Optional[np.ndarray] = None,
        save_path: Optional[Path] = None,
        title: Optional[str] = None
    ):
        """
        Plot cluster size distribution as bar chart.

        Args:
            labels: Cluster assignments (list or array)
            cluster_probs: Optional cluster probabilities (if None, uses counts)
            save_path: Path to save figure (optional)
            title: Plot title (optional)
        """
        # Ensure labels is numpy array
        labels = np.array(labels)

        n_clusters = len(np.unique(labels))
        cluster_ids = np.arange(n_clusters)

        if cluster_probs is None:
            cluster_sizes = np.bincount(labels)
            ylabel = 'Number of Samples'
        else:
            cluster_sizes = cluster_probs
            ylabel = 'Cluster Probability'

        # Create figure
        fig, ax = plt.subplots(figsize=(10, 6))

        bars = ax.bar(cluster_ids, cluster_sizes, color=plt.cm.Set3(np.linspace(0, 1, n_clusters)),
                     edgecolor='black', linewidth=1.5)

        # Add value labels on bars
        for i, (bar, value) in enumerate(zip(bars, cluster_sizes)):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{value:.2f}' if cluster_probs is not None else f'{int(value)}',
                   ha='center', va='bottom', fontweight='bold')

        if title:
            ax.set_title(title, fontweight='bold', fontsize=16)
        else:
            ax.set_title('Cluster Distribution', fontweight='bold', fontsize=16)

        ax.set_xlabel('Cluster ID', fontweight='bold')
        ax.set_ylabel(ylabel, fontweight='bold')
        ax.set_xticks(cluster_ids)
        ax.grid(True, alpha=0.3, axis='y')

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✓ Saved: {save_path}")
        else:
            plt.show()

        plt.close()


def create_visualization_suite(
    embeddings: np.ndarray,
    labels,
    texts: List[str],
    similarity_matrix: np.ndarray,
    threshold: float,
    output_dir: Path,
    prefix: str = "clustering"
):
    """
    Create full suite of visualizations and save to directory.

    Args:
        embeddings: Embeddings array
        labels: Cluster labels (list or array)
        texts: Original texts
        similarity_matrix: Similarity matrix
        threshold: Clustering threshold
        output_dir: Directory to save plots
        prefix: Filename prefix
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Ensure labels is numpy array
    labels = np.array(labels)

    visualizer = ClusteringVisualizer()

    print("\nGenerating visualization suite...")

    # 2D projections
    for method in ['pca', 'tsne']:
        try:
            visualizer.plot_2d_projection(
                embeddings, labels, texts,
                method=method,
                save_path=output_dir / f"{prefix}_2d_{method}.png"
            )
        except Exception as e:
            print(f"Warning: Could not create {method} plot: {e}")

    # 3D interactive
    if HAS_PLOTLY:
        try:
            visualizer.plot_3d_interactive(
                embeddings, labels, texts,
                save_path=output_dir / f"{prefix}_3d_interactive.html"
            )
        except Exception as e:
            print(f"Warning: Could not create 3D plot: {e}")

    # Heatmap
    visualizer.plot_similarity_heatmap(
        similarity_matrix, labels,
        save_path=output_dir / f"{prefix}_similarity_heatmap.png"
    )

    # Dendrogram
    visualizer.plot_dendrogram(
        embeddings, threshold,
        save_path=output_dir / f"{prefix}_dendrogram.png"
    )

    # Cluster distribution
    visualizer.plot_cluster_distribution(
        labels,
        save_path=output_dir / f"{prefix}_distribution.png"
    )

    print(f"\n✓ All visualizations saved to: {output_dir}")
