"""Clustering functionality for semantic similarity."""

import numpy as np
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics.pairwise import cosine_similarity
from typing import Tuple, Optional


class SemanticClusterer:
    """
    Clusters text embeddings based on semantic similarity using 
    agglomerative clustering with adaptive number of clusters.
    """
    
    def __init__(
        self,
        distance_threshold: float = 0.15,
        linkage: str = "average",
        metric: str = "cosine"
    ):
        """
        Initialize the semantic clusterer.
        
        Args:
            distance_threshold: Distance threshold for cluster merging.
                              For cosine: 1 - similarity_threshold
                              (e.g., 0.15 means merge if similarity > 0.85)
            linkage: Linkage criterion ('average', 'complete', 'single')
            metric: Distance metric ('cosine', 'euclidean')
        """
        self.distance_threshold = distance_threshold
        self.linkage = linkage
        self.metric = metric
        
        if linkage not in ["average", "complete", "single"]:
            raise ValueError(f"Unsupported linkage: {linkage}")
        
        if metric not in ["cosine", "euclidean"]:
            raise ValueError(f"Unsupported metric: {metric}")
    
    def cluster(self, embeddings: np.ndarray) -> Tuple[np.ndarray, int]:
        """
        Cluster embeddings based on semantic similarity.
        
        Args:
            embeddings: Array of shape (n_samples, embedding_dim)
            
        Returns:
            Tuple of (cluster_labels, n_clusters)
        """
        if len(embeddings) == 1:
            # Single sample - one cluster
            return np.array([0]), 1
        
        # Create agglomerative clustering model
        clustering = AgglomerativeClustering(
            n_clusters=None,  # Let distance_threshold determine this
            distance_threshold=self.distance_threshold,
            metric=self.metric,
            linkage=self.linkage
        )
        
        # Fit and predict
        labels = clustering.fit_predict(embeddings)
        n_clusters = len(np.unique(labels))
        
        return labels, n_clusters
    
    def get_cluster_representatives(
        self, 
        embeddings: np.ndarray, 
        labels: np.ndarray,
        texts: Optional[list] = None
    ) -> dict:
        """
        Get representative samples for each cluster (closest to centroid).
        
        Args:
            embeddings: Array of shape (n_samples, embedding_dim)
            labels: Cluster assignments
            texts: Optional list of original texts
            
        Returns:
            Dictionary mapping cluster_id to representative info
        """
        n_clusters = len(np.unique(labels))
        representatives = {}
        
        for cluster_id in range(n_clusters):
            # Get all embeddings in this cluster
            cluster_mask = labels == cluster_id
            cluster_embeddings = embeddings[cluster_mask]
            cluster_indices = np.where(cluster_mask)[0]
            
            # Compute centroid
            centroid = cluster_embeddings.mean(axis=0, keepdims=True)
            
            # Find closest sample to centroid
            similarities = cosine_similarity(cluster_embeddings, centroid).flatten()
            closest_idx = similarities.argmax()
            original_idx = cluster_indices[closest_idx]
            
            representatives[cluster_id] = {
                'index': int(original_idx),
                'size': int(cluster_mask.sum()),
                'text': texts[original_idx] if texts else None
            }
        
        return representatives
    
    def get_similarity_matrix(self, embeddings: np.ndarray) -> np.ndarray:
        """
        Compute pairwise cosine similarity matrix.
        
        Args:
            embeddings: Array of shape (n_samples, embedding_dim)
            
        Returns:
            Similarity matrix of shape (n_samples, n_samples)
        """
        return cosine_similarity(embeddings)
    
    def analyze_clusters(
        self, 
        embeddings: np.ndarray, 
        labels: np.ndarray,
        texts: Optional[list] = None
    ) -> dict:
        """
        Comprehensive cluster analysis.
        
        Args:
            embeddings: Array of shape (n_samples, embedding_dim)
            labels: Cluster assignments
            texts: Optional list of original texts
            
        Returns:
            Dictionary with cluster statistics and representatives
        """
        n_clusters = len(np.unique(labels))
        
        # Basic stats
        cluster_sizes = np.bincount(labels)
        
        # Representatives
        representatives = self.get_cluster_representatives(embeddings, labels, texts)
        
        # Intra-cluster cohesion (avg similarity within clusters)
        cohesion = {}
        for cluster_id in range(n_clusters):
            cluster_mask = labels == cluster_id
            cluster_embeddings = embeddings[cluster_mask]
            
            if len(cluster_embeddings) > 1:
                sim_matrix = cosine_similarity(cluster_embeddings)
                # Average similarity (excluding diagonal)
                mask = ~np.eye(len(sim_matrix), dtype=bool)
                avg_similarity = sim_matrix[mask].mean()
            else:
                avg_similarity = 1.0  # Single sample, perfect cohesion
            
            cohesion[cluster_id] = float(avg_similarity)
        
        return {
            'n_clusters': n_clusters,
            'cluster_sizes': cluster_sizes.tolist(),
            'representatives': representatives,
            'cohesion': cohesion
        }