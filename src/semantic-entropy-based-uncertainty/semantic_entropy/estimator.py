"""Main semantic uncertainty estimation class."""

import numpy as np
from sentence_transformers import SentenceTransformer
from typing import List, Optional, Dict, Any

from .clustering import SemanticClusterer
from .utils import (
    compute_entropy,
    normalize_entropy,
    cluster_probabilities_uniform,
    cluster_probabilities_weighted,
    format_uncertainty_report
)


class SemanticUncertaintyEstimator:
    """
    Estimates semantic uncertainty in text generations by clustering
    semantically similar outputs and computing entropy over clusters.
    """
    
    def __init__(
        self,
        encoder_model: str = "all-mpnet-base-v2",
        distance_threshold: float = 0.15,
        linkage: str = "average",
        use_weighted_probs: bool = False,
        device: Optional[str] = None
    ):
        """
        Initialize the semantic uncertainty estimator.
        
        Args:
            encoder_model: Sentence transformer model name
                          - 'all-mpnet-base-v2': Best quality (768-dim)
                          - 'all-MiniLM-L6-v2': Faster, smaller (384-dim)
            distance_threshold: Clustering distance threshold
                               (1 - cosine_similarity threshold)
                               e.g., 0.15 means merge if similarity > 0.85
            linkage: Agglomerative clustering linkage method
                    ('average', 'complete', 'single')
            use_weighted_probs: If True and log_probs provided, weight clusters
                               by sample probabilities instead of counts
            device: Device for encoder ('cpu', 'cuda', or None for auto)
        """
        self.encoder_model_name = encoder_model
        self.use_weighted_probs = use_weighted_probs
        
        # Initialize encoder
        self.encoder = SentenceTransformer(encoder_model, device=device)
        
        # Initialize clusterer
        self.clusterer = SemanticClusterer(
            distance_threshold=distance_threshold,
            linkage=linkage,
            metric="cosine"
        )
    
    def encode(self, texts: List[str], batch_size: int = 32) -> np.ndarray:
        """
        Encode texts into semantic embeddings.
        
        Args:
            texts: List of text strings to encode
            batch_size: Batch size for encoding
            
        Returns:
            Array of embeddings, shape (len(texts), embedding_dim)
        """
        embeddings = self.encoder.encode(
            texts,
            batch_size=batch_size,
            convert_to_numpy=True,
            show_progress_bar=False
        )
        return embeddings
    
    def compute(
        self,
        texts: List[str],
        log_probs: Optional[List[float]] = None,
        return_details: bool = False
    ) -> Dict[str, Any]:
        """
        Compute semantic uncertainty for a set of text generations.
        
        Args:
            texts: List of generated text samples
            log_probs: Optional list of log-probabilities for each sample
                      (e.g., from entropy-adaptive branching)
            return_details: If True, include detailed cluster analysis
            
        Returns:
            Dictionary containing:
                - entropy: Raw semantic entropy
                - normalized_entropy: Entropy normalized to [0, 1]
                - n_clusters: Number of semantic clusters found
                - cluster_labels: Cluster assignment for each sample
                - cluster_probs: Probability distribution over clusters
                - embeddings: Sentence embeddings (if return_details=True)
                - cluster_analysis: Detailed cluster info (if return_details=True)
        """
        if not texts:
            raise ValueError("texts list cannot be empty")
        
        if log_probs is not None and len(log_probs) != len(texts):
            raise ValueError("log_probs must have same length as texts")
        
        # Step 1: Encode texts into semantic embeddings
        embeddings = self.encode(texts)
        
        # Step 2: Cluster embeddings
        labels, n_clusters = self.clusterer.cluster(embeddings)
        
        # Step 3: Compute cluster probabilities
        if log_probs is not None and self.use_weighted_probs:
            cluster_probs = cluster_probabilities_weighted(
                labels, 
                np.array(log_probs)
            )
        else:
            cluster_probs = cluster_probabilities_uniform(labels)
        
        # Step 4: Compute semantic entropy
        entropy = compute_entropy(cluster_probs)
        normalized_entropy = normalize_entropy(entropy, n_clusters)
        
        # Prepare result
        result = {
            'entropy': entropy,
            'normalized_entropy': normalized_entropy,
            'n_clusters': n_clusters,
            'cluster_labels': labels.tolist(),
            'cluster_probs': cluster_probs.tolist()
        }
        
        # Add detailed analysis if requested
        if return_details:
            result['embeddings'] = embeddings
            result['cluster_analysis'] = self.clusterer.analyze_clusters(
                embeddings, 
                labels, 
                texts
            )
            result['similarity_matrix'] = self.clusterer.get_similarity_matrix(embeddings)
        
        return result
    
    def compute_with_report(
        self,
        texts: List[str],
        log_probs: Optional[List[float]] = None
    ) -> str:
        """
        Compute uncertainty and return a formatted text report.
        
        Args:
            texts: List of generated text samples
            log_probs: Optional list of log-probabilities
            
        Returns:
            Formatted string report
        """
        result = self.compute(texts, log_probs, return_details=True)
        return format_uncertainty_report(result)
    
    def get_cluster_representatives(
        self,
        texts: List[str],
        log_probs: Optional[List[float]] = None
    ) -> Dict[int, Dict[str, Any]]:
        """
        Get representative text for each semantic cluster.
        
        Args:
            texts: List of generated text samples
            log_probs: Optional list of log-probabilities
            
        Returns:
            Dictionary mapping cluster_id to representative info:
                - text: Representative text
                - size: Number of samples in cluster
                - probability: Cluster probability
        """
        result = self.compute(texts, log_probs, return_details=True)
        
        representatives = {}
        cluster_analysis = result['cluster_analysis']['representatives']
        cluster_probs = result['cluster_probs']
        
        for cluster_id, info in cluster_analysis.items():
            representatives[cluster_id] = {
                'text': info['text'],
                'size': info['size'],
                'probability': cluster_probs[cluster_id]
            }
        
        return representatives
    
    def interpret_uncertainty(self, normalized_entropy: float) -> str:
        """
        Interpret normalized entropy value.
        
        Args:
            normalized_entropy: Entropy value in [0, 1]
            
        Returns:
            Human-readable interpretation
        """
        if normalized_entropy < 0.3:
            return "HIGH CONFIDENCE - Responses are very consistent"
        elif normalized_entropy < 0.7:
            return "MODERATE UNCERTAINTY - Some variation in meanings"
        else:
            return "HIGH UNCERTAINTY - Many different interpretations"