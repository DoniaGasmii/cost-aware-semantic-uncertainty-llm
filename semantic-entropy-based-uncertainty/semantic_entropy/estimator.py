"""Main semantic uncertainty estimation class."""

import os
import numpy as np
from sentence_transformers import SentenceTransformer
from typing import List, Optional, Dict, Any


from .clustering import SemanticClusterer
from .utils import (
    compute_entropy,
    normalize_entropy,
    compute_uncertainty_score,
    cluster_probabilities_uniform,
    cluster_probabilities_weighted,
    format_uncertainty_report
)


class SemanticUncertaintyEstimator:
    """
    Estimates semantic uncertainty in text generations by clustering
    semantically similar outputs and computing entropy over clusters.
    """

    # Supported embedder shortcuts → full Hugging Face model IDs
    SUPPORTED_MODELS = {
        'mpnet': 'sentence-transformers/all-mpnet-base-v2',
        'sentence-t5': 'sentence-transformers/sentence-t5-xxl',
        'minilm': 'sentence-transformers/all-MiniLM-L6-v2',
    }

    def __init__(
        self,
        encoder_model: str = "sentence-t5",
        distance_threshold: float = 0.05,
        linkage: str = "average",
        use_weighted_probs: bool = False,
        device: Optional[str] = None,
        cache_dir: Optional[str] = None,
        batch_size: int = 32
    ):
        """
        Initialize the semantic uncertainty estimator.

        Args:
            encoder_model: Sentence transformer model name or shorthand
                          Shortcuts:
                          - 'sentence-t5': sentence-t5-xxl (768-dim, best quality, slower)
                          - 'mpnet': all-mpnet-base-v2 (768-dim, good quality, faster)
                          - 'minilm': all-MiniLM-L6-v2 (384-dim, fastest, lower quality)
                          Or provide full model name from sentence-transformers
            distance_threshold: Clustering distance threshold
                               (1 - cosine_similarity threshold)
                               e.g., 0.15 means merge if similarity > 0.85
            linkage: Agglomerative clustering linkage method
                    ('average', 'complete', 'single')
            use_weighted_probs: If True and log_probs provided, weight clusters
                               by sample probabilities instead of counts
            device: Device for encoder ('cpu', 'cuda', or None for auto)
            cache_dir: Directory to cache downloaded models (default: './models/cache')
            batch_size: Batch size for encoding (default: 32)
        """
        # Resolve model name using full HF IDs
        if encoder_model in self.SUPPORTED_MODELS:
            model_name = self.SUPPORTED_MODELS[encoder_model]
        else:
            model_name = encoder_model

        self.encoder_model_name = model_name
        self.use_weighted_probs = use_weighted_probs
        self.batch_size = batch_size

        # Determine device
        if device is None:
            import torch
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device

        # Set up cache folder
        cache_folder = cache_dir or "./models/cache"

        # Build potential local path (Hugging Face Hub cache layout)
        resolved_model_path = model_name
        if cache_folder:
            # Convert 'org/model' → 'models--org--model'
            safe_name = model_name.replace("/", "--")
            cache_base = os.path.join(cache_folder, f"models--{safe_name}")
            if os.path.exists(cache_base):
                snapshots_dir = os.path.join(cache_base, "snapshots")
                if os.path.exists(snapshots_dir):
                    snapshot_dirs = [d for d in os.listdir(snapshots_dir) 
                                   if os.path.isdir(os.path.join(snapshots_dir, d))]
                    if snapshot_dirs:
                        # Use the first (and typically only) snapshot
                        resolved_model_path = os.path.join(snapshots_dir, snapshot_dirs[0])
                        print(f"✅ Loading '{model_name}' from local cache: {resolved_model_path}")

        # Prepare model kwargs for memory efficiency
        model_kwargs = {}
        if self.device == "cuda":
            import torch
            model_kwargs = {'torch_dtype': torch.float16}

        # Load encoder — now with correct path resolution
        self.encoder = SentenceTransformer(
            resolved_model_path,
            device=self.device,
            cache_folder=cache_folder,
            model_kwargs=model_kwargs
        )
        
        # Initialize clusterer
        self.clusterer = SemanticClusterer(
            distance_threshold=distance_threshold,
            linkage=linkage,
            metric="cosine"
        )
    
    def encode(self, texts: List[str], batch_size: Optional[int] = None, show_progress: bool = False) -> np.ndarray:
        """
        Encode texts into semantic embeddings.

        Args:
            texts: List of text strings to encode
            batch_size: Batch size for encoding (uses constructor default if None)
            show_progress: Whether to show progress bar during encoding

        Returns:
            Array of embeddings, shape (len(texts), embedding_dim)
        """
        if batch_size is None:
            batch_size = self.batch_size

        embeddings = self.encoder.encode(
            texts,
            batch_size=batch_size,
            convert_to_numpy=True,
            show_progress_bar=show_progress,
            normalize_embeddings=True  # Critical for cosine similarity
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
                - uncertainty_score: Combined score considering both entropy and cluster count
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
        uncertainty_score = compute_uncertainty_score(entropy, n_clusters, len(texts))
        
        # Prepare result
        result = {
            'entropy': float(entropy),
            'normalized_entropy': float(normalized_entropy),
            'uncertainty_score': float(uncertainty_score),
            'n_clusters': int(n_clusters),
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
    
    def interpret_uncertainty(self, uncertainty_score: float) -> str:
        """
        Interpret uncertainty score value.
        
        Args:
            uncertainty_score: Combined uncertainty score in [0, 1]
            
        Returns:
            Human-readable interpretation
        """
        if uncertainty_score < 0.3:
            return "HIGH CONFIDENCE - Responses are very consistent"
        elif uncertainty_score < 0.7:
            return "MODERATE UNCERTAINTY - Some variation in meanings"
        else:
            return "HIGH UNCERTAINTY - Many different interpretations"