"""Utility functions for semantic entropy computation."""

import numpy as np
from typing import List, Dict, Any


def compute_entropy(probabilities: np.ndarray) -> float:
    """
    Compute Shannon entropy from probability distribution.
    
    Args:
        probabilities: Array of probabilities (must sum to 1)
        
    Returns:
        Entropy value
    """
    # Filter out zero probabilities to avoid log(0)
    probs = probabilities[probabilities > 0]
    
    if len(probs) == 0:
        return 0.0
    
    entropy = -np.sum(probs * np.log(probs))
    return float(entropy)


def normalize_entropy(entropy: float, n_clusters: int) -> float:
    """
    Normalize entropy to [0, 1] range.
    
    Args:
        entropy: Raw entropy value
        n_clusters: Number of clusters
        
    Returns:
        Normalized entropy in [0, 1]
    """
    if n_clusters <= 1:
        return 0.0
    
    max_entropy = np.log(n_clusters)
    return entropy / max_entropy if max_entropy > 0 else 0.0


def cluster_probabilities_uniform(labels: np.ndarray) -> np.ndarray:
    """
    Compute cluster probabilities using uniform weighting.
    
    Args:
        labels: Cluster assignment for each sample
        
    Returns:
        Array of probabilities for each cluster
    """
    n_clusters = len(np.unique(labels))
    cluster_probs = np.bincount(labels, minlength=n_clusters) / len(labels)
    return cluster_probs


def cluster_probabilities_weighted(labels: np.ndarray, log_probs: np.ndarray) -> np.ndarray:
    """
    Compute cluster probabilities using sample probability weights.
    
    Args:
        labels: Cluster assignment for each sample
        log_probs: Log probability of each sample
        
    Returns:
        Array of probabilities for each cluster
    """
    # Convert log probs to probabilities
    probs = np.exp(log_probs)
    probs = probs / probs.sum()  # Normalize
    
    n_clusters = len(np.unique(labels))
    cluster_probs = np.zeros(n_clusters)
    
    for i, label in enumerate(labels):
        cluster_probs[label] += probs[i]
    
    return cluster_probs


def format_uncertainty_report(result: Dict[str, Any]) -> str:
    """
    Format uncertainty computation results as a readable report.
    
    Args:
        result: Dictionary containing uncertainty metrics
        
    Returns:
        Formatted string report
    """
    report = []
    report.append("=" * 60)
    report.append("SEMANTIC UNCERTAINTY REPORT")
    report.append("=" * 60)
    report.append(f"Semantic Entropy:      {result['entropy']:.4f}")
    report.append(f"Normalized Entropy:    {result['normalized_entropy']:.4f}")
    report.append(f"Number of Clusters:    {result['n_clusters']}")
    report.append(f"Cluster Distribution:  {result['cluster_probs']}")
    report.append("")
    
    # Interpretation
    norm_ent = result['normalized_entropy']
    if norm_ent < 0.3:
        interpretation = "HIGH CONFIDENCE - Model responses are very consistent"
    elif norm_ent < 0.7:
        interpretation = "MODERATE UNCERTAINTY - Some variation in meanings"
    else:
        interpretation = "HIGH UNCERTAINTY - Many different interpretations"
    
    report.append(f"Interpretation: {interpretation}")
    report.append("=" * 60)
    
    return "\n".join(report)


def compute_uncertainty_score(entropy: float, n_clusters: int, n_samples: int) -> float:
    """
    Compute a combined uncertainty score that accounts for both entropy
    and number of clusters.

    This addresses the limitation of normalized entropy where 2 uniform clusters
    and 6 uniform clusters both get a score of 1.0.

    Args:
        entropy: Raw entropy value
        n_clusters: Number of clusters found
        n_samples: Total number of samples
        
    Returns:
        Combined uncertainty score in approximately [0, 1] range
        Higher = more uncertainty
    """
    if n_clusters <= 1:
        return 0.0

    # Component 1: Normalized entropy (how uniform is the distribution)
    normalized_ent = normalize_entropy(entropy, n_clusters)

    # Component 2: Cluster diversity (how many distinct meanings)
    # Normalize by theoretical maximum clusters (all samples different)
    cluster_diversity = (n_clusters - 1) / (n_samples - 1) if n_samples > 1 else 0.0

    # Combined score: weighted average
    # Entropy weight = 0.6, Diversity weight = 0.4
    # This gives more weight to distribution uniformity but also considers cluster count
    combined_score = 0.6 * normalized_ent + 0.4 * cluster_diversity

    return float(combined_score)