# scripts/se_utils.py
"""
Semantic entropy pipeline:
- embed candidate texts
- cluster embeddings
- compute entropy of cluster mass distribution
"""

from typing import List, Dict, Any
import numpy as np
from sklearn.cluster import KMeans, DBSCAN
from sentence_transformers import SentenceTransformer

_MODEL_CACHE: dict[str, SentenceTransformer] = {}

def _get_encoder(name: str) -> SentenceTransformer:
    m = _MODEL_CACHE.get(name)
    if m is None:
        m = SentenceTransformer(name)
        _MODEL_CACHE[name] = m
    return m

def embed_texts(texts: List[str], model_name: str) -> np.ndarray:
    if not texts:
        return np.zeros((0, 384), dtype=np.float32)
    enc = _get_encoder(model_name)
    emb = enc.encode(texts, show_progress_bar=False, normalize_embeddings=True)
    return np.asarray(emb, dtype=np.float32)

def cluster_embeddings(emb: np.ndarray, algo: str, max_clusters: int) -> np.ndarray:
    n = emb.shape[0]
    if n <= 2:
        return np.arange(n, dtype=np.int32)
    algo = algo.lower()
    if algo == "dbscan":
        # cosine distance = 1 - cosine sim
        lab = DBSCAN(eps=0.2, min_samples=2, metric="cosine").fit(emb).labels_
        lab = lab.astype(np.int32, copy=False)
        if (lab == -1).any():
            noise_idx = np.where(lab == -1)[0]
            start = (lab[lab >= 0].max() + 1) if (lab >= 0).any() else 0
            lab = lab.copy()
            for i, idx in enumerate(noise_idx):
                lab[idx] = start + i
        return lab
    # k-means with fixed n_init for portability
    k = int(min(max_clusters, max(1, n // 2)))
    km = KMeans(n_clusters=k, n_init=10, random_state=0)
    return km.fit_predict(emb).astype(np.int32, copy=False)

def semantic_entropy(labels: np.ndarray) -> float:
    if labels.size == 0:
        return 0.0
    _, counts = np.unique(labels, return_counts=True)
    p = counts / counts.sum()
    p = p[p > 0]
    return float(-(p * np.log(p)).sum())

def dominant_cluster_fraction(labels: np.ndarray) -> float:
    if labels.size == 0:
        return 0.0
    _, counts = np.unique(labels, return_counts=True)
    return float(counts.max() / counts.sum())

def compute_se(candidates: List[str], model_name: str, algo: str, max_clusters: int) -> Dict[str, Any]:
    emb = embed_texts(candidates, model_name=model_name)
    if emb.shape[0] == 0:
        return {"labels": [], "entropy": 0.0, "dominant_frac": 0.0}
    labels = cluster_embeddings(emb, algo=algo, max_clusters=max_clusters)
    se = semantic_entropy(labels)
    dom = dominant_cluster_fraction(labels)
    return {"labels": labels.tolist(), "entropy": se, "dominant_frac": dom}
