"""Tests for SemanticClusterer."""

import pytest
import numpy as np
from semantic_entropy.clustering import SemanticClusterer


class TestSemanticClusterer:
    
    @pytest.fixture
    def clusterer(self):
        """Create clusterer instance for testing."""
        return SemanticClusterer(
            distance_threshold=0.15,
            linkage="average"
        )
    
    def test_initialization(self):
        """Test clusterer initialization."""
        clusterer = SemanticClusterer()
        assert clusterer.distance_threshold == 0.15
        assert clusterer.linkage == "average"
        assert clusterer.metric == "cosine"
    
    def test_invalid_linkage(self):
        """Test error handling for invalid linkage."""
        with pytest.raises(ValueError):
            SemanticClusterer(linkage="invalid")
    
    def test_invalid_metric(self):
        """Test error handling for invalid metric."""
        with pytest.raises(ValueError):
            SemanticClusterer(metric="invalid")
    
    def test_single_embedding(self, clusterer):
        """Test clustering with single embedding."""
        embeddings = np.array([[1.0, 0.0, 0.0]])
        labels, n_clusters = clusterer.cluster(embeddings)
        
        assert n_clusters == 1
        assert labels[0] == 0
    
    def test_identical_embeddings(self, clusterer):
        """Test clustering with identical embeddings."""
        embeddings = np.array([
            [1.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
        ])
        labels, n_clusters = clusterer.cluster(embeddings)
        
        assert n_clusters == 1
        assert len(set(labels)) == 1
    
    def test_distinct_embeddings(self, clusterer):
        """Test clustering with distinct embeddings."""
        embeddings = np.array([
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ])
        labels, n_clusters = clusterer.cluster(embeddings)
        
        # These are orthogonal, should be in different clusters
        assert n_clusters > 1
    
    def test_similarity_threshold(self):
        """Test different distance thresholds."""
        embeddings = np.array([
            [1.0, 0.1, 0.0],
            [0.9, 0.2, 0.0],
            [0.0, 0.0, 1.0],
        ])
        
        # Strict threshold (low distance)
        strict_clusterer = SemanticClusterer(distance_threshold=0.05)
        _, n_strict = strict_clusterer.cluster(embeddings)
        
        # Lenient threshold (high distance)
        lenient_clusterer = SemanticClusterer(distance_threshold=0.5)
        _, n_lenient = lenient_clusterer.cluster(embeddings)
        
        # Strict should create more clusters
        assert n_strict >= n_lenient
    
    def test_get_similarity_matrix(self, clusterer):
        """Test similarity matrix computation."""
        embeddings = np.array([
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [1.0, 0.0, 0.0],
        ])
        
        sim_matrix = clusterer.get_similarity_matrix(embeddings)
        
        assert sim_matrix.shape == (3, 3)
        assert np.allclose(np.diag(sim_matrix), 1.0)  # Diagonal should be 1
        assert np.allclose(sim_matrix[0, 2], 1.0)  # Identical embeddings
        assert sim_matrix[0, 1] < 0.5  # Orthogonal embeddings
    
    def test_get_cluster_representatives(self, clusterer):
        """Test finding cluster representatives."""
        embeddings = np.array([
            [1.0, 0.0, 0.0],
            [0.9, 0.1, 0.0],
            [0.0, 1.0, 0.0],
        ])
        texts = ["Text A1", "Text A2", "Text B"]
        
        labels, _ = clusterer.cluster(embeddings)
        representatives = clusterer.get_cluster_representatives(
            embeddings, 
            labels, 
            texts
        )
        
        assert isinstance(representatives, dict)
        for cluster_id, info in representatives.items():
            assert 'index' in info
            assert 'size' in info
            assert 'text' in info
            assert info['text'] in texts
    
    def test_analyze_clusters(self, clusterer):
        """Test comprehensive cluster analysis."""
        embeddings = np.array([
            [1.0, 0.0, 0.0],
            [0.9, 0.1, 0.0],
            [0.0, 1.0, 0.0],
        ])
        texts = ["Text A1", "Text A2", "Text B"]
        
        labels, _ = clusterer.cluster(embeddings)
        analysis = clusterer.analyze_clusters(embeddings, labels, texts)
        
        assert 'n_clusters' in analysis
        assert 'cluster_sizes' in analysis
        assert 'representatives' in analysis
        assert 'cohesion' in analysis
        
        # Check cohesion values are in valid range [0, 1]
        for cluster_id, cohesion in analysis['cohesion'].items():
            assert 0.0 <= cohesion <= 1.0
    
    def test_linkage_methods(self):
        """Test different linkage methods."""
        embeddings = np.array([
            [1.0, 0.0, 0.0],
            [0.9, 0.1, 0.0],
            [0.8, 0.2, 0.0],
        ])
        
        for linkage in ['average', 'complete', 'single']:
            clusterer = SemanticClusterer(
                distance_threshold=0.15,
                linkage=linkage
            )
            labels, n_clusters = clusterer.cluster(embeddings)
            
            assert n_clusters >= 1
            assert len(labels) == 3


if __name__ == "__main__":
    pytest.main([__file__, "-v"])