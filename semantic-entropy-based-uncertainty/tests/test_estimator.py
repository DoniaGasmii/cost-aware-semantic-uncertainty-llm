"""Tests for SemanticUncertaintyEstimator."""

import pytest
import numpy as np
from semantic_entropy import SemanticUncertaintyEstimator


class TestSemanticUncertaintyEstimator:
    
    @pytest.fixture
    def estimator(self):
        """Create estimator instance for testing."""
        return SemanticUncertaintyEstimator(
            encoder_model="all-MiniLM-L6-v2",  # Smaller model for faster tests
            distance_threshold=0.15
        )
    
    def test_initialization(self):
        """Test estimator initialization."""
        estimator = SemanticUncertaintyEstimator()
        assert estimator is not None
        assert estimator.encoder is not None
        assert estimator.clusterer is not None
    
    def test_encode(self, estimator):
        """Test text encoding."""
        texts = ["Hello world", "Hi there"]
        embeddings = estimator.encode(texts)
        
        assert embeddings.shape[0] == 2
        assert embeddings.shape[1] == 384  # MiniLM embedding dimension
        assert embeddings.dtype == np.float32
    
    def test_single_text(self, estimator):
        """Test with single text input."""
        texts = ["The capital of France is Paris."]
        result = estimator.compute(texts)
        
        assert result['entropy'] == 0.0  # Single cluster = no entropy
        assert result['normalized_entropy'] == 0.0
        assert result['n_clusters'] == 1
        assert len(result['cluster_labels']) == 1
    
    def test_identical_texts(self, estimator):
        """Test with identical texts (should cluster into one)."""
        texts = ["Paris is the capital."] * 5
        result = estimator.compute(texts)
        
        assert result['n_clusters'] == 1
        assert result['entropy'] == 0.0
        assert result['normalized_entropy'] == 0.0
    
    def test_similar_texts(self, estimator):
        """Test with semantically similar but differently worded texts."""
        texts = [
            "The capital of France is Paris.",
            "Paris is the capital of France.",
            "It's Paris.",
        ]
        result = estimator.compute(texts)
        
        # Should cluster into 1 or 2 clusters (depending on threshold)
        assert result['n_clusters'] <= 2
        assert 0.0 <= result['normalized_entropy'] <= 1.0
    
    def test_different_texts(self, estimator):
        """Test with semantically different texts."""
        texts = [
            "The capital of France is Paris.",
            "The Earth orbits the Sun.",
            "Machine learning is a subset of AI.",
        ]
        result = estimator.compute(texts)
        
        # Should cluster into 3 separate clusters
        assert result['n_clusters'] >= 2
        assert result['entropy'] > 0
        assert result['normalized_entropy'] > 0
    
    def test_log_probs_length_mismatch(self, estimator):
        """Test error handling for mismatched log_probs."""
        texts = ["Text 1", "Text 2"]
        log_probs = [-1.0]  # Wrong length
        
        with pytest.raises(ValueError):
            estimator.compute(texts, log_probs=log_probs)
    
    def test_empty_texts(self, estimator):
        """Test error handling for empty input."""
        with pytest.raises(ValueError):
            estimator.compute([])
    
    def test_weighted_probabilities(self, estimator):
        """Test probability weighting with log_probs."""
        estimator.use_weighted_probs = True
        
        texts = [
            "Paris is the capital.",
            "Paris is the capital.",
            "Lyon is the capital.",
        ]
        
        # First two have high prob, third has low prob
        log_probs = [-1.0, -1.0, -10.0]
        
        result = estimator.compute(texts, log_probs=log_probs)
        
        # Should have 2 clusters
        assert result['n_clusters'] == 2
        
        # Paris cluster should have much higher probability
        cluster_probs = result['cluster_probs']
        assert max(cluster_probs) > 0.9  # Paris cluster dominates
    
    def test_return_details(self, estimator):
        """Test detailed output."""
        texts = ["Text 1", "Text 2", "Text 3"]
        result = estimator.compute(texts, return_details=True)
        
        assert 'embeddings' in result
        assert 'cluster_analysis' in result
        assert 'similarity_matrix' in result
        assert result['embeddings'].shape[0] == 3
        assert result['similarity_matrix'].shape == (3, 3)
    
    def test_cluster_representatives(self, estimator):
        """Test getting cluster representatives."""
        texts = [
            "Paris is the capital of France.",
            "The capital is Paris.",
            "London is the capital of England.",
        ]
        
        representatives = estimator.get_cluster_representatives(texts)
        
        assert len(representatives) >= 1
        for cluster_id, info in representatives.items():
            assert 'text' in info
            assert 'size' in info
            assert 'probability' in info
            assert info['text'] in texts
    
    def test_interpret_uncertainty(self, estimator):
        """Test uncertainty interpretation."""
        low = estimator.interpret_uncertainty(0.2)
        medium = estimator.interpret_uncertainty(0.5)
        high = estimator.interpret_uncertainty(0.8)
        
        assert "CONFIDENCE" in low
        assert "MODERATE" in medium
        assert "HIGH UNCERTAINTY" in high
    
    def test_compute_with_report(self, estimator):
        """Test report generation."""
        texts = ["Text 1", "Text 2"]
        report = estimator.compute_with_report(texts)
        
        assert isinstance(report, str)
        assert "SEMANTIC UNCERTAINTY REPORT" in report
        assert "Entropy" in report


if __name__ == "__main__":
    pytest.main([__file__, "-v"])