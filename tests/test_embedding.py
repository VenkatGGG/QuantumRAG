"""Tests for the embedding pipeline.

This module tests the embedding pipeline implementation including:
- Mean pooling formula: E = sum(T_i * M_i) / max(sum(M_i), epsilon)
- Attention mask application
- Output dimension verification (384)
- Model loading without external APIs
"""

import pytest
import torch
import numpy as np
from unittest.mock import Mock, patch

# Import the module to test (will be created)
from src.embedding import EmbeddingPipeline, mean_pooling


class TestMeanPooling:
    """Tests for the mean pooling implementation."""
    
    def test_mean_pooling_formula(self):
        """Test that mean pooling follows E = sum(T_i * M_i) / max(sum(M_i), epsilon).
        
        Uses mocked tensors with known expected outputs.
        """
        # Create simple test case: 2 tokens, 3 dimensions
        # last_hidden_state: [batch=1, seq_len=2, hidden_dim=3]
        last_hidden_state = torch.tensor([
            [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]
        ])  # shape: (1, 2, 3)
        
        # attention_mask: [batch=1, seq_len=2] - both tokens valid
        attention_mask = torch.tensor([[1, 1]])
        
        # Expected: sum over seq_len dimension
        # Token 1: [1, 2, 3], Token 2: [4, 5, 6]
        # Sum: [5, 7, 9]
        # Mask sum: 2
        # Mean: [5/2, 7/2, 9/2] = [2.5, 3.5, 4.5]
        expected = torch.tensor([[2.5, 3.5, 4.5]])
        
        result = mean_pooling(last_hidden_state, attention_mask)
        
        assert torch.allclose(result, expected, atol=1e-6)
    
    def test_attention_mask_applied(self):
        """Test that padding tokens are ignored in pooling calculation."""
        # Create test case with padding
        last_hidden_state = torch.tensor([
            [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [100.0, 100.0, 100.0]]
        ])  # shape: (1, 3, 3)
        
        # Mask: first two tokens valid, third is padding (0)
        attention_mask = torch.tensor([[1, 1, 0]])
        
        # Expected: only first two tokens contribute
        # Sum: [1+4, 2+5, 3+6] = [5, 7, 9]
        # Mask sum: 2
        # Mean: [2.5, 3.5, 4.5]
        expected = torch.tensor([[2.5, 3.5, 4.5]])
        
        result = mean_pooling(last_hidden_state, attention_mask)
        
        assert torch.allclose(result, expected, atol=1e-6)
        
        # Verify padding token values don't affect result
        # If padding was NOT applied, result would include the 100s
        wrong_result = torch.tensor([[35.0, 35.666667, 36.333333]])
        assert not torch.allclose(result, wrong_result, atol=1e-6)
    
    def test_epsilon_prevents_division_by_zero(self):
        """Test that epsilon prevents division by zero when all tokens are masked."""
        # Create test case where all tokens are padding
        last_hidden_state = torch.tensor([
            [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]
        ])  # shape: (1, 2, 3)
        
        # All tokens masked (padding)
        attention_mask = torch.tensor([[0, 0]])
        
        # Should not raise error - epsilon prevents division by zero
        result = mean_pooling(last_hidden_state, attention_mask)
        
        # With epsilon = 1e-9, sum(M_i) = 0, so denominator = max(0, 1e-9) = 1e-9
        # Sum of vectors is [0, 0, 0] (masked), divided by 1e-9 = [0, 0, 0]
        expected = torch.tensor([[0.0, 0.0, 0.0]])
        
        assert torch.allclose(result, expected, atol=1e-6)
        assert result.shape == (1, 3)


class TestEmbeddingPipeline:
    """Tests for the EmbeddingPipeline class."""
    
    @patch('src.embedding.AutoModel')
    @patch('src.embedding.AutoTokenizer')
    def test_model_loading(self, mock_tokenizer_class, mock_model_class):
        """Test that model loads without external API calls."""
        # Setup mocks
        mock_tokenizer = Mock()
        mock_tokenizer_class.from_pretrained.return_value = mock_tokenizer
        
        mock_model = Mock()
        mock_model_class.from_pretrained.return_value = mock_model
        
        # Create pipeline
        pipeline = EmbeddingPipeline()
        
        # Verify from_pretrained was called with local model only
        mock_model_class.from_pretrained.assert_called_once_with(
            "sentence-transformers/all-MiniLM-L6-v2",
            local_files_only=True
        )
        mock_tokenizer_class.from_pretrained.assert_called_once_with(
            "sentence-transformers/all-MiniLM-L6-v2",
            local_files_only=True
        )
    
    @patch('src.embedding.AutoModel')
    @patch('src.embedding.AutoTokenizer')
    def test_embedding_shape(self, mock_tokenizer_class, mock_model_class):
        """Test that output is 384-dimensional."""
        # Setup mocks
        mock_tokenizer = Mock()
        mock_tokenizer_class.from_pretrained.return_value = mock_tokenizer
        
        # Mock tokenizer output
        mock_tokenizer.return_value = {
            'input_ids': torch.tensor([[1, 2, 3]]),
            'attention_mask': torch.tensor([[1, 1, 1]])
        }
        
        # Mock model output - 384 dimensions
        mock_model = Mock()
        mock_outputs = Mock()
        mock_outputs.last_hidden_state = torch.randn(1, 3, 384)
        mock_model.return_value = mock_outputs
        mock_model_class.from_pretrained.return_value = mock_model
        
        # Create pipeline and embed
        pipeline = EmbeddingPipeline()
        result = pipeline.embed("test text")
        
        # Verify output shape is 384
        assert result.shape == (384,)
    
    @patch('src.embedding.AutoModel')
    @patch('src.embedding.AutoTokenizer')
    def test_deterministic_output(self, mock_tokenizer_class, mock_model_class):
        """Test that same input produces same embedding."""
        # Setup mocks with fixed output
        mock_tokenizer = Mock()
        mock_tokenizer_class.from_pretrained.return_value = mock_tokenizer
        
        # Mock tokenizer output
        mock_tokenizer.return_value = {
            'input_ids': torch.tensor([[1, 2, 3]]),
            'attention_mask': torch.tensor([[1, 1, 1]])
        }
        
        # Mock model with fixed output
        mock_model = Mock()
        fixed_output = torch.tensor([[[1.0] * 384, [2.0] * 384, [3.0] * 384]])
        mock_outputs = Mock()
        mock_outputs.last_hidden_state = fixed_output
        mock_model.return_value = mock_outputs
        mock_model_class.from_pretrained.return_value = mock_model
        
        # Create pipeline
        pipeline = EmbeddingPipeline()
        
        # Embed same text twice
        result1 = pipeline.embed("test text")
        result2 = pipeline.embed("test text")
        
        # Results should be identical
        assert np.allclose(result1, result2)
    
    @patch('src.embedding.AutoModel')
    @patch('src.embedding.AutoTokenizer')
    def test_uses_last_hidden_state_not_pooler(self, mock_tokenizer_class, mock_model_class):
        """Test that embedding uses last_hidden_state, not pooler_output."""
        # Setup mocks
        mock_tokenizer = Mock()
        mock_tokenizer_class.from_pretrained.return_value = mock_tokenizer
        
        mock_tokenizer.return_value = {
            'input_ids': torch.tensor([[1, 2, 3]]),
            'attention_mask': torch.tensor([[1, 1, 1]])
        }
        
        # Mock model with both last_hidden_state and pooler_output
        mock_model = Mock()
        last_hidden = torch.randn(1, 3, 384)
        pooler_output = torch.randn(1, 384)  # Different values
        
        mock_outputs = Mock()
        mock_outputs.last_hidden_state = last_hidden
        mock_outputs.pooler_output = pooler_output
        mock_model.return_value = mock_outputs
        mock_model_class.from_pretrained.return_value = mock_model
        
        # Create pipeline
        pipeline = EmbeddingPipeline()
        result = pipeline.embed("test text")
        
        # Calculate what the mean pooling should produce from last_hidden_state
        # With attention_mask all 1s, it's just the mean of the 3 tokens
        expected_from_last_hidden = last_hidden.squeeze(0).mean(dim=0).numpy()
        
        # Result should match mean pooling of last_hidden_state, not pooler_output
        assert np.allclose(result, expected_from_last_hidden, atol=1e-6)
        
        # Result should NOT match pooler_output
        assert not np.allclose(result, pooler_output.squeeze(0).numpy(), atol=1e-6)
