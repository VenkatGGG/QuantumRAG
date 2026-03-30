"""Embedding pipeline module for generating text embeddings.

This module implements a local embedding pipeline using raw transformers and torch.
Loads sentence-transformers/all-MiniLM-L6-v2 model and tokenizer.
Implements manual mean pooling with attention mask following formula:
    E = sum(T_i * M_i) / max(sum(M_i), 1e-9)

Output is 384-dimensional vectors.
"""

import torch
import numpy as np
from transformers import AutoModel, AutoTokenizer


def mean_pooling(last_hidden_state: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    """Perform mean pooling on token embeddings with attention mask.
    
    Implements the formula: E = sum(T_i * M_i) / max(sum(M_i), epsilon)
    where:
    - T_i is the token embedding at position i
    - M_i is the attention mask value at position i (0 or 1)
    - epsilon = 1e-9 prevents division by zero
    
    Args:
        last_hidden_state: Token embeddings from model output.
            Shape: (batch_size, seq_len, hidden_dim)
        attention_mask: Attention mask indicating valid tokens (1) and padding (0).
            Shape: (batch_size, seq_len)
    
    Returns:
        Pooled embeddings. Shape: (batch_size, hidden_dim)
    """
    # Expand attention mask to match embedding dimensions
    # Shape: (batch_size, seq_len, hidden_dim)
    mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
    
    # Apply mask to embeddings (zero out padding tokens)
    # sum(T_i * M_i) over seq_len dimension
    masked_embeddings = last_hidden_state * mask_expanded
    sum_embeddings = torch.sum(masked_embeddings, dim=1)
    
    # Sum of mask values per sample
    # sum(M_i) over seq_len dimension
    mask_sum = torch.sum(mask_expanded, dim=1)
    
    # Clamp to prevent division by zero: max(sum(M_i), epsilon)
    # epsilon = 1e-9
    mask_sum_clamped = torch.clamp(mask_sum, min=1e-9)
    
    # Divide: sum(T_i * M_i) / max(sum(M_i), epsilon)
    pooled = sum_embeddings / mask_sum_clamped
    
    return pooled


class EmbeddingPipeline:
    """Local embedding pipeline using sentence-transformers/all-MiniLM-L6-v2.
    
    Loads the model locally using transformers.AutoModel and implements
    manual mean pooling with attention mask. Produces 384-dimensional
    normalized embeddings.
    
    No external API calls are made - all processing is local.
    """
    
    # Model used for embeddings
    MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
    
    # Output dimension for all-MiniLM-L6-v2
    EMBEDDING_DIM = 384
    
    def __init__(self):
        """Initialize the embedding pipeline with model and tokenizer."""
        # Load model and tokenizer locally from cache
        # Use local_files_only=True to ensure we only use the pre-downloaded model
        # This prevents any runtime HF Hub requests and warnings
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.MODEL_NAME,
            local_files_only=True
        )
        self.model = AutoModel.from_pretrained(
            self.MODEL_NAME,
            local_files_only=True
        )
        
        # Set model to evaluation mode
        self.model.eval()
    
    def embed(self, text: str) -> np.ndarray:
        """Generate embedding for a text string.
        
        Tokenizes the text, runs through the model, and applies mean pooling
        to produce a 384-dimensional embedding vector.
        
        Args:
            text: Text to embed.
        
        Returns:
            384-dimensional embedding vector as numpy array.
        """
        # Tokenize the input text
        # return_tensors='pt' returns PyTorch tensors
        # padding=True ensures all sequences in batch have same length
        # truncation=True ensures we don't exceed model's max length
        encoded = self.tokenizer(
            text,
            padding=True,
            truncation=True,
            return_tensors='pt'
        )
        
        # Get input_ids and attention_mask
        input_ids = encoded['input_ids']
        attention_mask = encoded['attention_mask']
        
        # Run through model (no gradient computation needed)
        with torch.no_grad():
            outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
        
        # Use last_hidden_state, NOT pooler_output
        # last_hidden_state shape: (batch_size, seq_len, hidden_dim)
        last_hidden_state = outputs.last_hidden_state
        
        # Apply mean pooling with attention mask
        # pooled shape: (batch_size, hidden_dim)
        pooled = mean_pooling(last_hidden_state, attention_mask)
        
        # Convert to numpy and return single vector (remove batch dimension)
        embedding = pooled.squeeze(0).detach().cpu().numpy()
        
        return embedding
    
    def embed_batch(self, texts: list[str]) -> np.ndarray:
        """Generate embeddings for multiple texts.
        
        More efficient than calling embed() multiple times for batch processing.
        
        Args:
            texts: List of texts to embed.
        
        Returns:
            Array of embedding vectors. Shape: (len(texts), 384)
        """
        if not texts:
            return np.array([]).reshape(0, self.EMBEDDING_DIM)
        
        # Tokenize all texts together
        encoded = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            return_tensors='pt'
        )
        
        input_ids = encoded['input_ids']
        attention_mask = encoded['attention_mask']
        
        # Run through model
        with torch.no_grad():
            outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
        
        # Mean pooling
        last_hidden_state = outputs.last_hidden_state
        pooled = mean_pooling(last_hidden_state, attention_mask)
        
        # Convert to numpy
        embeddings = pooled.detach().cpu().numpy()
        
        return embeddings
