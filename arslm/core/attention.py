"""
Attention Mechanisms for ARSLM.

Implements various attention mechanisms including:
- Multi-head attention
- Self-attention
- Cross-attention
- Adaptive attention
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


class ScaledDotProductAttention(nn.Module):
    """Scaled Dot-Product Attention mechanism."""
    
    def __init__(self, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
    
    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute scaled dot-product attention.
        
        Args:
            query: Query tensor [batch, heads, seq_len, d_k]
            key: Key tensor [batch, heads, seq_len, d_k]
            value: Value tensor [batch, heads, seq_len, d_v]
            mask: Attention mask [batch, 1, seq_len, seq_len]
            
        Returns:
            - Attention output [batch, heads, seq_len, d_v]
            - Attention weights [batch, heads, seq_len, seq_len]
        """
        d_k = query.size(-1)
        
        # Calculate attention scores
        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
        
        # Apply mask if provided
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))
        
        # Apply softmax
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        # Apply attention to values
        output = torch.matmul(attention_weights, value)
        
        return output, attention_weights


class MultiHeadAttention(nn.Module):
    """
    Multi-Head Attention mechanism.
    
    Allows the model to jointly attend to information from different
    representation subspaces at different positions.
    """
    
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        dropout: float = 0.1,
        bias: bool = True
    ):
        """
        Initialize Multi-Head Attention.
        
        Args:
            hidden_size: Dimension of hidden states
            num_heads: Number of attention heads
            dropout: Dropout probability
            bias: Whether to use bias in linear layers
        """
        super().__init__()
        assert hidden_size % num_heads == 0, \
            "hidden_size must be divisible by num_heads"
        
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        
        # Linear projections
        self.query_proj = nn.Linear(hidden_size, hidden_size, bias=bias)
        self.key_proj = nn.Linear(hidden_size, hidden_size, bias=bias)
        self.value_proj = nn.Linear(hidden_size, hidden_size, bias=bias)
        self.output_proj = nn.Linear(hidden_size, hidden_size, bias=bias)
        
        # Attention mechanism
        self.attention = ScaledDotProductAttention(dropout)
        
        self.dropout = nn.Dropout(dropout)
        
    def _split_heads(self, x: torch.Tensor) -> torch.Tensor:
        """Split hidden states into multiple attention heads."""
        batch_size, seq_length, hidden_size = x.size()
        x = x.view(batch_size, seq_length, self.num_heads, self.head_dim)
        return x.transpose(1, 2)  # [batch, heads, seq_len, head_dim]
    
    def _merge_heads(self, x: torch.Tensor) -> torch.Tensor:
        """Merge attention heads back together."""
        batch_size = x.size(0)
        x = x.transpose(1, 2).contiguous()  # [batch, seq_len, heads, head_dim]
        return x.view(batch_size, -1, self.hidden_size)
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        key_value_states: Optional[torch.Tensor] = None,
        return_attention_weights: bool = False
    ) -> torch.Tensor:
        """
        Forward pass through multi-head attention.
        
        Args:
            hidden_states: Input tensor [batch_size, seq_length, hidden_size]
            attention_mask: Attention mask [batch_size, seq_length]
            key_value_states: Key/value states for cross-attention
            return_attention_weights: Whether to return attention weights
            
        Returns:
            Attention output tensor [batch_size, seq_length, hidden_size]
        """
        batch_size, seq_length, _ = hidden_states.size()
        
        # Linear projections
        query = self.query_proj(hidden_states)
        
        if key_value_states is None:
            # Self-attention
            key = self.key_proj(hidden_states)
            value = self.value_proj(hidden_states)
        else:
            # Cross-attention
            key = self.key_proj(key_value_states)
            value = self.value_proj(key_value_states)
        
        # Split into multiple heads
        query = self._split_heads(query)
        key = self._split_heads(key)
        value = self._split_heads(value)
        
        # Prepare attention mask
        if attention_mask is not None:
            # Expand mask for all heads
            attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
            attention_mask = attention_mask.expand(
                batch_size, 1, seq_length, seq_length
            )
        
        # Apply attention
        attn_output, attn_weights = self.attention(query, key, value, attention_mask)
        
        # Merge heads
        attn_output = self._merge_heads(attn_output)
        
        # Final linear projection
        output = self.output_proj(attn_output)
        output = self.dropout(output)
        
        if return_attention_weights:
            return output, attn_weights
        return output


class SelfAttention(nn.Module):
    """Self-Attention layer wrapper."""
    
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        dropout: float = 0.1
    ):
        super().__init__()
        self.attention = MultiHeadAttention(hidden_size, num_heads, dropout)
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Apply self-attention."""
        return self.attention(hidden_states, attention_mask)


class CrossAttention(nn.Module):
    """Cross-Attention layer for attending to external context."""
    
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        dropout: float = 0.1
    ):
        super().__init__()
        self.attention = MultiHeadAttention(hidden_size, num_heads, dropout)
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        context: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Apply cross-attention.
        
        Args:
            hidden_states: Query states
            context: Key/value states from external source
            attention_mask: Attention mask
            
        Returns:
            Attended output
        """
        return self.attention(
            hidden_states,
            attention_mask,
            key_value_states=context
        )


class AdaptiveAttention(nn.Module):
    """
    Adaptive Attention mechanism that dynamically adjusts attention patterns.
    
    This module learns to weight different attention mechanisms based on
    the input context.
    """
    
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.hidden_size = hidden_size
        
        # Multiple attention mechanisms
        self.self_attention = SelfAttention(hidden_size, num_heads, dropout)
        
        # Gating mechanism to choose attention pattern
        self.gate = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1),
            nn.Sigmoid()
        )
        
        # Alternative attention path
        self.alt_attention = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size)
        )
        
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass with adaptive attention selection.
        
        Args:
            hidden_states: Input tensor
            attention_mask: Attention mask
            
        Returns:
            Adaptively attended output
        """
        # Get attention output
        attn_output = self.self_attention(hidden_states, attention_mask)
        
        # Get alternative output
        alt_output = self.alt_attention(hidden_states)
        
        # Calculate gating weights based on mean pooling
        pooled = hidden_states.mean(dim=1, keepdim=True)
        gate_weight = self.gate(pooled)
        
        # Combine outputs
        output = gate_weight * attn_output + (1 - gate_weight) * alt_output
        
        return output


class LocalAttention(nn.Module):
    """
    Local Attention mechanism with limited attention window.
    
    More efficient for long sequences by restricting attention to a local window.
    """
    
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        window_size: int = 128,
        dropout: float = 0.1
    ):
        super().__init__()
        self.window_size = window_size
        self.attention = MultiHeadAttention(hidden_size, num_heads, dropout)
    
    def _create_local_mask(
        self,
        seq_length: int,
        device: torch.device
    ) -> torch.Tensor:
        """Create mask for local attention window."""
        mask = torch.ones(seq_length, seq_length, device=device)
        
        for i in range(seq_length):
            start = max(0, i - self.window_size // 2)
            end = min(seq_length, i + self.window_size // 2 + 1)
            mask[i, :start] = 0
            mask[i, end:] = 0
        
        return mask
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Apply local attention."""
        seq_length = hidden_states.size(1)
        device = hidden_states.device
        
        # Create local attention mask
        local_mask = self._create_local_mask(seq_length, device)
        
        # Combine with existing mask if provided
        if attention_mask is not None:
            local_mask = local_mask * attention_mask
        
        return self.attention(hidden_states, local_mask)