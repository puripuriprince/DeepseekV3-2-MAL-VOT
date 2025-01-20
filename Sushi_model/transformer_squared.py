import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict

class TransformerSquared(nn.Module):
    """
    Implementation of Transformer² (Transformer Squared) from the paper:
    'TRANSFORMER2: SELF-ADAPTIVE LLMS'
    """
    def __init__(
        self,
        dim: int = 768,
        num_layers: int = 12,
        num_heads: int = 12,
        mlp_ratio: int = 4,
        dropout: float = 0.1,
        max_sequence_length: int = 8192,
        use_gradient_checkpointing: bool = True
    ):
        super().__init__()
        
        # Initialize SVD components for each layer
        self.layers = nn.ModuleList([
            TransformerLayerSVD(
                dim=dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                dropout=dropout
            ) for _ in range(num_layers)
        ])
        
        self.use_gradient_checkpointing = use_gradient_checkpointing
        self.norm = nn.LayerNorm(dim)

    def forward(self,
                x: torch.Tensor,
                z_vectors: Optional[Dict[str, torch.Tensor]] = None) -> Dict[str, torch.Tensor]:
        """
        Forward pass with SVD-based adaptation
        Args:
            x: Input tensor
            z_vectors: Optional SVD scaling vectors for adaptation
        """
        attention_maps = []
        
        # Process through layers with SVD adaptation
        for i, layer in enumerate(self.layers):
            # Get layer-specific z vector if provided
            z = None if z_vectors is None else z_vectors.get(f'layer_{i}')
            
            if self.use_gradient_checkpointing and self.training:
                x, attn = torch.utils.checkpoint.checkpoint(layer, x, z)
            else:
                x, attn = layer(x, z_vector=z)
            attention_maps.append(attn)
            
        # Apply final layer norm
        x = self.norm(x)
        
        return {
            'hidden_states': x,
            'attention_maps': attention_maps
        }

class TransformerLayerSVD(nn.Module):
    """Transformer layer with SVD-based adaptation"""
    def __init__(self,
                 dim: int,
                 num_heads: int,
                 mlp_ratio: int = 4,
                 dropout: float = 0.1):
        super().__init__()
        
        # Attention with SVD adaptation
        self.attention = SVDAttention(
            dim=dim,
            num_heads=num_heads,
            dropout=dropout
        )
        
        # MLP
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * mlp_ratio),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * mlp_ratio, dim),
            nn.Dropout(dropout)
        )
        
        # Layer norms
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        
    def forward(self, 
                x: torch.Tensor,
                z_vector: Optional[torch.Tensor] = None) -> tuple:
        # Attention with SVD adaptation
        attn_out, attn_maps = self.attention(self.norm1(x), z_vector=z_vector)
        x = x + attn_out
        
        # MLP
        x = x + self.mlp(self.norm2(x))
        
        return x, attn_maps

class SVDAttention(nn.Module):
    """Attention module with SVD-based adaptation"""
    def __init__(self,
                 dim: int,
                 num_heads: int,
                 dropout: float = 0.1):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        
        # QKV projections
        self.to_qkv = nn.Linear(dim, dim * 3)
        self.to_out = nn.Linear(dim, dim)
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self,
                x: torch.Tensor,
                z_vector: Optional[torch.Tensor] = None) -> tuple:
        batch_size, seq_len, _ = x.shape
        
        # QKV projection
        qkv = self.to_qkv(x)
        q, k, v = qkv.chunk(3, dim=-1)
        
        # Reshape for attention
        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Apply SVD adaptation if z_vector provided
        if z_vector is not None:
            # Scale singular values
            q = q * z_vector.view(1, -1, 1, 1)
            k = k * z_vector.view(1, -1, 1, 1)
            v = v * z_vector.view(1, -1, 1, 1)
        
        # Compute attention
        attn = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)
        
        # Apply attention to values
        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).reshape(batch_size, seq_len, -1)
        
        # Output projection
        out = self.to_out(out)
        
        return out, attn