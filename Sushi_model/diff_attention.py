import torch 
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Tuple
from titans_pytorch import MemoryAsContextTransformer

class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5, elementwise_affine: bool = True):
        super().__init__()
        self.eps = eps
        self.elementwise_affine = elementwise_affine
        if elementwise_affine:
            self.weight = nn.Parameter(torch.ones(dim))
        else:
            self.register_parameter('weight', None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Calculate RMS
        rms = torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        # Normalize
        x = x * rms
        # Scale if needed
        if self.weight is not None:
            x = x * self.weight
        return x

class DifferentialAttention(nn.Module):
    def __init__(self, 
                 dim: int, 
                 num_heads: int,
                 dropout: float = 0.1,
                 init_lambda: float = 0.8):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        
        # QKV projections
        self.to_q = nn.Linear(dim, dim)
        self.to_k = nn.Linear(dim, dim)
        self.to_v = nn.Linear(dim, dim)
        
        # Output projection
        self.to_out = nn.Linear(dim, dim)
        
        # Differential attention parameters
        self.lambda_param = nn.Parameter(torch.ones(1) * init_lambda)
        
        # GroupNorm for attention scores
        self.group_norm = nn.GroupNorm(num_groups=num_heads, 
                                     num_channels=num_heads,
                                     eps=1e-6)
        
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size, seq_len, _ = x.shape
        
        # Project to Q, K, V
        q = self.to_q(x).view(batch_size, seq_len, self.num_heads, self.head_dim)
        k = self.to_k(x).view(batch_size, seq_len, self.num_heads, self.head_dim)
        v = self.to_v(x).view(batch_size, seq_len, self.num_heads, self.head_dim)
        
        # Transpose for attention computation
        q = q.transpose(1, 2)  # [batch, heads, seq_len, head_dim]
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        
        # Split heads for differential attention (only Q and K)
        q1, q2 = q.chunk(2, dim=-1)  # Split along head_dim instead of seq_len
        k1, k2 = k.chunk(2, dim=-1)
        
        # Compute attention scores with adjusted dimensions
        attn1 = torch.matmul(q1, k1.transpose(-2, -1)) * self.scale
        attn2 = torch.matmul(q2, k2.transpose(-2, -1)) * self.scale
        
        # Apply differential attention mechanism
        attn = F.softmax(attn1, dim=-1) - self.lambda_param * F.softmax(attn2, dim=-1)
        
        # Apply GroupNorm to attention scores
        attn = self.group_norm(attn)
        attn = self.dropout(attn)
        
        # Apply attention to values
        out = torch.matmul(attn, v)
        
        # Reshape and project output
        out = out.transpose(1, 2).contiguous().view(batch_size, seq_len, -1)
        out = self.to_out(out)
        
        return out, attn

class TransformerLayerWithMemory(nn.Module):
    def __init__(self,
                 dim: int,
                 num_heads: int,
                 mlp_ratio: int = 4,
                 dropout: float = 0.1,
                 memory_size: int = 512,
                 segment_len: int = 128,
                 num_persist_mem_tokens: int = 4,
                 num_longterm_mem_tokens: int = 16):
        super().__init__()
        
        # Differential attention
        self.diff_attn = DifferentialAttention(
            dim=dim,
            num_heads=num_heads,
            dropout=dropout
        )
        
        # Memory components
        self.memory = MemoryAsContextTransformer(
            num_tokens=memory_size,
            dim=dim,
            depth=2,
            segment_len=segment_len,
            num_persist_mem_tokens=num_persist_mem_tokens,
            num_longterm_mem_tokens=num_longterm_mem_tokens,
            neural_memory_kwargs=dict(
                dim_head=64,
                heads=4,
                default_model_kwargs=dict(
                    depth=2
                )
            )
        )
        
        # Memory gating
        self.memory_gate = nn.Sequential(
            nn.Linear(dim, dim, bias=True),
            nn.GELU(),
            nn.Linear(dim, 1, bias=True)
        )
        
        # MLP with bias
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * mlp_ratio, bias=True),
            nn.GELU(),
            nn.Linear(dim * mlp_ratio, dim, bias=True)
        )
        
        # Layer norms with bias
        self.norm1 = nn.LayerNorm(dim, elementwise_affine=True)
        self.norm2 = nn.LayerNorm(dim, elementwise_affine=True)
        self.norm3 = nn.LayerNorm(dim, elementwise_affine=True)
