import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Union
import math
import logging

try:
    import xformers.ops as xops
    HAVE_XFORMERS = True
except ImportError:
    HAVE_XFORMERS = False

logger = logging.getLogger(__name__)

def create_local_block_mask(
    seq_len: int,
    block_size: int,
    device: torch.device
) -> torch.Tensor:
    """Create attention mask for local block attention"""
    # Create causal mask
    mask = torch.triu(
        torch.ones(seq_len, seq_len, device=device),
        diagonal=1
    ).bool()
    
    # Add block structure
    for i in range(seq_len):
        start = max(0, i - block_size)
        end = min(seq_len, i + block_size + 1)
        mask[i, start:end] = False
        
    return mask

class MultiHeadDifferentialAttention(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int = 8,
        dropout: float = 0.1,
        bias: bool = True,
        lambda_init: float = 0.8,
        config = None
    ):
        super().__init__()
        self.lambda_init = lambda_init  # Store lambda_init as instance variable
        super().__init__()
        if config is None:
            from config.model_config import ModelConfig
            config = ModelConfig(
                dim=d_model,
                num_heads=num_heads,
                lambda_init=lambda_init,
                dropout=dropout
            )
        self.config = config
        self.dim = d_model
        self.num_heads = num_heads
        self.dropout = dropout
        self.head_dim = self.dim // num_heads
        self.scale = self.head_dim ** -0.5
        
        # Linear layers
        self.q_proj = nn.Linear(self.dim, self.dim, bias=bias)
        self.k_proj = nn.Linear(self.dim, self.dim, bias=bias)
        self.v_proj = nn.Linear(self.dim, self.dim, bias=bias)
        self.out_proj = nn.Linear(self.dim, self.dim, bias=bias)
        
        # Dropout
        self.attn_dropout = nn.Dropout(dropout)
        self.resid_dropout = nn.Dropout(dropout)
        
        # Group norm per attention head
        self.group_norm = nn.GroupNorm(num_heads, self.dim, eps=1e-5)
        
        # Initialize layers
        with torch.no_grad():
            nn.init.zeros_(self.out_proj.weight)
            if bias:
                nn.init.zeros_(self.out_proj.bias)
                
        # Lambda parameters for differential attention (per head)
        self.lambda_q1 = nn.Parameter(torch.randn(num_heads, self.head_dim))
        self.lambda_k1 = nn.Parameter(torch.randn(num_heads, self.head_dim))
        self.lambda_q2 = nn.Parameter(torch.randn(num_heads, self.head_dim))
        self.lambda_k2 = nn.Parameter(torch.randn(num_heads, self.head_dim))
        
        # Initialize lambda parameters
        for param in [self.lambda_q1, self.lambda_k1, self.lambda_q2, self.lambda_k2]:
            nn.init.normal_(param, mean=0.0, std=lambda_init / math.sqrt(self.head_dim))
                
        # Create separate group norm for each head
        self.group_norms = nn.ModuleList([
            nn.GroupNorm(1, self.head_dim, eps=1e-5)
            for _ in range(num_heads)
        ])
                
        # Stats for monitoring
        self.stats = {}
        
    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        chunk_size: Optional[int] = None,
        local_block_size: Optional[int] = None
    ) -> Tuple[torch.Tensor, dict]:
        """
        Args:
            x: Input of shape [batch_size, seq_len, dim]
            mask: Optional attention mask
            chunk_size: Optional chunk size for chunked attention
            local_block_size: Optional local block size
        """
        batch_size, seq_len, _ = x.shape
        
        # Project to q, k, v
        q = self.q_proj(x)
        k = self.k_proj(x) 
        v = self.v_proj(x)
        
        # Reshape to multiple heads
        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim)
        k = k.view(batch_size, seq_len, self.num_heads, self.head_dim)
        v = v.view(batch_size, seq_len, self.num_heads, self.head_dim)
        
        # Create attention mask
        if local_block_size is not None:
            local_mask = create_local_block_mask(seq_len, local_block_size, x.device)
            mask = local_mask if mask is None else mask | local_mask
            
        # Use xFormers if available and enabled
        use_xformers = getattr(self.config, 'use_xformers', False)
        if use_xformers and HAVE_XFORMERS:
            # Reshape for xFormers
            q = q.transpose(1, 2)  # [batch, heads, seq, head_dim]
            k = k.transpose(1, 2)
            v = v.transpose(1, 2)
            
            # Convert mask for xFormers
            if mask is not None:
                mask = mask.to(torch.bool)
            
            # Run attention
            attn_output = xops.memory_efficient_attention(
                q, k, v,
                attn_bias=mask,
                p=self.dropout if self.training else 0.0
            )
            
            # Reshape output
            attn_output = attn_output.transpose(1, 2).contiguous()
            
        else:
            # Standard scaled dot-product attention
            q = q.transpose(1, 2)  # [batch, heads, seq, head_dim]
            k = k.transpose(1, 2)
            v = v.transpose(1, 2)            # Compute differential attention scores (per head)
            lambda_term1 = torch.exp((self.lambda_q1 * self.lambda_k1).sum(-1))  # [num_heads]
            lambda_term2 = torch.exp((self.lambda_q2 * self.lambda_k2).sum(-1))  # [num_heads]
            lambda_weights = lambda_term1 - lambda_term2 + self.lambda_init  # [num_heads]
            
            # Reshape lambda weights for broadcasting: [num_heads] -> [batch, num_heads, 1, 1]
            lambda_weights = lambda_weights.view(1, -1, 1, 1)

            # Apply scaled dot-product attention with lambda weights
            attn_weights = torch.matmul(q, k.transpose(-2, -1)) * self.scale * lambda_weights
            
            # Apply mask
            if mask is not None:
                attn_weights = attn_weights.masked_fill(mask, float('-inf'))
                
            # Softmax and dropout
            attn_weights = F.softmax(attn_weights, dim=-1)
            attn_weights = self.attn_dropout(attn_weights)
            
            # Compute output
            attn_output = torch.matmul(attn_weights, v)
            attn_output = attn_output.transpose(1, 2).contiguous()
            
        # Apply group norm per head
        attn_output = attn_output.view(batch_size, seq_len, self.num_heads, self.head_dim)
        normed_heads = []
        for i in range(self.num_heads):
            head_output = attn_output[..., i, :]  # [batch, seq, head_dim]
            normed_head = self.group_norms[i](head_output.transpose(1, 2)).transpose(1, 2)  # Apply norm across head_dim
            normed_heads.append(normed_head)
        attn_output = torch.stack(normed_heads, dim=2)  # [batch, seq, heads, head_dim]
        attn_output = attn_output.reshape(batch_size, seq_len, self.dim)
        
        # Project output
        output = self.out_proj(attn_output)
        output = self.resid_dropout(output)
        
        # Update stats
        if not self.config.use_xformers:
            entropy = -(attn_weights * torch.log(attn_weights + 1e-10)).sum(-1).mean().item()
            # Calculate sparsity as fraction of near-zero attention weights
            sparsity = (attn_weights < 0.01).float().mean().item()
            entropy = -(attn_weights * torch.log(attn_weights + 1e-10)).sum(-1).mean().item()
            
            self.stats = {
                'max_attn': attn_weights.max().item(),
                'mean_attn': attn_weights.mean().item(),
                'head_entropy': entropy,
                'attention_entropy': entropy,
                'sparsity': sparsity
            }
        else:
            self.stats.update({
                'max_attn': 0.0,
                'mean_attn': 0.0,
                'head_entropy': 0.0,
                'attention_entropy': 0.0
            })
        
        return output, self.stats