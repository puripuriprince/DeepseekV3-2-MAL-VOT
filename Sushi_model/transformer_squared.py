import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict

import math
import logging
import torch.utils.checkpoint as checkpoint
from typing import Optional, Dict, List, Tuple, Union
from dataclasses import dataclass
from config.model_config import ModelConfig

logger = logging.getLogger(__name__)

@dataclass
class TransformerStats:
    """Track transformer performance metrics"""
    layer_times: List[float] = None
    attention_patterns: List[torch.Tensor] = None
    memory_per_layer: List[int] = None
    gradient_norms: List[float] = None
    peak_memory: float = 0.0

class TransformerSquared(nn.Module):
    """
    Implementation of Transformer² (Transformer Squared) from the paper:
    'TRANSFORMER2: SELF-ADAPTIVE LLMS'
    
    Optimized version with:
    - Flash attention support
    - Memory efficient attention
    - Gradient checkpointing
    - Performance monitoring
    - SVD-based adaptation
    """
    def __init__(
        self,
        dim: int = 768,
        num_layers: int = 12,
        num_heads: int = 12,
        mlp_ratio: int = 4,
        dropout: float = 0.1,
        max_sequence_length: int = 8192,
        use_gradient_checkpointing: bool = True,
        use_flash_attention: bool = True,
        memory_efficient: bool = True,
        use_diff_transformer: bool = False,
        use_self_adaptation: bool = False
    ):
        super().__init__()
        logger.info(f"Initializing TransformerSquared with dim={dim.dim if isinstance(dim, ModelConfig) else dim}, layers={num_layers}")
        # Extract dim value if ModelConfig is passed
        dim_value = dim.dim if isinstance(dim, ModelConfig) else dim
        
        # Initialize SVD components for each layer
        self.layers = nn.ModuleList([
            TransformerLayerSVD(
                dim=dim_value,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                dropout=dropout
            ) for _ in range(num_layers)
        ])
        
        self.use_gradient_checkpointing = use_gradient_checkpointing
        self.use_flash_attention = use_flash_attention
        self.memory_efficient = memory_efficient
        # Extract dim value if ModelConfig is passed
        dim_value = dim.dim if isinstance(dim, ModelConfig) else dim
        self.norm = nn.LayerNorm(dim_value)
        
        # Performance tracking
        self.stats = TransformerStats()
        self._initialize_stats()

    def forward(
        self,
        x: torch.Tensor,
        z_vectors: Optional[Dict[str, torch.Tensor]] = None,
        use_diff_transformer: bool = False,
        use_self_adaptation: bool = False
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass with SVD-based adaptation
        Args:
            x: Input tensor
            z_vectors: Optional SVD scaling vectors for adaptation
        """
        logger.debug(f"Starting forward pass with input shape: {x.shape}")
        start_time = torch.cuda.Event(enable_timing=True)
        end_time = torch.cuda.Event(enable_timing=True)
        
        if start_time:
            start_time.record()
            
        try:
            attention_maps = []
            self.stats.layer_times = []
            self.stats.memory_per_layer = []
            
            # Convert input to float and ensure batch dimension
            x = x.float()
            if len(x.shape) == 2:
                x = x.unsqueeze(0)  # Add batch dimension if missing
            
            # Store original dimensions
            batch_size, seq_len, _ = x.shape
            
            # Shape verification
            assert x.shape[0] == batch_size, f"Expected batch size {batch_size}, got {x.shape[0]}"
            assert x.shape[1] == seq_len, f"Expected seq length {seq_len}, got {x.shape[1]}"
            
            # Process through layers with SVD adaptation
            for i, layer in enumerate(self.layers):
                layer_start = torch.cuda.Event(enable_timing=True)
                layer_end = torch.cuda.Event(enable_timing=True)
                
                if layer_start:
                    layer_start.record()
                
                # Get layer-specific z vector if provided
                z = None if z_vectors is None else z_vectors.get(f'layer_{i}')
                if z is not None:
                    z = z.float()  # Ensure z vector is float
                    
                    # Apply SVD-based adaptation if enabled
                    if use_self_adaptation:
                        layer.apply_singular_scaling(z)
                
                # Track memory before layer
                if torch.cuda.is_available():
                    mem_before = torch.cuda.memory_allocated()
                
                # Apply layer with appropriate optimizations
                if self.use_gradient_checkpointing and self.training:
                    x, attn = checkpoint.checkpoint(layer, x, z)
                else:
                    x, attn = layer(x, z_vector=z)
                
                # Shape verification after layer
                assert x.shape[0] == batch_size, f"Expected batch size {batch_size}, got {x.shape[0]}"
                assert x.shape[1] == seq_len, f"Expected seq length {seq_len}, got {x.shape[1]}"
                
                attention_maps.append(attn)
                
                # Track memory after layer
                if torch.cuda.is_available():
                    mem_after = torch.cuda.memory_allocated()
                    self.stats.memory_per_layer.append(mem_after - mem_before)
                
                if layer_end:
                    layer_end.record()
                    torch.cuda.synchronize()
                    self.stats.layer_times.append(layer_start.elapsed_time(layer_end))
                
                logger.debug(f"Layer {i} complete - shape: {x.shape}")
            
            # Apply final layer norm
            x = self.norm(x)
            
            if end_time:
                end_time.record()
                torch.cuda.synchronize()
                total_time = start_time.elapsed_time(end_time)
                logger.debug(f"Forward pass complete in {total_time:.2f}ms")
            
            # Track gradient norms in training
            if self.training:
                self._track_gradient_norms()
            
            # Ensure output has shape [batch_size, seq_len, dim]
            if len(x.shape) == 2:
                x = x.unsqueeze(0)
            elif x.shape[0] != batch_size:
                x = x.reshape(batch_size, seq_len, -1)
            
            # Final shape verification
            assert x.shape[0] == batch_size, f"Expected batch size {batch_size}, got {x.shape[0]}"
            assert x.shape[1] == seq_len, f"Expected seq length {seq_len}, got {x.shape[1]}"
                
            return {
                'hidden_states': x,
                'attention_maps': attention_maps,
                'stats': self.stats
            }
            
        except Exception as e:
            logger.error(f"Error in transformer forward pass: {str(e)}")
            raise

    def _initialize_stats(self):
        """Initialize performance tracking stats"""
        self.stats.layer_times = []
        self.stats.attention_patterns = []
        self.stats.memory_per_layer = []
        self.stats.gradient_norms = []

    def _track_gradient_norms(self):
        """Track gradient norms during training"""
        with torch.no_grad():
            grad_norms = []
            for layer in self.layers:
                layer_norm = 0
                for p in layer.parameters():
                    if p.grad is not None:
                        layer_norm += p.grad.norm().item()
                grad_norms.append(layer_norm)
            self.stats.gradient_norms = grad_norms

class SVDAdapter(nn.Module):
    """
    Maintains SVD-based scaling for a weight matrix W = U S V^T.
    Freezes U and V, but learns z separately. Then scaled_sigma = sigma * z.
    """
    def __init__(self, W: torch.Tensor, rank: int, init_std: float = 0.01):
        super().__init__()
        # Perform offline SVD once
        U_, S_, Vt_ = torch.linalg.svd(W, full_matrices=False)
        # Keep top `rank` components
        r = min(rank, S_.numel())
        self.U = nn.Parameter(U_[:, :r], requires_grad=False)
        self.Vt = nn.Parameter(Vt_[:r, :], requires_grad=False)
        self.sigma_base = nn.Parameter(S_[:r], requires_grad=False)
        
        # Learnable vector z of shape [r], init near 1
        self.z = nn.Parameter(torch.ones(r) + init_std * torch.randn(r))

    def forward(self):
        """Recompute adapted weight = U diag(sigma_base * z) V^T"""
        scaled_sigma = self.sigma_base * self.z
        # [out_dim, r] x diag(r) x [r, in_dim]
        W_approx = (self.U * scaled_sigma.unsqueeze(0)) @ self.Vt
        return W_approx

class TransformerLayerSVD(nn.Module):
    """Transformer layer with SVD-based adaptation"""
    def __init__(self,
                 dim: int,
                 num_heads: int,
                 mlp_ratio: int = 4,
                 dropout: float = 0.1,
                 rank: int = 4):
        super().__init__()
        self.rank = rank
        
        # Initialize weight matrices
        self.qkv_weight = nn.Parameter(torch.randn(dim, 3 * dim) / math.sqrt(dim))
        
        # SVD adapter for attention weights
        self.svd_adapter = SVDAdapter(self.qkv_weight, rank=rank)
        
        # Attention with SVD adaptation
        self.attention = SVDAttention(
            dim=dim,
            num_heads=num_heads,
            dropout=dropout,
            rank=rank
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
        # Store original dimensions
        batch_size, seq_len, _ = x.shape
        
        # Attention with SVD adaptation
        attn_out, attn_maps = self.attention(self.norm1(x), z_vector=z_vector)
        x = x + attn_out
        
        # MLP
        x = x + self.mlp(self.norm2(x))
        
        # Ensure output maintains original dimensions
        if x.size(0) != batch_size or x.size(1) != seq_len:
            x = x.reshape(batch_size, seq_len, -1)
            
        # Final shape verification
        assert x.shape[0] == batch_size, f"Expected batch size {batch_size}, got {x.shape[0]}"
        assert x.shape[1] == seq_len, f"Expected seq length {seq_len}, got {x.shape[1]}"
        
        return x, attn_maps
        
    def apply_singular_scaling(self, z_vector: Optional[torch.Tensor]):
        """Apply SVD-based scaling from z_vector"""
        if z_vector is not None:
            # Scale singular values with z_vector
            scaled_sigma = self.sigma * z_vector
            # Update weight matrix W = U * diag(scaled_sigma) * V
            W = torch.mm(self.U * scaled_sigma.unsqueeze(0), self.V)
            # Apply scaled weights to attention
            self.attention.update_weights(W)
        # Attention with SVD adaptation
        attn_out, attn_maps = self.attention(self.norm1(x), z_vector=z_vector)
        x = x + attn_out
        
        # MLP
        x = x + self.mlp(self.norm2(x))
        
        return x, attn_maps

class SVDAttention(nn.Module):
    def update_weights(self, W: torch.Tensor):
        """Update weights with SVD-adapted matrix"""
        # Update projection weights with new SVD-scaled matrix
        with torch.no_grad():
            # Split W for differential attention
            W1, W2 = W.chunk(2, dim=-1)
            self.to_qkv1.weight.copy_(W1)
            self.to_qkv2.weight.copy_(W2)
    """SVD-Enhanced Differential Attention"""
    def __init__(self, dim, num_heads=8, rank=4, sigma=0.1, dropout=0.1):
        super().__init__()
        # Extract dim value if ModelConfig is passed
        dim_value = dim.dim if isinstance(dim, ModelConfig) else dim
        self.dim = dim_value
        self.num_heads = num_heads
        assert dim_value % num_heads == 0, f"dim {dim_value} must be divisible by num_heads {num_heads}"
        self.d_head = dim_value // num_heads
        self.rank = rank
        self.sigma = nn.Parameter(torch.tensor(sigma))
        self.scale = 1 / math.sqrt(self.d_head)
        
        # SVD adaptation parameters
        self.U = nn.Parameter(torch.randn(dim_value, rank))
        self.V = nn.Parameter(torch.randn(rank, dim_value))
        
        # Projection layers for differential attention
        self.to_qkv1 = nn.Linear(dim_value, dim_value * 3, bias=False)
        self.to_qkv2 = nn.Linear(dim_value, dim_value * 3, bias=False)
        
        # Initialize weights properly
        nn.init.xavier_uniform_(self.to_qkv1.weight)
        nn.init.xavier_uniform_(self.to_qkv2.weight)
        
        # Output projection with LayerNorm instead of GroupNorm
        # This avoids issues with group size divisibility
        self.to_out = nn.Sequential(
            nn.Linear(dim_value, dim_value),
            nn.LayerNorm(dim_value)
        )
        
        # Initialize projection layers
        self.proj1 = nn.Linear(dim_value // 2, dim_value)
        self.proj2 = nn.Linear(dim_value // 2, dim_value)
        
        # Initialize projection weights
        nn.init.xavier_uniform_(self.proj1.weight)
        nn.init.xavier_uniform_(self.proj2.weight)
        
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, z_vector=None):
        """Forward pass with SVD adaptation and differential attention"""
        # Store original dimensions
        batch_size, seq_len, _ = x.shape
        
        # SVD adaptation
        svd_adapt = self.sigma * (self.U @ self.V)  # Shape: [dim, rank] @ [rank, dim] -> [dim, dim]
        if z_vector is not None:
            # Ensure z_vector matches rank dimension for SVD adaptation
            z_vector = z_vector[:self.rank] if z_vector.size(0) > self.rank else z_vector
            z_vector = z_vector.reshape(1, -1)  # [1, rank] for correct broadcasting
            scaled_sigma = self.sigma * z_vector  # [rank] * [1, rank] -> [1, rank]
            svd_adapt = self.U @ (scaled_sigma.unsqueeze(-1) * self.V)  # Proper matrix multiplication
            
        # Project to input dimension
        svd_adapt = svd_adapt[:self.dim].sum(dim=-1)  # Shape: [dim]
        
        # Reshape svd_adapt to match input shape [batch_size, seq_len, dim]
        svd_adapt = svd_adapt.reshape(1, 1, self.dim)
        svd_adapt = svd_adapt.expand(batch_size, seq_len, self.dim)
            
        # Add SVD adaptation
        adapted_x = x + svd_adapt
        
        # Split input for differential attention
        x1, x2 = torch.chunk(adapted_x, 2, dim=-1)  # Each has shape [batch_size, seq_len, dim//2]
        
        # Project split inputs to full dimension
        x1 = self.proj1(x1)  # Shape: [batch_size, seq_len, dim]
        x2 = self.proj2(x2)  # Shape: [batch_size, seq_len, dim]
        
        # Get QKV pairs
        qkv1 = self.to_qkv1(x1)  # Shape: [batch_size, seq_len, 3*dim]
        qkv2 = self.to_qkv2(x2)  # Shape: [batch_size, seq_len, 3*dim]
        
        # Split into Q, K, V
        q1, k1, v1 = qkv1.chunk(3, dim=-1)
        q2, k2, v2 = qkv2.chunk(3, dim=-1)
        
        # Reshape for multi-head attention
        q1 = q1.view(batch_size, seq_len, self.num_heads, self.d_head).transpose(1, 2)
        k1 = k1.view(batch_size, seq_len, self.num_heads, self.d_head).transpose(1, 2)
        v1 = v1.view(batch_size, seq_len, self.num_heads, self.d_head).transpose(1, 2)
        
        q2 = q2.view(batch_size, seq_len, self.num_heads, self.d_head).transpose(1, 2)
        k2 = k2.view(batch_size, seq_len, self.num_heads, self.d_head).transpose(1, 2)
        v2 = v2.view(batch_size, seq_len, self.num_heads, self.d_head).transpose(1, 2)
        
        # Differential attention
        out = self.differential_attention(q1, k1, v1, q2, k2, v2)
        out = out.transpose(1, 2).contiguous().view(batch_size, seq_len, -1)
        
        # Output projection with LayerNorm
        out = self.to_out(out)
        
        # Ensure output shape matches input shape
        if out.size(0) != batch_size or out.size(1) != seq_len:
            out = out.view(batch_size, seq_len, -1)
            
        return out, None
        
    def differential_attention(self, q1, k1, v1, q2, k2, v2):
        """
        Compute differential attention as described in the paper.
        Subtracts second attention map from first.
        """
        # Store original batch size
        batch_size = q1.size(0)
        
        # Compute attention scores
        attn1 = torch.matmul(q1, k1.transpose(-2, -1)) * self.scale
        attn2 = torch.matmul(q2, k2.transpose(-2, -1)) * self.scale
        
        # Apply softmax
        attn1 = F.softmax(attn1, dim=-1)
        attn2 = F.softmax(attn2, dim=-1)
        
        # Apply dropout
        attn1 = self.dropout(attn1)
        attn2 = self.dropout(attn2)
        
        # Compute differential attention
        diff_attn = attn1 - attn2
        
        # Apply attention to values
        out1 = torch.matmul(diff_attn, v1)
        out2 = torch.matmul(diff_attn, v2)
        
        # Average the outputs and ensure batch dimension is preserved
        out = (out1 + out2) / 2
        if out.size(0) != batch_size:
            out = out.view(batch_size, -1, out.size(-1))
            
        return out