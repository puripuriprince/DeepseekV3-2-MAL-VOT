import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict

from .transformer_squared import TransformerSquared
from .byte_latent_patches import ByteLevelTokenizer
from .diff_attention import DifferentialAttention, TransformerLayerWithMemory

class LocalEncoder(nn.Module):
    """Light-weight transformer for encoding bytes into patch representations"""
    def __init__(
        self,
        dim: int,
        num_layers: int,
        num_heads: int,
        window_size: int = 128,
        dropout: float = 0.1
    ):
        super().__init__()
        self.window_size = window_size
        
        # Byte embeddings
        self.byte_embeddings = nn.Embedding(256, dim)
        
        # Hash n-gram embeddings for n=3 to 8
        self.hash_size = 10000
        self.n_gram_embeddings = nn.ModuleDict({
            f'n{n}': nn.Embedding(self.hash_size, dim)
            for n in range(3, 9)
        })
        
        # Local transformer layers
        self.layers = nn.ModuleList([
            nn.ModuleDict({
                'transformer': nn.TransformerEncoderLayer(
                    d_model=dim,
                    nhead=num_heads,
                    dim_feedforward=dim * 4,
                    dropout=dropout,
                    batch_first=True
                ),
                'cross_attention': CrossAttention(dim)
            }) for _ in range(num_layers)
        ])
        
        self.norm = nn.LayerNorm(dim)
        
    def compute_ngram_hash(self, bytes_seq: torch.Tensor, n: int) -> torch.Tensor:
        """Compute rolling hash for n-grams"""
        batch_size, seq_len = bytes_seq.shape
        device = bytes_seq.device
        
        # Create n-gram windows
        indices = torch.arange(n, device=device)
        windows = torch.stack([
            torch.roll(bytes_seq, -i, dims=1) 
            for i in indices
        ], dim=-1)
        
        # Compute rolling hash
        base = 257
        mod = self.hash_size
        multiplier = torch.pow(base, indices, device=device)
        hashed = torch.sum(windows * multiplier, dim=-1) % mod
        
        # Mask invalid positions
        mask = torch.arange(seq_len, device=device) < (seq_len - n + 1)
        hashed = hashed * mask
        
        return hashed
        
    def forward(self, 
                bytes_seq: torch.Tensor,
                patch_indices: torch.Tensor) -> torch.Tensor:
        """
        Args:
            bytes_seq: [batch_size, seq_len] byte-level input
            patch_indices: [batch_size, num_patches] indices where patches start
        """
        # Get byte embeddings
        x = self.byte_embeddings(bytes_seq)
        
        # Add n-gram embeddings
        for n in range(3, 9):
            hashed = self.compute_ngram_hash(bytes_seq, n)
            x = x + self.n_gram_embeddings[f'n{n}'](hashed)
            
        # Process through local transformer and cross attention
        for layer in self.layers:
            # Local causal attention
            attn_mask = self.get_local_mask(x.size(1))
            x = layer['transformer'](x, src_mask=attn_mask)
            
            # Cross attention to form patch representations
            x = layer['cross_attention'](x, patch_indices)
            
        return self.norm(x)
    
    def get_local_mask(self, seq_len: int) -> torch.Tensor:
        """Create causal mask with local window"""
        device = next(self.parameters()).device
        mask = torch.ones(seq_len, seq_len, device=device) * float('-inf')
        
        for i in range(seq_len):
            start = max(0, i - self.window_size)
            mask[i, start:i+1] = 0
            
        return mask

class CrossAttention(nn.Module):
    """Cross attention for pooling byte representations into patches"""
    def __init__(self, dim: int):
        super().__init__()
        self.to_q = nn.Linear(dim, dim)
        self.to_k = nn.Linear(dim, dim)
        self.to_v = nn.Linear(dim, dim)
        self.scale = dim ** -0.5
        
    def forward(self, 
                x: torch.Tensor,
                patch_indices: torch.Tensor) -> torch.Tensor:
        """
        Pool byte representations into patch representations using cross attention
        """
        batch_size, seq_len, dim = x.shape
        num_patches = patch_indices.size(1)
        
        # Create patch masks
        patch_masks = []
        for i in range(num_patches-1):
            start = patch_indices[:, i:i+1]
            end = patch_indices[:, i+1:i+2]
            mask = (torch.arange(seq_len, device=x.device) >= start) & \
                   (torch.arange(seq_len, device=x.device) < end)
            patch_masks.append(mask)
        
        # Add final patch
        mask = torch.arange(seq_len, device=x.device) >= patch_indices[:, -1:]
        patch_masks.append(mask)
        
        patch_masks = torch.stack(patch_masks, dim=1)
        
        # Cross attention
        q = self.to_q(x)
        k = self.to_k(x) 
        v = self.to_v(x)
        
        attn = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        attn = attn.masked_fill(~patch_masks.unsqueeze(2), float('-inf'))
        attn = F.softmax(attn, dim=-1)
        
        return torch.matmul(attn, v)

class LocalDecoder(nn.Module):
    """Light-weight transformer for decoding patches back to bytes"""
    def __init__(
        self,
        dim: int,
        num_layers: int,
        num_heads: int,
        window_size: int = 128,
        dropout: float = 0.1
    ):
        super().__init__()
        self.window_size = window_size
        
        # Cross attention to incorporate patch information
        self.cross_attention = CrossAttention(dim)
        
        # Local transformer layers
        self.layers = nn.ModuleList([
            nn.TransformerDecoderLayer(
                d_model=dim,
                nhead=num_heads,
                dim_feedforward=dim * 4,
                dropout=dropout,
                batch_first=True
            ) for _ in range(num_layers)
        ])
        
        self.norm = nn.LayerNorm(dim)
        self.to_bytes = nn.Linear(dim, 256)
        
    def forward(self,
                patch_repr: torch.Tensor,
                bytes_seq: torch.Tensor,
                patch_indices: torch.Tensor) -> torch.Tensor:
        """
        Args:
            patch_repr: [batch_size, num_patches, dim] patch representations
            bytes_seq: [batch_size, seq_len] byte sequence for teacher forcing
            patch_indices: [batch_size, num_patches] patch boundary indices
        """
        x = self.cross_attention(patch_repr, patch_indices)
        
        # Local causal decoding
        attn_mask = self.get_local_mask(bytes_seq.size(1))
        for layer in self.layers:
            x = layer(x, patch_repr, tgt_mask=attn_mask)
            
        x = self.norm(x)
        return self.to_bytes(x)
    
    def get_local_mask(self, seq_len: int) -> torch.Tensor:
        """Create causal mask with local window"""
        device = next(self.parameters()).device
        mask = torch.ones(seq_len, seq_len, device=device) * float('-inf')
        
        for i in range(seq_len):
            start = max(0, i - self.window_size)
            mask[i, start:i+1] = 0
            
        return mask

class SushiHybrid(nn.Module):
    """
    Hybrid architecture combining:
    1. Byte-level processing (BLT)
    2. SVD-based adaptation (Transformer²)
    3. Differential attention
    4. Neural memory
    """
    def __init__(
        self,
        dim: int = 768,
        num_layers: int = 12,
        num_heads: int = 12,
        mlp_ratio: int = 4,
        dropout: float = 0.1,
        max_sequence_length: int = 8192,
        memory_size: int = 512,
        chunk_size: int = 64,
        num_experts: int = 3,
        local_window: int = 128,
        use_gradient_checkpointing: bool = True
    ):
        super().__init__()
        
        # Byte-level tokenizer and processing
        self.tokenizer = ByteLevelTokenizer()
        
        # Local byte-level encoder
        self.byte_encoder = ByteEncoder(
            dim=dim,
            num_layers=2,  # Light-weight local processing
            num_heads=num_heads,
            window_size=local_window,
            dropout=dropout
        )
        
        # Core latent transformer with SVD adaptation
        self.latent_transformer = TransformerSquared(
            dim=dim,
            num_layers=num_layers,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            dropout=dropout,
            max_sequence_length=max_sequence_length,
            memory_size=memory_size,
            use_gradient_checkpointing=use_gradient_checkpointing
        )
        
        # Differential attention layers
        self.diff_attention = nn.ModuleList([
            DifferentialAttention(
                dim=dim,
                num_heads=num_heads,
                dropout=dropout
            ) for _ in range(num_layers)
        ])
        
        # Memory-augmented transformer layers
        self.memory_layers = nn.ModuleList([
            TransformerLayerWithMemory(
                dim=dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                dropout=dropout,
                memory_size=memory_size,
                chunk_size=chunk_size
            ) for _ in range(num_layers)
        ])
        
        # Expert system
        self.num_experts = num_experts
        self.experts = nn.ModuleList([
            nn.Linear(dim, dim) for _ in range(num_experts)
        ])
        self.expert_gate = nn.Linear(dim, num_experts)
        
        # Output decoder
        self.decoder = ByteDecoder(
            dim=dim,
            num_layers=2,  # Light-weight local processing
            num_heads=num_heads,
            window_size=local_window,
            dropout=dropout
        )
        
    def forward(
        self,
        input_ids: torch.Tensor,
        expert_weights: Optional[torch.Tensor] = None,
        z_vectors: Optional[Dict[str, torch.Tensor]] = None,
        use_memory: bool = True
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass combining all architectural components
        
        Args:
            input_ids: Input byte sequence
            expert_weights: Optional pre-computed expert weights
            z_vectors: Optional SVD adaptation vectors
            use_memory: Whether to use memory components
        """
        # Get byte-level representations
        byte_repr, patch_indices = self.byte_encoder(input_ids)
        
        # Process through latent transformer with SVD adaptation
        latent_outputs = self.latent_transformer(
            byte_repr,
            expert_weights=expert_weights,
            z_vectors=z_vectors
        )
        hidden_states = latent_outputs['hidden_states']
        
        # Apply differential attention
        diff_attention_maps = []
        for diff_attn in self.diff_attention:
            diff_out, diff_map = diff_attn(hidden_states)
            hidden_states = hidden_states + diff_out
            diff_attention_maps.append(diff_map)
            
        # Memory integration
        if use_memory:
            memory_states = []
            for mem_layer in self.memory_layers:
                hidden_states, mem_state, _ = mem_layer(
                    hidden_states,
                    None,  # Initial memory state
                    None   # Memory mask
                )
                memory_states.append(mem_state)
                
        # Expert processing
        if expert_weights is None:
            # Compute expert weights if not provided
            pooled = hidden_states.mean(dim=1)
            expert_logits = self.expert_gate(pooled)
            expert_weights = F.softmax(expert_logits, dim=-1)
            
        # Apply expert weights
        expert_outputs = []
        for i, expert in enumerate(self.experts):
            expert_out = expert(hidden_states)
            expert_outputs.append(expert_out * expert_weights[:, i:i+1, None])
        hidden_states = sum(expert_outputs)
        
        # Decode back to bytes
        byte_logits = self.decoder(
            hidden_states,
            patch_indices
        )
        
        return {
            'byte_logits': byte_logits,
            'hidden_states': hidden_states,
            'attention_maps': latent_outputs['attention_maps'],
            'diff_attention_maps': diff_attention_maps,
            'expert_weights': expert_weights,
            'memory_states': memory_states if use_memory else None,
            'patch_indices': patch_indices
        }
class ByteEncoder(nn.Module):
    """Lightweight encoder for byte-level processing"""
    def __init__(self, dim, num_layers, num_heads, window_size, dropout):
        super().__init__()
        # Implementation moved to separate file for clarity
        pass

class ByteDecoder(nn.Module):
    """Lightweight decoder for byte-level processing"""
    def __init__(self, dim, num_layers, num_heads, window_size, dropout):
        super().__init__()
        # Implementation moved to separate file for clarity
        pass

