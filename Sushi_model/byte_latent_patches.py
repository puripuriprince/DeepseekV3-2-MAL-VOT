import torch
import torch.nn as nn
import logging
from typing import Optional, Dict, List, Tuple
import math
from .diff_attention import MultiHeadDifferentialAttention

logger = logging.getLogger(__name__)

class ByteLatentPatches(nn.Module):
    """
    Implements byte-level tokenization and latent patch processing
    from the BLT paper.
    """
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.expansion_rate = getattr(config, 'expansion_rate', 2)
        
        # Initialize components
        self.byte_embeddings = nn.Embedding(256, config.hidden_size)
        self.ngram_embeddings = nn.ModuleList([
            nn.Embedding(config.ngram_vocab_size, config.hidden_size)
            for _ in range(6)  # 6 different n-gram sizes (3-8)
        ])
        
        # Projection layers
        self.encode_proj = nn.Linear(config.hidden_size * 2, config.hidden_size)
        self.decode_proj = nn.Linear(config.hidden_size, 256)
        
        # Normalization layers
        self.input_norm = nn.LayerNorm(config.hidden_size)
        self.output_norm = nn.LayerNorm(config.hidden_size)
        self.post_combined_norm = nn.LayerNorm(config.hidden_size)
        
        # Transformer layers
        self.layers = nn.ModuleList([
            EncoderLayer(config) for _ in range(config.encoder_layers)
        ])
        
        # Gating mechanisms
        self.laurel_gates = nn.ModuleList([
            nn.Linear(config.hidden_size * 2, 1)
            for _ in range(config.encoder_layers - 1)
        ])
        
        self.hyper_projections = nn.ModuleList([
            nn.Linear(config.hidden_size, config.hidden_size)
            for _ in range(config.encoder_layers)
        ])
        
        self.hyper_gates = nn.ModuleList([
            nn.Linear(config.hidden_size * 2, 1)
            for _ in range(config.encoder_layers)
        ])
        
        self.memory_projections = nn.ModuleList([
            nn.Linear(config.hidden_size, config.hidden_size)
            for _ in range(config.encoder_layers)
        ])
        
        self.memory_gates = nn.ModuleList([
            nn.Linear(config.hidden_size * 2, 1)
            for _ in range(config.encoder_layers)
        ])
        
        # Initialize hierarchical memories
        self.hierarchical_memories = nn.ParameterList([
            nn.Parameter(torch.zeros(1, 1, config.hidden_size))
            for _ in range(config.encoder_layers)
        ])
        
        # Dropout
        self.dropout = nn.Dropout(config.dropout)
        self.gate_dropout = nn.Dropout(config.dropout)
        
        # Adaptive gating parameters
        self.ngram_gates = nn.Parameter(torch.ones(6))

    def compute_ngram_hash(self, bytes_sequence, n):
        """Compute hash indices for n-grams"""
        # Store original dimensions
        batch_size, seq_length = bytes_sequence.shape
        device = bytes_sequence.device
        
        if seq_length < n:
            return torch.empty((batch_size, 0), dtype=torch.long, device=device)
            
        # Compute n-grams while preserving batch dimension
        ngrams = bytes_sequence.unfold(dimension=1, size=n, step=1)  # [batch_size, seq_len-n+1, n]
        weights = (256 ** torch.arange(n, device=device)).float()
        hash_values = (ngrams * weights).sum(dim=-1)  # [batch_size, seq_len-n+1]
        
        vocab_size = self.ngram_embeddings[n-3].num_embeddings
        hash_indices = (hash_values % vocab_size).long()  # [batch_size, seq_len-n+1]
        
        # Ensure batch dimension is preserved
        if hash_indices.size(0) != batch_size:
            hash_indices = hash_indices.reshape(batch_size, -1)
            
        # Shape verification
        assert hash_indices.shape[0] == batch_size, f"Expected batch size {batch_size}, got {hash_indices.shape[0]}"
        assert hash_indices.shape[1] == seq_length - n + 1, f"Expected seq length {seq_length - n + 1}, got {hash_indices.shape[1]}"
        
        return hash_indices

    def encode(self, bytes_input):
        """Encode byte sequence into embeddings"""
        # Handle different input shapes
        if isinstance(bytes_input, dict):
            # Multimodal input
            if 'image' in bytes_input:
                return self._encode_image(bytes_input['image'])
            elif 'mesh' in bytes_input:
                return self._encode_mesh(bytes_input['mesh'])
            else:
                raise ValueError(f"Unknown input type in dict: {bytes_input.keys()}")
        
        # Ensure input has batch dimension
        if len(bytes_input.shape) == 1:
            bytes_input = bytes_input.unsqueeze(0)
            
        # Store original dimensions
        batch_size, seq_length = bytes_input.shape
        
        # Shape verification
        assert len(bytes_input.shape) == 2, f"Expected 2D input, got shape {bytes_input.shape}"
        
        # Get byte-level embeddings
        byte_embeds = self.byte_embeddings(bytes_input)  # [batch_size, seq_len, hidden_size]
        byte_embeds = self.dropout(byte_embeds)
        
        # Shape verification after embedding
        assert byte_embeds.shape[0] == batch_size, f"Expected batch size {batch_size}, got {byte_embeds.shape[0]}"
        assert byte_embeds.shape[1] == seq_length, f"Expected seq length {seq_length}, got {byte_embeds.shape[1]}"
        
        # Store original dimensions
        batch_size, seq_length = bytes_input.shape
        
        # Process n-grams (3 to 8)
        ngram_features = []
        
        for i, n in enumerate(range(3, 9)):
            if seq_length >= n:
                ngram_hashes = self.compute_ngram_hash(bytes_input, n)  # [batch_size, seq_len-n+1]
                ngram_embeds = self.ngram_embeddings[i](ngram_hashes)  # [batch_size, seq_len-n+1, hidden_size]
                
                gate = torch.sigmoid(self.ngram_gates[i])
                gate = self.gate_dropout(gate)
                
                scaled_embeds = (ngram_embeds / n) * gate
                
                # Pad to match sequence length
                padded = torch.zeros(
                    batch_size, seq_length, self.hidden_size,
                    device=bytes_input.device,
                    dtype=scaled_embeds.dtype
                )
                padded[:, :ngram_hashes.size(1)] = scaled_embeds
                
                # Ensure padded has correct batch dimension
                if padded.size(0) != batch_size:
                    padded = padded.reshape(batch_size, seq_length, -1)
                
                # Shape verification
                assert padded.shape[0] == batch_size, f"Expected batch size {batch_size}, got {padded.shape[0]}"
                assert padded.shape[1] == seq_length, f"Expected seq length {seq_length}, got {padded.shape[1]}"
                
                ngram_features.append(padded)
        
        if ngram_features:
            # Stack and mean while preserving batch dimension
            ngram_embeds = torch.stack(ngram_features).mean(dim=0)  # [batch_size, seq_len, hidden_size]
            combined = torch.cat([byte_embeds, ngram_embeds], dim=-1)
            encoded = self.encode_proj(combined)
        else:
            encoded = byte_embeds
            
        # Shape verification
        assert encoded.shape[0] == batch_size, f"Expected batch size {batch_size}, got {encoded.shape[0]}"
        assert encoded.shape[1] == seq_length, f"Expected seq length {seq_length}, got {encoded.shape[1]}"
            
        # Ensure output has shape [batch_size, seq_len, hidden_size]
        if len(encoded.shape) == 2:
            encoded = encoded.unsqueeze(0)
        elif encoded.shape[0] != batch_size:
            encoded = encoded.reshape(batch_size, seq_length, -1)
            
        # Final shape verification
        assert encoded.shape[0] == batch_size, f"Expected batch size {batch_size}, got {encoded.shape[0]}"
        assert encoded.shape[1] == seq_length, f"Expected seq length {seq_length}, got {encoded.shape[1]}"
        assert encoded.shape[2] == self.hidden_size, f"Expected hidden size {self.hidden_size}, got {encoded.shape[2]}"
            
        # Final shape verification
        assert encoded.shape[0] == batch_size, f"Expected batch size {batch_size}, got {encoded.shape[0]}"
        assert encoded.shape[1] == seq_length, f"Expected seq length {seq_length}, got {encoded.shape[1]}"
            
        return encoded

    def decode(self, hidden_states):
        """Decode hidden states back to byte logits"""
        return self.decode_proj(hidden_states)

    def forward(self, patch_embeddings, attention_mask=None):
        """Forward pass with hierarchical processing"""
        batch_size = patch_embeddings.size(0)
        
        h = self.input_norm(patch_embeddings)
        h = self.dropout(h)
        
        hyper_h = h.unsqueeze(2).expand(-1, -1, self.expansion_rate, -1)
        positions = torch.arange(patch_embeddings.size(1), device=patch_embeddings.device)
        
        for idx, layer in enumerate(self.layers):
            prev_h = self.dropout(h)
            layer_out = layer(h, self_mask=attention_mask, positions=positions)
            
            combined_out = self._apply_skip_and_laurel(prev_h, prev_h, layer_out, idx)
            hyper_h = self._apply_hyper_connections(combined_out, hyper_h, idx)
            
            combined_features = combined_out + hyper_h.mean(dim=2)
            combined_features = self.post_combined_norm(combined_features)
            
            h = self._apply_hierarchical_memory(combined_features, idx, batch_size)
            h = self.dropout(h)
        
        h = self.output_norm(h)
        h = self.dropout(h)
        return h

    def _apply_skip_and_laurel(self, prev_h, skip_h, layer_out, layer_idx):
        """Apply skip connection and Laurel combination"""
        if layer_idx == 0:
            return layer_out + skip_h
            
        alpha = torch.sigmoid(
            self.laurel_gates[layer_idx-1](
                torch.cat([prev_h, layer_out], dim=-1)
            )
        )
        return alpha * layer_out + (1 - alpha) * skip_h
        
    def _apply_hyper_connections(self, curr_h, hyper_h, layer_idx):
        """Apply hyper-connections between layers"""
        hyper_update = self.hyper_projections[layer_idx](curr_h)
        hyper_update = hyper_update.unsqueeze(2)
        
        gate = torch.sigmoid(
            self.hyper_gates[layer_idx](
                torch.cat([hyper_h, hyper_update], dim=-1)
            )
        )
        return gate * hyper_update + (1 - gate) * hyper_h
        
    def _apply_hierarchical_memory(self, features, layer_idx, batch_size):
        """Apply hierarchical memory updates"""
        mem = self.hierarchical_memories[layer_idx]
        mem_proj = self.memory_projections[layer_idx](features)
        
        gate = torch.sigmoid(
            self.memory_gates[layer_idx](
                torch.cat([mem.expand(batch_size, -1, -1), mem_proj], dim=-1)
            )
        )
        
        self.hierarchical_memories[layer_idx] = (
            gate * mem_proj + (1 - gate) * mem
        ).mean(dim=0, keepdim=True)
        
        return features + mem.expand(batch_size, -1, -1)

class ByteLevelTokenizer:
    """Tokenizes text into bytes"""
    def __init__(self):
        pass
        
    def encode(self, text: str) -> torch.Tensor:
        """Convert text to byte tensor"""
        return torch.tensor([ord(c) for c in text], dtype=torch.long)
        
    def decode(self, tokens: torch.Tensor) -> str:
        """Convert byte tensor back to text"""
        return ''.join(chr(t) for t in tokens.tolist())

class EncoderLayer(nn.Module):
    """Transformer encoder layer with hierarchical memory"""
    def __init__(self, config):
        super().__init__()
        self.attention = MultiHeadDifferentialAttention(
            config.hidden_size,
            config.heads,
            config.lambda_init
        )
        self.feed_forward = nn.Sequential(
            nn.Linear(config.hidden_size, config.hidden_size * 4),
            nn.GELU(),
            nn.Linear(config.hidden_size * 4, config.hidden_size)
        )
        self.norm1 = nn.LayerNorm(config.hidden_size)
        self.norm2 = nn.LayerNorm(config.hidden_size)
        self.dropout = nn.Dropout(config.dropout)
        
    def forward(self, x, self_mask=None, positions=None):
        attended = self.attention(self.norm1(x), mask=self_mask)[0]
        x = x + self.dropout(attended)
        x = x + self.dropout(self.feed_forward(self.norm2(x)))
        return x
