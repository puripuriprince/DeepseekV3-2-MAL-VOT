import torch
import torch.nn as nn

class ByteLevelTokenizer:
    def __init__(self):
        self.vocab_size = 256  # Full byte range
        
    def encode(self, text: str) -> torch.Tensor:
        """Convert text to byte-level tokens"""
        bytes_data = text.encode('utf-8')
        return torch.tensor([b for b in bytes_data], dtype=torch.long)
    
    def decode(self, tokens: torch.Tensor) -> str:
        """Convert byte-level tokens back to text"""
        return bytes(tokens.cpu().tolist()).decode('utf-8', errors='replace')

class ByteLatentPatches(nn.Module):
    def __init__(self, dim: int, max_sequence_length: int = 8192):
        super().__init__()
        self.dim = dim
        self.max_sequence_length = max_sequence_length
        
        # Byte-level embedding
        self.byte_embeddings = nn.Embedding(256, dim)
        self.pos_embedding = nn.Parameter(torch.randn(1, max_sequence_length, dim))
        
    def forward(self, byte_tokens: torch.Tensor) -> torch.Tensor:
        B, N = byte_tokens.shape
        x = self.byte_embeddings(byte_tokens)
        x = x + self.pos_embedding[:, :N]
        return x 