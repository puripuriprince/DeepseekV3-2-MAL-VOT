import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, Any, Union

from .universal_processor import UniversalProcessor
from .multimodal_head import MultimodalHead
from .diff_attention import DifferentialAttention
from titans_pytorch import MemoryAsContextTransformer, NeuralMemory

class TransformerLayerWithMemory(nn.Module):
    def __init__(self,
                 dim: int,
                 num_heads: int,
                 mlp_ratio: int = 4,
                 dropout: float = 0.1,
                 memory_size: int = 512,
                 segment_len: int = 128,
                 chunk_size: int = 64,
                 num_persist_mem_tokens: int = 4,
                 num_longterm_mem_tokens: int = 16):
        super().__init__()
        
        # Differential attention
        self.diff_attn = DifferentialAttention(
            dim=dim,
            num_heads=num_heads,
            dropout=dropout
        )
        
        # Replace MemoryAsContextTransformer with NeuralMemory
        self.memory = NeuralMemory(
            dim=dim,
            chunk_size=chunk_size,
            pre_rmsnorm=True
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
        self.norm3 = nn.LayerNorm(dim)
        
    def forward(self,
                x: torch.Tensor,
                memory: torch.Tensor,
                memory_mask: torch.Tensor,
                use_checkpoint: bool = True) -> tuple:
        """Forward pass with differential attention and memory integration"""
        # Differential attention
        attn_out, attn_maps = self.diff_attn(self.norm1(x))
        x = x + attn_out
        
        # Memory retrieval and integration
        memory_out = self.memory(self.norm2(x))
        x = x + memory_out  # NeuralMemory returns tensor of same shape as input
        
        # MLP
        x = x + self.mlp(self.norm3(x))
        
        return x, memory, attn_maps

class TransformerSquared(nn.Module):
    def __init__(
        self,
        dim: int = 768,
        num_layers: int = 12,
        num_heads: int = 12,
        num_experts: int = 3,
        mlp_ratio: int = 4,
        dropout: float = 0.1,
        max_sequence_length: int = 8192,
        image_size: int = 256,
        patch_size: int = 16,
        memory_size: int = 512,
        chunk_size: int = 64,
        num_reasoning_steps: int = 3,
        segment_len: int = 128,
        num_persist_mem_tokens: int = 4,
        num_longterm_mem_tokens: int = 16,
        use_gradient_checkpointing: bool = True
    ):
        super().__init__()
        self.test_time_enabled = False
        
        # Universal input processor
        self.processor = UniversalProcessor(
            dim=dim,
            max_sequence_length=max_sequence_length,
            image_size=image_size,
            patch_size=patch_size
        )
        
        # Memory transformer layers with gradient checkpointing
        self.layers = nn.ModuleList([
            TransformerLayerWithMemory(
                dim=dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                dropout=dropout,
                memory_size=memory_size,
                chunk_size=chunk_size,
                segment_len=segment_len,
                num_persist_mem_tokens=num_persist_mem_tokens,
                num_longterm_mem_tokens=num_longterm_mem_tokens
            ) for _ in range(num_layers)
        ])
        self.use_gradient_checkpointing = use_gradient_checkpointing
        
        # Layer norm
        self.norm = nn.LayerNorm(dim)
        
        # Multimodal output head
        self.output_head = MultimodalHead(
            dim=dim,
            image_size=image_size,
            patch_size=patch_size
        )
        
        # Initialize memory
        self.memory_size = memory_size
        self.register_buffer('memory_tokens', torch.randn(1, memory_size, dim))
        self.register_buffer('memory_mask', torch.ones(1, memory_size))
        
        # Initialize reasoning components
        self.num_reasoning_steps = num_reasoning_steps
        self.register_buffer('reasoning_memory', torch.randn(num_reasoning_steps, dim))
        self.thought_projector = nn.Linear(dim, dim)
        self.reasoning_gate = nn.Sequential(
            nn.Linear(dim, 1),
            nn.Sigmoid()
        )
        
        # Expert system
        self.num_experts = num_experts
        self.experts = nn.ModuleList([
            nn.Linear(dim, dim) for _ in range(num_experts)
        ])
        self.expert_gate = nn.Linear(dim, num_experts)
        
    def enable_test_time(self):
        """Enable test-time computation tracking"""
        self.test_time_enabled = True
        
    def disable_test_time(self):
        """Disable test-time computation tracking"""
        self.test_time_enabled = False
        
    def get_compute_stats(self):
        """Get computation statistics when test_time is enabled"""
        if not self.test_time_enabled:
            return {}
        return {
            'reasoning_steps': self.reasoning_steps if hasattr(self, 'reasoning_steps') else [],
            'attention_maps': self.attention_maps if hasattr(self, 'attention_maps') else []
        }
        
    def compute_expert_weights(self, x: torch.Tensor) -> torch.Tensor:
        """Compute expert weights based on input embeddings"""
        # Average over sequence length dimension
        pooled = x.mean(dim=1)
        # Compute expert logits and apply softmax
        expert_logits = self.expert_gate(pooled)
        return F.softmax(expert_logits, dim=-1)

    def forward(self,
                inputs: Union[str, torch.Tensor, Dict, Any],
                input_type: Optional[str] = None,
                output_type: Optional[str] = None,
                temperature: float = 1.0,
                spatial_task: Optional[Dict] = None,
                expert_weights: Optional[torch.Tensor] = None,
                use_cache: bool = True) -> Dict[str, torch.Tensor]:
        """Forward pass with universal input processing and multimodal output generation"""
        # Process input into embeddings
        x = self.processor(inputs, input_type)
        batch_size = x.shape[0]
        
        # Expand memory for batch size
        memory = self.memory_tokens.expand(batch_size, -1, -1)
        memory_mask = self.memory_mask.expand(batch_size, -1)
        
        # Store attention maps for visualization
        attention_maps = []
        
        # Expert gating - use provided weights or compute new ones
        if expert_weights is None:
            expert_weights = self.compute_expert_weights(x)
            
        # Apply expert weights
        expert_outputs = []
        for i, expert in enumerate(self.experts):
            expert_out = expert(x)
            expert_outputs.append(expert_out * expert_weights[:, i:i+1, None])
        x = sum(expert_outputs)
        
        # Process through layers
        for layer in self.layers:
            x, memory, attn = layer(
                x, 
                memory, 
                memory_mask,
                use_checkpoint=self.use_gradient_checkpointing
            )
            attention_maps.append(attn)
        
        # Apply final layer norm
        x = self.norm(x)
        
        # Generate output using multimodal head
        outputs = self.output_head(
            x,
            output_type=output_type,
            temperature=temperature,
            spatial_task=spatial_task
        )
        
        # Add attention maps and expert weights to outputs
        outputs['attention_maps'] = attention_maps
        outputs['expert_weights'] = expert_weights
        
        return outputs