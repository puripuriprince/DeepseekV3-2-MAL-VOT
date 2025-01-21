import torch
import torch.nn as nn
import logging
from typing import Optional, Dict
from config.model_config import ModelConfig

logger = logging.getLogger(__name__)

class ChainOfThought(nn.Module):
    """Implements chain-of-thought reasoning with multimodal context"""
    def __init__(
        self,
        dim: int = 768,
        num_reasoning_steps: int = 3,
        max_thought_len: int = 512,
        dropout: float = 0.1
    ):
        super().__init__()
        
        # Extract dim value if ModelConfig is passed
        dim_value = dim.dim if isinstance(dim, ModelConfig) else dim
        self.dim = dim_value
        self.num_steps = num_reasoning_steps
        self.max_thought_len = max_thought_len

        # Thought generation layers
        self.thought_generator = nn.ModuleList([
            nn.TransformerDecoderLayer(
                d_model=dim_value,
                nhead=8,
                dim_feedforward=dim_value * 4,
                dropout=dropout
            ) for _ in range(num_reasoning_steps)
        ])
        
        # Output projection
        self.output_proj = nn.Linear(dim_value, dim_value)
        
    def forward(self, hidden_states: torch.Tensor, multimodal_context: Optional[Dict[str, torch.Tensor]] = None):
        """
        Apply chain-of-thought reasoning to input states
        Args:
            hidden_states: Input tensor
            multimodal_context: Optional dict of multimodal tensors
        """
        # Store original dimensions
        batch_size, seq_len, _ = hidden_states.shape
        x = hidden_states
        reasoning_steps = []
        
        # Shape verification
        assert x.shape[0] == batch_size, f"Expected batch size {batch_size}, got {x.shape[0]}"
        assert x.shape[1] == seq_len, f"Expected seq length {seq_len}, got {x.shape[1]}"
        
        # Apply reasoning steps
        for step in range(self.num_steps):
            # Generate thought
            prev_x = x
            x = self.thought_generator[step](x, x)
            
            # Shape verification after thought generation
            assert x.shape[0] == batch_size, f"Expected batch size {batch_size}, got {x.shape[0]}"
            assert x.shape[1] == seq_len, f"Expected seq length {seq_len}, got {x.shape[1]}"
            
            # Track reasoning step
            step_info = {
                'step': step,
                'thought_vector': x - prev_x,
                'gate': torch.sigmoid(self.output_proj(x)).mean(),
                'context_used': bool(multimodal_context)
            }
            reasoning_steps.append(step_info)
            
            # Incorporate multimodal context if provided
            if multimodal_context:
                for modality, tensor in multimodal_context.items():
                    # Reshape tensor to match sequence length
                    if modality == 'image':
                        # Average over spatial dimensions for images
                        tensor = tensor.mean(dim=(2, 3))  # [batch, channels]
                    elif modality == 'mesh':
                        # Average over vertex dimension for meshes
                        tensor = tensor.mean(dim=1)  # [batch, features]
                    
                    # Project to match hidden dimension
                    if not hasattr(self, f'{modality}_proj'):
                        setattr(self, f'{modality}_proj', 
                               nn.Linear(tensor.size(-1), self.dim).to(tensor.device))
                    
                    proj = getattr(self, f'{modality}_proj')
                    tensor = proj(tensor)  # [batch, dim]
                    
                    # Expand to match sequence length
                    tensor = tensor.unsqueeze(1).expand(batch_size, seq_len, -1)
                    x = x + tensor
                    # Shape verification after adding context
                    assert x.shape[0] == batch_size, f"Expected batch size {batch_size}, got {x.shape[0]}"
                    assert x.shape[1] == seq_len, f"Expected seq length {seq_len}, got {x.shape[1]}"
                    
        # Store reasoning steps in stats
        self.reasoning_stats = reasoning_steps
        
            
        # Final output projection
        out = self.output_proj(x)
        
        # Final shape verification
        assert out.shape[0] == batch_size, f"Expected batch size {batch_size}, got {out.shape[0]}"
        assert out.shape[1] == seq_len, f"Expected seq length {seq_len}, got {out.shape[1]}"
        
        return out
