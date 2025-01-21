import torch
import torch.nn as nn
import logging
from typing import Dict, Optional, List, Tuple, Union

logger = logging.getLogger(__name__)

class TestTimeCompute:
    """Handles test-time computation optimization with multimodal support"""
    
    def __init__(
        self,
        model: nn.Module,
        cache_size: int = 1024,
        chunk_size: int = 64
    ):
        self.model = model
        self.cache_size = cache_size
        self.chunk_size = chunk_size
        self.cache = {}
        
    @torch.no_grad()
    def compute_with_cache(
        self,
        input_ids: Union[torch.Tensor, Dict[str, torch.Tensor]],
        context: Dict[str, torch.Tensor]
    ) -> torch.Tensor:
        """
        Compute with caching of key-value pairs. Supports multimodal inputs.
        """
        # Handle dictionary input for multimodal
        if isinstance(input_ids, dict):
            cache_key = tuple([
                tuple(v.cpu().numpy().flatten()) 
                for v in input_ids.values()
                if torch.is_tensor(v)
            ])
        else:
            cache_key = tuple(input_ids.cpu().numpy().flatten())
            
        if cache_key in self.cache:
            return self.cache[cache_key]
            
        # Process text inputs in chunks
        if isinstance(input_ids, torch.Tensor):
            chunks = torch.split(input_ids, self.chunk_size, dim=1)
            outputs = []
            
            for chunk in chunks:
                output = self.model(chunk, context)
                outputs.append(output[0])  # Only keep final output
                
            final_output = torch.cat(outputs, dim=1)
            
        # Process multimodal input
        else:
            # No chunking for images/multimodal
            final_output = self.model(input_ids, context)[0]
                
        # Update cache
        if len(self.cache) > self.cache_size:
            # Remove oldest entries
            remove_keys = list(self.cache.keys())[:len(self.cache) - self.cache_size]
            for k in remove_keys:
                del self.cache[k]
                
        self.cache[cache_key] = final_output
        return final_output

    def __call__(self, x: Union[torch.Tensor, Dict[str, torch.Tensor]]) -> torch.Tensor:
        """Forward pass with test-time optimizations"""
        return self.compute_with_cache(x, {})
