import torch
import torch.nn as nn
import logging
from typing import Dict, Optional, Union, List, Tuple
from dataclasses import dataclass

from .transformer_squared import TransformerSquared
from .diff_attention import MultiHeadDifferentialAttention # why isnt this being used?
from .thinking import ChainOfThought
from .test_time_compute import TestTimeCompute
from .byte_latent_patches import ByteLatentPatches, ByteLevelTokenizer
from .memory_manager import MemoryManager
from .adaptation import (
    BaseAdaptationStrategy,
    PromptBasedAdaptationStrategy,# why arent these being used?
    ClassificationExpertAdaptationStrategy, 
    CEMAdaptationStrategy,
    SVFAdaptationStrategy
)

logger = logging.getLogger(__name__)

@dataclass 
class ModelStats:
    """Track model statistics for debugging and optimization"""
    attention_scores: Dict[str, torch.Tensor] = None
    memory_usage: Dict[str, float] = None
    computation_time: Dict[str, float] = None
    peak_memory: float = 0.0

class SushiFull(nn.Module):
    """
    Unified multimodal model with memory, differential attention,
    and chain-of-thought reasoning capabilities.
    """
    def __init__(self, config):
        super().__init__()
        logger.info("Initializing SushiFull model")
        
        # Store config
        self.config = config
        
        # Performance monitoring
        self.stats = ModelStats()
        self._profile_enabled = config.profile_mode
        
        # Store configuration flags for new features
        self.use_diff_transformer = config.use_diff_transformer
        self.use_self_adaptation = config.use_self_adaptation
        
        # Test-time compute state
        self._test_time_enabled = False
        self._compute_stats = []
        
        try:
            # Initialize tokenizer
            logger.debug("Initializing tokenizer")
            self.tokenizer = ByteLevelTokenizer()
            
            # Initialize components as submodules
            logger.debug("Initializing transformer component")
            self.transformer = TransformerSquared(config)
            
            logger.debug("Initializing chain of thought component")
            self.thinking = ChainOfThought(config)
            
            logger.debug("Initializing test time compute")
            self.ttc = TestTimeCompute(config)
            
            logger.debug("Initializing memory manager")
            self.memory = MemoryManager(config)
            
            # Multimodal components
            logger.debug("Initializing encoders")
            self.byte_encoder = ByteLatentPatches(config)
            self.patch_embed = ByteLatentPatches(config)
            
            # Expert vectors for adaptation
            self.expert_z_dict = {}
            
            logger.info("SushiFull model initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize SushiFull: {str(e)}")
            raise

    def forward(
        self,
        x: Union[torch.Tensor, Dict[str, torch.Tensor]],
        multimodal_context: Optional[Dict[str, torch.Tensor]] = None,
        adaptation_strategy: Optional[BaseAdaptationStrategy] = None,
        user_prompt: str = "",
        few_shot_samples: List[torch.Tensor] = None,
        scoring_fn = None,
        use_memory: bool = False,
        return_stats: bool = False,
        **kwargs
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, ModelStats]]:
        """
        Unified forward pass combining all components
        """
        logger.debug(f"Forward pass started with input shape: {x.shape}")
        
        try:
            # Start profiling if enabled
            if self._profile_enabled:
                self._start_profile()
                
            # Reset compute stats for this forward pass
            if self._test_time_enabled:
                self._compute_stats = []
            
            # Track memory usage
            if torch.cuda.is_available():
                self.stats.peak_memory = torch.cuda.max_memory_allocated()

            # 1. Get adapted z_vectors if strategy provided
            z_vectors = None
            if adaptation_strategy is not None and self.expert_z_dict:
                if isinstance(adaptation_strategy, SVFAdaptationStrategy) and self.use_self_adaptation:
                    # Use SVF adaptation from Transformer2
                    z_vectors = adaptation_strategy.adapt_z_vectors(
                        input_ids=x,
                        user_prompt=user_prompt,
                        experts=self.expert_z_dict,
                        few_shot_samples=few_shot_samples,
                        scoring_fn=scoring_fn
                    )
                else:
                    # Use traditional adaptation
                    z_vectors = adaptation_strategy.adapt_z_vectors(
                        input_ids=x,
                        user_prompt=user_prompt,
                        experts=self.expert_z_dict,
                        few_shot_samples=few_shot_samples,
                        scoring_fn=scoring_fn
                    )
                    
                # Ensure z_vectors match batch dimension
                if isinstance(x, torch.Tensor):
                    batch_size = x.size(0)
                    z_vectors = {
                        k: v.expand(batch_size, -1) if len(v.shape) == 1 else v
                        for k, v in z_vectors.items()
                    }
            
            # 2. Apply memory if requested
            if use_memory:
                logger.debug("Retrieving from memory")
                memory_context = self.memory.retrieve(x)
                x = torch.cat([memory_context, x], dim=1)
                
            # Store original dimensions
            if isinstance(x, torch.Tensor):
                # Add batch dimension if missing
                if len(x.shape) == 1:
                    x = x.unsqueeze(0)
                batch_size, seq_len = x.shape
            else:
                raise ValueError(f"Unsupported input type: {type(x)}")

            # 3. Embed input before transformer 
            logger.debug("Embedding input")
            x = self.byte_encoder.encode(x)
            
            # Verify shape after encoding
            if x.size(0) != batch_size:
                x = x.reshape(batch_size, seq_len, -1)
            
            # Shape verification
            assert x.shape[0] == batch_size, f"Expected batch size {batch_size}, got {x.shape[0]}"
            assert x.shape[1] == seq_len, f"Expected seq length {seq_len}, got {x.shape[1]}"
            
            # 3. Process through transformer with z_vectors and feature flags
            logger.debug("Running transformer")
            transformer_out = self.transformer(
                x,
                z_vectors=z_vectors,
                use_diff_transformer=self.use_diff_transformer,
                use_self_adaptation=self.use_self_adaptation
            )
            hidden_states = transformer_out['hidden_states']
            
            # Verify shape after transformer
            if hidden_states.size(0) != batch_size:
                hidden_states = hidden_states.reshape(batch_size, seq_len, -1)
            
            # Shape verification
            assert hidden_states.shape[0] == batch_size, f"Expected batch size {batch_size}, got {hidden_states.shape[0]}"
            assert hidden_states.shape[1] == seq_len, f"Expected seq length {seq_len}, got {hidden_states.shape[1]}"
            
            # 4. Apply chain of thought reasoning with multimodal context
            logger.debug("Applying chain of thought")
            if multimodal_context:
                hidden_states = self.thinking(hidden_states, multimodal_context)
            else:
                hidden_states = self.thinking(hidden_states)
                
            # Verify shape after thinking
            if hidden_states.size(0) != batch_size:
                hidden_states = hidden_states.reshape(batch_size, seq_len, -1)
            
            # Final shape verification
            assert hidden_states.shape[0] == batch_size, f"Expected batch size {batch_size}, got {hidden_states.shape[0]}"
            assert hidden_states.shape[1] == seq_len, f"Expected seq length {seq_len}, got {hidden_states.shape[1]}"
            
            # 5. Apply test-time computation if needed
            if not self.training:
                logger.debug("Applying test-time computation")
                hidden_states = self.ttc(hidden_states)
            
            # 6. Store in memory if requested
            if use_memory:
                logger.debug("Storing in memory")
                self.memory.store(
                    hidden_states,
                    attention_scores=transformer_out.get('attention_maps')
                )
            
            # 7. Update stats
            if self._profile_enabled:
                self._update_stats(transformer_out)
                self._end_profile()
            
            # Extract hidden states from transformer output
            if isinstance(hidden_states, dict):
                hidden_states = hidden_states['hidden_states']
            
            # Return appropriate output based on return_stats flag
            if return_stats:
                stats = {
                    **transformer_out['stats'].__dict__,
                    'reasoning_steps': self._compute_stats if self._test_time_enabled else []
                }
                return hidden_states, stats
            return hidden_states
            
        except Exception as e:
            logger.error(f"Error in forward pass: {str(e)}")
            raise
            
    def enable_test_time(self):
        """Enable test-time computation and reasoning tracking"""
        self._test_time_enabled = True
        self._compute_stats = []
        
    def disable_test_time(self):
        """Disable test-time computation"""
        self._test_time_enabled = False
        self._compute_stats = []
        
    def _update_stats(self, transformer_out):
        """Update performance statistics"""
        self.stats.attention_scores = transformer_out.get('attention_maps')
        if hasattr(self.memory, 'stats'):
            self.stats.memory_usage = {
                'stored_tokens': self.memory.stats.stored_tokens,
                'retrieved_tokens': self.memory.stats.retrieved_tokens,
                'hit_rate': self.memory.stats.hit_rate
            }
