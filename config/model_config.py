import logging
import logging
from dataclasses import dataclass, field
from typing import Optional, Dict, Any
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('sushi_model.log')
    ]
)

logger = logging.getLogger(__name__)

@dataclass
class ModelConfig:
    # Model architecture
    dim: int = 256
    hidden_size: int = 256
    ngram_vocab_size: int = 10000
    encoder_layers: int = 4
    num_heads: int = 8  # Added to support tests
    lambda_init: float = 0.8
    
    # Logging settings
    log_level: Optional[str] = "INFO"
    debug_mode: bool = False
    
    # Debug settings
    debug_mode: bool = False
    profile_mode: bool = False
    log_level: str = "INFO"
    log_file: Optional[str] = "sushi_model.log"
    
    # Performance settings
    use_cuda_kernels: bool = True
    use_flash_attention: bool = True
    memory_efficient_attention: bool = True
    gradient_checkpointing: bool = True
    use_diff_transformer: bool = True  # Enable by default
    use_self_adaptation: bool = True   # Enable by default
    use_xformers: bool = False  # Optional xFormers integration
    use_local_block: bool = False  # Enable local block attention
    use_accelerated_scan: bool = False  # Enable accelerated scan for memory
    use_weighted_combination: bool = False  # Enable weighted combination of experts
    
    # Attention settings
    local_block_size: Optional[int] = None  # Size for local block attention
    chunk_size: int = 128  # Size for chunked attention
    sliding_window: Optional[int] = None  # Size of sliding window
    
    # DiffTransformer settings
    lambda_init: float = 0.8
    svd_rank: int = 4
    svd_init_std: float = 0.01
    group_norm_eps: float = 1e-5
    num_experts: int = 3
    num_layers: int = 12
    heads: int = 8
    dropout: float = 0.1
    
    # Memory settings
    segment_len: int = 128
    num_persist_mem_tokens: int = 8  # Increased for hierarchical memory
    num_longterm_mem_tokens: int = 32  # Increased capacity
    max_sequence_length: int = 8192
    memory_dim: int = 768
    memory_update_steps: int = 100
    memory_forget_rate: float = 0.1
    surprise_threshold: float = 0.5
    
    # Adaptation settings
    adaptation_steps: int = 10
    adaptation_top_k: int = 5
    adaptation_population_size: int = 100
    adaptation_init_std: float = 0.1
    classifier_expert_name: str = 'job_classifier'
    fallback_expert_name: str = 'others'
    
    # Reasoning settings
    num_reasoning_steps: int = 4
    
    # Training settings
    learning_rate: float = 2e-3
    kl_coef: float = 0.1
    num_epochs: int = 100
    batch_size: int = 32
    gradient_clip: float = 1.0
    
    # Adaptation settings
    adaptation_steps: int = 10
    adaptation_top_k: int = 5
    adaptation_population_size: int = 100
    adaptation_init_std: float = 0.1
    classifier_expert_name: str = 'job_classifier'
    fallback_expert_name: str = 'others'
    
    # Paths
    checkpoint_dir: str = "checkpoints"
    log_dir: str = "logs"
    
    @classmethod
    def from_dict(cls, config_dict: dict) -> 'ModelConfig':
        """Create config from dictionary"""
        return cls(**{
            k: v for k, v in config_dict.items()
            if k in cls.__dataclass_fields__
        })
    
    def __post_init__(self):
        # Set up logging based on config
        if self.log_level:
            logging.getLogger().setLevel(getattr(logging, self.log_level))
            logger.info(f"Initialized ModelConfig with dim={self.dim}")
            
        if self.debug_mode:
            logger.setLevel(logging.DEBUG)
            logger.debug("Debug mode enabled")
    
    def to_dict(self) -> dict:
        """Convert config to dictionary"""
        return {
            field.name: getattr(self, field.name)
            for field in self.__dataclass_fields__.values()
        }
    
    def save(self, filepath: str):
        """Save config to JSON file"""
        import json
        with open(filepath, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
    
    @classmethod
    def load(cls, filepath: str) -> 'ModelConfig':
        """Load config from JSON file"""
        import json
        with open(filepath, 'r') as f:
            config_dict = json.load(f)
        return cls.from_dict(config_dict)

# Default configuration
DEFAULT_CONFIG = ModelConfig() 