from dataclasses import dataclass
from typing import Optional

@dataclass
class ModelConfig:
    # Model architecture
    dim: int = 256
    num_experts: int = 3
    num_layers: int = 12
    heads: int = 8
    dropout: float = 0.1
    
    # Memory settings
    segment_len: int = 128
    num_persist_mem_tokens: int = 4
    num_longterm_mem_tokens: int = 16
    max_sequence_length: int = 8192
    
    # Reasoning settings
    num_reasoning_steps: int = 4
    
    # Training settings
    learning_rate: float = 2e-3
    kl_coef: float = 0.1
    num_epochs: int = 100
    batch_size: int = 32
    gradient_clip: float = 1.0
    
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