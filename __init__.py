"""
DeepseekV3-2-MAL-VOT: A Transformer² model with byte-level processing, differential attention,
and self-adaptive memory for enhanced reasoning capabilities.
"""

from models import (
    TransformerSquared,
    ByteLatentPatches,
    ByteLevelTokenizer,
    DifferentialAttention,
    TitanMemoryLayer
)
from config import ModelConfig, DEFAULT_CONFIG

__version__ = "0.1.0"
__author__ = "Your Name"

__all__ = [
    # Models
    'TransformerSquared',
    'ByteLatentPatches',
    'ByteLevelTokenizer',
    'DifferentialAttention',
    'TitanMemoryLayer',
    
    # Configuration
    'ModelConfig',
    'DEFAULT_CONFIG'
]