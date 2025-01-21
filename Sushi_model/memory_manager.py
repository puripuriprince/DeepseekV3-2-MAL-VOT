import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import logging
from typing import Optional, Dict, Any, Tuple
from dataclasses import dataclass
from config.model_config import ModelConfig

logger = logging.getLogger(__name__)

@dataclass
class MemoryStats:
    """Track memory usage statistics"""
    stored_tokens: int = 0
    retrieved_tokens: int = 0
    peak_memory_mb: float = 0.0
    hit_rate: float = 0.0
    surprise_scores: Optional[torch.Tensor] = None
    prediction_errors: Optional[torch.Tensor] = None

class MemoryManager(nn.Module):
    """
    Hierarchical memory management with surprise-based storage from papers.
    Features:
    - Working memory for recent surprising states
    - Persistent memory for fixed context
    - Long-term memory with gated updates
    - Surprise-based storage mechanism
    """
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        # Initialize stats
        self.stats = {
            'retrieved_tokens': 0,
            'stored_tokens': 0,
            'prediction_errors': None,
            'surprise_scores': None,
            'hit_rate': 0.0
        }
        
        # Extract dimension
        dim = config.dim if isinstance(config, ModelConfig) else config
        
        # Memory hierarchy
        self.persistent_memory = nn.Parameter(
            torch.randn(1, config.num_persist_mem_tokens, dim),
            requires_grad=False  # Fixed context
        )
        self.long_term_buffer = nn.Parameter(
            torch.zeros(1, config.num_longterm_mem_tokens, dim)
        )
        self.working_memory = nn.Parameter(
            torch.zeros(1, config.num_persist_mem_tokens // 2, dim)
        )
        
        # Enhanced predictor network for better surprise estimation
        self.predictor = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim * 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(dim * 2, dim),
            nn.LayerNorm(dim)
        )
        
        # Add a running mean estimator for better surprise detection
        self.register_buffer('running_mean', torch.zeros(dim))
        self.register_buffer('running_var', torch.ones(dim))
        self.momentum = 0.1
        
        # Improved gating with normalization
        self.surprise_gate = nn.Sequential(
            nn.LayerNorm(dim * 2),
            nn.Linear(dim * 2, dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(dim, 1),
            nn.Sigmoid()
        )
        
        # Projection layers
        self.memory_proj = nn.Linear(dim, dim)
        self.context_proj = nn.Linear(dim, dim)
        
        # Layer norm for predictions
        self.pred_norm = nn.LayerNorm(dim)
        
    def compute_surprise(self, hidden_states: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute surprise scores based on prediction errors.
        Uses improved error calculation from papers.
        """        # Make predictions
        predictions = self.predictor(hidden_states)

        # Update running statistics
        with torch.no_grad():
            mean = hidden_states.mean(dim=(0, 1))
            var = hidden_states.var(dim=(0, 1))
            self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * mean
            self.running_var = (1 - self.momentum) * self.running_var + self.momentum * var

        # Compute prediction errors
        local_error = torch.norm(predictions - hidden_states, dim=-1)
        normalized_states = (hidden_states - self.running_mean) / (self.running_var.sqrt() + 1e-5)
        global_error = torch.norm(normalized_states, dim=-1)        # For random sequences:
        # - Higher prediction errors (local_error)
        # - Higher deviation from mean (global_error)
        # - Higher variance in both errors

        # Scale errors to emphasize differences
        # For random sequences, both errors will be high and varied
        # For constant sequences, both errors will be low and uniform
        scaled_local = local_error / (local_error.mean() + 1e-8)
        scaled_global = global_error / (global_error.mean() + 1e-8)

        # Compute variance as a measure of sequence randomness
        local_var = scaled_local.var(dim=1, keepdim=True)
        global_var = scaled_global.var(dim=1, keepdim=True)

        # Combine errors with variance weighting
        # This ensures random sequences (high variance) get higher scores
        combined_error = (scaled_local + scaled_global) * (local_var + global_var)

        # Normalize to [0,1] range per batch
        combined_error = (combined_error - combined_error.min(dim=1, keepdim=True)[0]) / (
            combined_error.max(dim=1, keepdim=True)[0] - combined_error.min(dim=1, keepdim=True)[0] + 1e-8
        )

        # Convert to surprise scores
        surprise_scores = combined_error.unsqueeze(-1)

        # Return both scores and errors for testing
        return surprise_scores, combined_error
        self.stats.surprise_scores = surprise_scores
        
        return surprise_scores, errors
        
    def store(self, hidden_states: torch.Tensor, attention_scores: Optional[torch.Tensor] = None):
        """
        Store new context in memory using surprise-based gating.
        Implements hierarchical storage with exponential moving average updates.
        Optionally uses accelerated scan operations from Titans paper.
        """
        batch_size = hidden_states.size(0)
        
        # Compute surprise scores
        surprise_scores, errors = self.compute_surprise(hidden_states)
        
        if self.config.use_accelerated_scan:
            try:
                from titans_pytorch.associative_scan import associative_scan, binary_operator
                
                # Prepare gates and values for scan
                gates = surprise_scores.view(-1, 1)  # [batch*seq, 1]
                values = hidden_states.view(-1, self.config.dim)  # [batch*seq, dim]
                
                # Run associative scan
                _, scanned_values = associative_scan(
                    binary_operator,
                    (gates, values)
                )
                
                # Reshape back
                scanned_values = scanned_values.view(batch_size, -1, self.config.dim)
                
                # Use scanned values for memory updates
                working_update = scanned_values[:, -top_k:]
                
            except ImportError:
                logger.warning("Accelerated scan not available, falling back to standard update")
                # Fallback to standard topk approach
                top_k = self.config.num_persist_mem_tokens // 2
                _, top_indices = torch.topk(surprise_scores.squeeze(-1), k=top_k, dim=1)
                working_update = torch.zeros_like(self.working_memory)
                for b in range(batch_size):
                    working_update[0, :top_k] = hidden_states[b, top_indices[b]]
        else:
            # Standard topk approach
            top_k = self.config.num_persist_mem_tokens // 2
            _, top_indices = torch.topk(surprise_scores.squeeze(-1), k=top_k, dim=1)
            working_update = torch.zeros_like(self.working_memory)
            for b in range(batch_size):
                working_update[0, :top_k] = hidden_states[b, top_indices[b]]
        
        # Use scatter to update working memory efficiently
        working_update = torch.zeros_like(self.working_memory)
        for b in range(batch_size):
            working_update[0, :top_k] = hidden_states[b, top_indices[b]]
        
        # Exponential moving average update for working memory
        alpha_work = 0.3  # Faster update rate for working memory
        self.working_memory.data = (
            (1 - alpha_work) * self.working_memory + 
            alpha_work * working_update
        )
        
        # Update long-term memory with gated write based on surprise
        memory_update = self.memory_proj(hidden_states)
        
        # Weight updates by normalized surprise scores
        surprise_weights = F.softmax(surprise_scores.squeeze(-1), dim=-1)
        weighted_update = memory_update * surprise_weights.unsqueeze(-1)
        
        # Compute average update with attention context if available
        if attention_scores is not None:
            # Use attention patterns to guide memory updates
            attn_weights = attention_scores.mean(dim=1)  # Average over heads
            weighted_update = weighted_update * attn_weights.unsqueeze(-1)
            
        avg_update = weighted_update.mean(dim=(0, 1), keepdim=True)
        
        # Expand to match long-term buffer size
        avg_update = avg_update.expand(-1, self.config.num_longterm_mem_tokens, -1)
        
        # Exponential moving average update
        alpha = 0.1  # Update rate
        self.long_term_buffer.data = (
            (1 - alpha) * self.long_term_buffer + 
            alpha * avg_update
        )
        
        # Update stats
        self.stats['stored_tokens'] += hidden_states.size(1)
        if torch.cuda.is_available():
            self.stats['peak_memory_mb'] = torch.cuda.max_memory_allocated() / 1024 / 1024
            
    def retrieve(self, query_states: torch.Tensor) -> torch.Tensor:
        """Retrieve relevant memories using hierarchical attention"""
        batch_size = query_states.size(0)
        
        # Project query
        query_proj = self.context_proj(query_states)
        
        # Combine memories hierarchically
        memories = [
            self.working_memory,  # Most recent surprising states
            self.persistent_memory,  # Fixed memories
            self.long_term_buffer  # Long-term gated memories
        ]
        
        # Compute attention for each memory level
        retrieved_memories = []
        for memory in memories:
            # Compute attention scores
            scores = torch.matmul(
                query_proj, memory.transpose(-2, -1)
            ) / math.sqrt(self.config.dim)
            
            # Apply softmax
            attn_probs = torch.softmax(scores, dim=-1)
            
            # Weighted sum of memories
            retrieved = torch.matmul(attn_probs, memory)
            retrieved_memories.append(retrieved)
        
        # Combine retrieved memories with learned weights
        combined_memory = sum(retrieved_memories) / len(memories)
        
        # Update stats
        self.stats['retrieved_tokens'] += query_states.size(1)
        self.stats['hit_rate'] = (attn_probs > 0.1).float().mean().item()
        
        return combined_memory
        
    def forget(self):
        """Clear memory buffers"""
        self.long_term_buffer.data.zero_()
        self.working_memory.data.zero_()
        logger.info("Cleared memory buffers")
