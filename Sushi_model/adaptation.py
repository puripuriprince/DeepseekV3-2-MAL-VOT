import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from config.model_config import ModelConfig

class BaseAdaptationStrategy:
    """Base class for adaptation strategies"""
    def __init__(self):
        self.training = True

    def train(self, mode=True):
        self.training = mode
        return self

    def eval(self):
        return self.train(False)

    def adapt_z_vectors(
        self,
        input_ids: torch.Tensor,
        user_prompt: str,
        experts: Dict[str, Dict[str, torch.Tensor]],
        few_shot_samples: Optional[List[torch.Tensor]] = None,
        scoring_fn = None
    ) -> Dict[str, torch.Tensor]:
        raise NotImplementedError

class PromptBasedAdaptationStrategy(BaseAdaptationStrategy):
    """Adaptation strategy that uses prompt text to select expert"""
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.fallback_expert = config.fallback_expert_name

    def adapt_z_vectors(
        self,
        input_ids: torch.Tensor,
        user_prompt: str,
        experts: Dict[str, Dict[str, torch.Tensor]],
        few_shot_samples: Optional[List[torch.Tensor]] = None,
        scoring_fn = None
    ) -> Dict[str, torch.Tensor]:
        # Simple keyword matching for expert selection
        selected_expert = self.fallback_expert
        for expert_name in experts.keys():
            if expert_name.lower() in user_prompt.lower():
                selected_expert = expert_name
                break
                
        return experts[selected_expert]

class ClassificationExpertAdaptationStrategy(BaseAdaptationStrategy):
    """Adaptation strategy that uses a classifier to select expert"""
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.classifier = nn.Linear(config.dim, len(config.expert_names))
        self.fallback_expert = config.fallback_expert_name

    def adapt_z_vectors(
        self,
        input_ids: torch.Tensor,
        user_prompt: str,
        experts: Dict[str, Dict[str, torch.Tensor]],
        few_shot_samples: Optional[List[torch.Tensor]] = None,
        scoring_fn = None
    ) -> Dict[str, torch.Tensor]:
        # Get embeddings from input_ids (simplified)
        hidden = torch.mean(input_ids, dim=1)  # [batch, dim]
        logits = self.classifier(hidden)  # [batch, num_experts]
        expert_probs = F.softmax(logits, dim=-1)
        
        # Select expert with highest probability
        expert_idx = torch.argmax(expert_probs, dim=-1)[0]
        selected_expert = list(experts.keys())[expert_idx]
        
        return experts[selected_expert]

class CEMAdaptationStrategy(BaseAdaptationStrategy):
    """
    Cross-Entropy Method (CEM) based adaptation strategy.
    Optimizes mixture of experts through iterative sampling.
    """
    def __init__(
        self,
        config: ModelConfig,
        num_samples: int = 100,
        elite_fraction: float = 0.1,
        num_iterations: int = 5
    ):
        super().__init__()
        self.config = config
        self.num_samples = num_samples
        self.elite_fraction = elite_fraction
        self.num_iterations = num_iterations
        self.elite_size = max(1, int(num_samples * elite_fraction))

    def adapt_z_vectors(
        self,
        input_ids: torch.Tensor,
        user_prompt: str,
        experts: Dict[str, Dict[str, torch.Tensor]],
        few_shot_samples: Optional[List[torch.Tensor]] = None,
        scoring_fn = None
    ) -> Dict[str, torch.Tensor]:
        if scoring_fn is None:
            return experts[self.config.fallback_expert_name]
            
        num_experts = len(experts)
        device = next(iter(experts.values()))[next(iter(experts.values()))].device
        
        # Initialize distribution parameters
        mu = torch.ones(num_experts, device=device) / num_experts
        sigma = torch.ones(num_experts, device=device) * 0.1
        
        for _ in range(self.num_iterations):
            # Sample weights
            weights = torch.normal(mu.unsqueeze(0).expand(self.num_samples, -1),
                                 sigma.unsqueeze(0).expand(self.num_samples, -1))
            weights = F.softmax(weights, dim=-1)
            
            # Evaluate samples
            scores = []
            for w in weights:
                # Combine experts
                combined = {}
                for layer_key in next(iter(experts.values())).keys():
                    layer_vectors = []
                    for i, (_, expert) in enumerate(experts.items()):
                        layer_vectors.append(expert[layer_key])
                    stacked = torch.stack(layer_vectors)
                    combined[layer_key] = torch.sum(stacked * w.view(-1, 1), dim=0)
                
                # Score combination
                score = scoring_fn(combined)
                scores.append(score)
            
            # Select elite samples
            scores = torch.tensor(scores, device=device)
            _, elite_idx = torch.topk(scores, self.elite_size)
            elite_weights = weights[elite_idx]
            
            # Update distribution
            mu = elite_weights.mean(dim=0)
            sigma = elite_weights.std(dim=0)
        
        # Return best mixture
        combined = {}
        for layer_key in next(iter(experts.values())).keys():
            layer_vectors = []
            for i, (_, expert) in enumerate(experts.items()):
                layer_vectors.append(expert[layer_key])
            stacked = torch.stack(layer_vectors)
            combined[layer_key] = torch.sum(stacked * mu.view(-1, 1), dim=0)
            
        return combined

class SVFAdaptationStrategy(BaseAdaptationStrategy):
    """
    Self-Verification Fine-tuning (SVF) adaptation strategy.
    Uses self-verification to adapt model parameters based on task performance.
    """
    def __init__(
        self,
        config: ModelConfig,
        verification_steps: int = 3,
        learning_rate: float = 0.01
    ):
        super().__init__()
        self.config = config
        self.verification_steps = verification_steps
        self.learning_rate = learning_rate
        self.fallback_expert = config.fallback_expert_name

    def adapt_z_vectors(
        self,
        input_ids: torch.Tensor,
        user_prompt: str,
        experts: Dict[str, Dict[str, torch.Tensor]],
        few_shot_samples: Optional[List[torch.Tensor]] = None,
        scoring_fn = None
    ) -> Dict[str, torch.Tensor]:
        if scoring_fn is None or few_shot_samples is None:
            return experts[self.fallback_expert]
            
        # Start with fallback expert
        best_expert = experts[self.fallback_expert]
        best_score = float('-inf')
        
        # Try verification steps with each expert
        for expert_name, expert in experts.items():
            total_score = 0
            for sample in few_shot_samples:
                # Score the expert on this sample
                score = scoring_fn(expert, sample)
                total_score += score
                
            avg_score = total_score / len(few_shot_samples)
            if avg_score > best_score:
                best_score = avg_score
                best_expert = expert
                
        return best_expert

class WeightedCombinationAdaptationStrategy(BaseAdaptationStrategy):
    """
    Adaptation strategy that combines multiple experts using learned weights.
    Inspired by Self-Adaptive LLMs paper.
    """
    def __init__(
        self,
        expert_names: List[str],
        config: ModelConfig,
        per_layer: bool = False,
        norm: bool = True,
        init_std: float = 0.01
    ):
        super().__init__()
        self.expert_names = expert_names
        self.config = config
        self.per_layer = per_layer
        self.norm = norm
        
        # Initialize weights close to uniform
        init_val = 1.0 / len(expert_names)
        if per_layer:
            self.adaptive_weights = nn.Parameter(
                torch.full((len(expert_names), config.num_layers), init_val)
            )
        else:
            self.adaptive_weights = nn.Parameter(
                torch.full((len(expert_names),), init_val)
            )
            
        # Add small noise for symmetry breaking
        with torch.no_grad():
            self.adaptive_weights.add_(torch.randn_like(self.adaptive_weights) * init_std)

    def get_weights(self) -> torch.Tensor:
        """Get normalized combination weights"""
        if self.norm:
            # Normalize across experts dimension
            if self.per_layer:
                return F.softmax(self.adaptive_weights, dim=0)
            return F.softmax(self.adaptive_weights, dim=0)
        return self.adaptive_weights

    def adapt_z_vectors(
        self,
        input_ids: torch.Tensor,
        user_prompt: str,
        experts: Dict[str, Dict[str, torch.Tensor]],
        few_shot_samples: Optional[List[torch.Tensor]] = None,
        scoring_fn = None
    ) -> Dict[str, torch.Tensor]:
        """
        Combine expert z-vectors using learned weights
        """
        weights = self.get_weights()
        
        # Weighted combination of experts
        combined = {}
        for layer_key in experts[self.expert_names[0]].keys():
            layer_vectors = []
            for exp_name in self.expert_names:
                if exp_name in experts:
                    layer_vectors.append(experts[exp_name][layer_key])
                
            if not layer_vectors:
                continue
                
            # Stack and combine
            stacked = torch.stack(layer_vectors)
            if self.per_layer:
                # Use per-layer weights
                layer_idx = int(layer_key.split('.')[-1])
                w = weights[:, layer_idx:layer_idx+1]
            else:
                # Use same weights for all layers
                w = weights
                
            # Weighted sum
            combined[layer_key] = torch.sum(stacked * w.unsqueeze(-1), dim=0)
            
        return combined

    def update_weights(
        self,
        rewards: torch.Tensor,
        expert_outputs: Dict[str, torch.Tensor]
    ):
        """
        Update combination weights based on rewards
        """
        if not self.training:
            return
            
        # Compute gradient for weights
        reward_tensor = torch.tensor(rewards, device=self.adaptive_weights.device)
        
        # Simple policy gradient update
        if self.per_layer:
            grad = torch.zeros_like(self.adaptive_weights)
            for i, expert in enumerate(self.expert_names):
                if expert in expert_outputs:
                    grad[i] = reward_tensor.mean()
        else:
            grad = reward_tensor.mean() * torch.ones_like(self.adaptive_weights)
            
        # Update with small step size
        with torch.no_grad():
            self.adaptive_weights.add_(grad * 0.01)

    def get_expert_weights(self) -> Dict[str, float]:
        """Get current expert weights for logging"""
        weights = self.get_weights().detach()
        return {
            name: float(w) 
            for name, w in zip(self.expert_names, weights)
        }
