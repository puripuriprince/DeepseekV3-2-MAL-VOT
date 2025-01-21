# Standard library imports
import os
import json
import logging
from typing import List, Dict, Any, Optional

# Third-party imports
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.distributions import Categorical
from tqdm import tqdm
import argparse
from transformers import AutoModelForCausalLM, AutoTokenizer

# Local imports
from Sushi_model.transformer_squared import TransformerSquared
from Sushi_model.byte_latent_patches import ByteLevelTokenizer


# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class Trainer:
    def __init__(
        self,
        model: TransformerSquared,
        tokenizer: ByteLevelTokenizer,
        learning_rate: float = 2e-3,
        kl_coef: float = 0.1,
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ):
        self.model = model.to(device)
        self.tokenizer = tokenizer
        self.device = device
        self.kl_coef = kl_coef
        
        # Optimizer
        self.optimizer = optim.AdamW(model.parameters(), lr=learning_rate)
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=1000)
        
    def analyze_reasoning(self, reasoning_history: List[Dict]) -> Dict[str, Any]:
        """Analyze the reasoning process from test-time compute"""
        analysis = {
            'gate_values': [],
            'thought_vectors': [],
            'step_impacts': []
        }
        
        for step_info in reasoning_history:
            # Track gate values (how much each reasoning step contributes)
            analysis['gate_values'].append(step_info['gate'].mean().item())
            
            # Track thought vector norms (magnitude of reasoning at each step)
            analysis['thought_vectors'].append(
                step_info['thought_vector'].norm().item()
            )
            
            # Compute impact of this reasoning step
            step_impact = (
                step_info['gate'] * step_info['thought_vector'].norm()
            ).item()
            analysis['step_impacts'].append(step_impact)
            
        return analysis
    
    def compute_rewards(self, 
                       outputs: torch.Tensor, 
                       targets: torch.Tensor,
                       reasoning_stats: Optional[Dict] = None) -> torch.Tensor:
        """Compute rewards for REINFORCE with reasoning quality bonus"""
        # Convert to probabilities
        probs = F.softmax(outputs, dim=-1)
        # Sample from distribution
        dist = Categorical(probs)
        sampled = dist.sample()
        
        # Base rewards (1 for correct, -1 for incorrect)
        rewards = (sampled == targets).float() * 2 - 1
        
        # Add bonus for good reasoning (if stats available)
        if reasoning_stats is not None:
            # Bonus for consistent gate values (smooth reasoning)
            gate_std = np.std(reasoning_stats['gate_values'])
            gate_bonus = torch.full_like(rewards, 1.0 / (1.0 + gate_std))
            
            # Bonus for impactful reasoning steps
            impact_mean = np.mean(reasoning_stats['step_impacts'])
            impact_bonus = torch.full_like(rewards, impact_mean)
            
            # Combine bonuses
            rewards = rewards + 0.1 * (gate_bonus + impact_bonus)
        
        return rewards, dist.log_prob(sampled)
    
    def train_step(
        self,
        batch: Dict[str, torch.Tensor],
        base_model: Optional[TransformerSquared] = None
    ) -> Dict[str, float]:
        """
        Single training step with performance monitoring and error handling
        
        Args:
            batch: Input batch
            base_model: Optional teacher model for distillation
            
        Returns:
            Dict of metrics
        """
        """Single training step with chain of thought reasoning"""
        logger.debug("Starting training step")
        start_time = torch.cuda.Event(enable_timing=True)
        end_time = torch.cuda.Event(enable_timing=True)
        
        try:
            self.model.train()
            self.optimizer.zero_grad()
            
            # Enable test-time compute for reasoning analysis
            self.model.enable_test_time()
            
            # Track memory usage
            if torch.cuda.is_available():
                start_mem = torch.cuda.memory_allocated()
                
            # Get inputs and targets
            input_ids = batch["input_ids"].to(self.device)
            target_ids = batch["target_ids"].to(self.device)
            
            logger.debug(f"Input shape: {input_ids.shape}, Target shape: {target_ids.shape}")
            
            start_time.record()
            
            # First pass - task identification with reasoning
            first_pass_output, stats = self.model(input_ids)
            
            # Log performance stats
            if stats:
                logger.debug(f"First pass stats: {stats}")
        
        # Get reasoning statistics
        stats = self.model.get_compute_stats()
        reasoning_analysis = self.analyze_reasoning(stats['reasoning_steps'][-1])
        
        # Compute expert weights based on first pass
        expert_logits = self.model.compute_expert_weights(first_pass_output)
        expert_weights = F.softmax(expert_logits, dim=-1)
        
        # Second pass with expert weights and reasoning
        outputs, _ = self.model(input_ids, expert_weights=expert_weights)
        
        # Compute REINFORCE loss with reasoning-aware rewards
        rewards, log_probs = self.compute_rewards(
            outputs, 
            target_ids,
            reasoning_stats=reasoning_analysis
        )
        reinforce_loss = -(log_probs * rewards).mean()
        
        # Compute KL divergence if base model provided
        kl_loss = 0
        if base_model is not None:
            with torch.no_grad():
                base_outputs, _ = base_model(input_ids)
            kl_loss = F.kl_div(
                F.log_softmax(outputs, dim=-1),
                F.softmax(base_outputs, dim=-1),
                reduction='batchmean'
            )
        
        # Total loss
        loss = reinforce_loss + self.kl_coef * kl_loss
        
        # Backward pass
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.optimizer.step()
        self.scheduler.step()
        
        # Disable test-time compute
        self.model.disable_test_time()
        
        end_time.record()
        torch.cuda.synchronize()
        
        # Calculate memory usage
        if torch.cuda.is_available():
            end_mem = torch.cuda.memory_allocated()
            mem_diff = end_mem - start_mem
            logger.debug(f"Memory usage for step: {mem_diff / 1024**2:.2f}MB")
        
        # Log timing
        step_time = start_time.elapsed_time(end_time)
        logger.debug(f"Step time: {step_time:.2f}ms")
        
        metrics = {
            "loss": loss.item(),
            "reinforce_loss": reinforce_loss.item(),
            "kl_loss": kl_loss.item() if base_model else 0,
            "mean_gate_value": np.mean(reasoning_analysis['gate_values']),
            "mean_impact": np.mean(reasoning_analysis['step_impacts']),
            "step_time_ms": step_time,
            "memory_mb": mem_diff / 1024**2 if torch.cuda.is_available() else 0
        }
        
        logger.debug(f"Step metrics: {metrics}")
        return metrics
    
    def train(
        self,
        train_dataloader: torch.utils.data.DataLoader,
        val_dataloader: torch.utils.data.DataLoader,
        num_epochs: int,
        base_model: Optional[TransformerSquared] = None,
        save_dir: str = "checkpoints"
    ):
        """Full training loop with reasoning analysis"""
        best_val_loss = float('inf')
        
        for epoch in range(num_epochs):
            # Training
            self.model.train()
            train_metrics = []
            
            pbar = tqdm(train_dataloader, desc=f"Epoch {epoch}")
            for batch in pbar:
                metrics = self.train_step(batch, base_model)
                train_metrics.append(metrics)
                pbar.set_postfix(metrics)
            
            # Validation with reasoning analysis
            self.model.eval()
            val_losses = []
            val_reasoning_stats = []
            
            with torch.no_grad():
                for batch in val_dataloader:
                    # Enable test-time compute for validation
                    self.model.enable_test_time()
                    
                    input_ids = batch["input_ids"].to(self.device)
                    target_ids = batch["target_ids"].to(self.device)
                    
                    # Two-pass inference with reasoning
                    first_pass, _ = self.model(input_ids)
                    expert_logits = self.model.compute_expert_weights(first_pass)
                    expert_weights = F.softmax(expert_logits, dim=-1)
                    outputs, _ = self.model(input_ids, expert_weights=expert_weights)
                    
                    # Get reasoning statistics
                    stats = self.model.get_compute_stats()
                    reasoning_analysis = self.analyze_reasoning(stats['reasoning_steps'][-1])
                    val_reasoning_stats.append(reasoning_analysis)
                    
                    loss = F.cross_entropy(outputs.view(-1, outputs.size(-1)), target_ids.view(-1))
                    val_losses.append(loss.item())
                    
                    # Disable test-time compute
                    self.model.disable_test_time()
            
            val_loss = sum(val_losses) / len(val_losses)
            
            # Compute average reasoning metrics
            avg_reasoning = {
                'gate_values': np.mean([s['gate_values'] for s in val_reasoning_stats]),
                'step_impacts': np.mean([s['step_impacts'] for s in val_reasoning_stats])
            }
            
            # Logging
            logger.info(
                f"Epoch {epoch}: "
                f"train_loss={sum(m['loss'] for m in train_metrics)/len(train_metrics):.4f}, "
                f"val_loss={val_loss:.4f}, "
                f"avg_gate={avg_reasoning['gate_values']:.4f}, "
                f"avg_impact={avg_reasoning['step_impacts']:.4f}"
            )
            
            # Save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'val_loss': val_loss,
                    'reasoning_stats': avg_reasoning
                }, f"{save_dir}/best_model.pt")

class DistillationTrainer:
    def __init__(
        self,
        student_model: TransformerSquared,
        tokenizer: ByteLevelTokenizer,
        learning_rate: float = 2e-4,
        temperature: float = 2.0,
        alpha: float = 0.5,
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ):
        self.student = student_model.to(device)
        self.tokenizer = tokenizer
        self.device = device
        self.temperature = temperature
        self.alpha = alpha
        
        # Initialize DeepSeek teacher model
        self.teacher = AutoModelForCausalLM.from_pretrained(
            "deepseek-ai/deepseek-coder-33b-instruct",
            torch_dtype=torch.bfloat16,
            device_map="auto"
        )
        self.teacher.eval()
        self.teacher_tokenizer = AutoTokenizer.from_pretrained(
            "deepseek-ai/deepseek-coder-33b-instruct"
        )
        
        # Optimizer
        self.optimizer = optim.AdamW(student_model.parameters(), lr=learning_rate)
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=1000)

    def compute_architecture_loss(
        self,
        outputs: Dict[str, torch.Tensor],
        memory: torch.Tensor,
        attention_maps: List[torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """Compute losses for the architectural components"""
        losses = {}
        
        # Memory coherence loss
        if memory is not None:
            memory_coherence = torch.var(memory, dim=1).mean()
            losses['memory'] = memory_coherence
        
        # Differential attention diversity loss
        if attention_maps:
            attn_entropy = -torch.mean(
                torch.sum(
                    F.softmax(attention_maps[-1], dim=-1) * 
                    F.log_softmax(attention_maps[-1], dim=-1),
                    dim=-1
                )
            )
            losses['attention'] = -attn_entropy  # Maximize entropy for diverse attention
        
        # Expert utilization loss
        if 'expert_weights' in outputs:
            expert_entropy = -torch.mean(
                torch.sum(
                    outputs['expert_weights'] * 
                    torch.log(outputs['expert_weights'] + 1e-10),
                    dim=-1
                )
            )
            losses['expert'] = -expert_entropy  # Maximize entropy for balanced expert use
        
        return losses

    def train_step(
        self,
        batch: Dict[str, torch.Tensor],
    ) -> Dict[str, float]:
        """Single training step with integrated architectural components"""
        self.student.train()
        self.optimizer.zero_grad()
        
        input_ids = batch["input_ids"].to(self.device)
        target_ids = batch["target_ids"].to(self.device)
        
        # Enable test-time compute for reasoning analysis
        self.student.enable_test_time()
        
        # Forward pass through student - using the integrated components
        student_outputs = self.student(input_ids)
        
        # Get teacher predictions
        with torch.no_grad():
            teacher_inputs = self.convert_to_teacher_tokens(input_ids)
            teacher_outputs = self.teacher(teacher_inputs.to(self.teacher.device))
            teacher_logits = teacher_outputs.logits.to(self.device)
        
        # Get reasoning statistics
        stats = self.student.get_compute_stats()
        reasoning_analysis = self.analyze_reasoning(stats['reasoning_steps'][-1])
        
        # Compute architectural losses using the integrated components
        architecture_losses = self.compute_architecture_loss(
            student_outputs,
            student_outputs.get('memory', None),
            student_outputs.get('attention_maps', [])
        )
        
        # Main losses
        distillation_loss = self.compute_distillation_loss(
            student_outputs['logits'], 
            teacher_logits
        )
        
        task_loss = F.cross_entropy(
            student_outputs['logits'].view(-1, student_outputs['logits'].size(-1)),
            target_ids.view(-1)
        )
        
        reasoning_loss = self.compute_reasoning_loss(reasoning_analysis)
        
        # Combine all losses
        architecture_loss = sum(architecture_losses.values())
        total_loss = (
            self.alpha * distillation_loss +
            (1 - self.alpha) * task_loss +
            0.1 * reasoning_loss +
            0.1 * architecture_loss
        )
        
        # Backward pass
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.student.parameters(), 1.0)
        self.optimizer.step()
        self.scheduler.step()
        
        # Disable test-time compute
        self.student.disable_test_time()
        
        # Return metrics
        metrics = {
            "total_loss": total_loss.item(),
            "distillation_loss": distillation_loss.item(),
            "task_loss": task_loss.item(),
            "reasoning_loss": reasoning_loss.item(),
            "architecture_loss": architecture_loss.item(),
            "mean_gate_value": np.mean(reasoning_analysis['gate_values']),
            "mean_impact": np.mean(reasoning_analysis['step_impacts'])
        }
        
        # Add individual architecture losses to metrics
        for name, loss in architecture_losses.items():
            metrics[f"{name}_loss"] = loss.item()
        
        return metrics

    def convert_to_teacher_tokens(self, byte_tokens: torch.Tensor) -> torch.Tensor:
        """Convert byte-level tokens to teacher model tokens"""
        # Decode byte tokens to text
        texts = self.tokenizer.batch_decode(byte_tokens)
        # Encode with teacher tokenizer
        teacher_tokens = self.teacher_tokenizer(
            texts,
            padding=True,
            truncation=True,
            return_tensors="pt"
        )
        return teacher_tokens.input_ids
    
    def compute_reasoning_loss(self, reasoning_stats: Dict) -> torch.Tensor:
        """Compute loss based on reasoning quality"""
        # Encourage consistent gate values
        gate_std = torch.tensor(np.std(reasoning_stats['gate_values'])).to(self.device)
        gate_loss = 1.0 / (1.0 + gate_std)
        
        # Encourage impactful reasoning steps
        impact_mean = torch.tensor(np.mean(reasoning_stats['step_impacts'])).to(self.device)
        impact_loss = -impact_mean  # Negative because we want to maximize impact
        
        return gate_loss + impact_loss

    def train(
        self,
        train_dataloader: DataLoader,
        val_dataloader: DataLoader,
        num_epochs: int,
        save_dir: str = "checkpoints"
    ):
        """Full training loop with knowledge distillation"""
        best_val_loss = float('inf')
        
        for epoch in range(num_epochs):
            # Training
            self.student.train()
            train_metrics = []
            
            pbar = tqdm(train_dataloader, desc=f"Epoch {epoch}")
            for batch in pbar:
                metrics = self.train_step(batch)
                train_metrics.append(metrics)
                pbar.set_postfix(metrics)
            
            # Validation
            self.student.eval()
            val_losses = []
            val_reasoning_stats = []
            
            with torch.no_grad():
                for batch in val_dataloader:
                    self.student.enable_test_time()
                    
                    input_ids = batch["input_ids"].to(self.device)
                    target_ids = batch["target_ids"].to(self.device)
                    
                    # Get student and teacher predictions
                    student_outputs = self.student(input_ids)
                    teacher_inputs = self.convert_to_teacher_tokens(input_ids)
                    teacher_outputs = self.teacher(teacher_inputs.to(self.teacher.device))
                    
                    # Compute validation losses
                    distill_loss = self.compute_distillation_loss(
                        student_outputs['logits'],
                        teacher_outputs.logits.to(self.device)
                    )
                    task_loss = F.cross_entropy(
                        student_outputs['logits'].view(-1, student_outputs['logits'].size(-1)),
                        target_ids.view(-1)
                    )
                    
                    # Get reasoning statistics
                    stats = self.student.get_compute_stats()
                    reasoning_analysis = self.analyze_reasoning(stats['reasoning_steps'][-1])
                    val_reasoning_stats.append(reasoning_analysis)
                    
                    total_loss = self.alpha * distill_loss + (1 - self.alpha) * task_loss
                    val_losses.append(total_loss.item())
                    
                    self.student.disable_test_time()
            
            val_loss = sum(val_losses) / len(val_losses)
            
            # Compute average reasoning metrics
            avg_reasoning = {
                'gate_values': np.mean([s['gate_values'] for s in val_reasoning_stats]),
                'step_impacts': np.mean([s['step_impacts'] for s in val_reasoning_stats])
            }
            
            # Logging
            logger.info(
                f"Epoch {epoch}: "
                f"train_loss={sum(m['total_loss'] for m in train_metrics)/len(train_metrics):.4f}, "
                f"val_loss={val_loss:.4f}, "
                f"avg_gate={avg_reasoning['gate_values']:.4f}, "
                f"avg_impact={avg_reasoning['step_impacts']:.4f}"
            )
            
            # Save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.student.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'val_loss': val_loss,
                    'reasoning_stats': avg_reasoning
                }, f"{save_dir}/best_model.pt")

def test_model(
    model: TransformerSquared,
    tokenizer: ByteLevelTokenizer,
    test_input: str,
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
):
    """Test the model with chain of thought reasoning"""
    model.eval()
    model.enable_test_time()
    
    # Tokenize input - ensure we get a proper tensor
    if isinstance(test_input, str):
        # Convert string to bytes then to tensor
        input_bytes = test_input.encode('utf-8')
        input_ids = torch.tensor([[b for b in input_bytes]], dtype=torch.long).to(device)
    else:
        input_ids = test_input.to(device)
    
    with torch.no_grad():
        # First pass with reasoning
        first_pass = model(input_ids, input_type='bytes')  # Specify input_type
        
        # Get reasoning statistics
        stats = model.get_compute_stats()
        reasoning_analysis = Trainer.analyze_reasoning(None, stats['reasoning_steps'][-1])
        
        # Print reasoning steps
        print("\nReasoning Process:")
        for step, (gate, impact) in enumerate(zip(
            reasoning_analysis['gate_values'],
            reasoning_analysis['step_impacts']
        )):
            print(f"Step {step}:")
            print(f"- Contribution: {gate:.3f}")
            print(f"- Impact: {impact:.3f}")
        
        # Compute expert weights and second pass
        expert_weights = model.compute_expert_weights(first_pass['hidden_states'])
        outputs = model(input_ids, input_type='bytes', expert_weights=expert_weights)
        
        # Decode output
        if 'logits' in outputs:
            output_bytes = outputs['logits'].argmax(dim=-1)
            output_text = tokenizer.decode(output_bytes[0])
        else:
            output_text = "[No text output available]"
        
        print("\nFinal Output:", output_text)
        print("\nExpert Weights:", expert_weights.tolist())
    
    model.disable_test_time()
    return output_text, reasoning_analysis

def test_model_inputs(model: TransformerSquared, config: Dict, device: str):
    """Test the model with different types of inputs"""
    model.eval()
    logger.info("Testing model with different input types...")
    
    with torch.no_grad():
        # Test text input
        text_input = "Hello, this is a test of the multimodal transformer."
        logger.info(f"\nTesting text input: {text_input}")
        output = model(text_input, input_type='text', output_type='text')
        logger.info(f"Text output shape: {output['text'].shape}")
        
        # Test image input
        image = torch.randn(1, 3, config['image_size'], config['image_size']).to(device)
        logger.info(f"\nTesting image input with shape: {image.shape}")
        output = model(image, input_type='image', output_type='image')
        logger.info(f"Image output shape: {output['image'].shape}")
        
        # Test mesh input (random vertices and faces)
        vertices = torch.randn(1, config['mesh_max_vertices'], 3).to(device)
        faces = torch.randint(0, config['mesh_max_vertices'], (1, config['mesh_max_vertices']-2, 3)).to(device)
        mesh_input = (vertices, faces)
        logger.info(f"\nTesting mesh input with vertices shape: {vertices.shape}")
        output = model(mesh_input, input_type='mesh', output_type='mesh')
        logger.info(f"Mesh output - vertices shape: {output['mesh']['vertices'].shape}")
        
        # Test multimodal input
        multimodal_input = {
            'text': text_input,
            'image': image
        }
        logger.info("\nTesting multimodal input")
        output = model(multimodal_input, input_type='mixed', output_type='multimodal')
        logger.info("Multimodal output shapes:")
        for key, value in output.items():
            if isinstance(value, torch.Tensor):
                logger.info(f"- {key}: {value.shape}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train or test the TransformerSquared model')
    parser.add_argument('--config', type=str, default='config/model_config.json',
                        help='Path to model configuration file')
    parser.add_argument('--test', action='store_true',
                        help='Run test forward pass with different input types')
    parser.add_argument('--train', action='store_true',
                        help='Run training loop')
    parser.add_argument('--model_path', type=str,
                        help='Path to pre-trained model checkpoint to load')
    args = parser.parse_args()
    
    # Load configuration
    with open(args.config, 'r') as f:
        config = json.load(f)
    
    # Create model
    model = TransformerSquared(
        dim=config['dim'],
        num_layers=config['num_layers'],
        num_heads=config['heads'],
        mlp_ratio=4,
        dropout=config['dropout'],
        max_sequence_length=config['max_sequence_length'],
        image_size=config['image_size'],
        patch_size=config['patch_size'],
        memory_size=config['memory_size'],
        num_reasoning_steps=config['num_reasoning_steps'],
        use_gradient_checkpointing=config['use_gradient_checkpointing']
    )
    
    # Load pre-trained model if specified
    if args.model_path:
        logger.info(f"Loading pre-trained model from {args.model_path}")
        checkpoint = torch.load(args.model_path, map_location='cpu')
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    if args.test:
        test_model_inputs(model, config, device)
        
    if args.train:
        # Create trainer
        tokenizer = ByteLevelTokenizer()
        trainer = DistillationTrainer(
            student_model=model,
            tokenizer=tokenizer,
            learning_rate=config['learning_rate'],
            temperature=config['temperature'],
            alpha=config['alpha']
        )
