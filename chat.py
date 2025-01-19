import torch
import torch.nn.functional as F
from models.transformer_squared import TransformerSquared
from models.byte_latent_patches import ByteLevelTokenizer
import argparse
import json
from typing import Dict, List, Tuple
import numpy as np
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.progress import Progress
import os
import logging
from tqdm import tqdm
from models.test_processor import TestProcessor

console = Console()
logger = logging.getLogger(__name__)

class ModelChat:
    def __init__(
        self,
        model_path: str,
        config_path: str = "config/model_config.json",
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        test_mode: bool = False
    ):
        # Load model configuration
        with open(config_path, 'r') as f:
            raw_config = json.load(f)
            
        # Map config keys to match TransformerSquared parameters
        self.config = {
            'dim': raw_config.get('dim', 768),
            'num_layers': raw_config.get('num_layers', 12),
            'num_heads': raw_config.get('heads', 12),
            'num_experts': raw_config.get('num_experts', 3),
            'mlp_ratio': raw_config.get('mlp_ratio', 4),
            'dropout': raw_config.get('dropout', 0.1),
            'max_sequence_length': raw_config.get('max_sequence_length', 2048),
            'image_size': raw_config.get('image_size', 256),
            'patch_size': raw_config.get('patch_size', 16),
            'memory_size': raw_config.get('memory_size', 256),
            'chunk_size': raw_config.get('chunk_size', 64),
            'num_reasoning_steps': raw_config.get('num_reasoning_steps', 3),
            'segment_len': raw_config.get('segment_len', 128),
            'num_persist_mem_tokens': raw_config.get('num_persist_mem_tokens', 4),
            'num_longterm_mem_tokens': raw_config.get('num_longterm_mem_tokens', 8),
            'use_gradient_checkpointing': raw_config.get('use_gradient_checkpointing', False)
        }
        
        try:
            # Initialize model and processor
            logger.info("Initializing model...")
            self.model = TransformerSquared(**self.config)
            self.device = device
            self.test_mode = test_mode
            
            # Choose appropriate processor
            if test_mode:
                logger.info("Using test processor...")
                self.processor = TestProcessor(device=device)
            else:
                logger.info("Using byte level tokenizer...")
                self.tokenizer = ByteLevelTokenizer()
            
            # Load model weights if available
            if os.path.exists(model_path) and model_path.endswith('.pt'):
                logger.info(f"Loading model weights from {model_path}")
                try:
                    checkpoint = torch.load(model_path, map_location=device)
                    if isinstance(checkpoint, dict):
                        state_dict = checkpoint.get('model_state_dict', checkpoint)
                    else:
                        state_dict = checkpoint
                    
                    missing_keys, unexpected_keys = self.model.load_state_dict(
                        state_dict, strict=False
                    )
                    
                    if missing_keys:
                        logger.warning(f"Missing keys when loading checkpoint: {len(missing_keys)} keys")
                    if unexpected_keys:
                        logger.warning(f"Unexpected keys in checkpoint: {len(unexpected_keys)} keys")
                        
                except Exception as e:
                    logger.error(f"Error loading checkpoint: {str(e)}")
                    logger.warning("Proceeding with freshly initialized model")
            else:
                logger.warning(f"No compatible checkpoint found at {model_path}. Initializing fresh model.")
            
            # Move model to device and set to eval mode
            self.model = self.model.to(device)
            self.model.eval()
            
            # Initialize conversation history
            self.conversation_history = []
            
        except Exception as e:
            logger.error(f"Failed to initialize model: {str(e)}")
            raise
    
    def format_reasoning_analysis(self, reasoning_stats: Dict) -> str:
        """Format reasoning statistics for display"""
        formatted = []
        
        # Format gate values and impacts
        for step, (gate, impact) in enumerate(zip(
            reasoning_stats['gate_values'],
            reasoning_stats['step_impacts']
        )):
            step_info = f"Step {step}:\n"
            step_info += f"├── Contribution: {gate:.3f}\n"
            step_info += f"└── Impact: {impact:.3f}\n"
            formatted.append(step_info)
            
        return "\n".join(formatted)
    
    def visualize_expert_weights(self, weights: List[float]) -> str:
        """Create a visual representation of expert weights"""
        max_width = 40
        visualization = []
        
        for i, weight in enumerate(weights):
            bar_length = int(weight * max_width)
            bar = "█" * bar_length + "░" * (max_width - bar_length)
            visualization.append(f"Expert {i}: {bar} {weight:.3f}")
            
        return "\n".join(visualization)
    
    def process_response(self, input_text: str) -> Tuple[str, Dict, List[float]]:
        """Process user input and generate response with reasoning steps"""
        try:
            # Process input based on mode
            try:
                if self.test_mode:
                    logger.info("Using test processor for input...")
                    inputs = self.processor.encode_single(input_text)
                    input_ids = inputs['input_ids']
                    # Validate input dimensions
                    if input_ids.dim() == 1:
                        input_ids = input_ids.unsqueeze(0)
                    logger.info(f"Input shape: {input_ids.shape}, device: {input_ids.device}")
                else:
                    logger.info("Using byte tokenizer for input...")
                    input_bytes = input_text.encode('utf-8')
                    input_ids = torch.tensor([[b for b in input_bytes]], dtype=torch.long)
                    logger.info(f"Input shape: {input_ids.shape}")
                
                # Ensure input is on correct device
                input_ids = input_ids.to(self.device)
                
            except Exception as e:
                logger.error(f"Error during input processing: {str(e)}")
                raise
            
            # Generate response with reasoning
            with torch.no_grad():
                try:
                    logger.info("Enabling test time mode...")
                    self.model.enable_test_time()
                    
                    # First pass to get reasoning
                    logger.info("Running first pass for reasoning...")
                    logger.info(f"Model device: {next(self.model.parameters()).device}")
                    
                    try:
                        # Try CUDA first
                        first_pass = self.model(
                            input_ids,
                            input_type='text' if self.test_mode else 'bytes',
                            output_type='text'
                        )
                    except RuntimeError as e:
                        if "CUDA" in str(e):
                            logger.warning("CUDA error encountered, falling back to CPU")
                            # Move model and inputs to CPU
                            self.model = self.model.cpu()
                            input_ids = input_ids.cpu()
                            self.device = "cpu"
                            first_pass = self.model(
                                input_ids,
                                input_type='text' if self.test_mode else 'bytes',
                                output_type='text'
                            )
                        else:
                            raise
                    
                    logger.info("First pass completed successfully")
                    
                    # Get reasoning statistics with safety checks
                    logger.info("Getting compute stats...")
                    stats = self.model.get_compute_stats()
                    logger.info(f"Compute stats: {stats}")
                    
                    reasoning_stats = {
                        'gate_values': [],
                        'step_impacts': []
                    }
                    
                    if 'reasoning_steps' in stats:
                        logger.info(f"Found {len(stats['reasoning_steps'])} reasoning steps")
                        if len(stats['reasoning_steps']) > 0:
                            last_step = stats['reasoning_steps'][0]
                            logger.info(f"Last step content: {last_step}")
                            if isinstance(last_step, dict):
                                reasoning_stats['gate_values'] = last_step.get('gate_values', [0.0])
                                reasoning_stats['step_impacts'] = last_step.get('step_impacts', [0.0])
                                logger.info(f"Extracted gate values: {reasoning_stats['gate_values']}")
                                logger.info(f"Extracted step impacts: {reasoning_stats['step_impacts']}")
                    else:
                        logger.warning("No reasoning_steps found in compute stats")
                    
                    # Get expert weights from first pass
                    try:
                        expert_weights = first_pass.get('expert_weights', 
                            torch.ones(self.config['num_experts']).to(self.device) / self.config['num_experts']
                        )
                        logger.info(f"Expert weights shape: {expert_weights.shape}")
                    except Exception as e:
                        logger.error(f"Error getting expert weights: {str(e)}")
                        expert_weights = torch.ones(self.config['num_experts']).to(self.device) / self.config['num_experts']
                    
                    # Generate final output
                    logger.info("Generating final output...")
                    outputs = self.model(
                        input_ids,
                        input_type='text' if self.test_mode else 'bytes',
                        output_type='text',
                        expert_weights=expert_weights
                    )
                    logger.info("Final output generated successfully")
                    
                    # Decode output based on mode
                    try:
                        if 'text' in outputs:
                            if self.test_mode:
                                logger.info("Decoding with test processor...")
                                response = self.processor.decode(outputs['text'][0])
                            else:
                                logger.info("Decoding with byte tokenizer...")
                                logits = outputs['text'][0]
                                output_bytes = logits.argmax(dim=-1)
                                response = bytes([b.item() for b in output_bytes]).decode('utf-8', errors='replace')
                            
                            if not response.strip():
                                response = "[No coherent text generated]"
                        else:
                            logger.warning("No text output in model response")
                            response = "[No text output available]"
                        
                        expert_weights = expert_weights.mean(dim=0).tolist()
                        
                    except Exception as e:
                        logger.error(f"Error during output decoding: {str(e)}")
                        raise
                        
                except Exception as e:
                    logger.error(f"Error during model inference: {str(e)}")
                    raise
                    
                finally:
                    # Always disable test time mode
                    self.model.disable_test_time()
                    
            return response, reasoning_stats, expert_weights
            
        except Exception as e:
            logger.error(f"Error in process_response: {str(e)}\nTraceback:", exc_info=True)
            return "[Error processing response]", {'gate_values': [], 'step_impacts': []}, []
    
    def display_response(
        self,
        input_text: str,
        output_text: str,
        reasoning_stats: Dict,
        expert_weights: List[float],
        show_reasoning: bool = True
    ):
        """Display formatted response with reasoning"""
        # Show input
        console.print("\n[bold blue]You:[/bold blue]")
        console.print(Panel(input_text))
        
        # Show model response
        console.print("\n[bold green]Assistant:[/bold green]")
        console.print(Panel(Markdown(output_text)))
        
        if show_reasoning:
            # Show reasoning process
            logger.info("Displaying reasoning process...")
            logger.info(f"Reasoning stats: {reasoning_stats}")
            
            if not reasoning_stats['gate_values'] and not reasoning_stats['step_impacts']:
                logger.warning("No reasoning steps found to display")
                console.print("\n[bold yellow]No reasoning steps available[/bold yellow]")
            else:
                console.print("\n[bold yellow]Reasoning Process:[/bold yellow]")
                console.print(Panel(self.format_reasoning_analysis(reasoning_stats)))
            
            # Show expert weights
            logger.info(f"Expert weights: {expert_weights}")
            console.print("\n[bold magenta]Expert Contributions:[/bold magenta]")
            console.print(Panel(self.visualize_expert_weights(expert_weights)))
    
    def save_conversation(self, filepath: str):
        """Save conversation history to file"""
        with open(filepath, 'w') as f:
            json.dump(self.conversation_history, f, indent=2)
    
    def chat(self):
        """Interactive chat loop"""
        print("Welcome to the AI Assistant!")
        print("Type 'exit' to end the conversation, 'save' to save the chat history.")
        
        while True:
            # Get user input
            user_input = input("\nYou: ").strip()
            if not user_input:
                continue
            
            if user_input.lower() == 'exit':
                break
            
            if user_input.lower() == 'save':
                filepath = input("Enter filepath to save conversation: ").strip()
                self.save_conversation(filepath)
                print("Conversation saved!")
                continue
            
            # Process response with progress bar
            with Progress() as progress:
                task = progress.add_task("Thinking...", total=100)
                try:
                    output, reasoning_stats, expert_weights = self.process_response(user_input)
                    progress.update(task, completed=100)
                except Exception as e:
                    print(f"\nError generating response: {str(e)}")
                    continue
            
            # Store conversation
            self.conversation_history.append({
                'user': user_input,
                'assistant': output,
                'reasoning_stats': reasoning_stats,
                'expert_weights': expert_weights
            })
            
            # Display response
            self.display_response(
                user_input,
                output,
                reasoning_stats,
                expert_weights
            )

def main():
    parser = argparse.ArgumentParser(description="Chat with the trained model")
    parser.add_argument("--model_path", type=str, required=True, help="Path to trained model weights")
    parser.add_argument("--config_path", type=str, default="config/model_config.json", help="Path to model configuration")
    parser.add_argument("--no_reasoning", action="store_true", help="Disable reasoning visualization")
    parser.add_argument("--test", action="store_true", help="Use test processor for text-only processing")
    args = parser.parse_args()
    
    # Initialize chat interface
    chat_interface = ModelChat(
        model_path=args.model_path,
        config_path=args.config_path,
        test_mode=args.test
    )
    
    # Start chat
    try:
        chat_interface.chat()
    except KeyboardInterrupt:
        console.print("\n[bold red]Chat ended by user[/bold red]")
    
    # Offer to save conversation
    if chat_interface.conversation_history:
        save = console.input("\nSave conversation history? (y/n): ").strip().lower()
        if save == 'y':
            filepath = console.input("Enter filepath to save conversation: ").strip()
            chat_interface.save_conversation(filepath)
            console.print("[green]Conversation saved![/green]")

if __name__ == "__main__":
    main() 