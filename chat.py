import torch
import torch.nn.functional as F
from Sushi_model.transformer_squared import TransformerSquared
from Sushi_model.byte_latent_patches import ByteLevelTokenizer
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
            
        # Create ModelConfig from raw config
        from config.model_config import ModelConfig
        self.config = ModelConfig.from_dict(raw_config)
        
        try:
            # Initialize model and tokenizer
            logger.info("Initializing model...")
            from Sushi_model.sushiFull import SushiFull
            self.model = SushiFull(self.config)
            self.device = device
            self.test_mode = test_mode
            
            # The tokenizer is built into SushiHybrid
            self.tokenizer = self.model.tokenizer
            
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
            
            # Initialize conversation history and memory states
            self.conversation_history = []
            self.memory_states = None
            self.expert_weights = None
            self.z_vectors = None
            
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
            # Convert input text to bytes
            input_bytes = input_text.encode('utf-8')
            input_ids = torch.tensor([[b for b in input_bytes]], dtype=torch.long).to(self.device)
            
            # Generate response with all features
            with torch.no_grad():
                try:
                    outputs = self.model(
                        input_ids=input_ids,
                        expert_weights=self.expert_weights,
                        z_vectors=self.z_vectors,
                        use_memory=True
                    )
                    
                    # Update memory states for next interaction
                    self.memory_states = outputs['memory_states']
                    self.expert_weights = outputs['expert_weights']
                    
                    # Convert byte logits to text
                    byte_logits = outputs['byte_logits']
                    output_bytes = byte_logits.argmax(dim=-1)
                    response = bytes([b.item() for b in output_bytes[0]]).decode('utf-8', errors='replace')
                    
                    # Extract reasoning information
                    reasoning_stats = {
                        'gate_values': [],
                        'step_impacts': []
                    }
                    
                    # Process attention maps for reasoning steps
                    for attn_map, diff_map in zip(
                        outputs['attention_maps'],
                        outputs['diff_attention_maps']
                    ):
                        # Calculate impact from attention and differential attention
                        gate_value = attn_map.mean().item()
                        step_impact = diff_map.mean().item()
                        
                        reasoning_stats['gate_values'].append(gate_value)
                        reasoning_stats['step_impacts'].append(step_impact)
                    
                    # Get expert weights for visualization
                    expert_weights = outputs['expert_weights'][0].tolist()
                    
                    return response, reasoning_stats, expert_weights
                    
                except Exception as e:
                    logger.error(f"Error during model inference: {str(e)}")
                    raise
                    
        except Exception as e:
            logger.error(f"Error in process_response: {str(e)}")
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