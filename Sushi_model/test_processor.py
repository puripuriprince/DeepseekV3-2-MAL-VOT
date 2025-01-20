import torch
from transformers import AutoTokenizer
from typing import Union, List, Dict

class TestProcessor:
    def __init__(
        self,
        model_name: str = "deepseek-ai/deepseek-coder-33b-instruct",
        max_length: int = 2048,
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.max_length = max_length
        self.device = device
        
        # Special tokens
        self.bos_token = "<|im_start|>"
        self.eos_token = "<|im_end|>"
        self.system_token = "system"
        self.assistant_token = "assistant"
        self.user_token = "user"
        
    def format_message(self, role: str, content: str) -> str:
        """Format a message with role and content."""
        return f"{self.bos_token}{role}\n{content}{self.eos_token}"
    
    def process_conversation(
        self,
        messages: List[Dict[str, str]],
        add_generation_prompt: bool = True
    ) -> Dict[str, torch.Tensor]:
        """Process a conversation into model inputs."""
        # Format conversation
        formatted_text = ""
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            formatted_text += self.format_message(role, content)
        
        # Add generation prompt if needed
        if add_generation_prompt:
            formatted_text += f"{self.bos_token}{self.assistant_token}\n"
        
        # Tokenize
        inputs = self.tokenizer(
            formatted_text,
            max_length=self.max_length,
            truncation=True,
            padding=True,
            return_tensors="pt"
        )
        
        # Move to device
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        return inputs
    
    def decode(self, token_ids: torch.Tensor) -> str:
        """Decode token IDs back to text."""
        return self.tokenizer.decode(token_ids, skip_special_tokens=True)
    
    def encode_single(self, text: str) -> Dict[str, torch.Tensor]:
        """Encode a single text input."""
        inputs = self.tokenizer(
            text,
            max_length=self.max_length,
            truncation=True,
            padding=True,
            return_tensors="pt"
        )
        return {k: v.to(self.device) for k, v in inputs.items()}
    
    def process_image_text_pair(
        self,
        image_tensor: torch.Tensor,
        text: str
    ) -> Dict[str, torch.Tensor]:
        """Process an image-text pair for training."""
        # Encode text
        text_inputs = self.encode_single(text)
        
        # Combine with image
        inputs = {
            "pixel_values": image_tensor.to(self.device),
            **text_inputs
        }
        
        return inputs
    
    @property
    def vocab_size(self) -> int:
        """Get the vocabulary size of the tokenizer."""
        return self.tokenizer.vocab_size
    
    @property
    def pad_token_id(self) -> int:
        """Get the pad token ID."""
        return self.tokenizer.pad_token_id
    
    @property
    def eos_token_id(self) -> int:
        """Get the end of sequence token ID."""
        return self.tokenizer.eos_token_id
    
    @property
    def bos_token_id(self) -> int:
        """Get the beginning of sequence token ID."""
        return self.tokenizer.bos_token_id 