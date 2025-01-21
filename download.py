from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import os
import time
from requests.exceptions import ConnectionError, Timeout
from huggingface_hub.utils import RepositoryNotFoundError, EntryNotFoundError

def download_with_retries(model_id, download_type, max_retries=5, initial_delay=10, save_directory=None):
    """Generic retry handler for model downloads"""
    last_error = None
    for attempt in range(max_retries):
        try:
            if download_type == "tokenizer":
                return AutoTokenizer.from_pretrained(
                    model_id,
                    timeout=120,
                    resume_download=True
                )
            elif download_type == "model":
                return AutoModelForCausalLM.from_pretrained(
                    model_id,
                    torch_dtype=torch.float16,
                    device_map="auto",
                    timeout=512,
                    resume_download=True
                )
        except (ConnectionError, Timeout, RepositoryNotFoundError, EntryNotFoundError) as e:
            last_error = e
            if attempt < max_retries - 1:
                delay = initial_delay * (2 ** attempt)
                print(f"Download failed ({str(e)}). Retrying in {delay} seconds... (Attempt {attempt + 2}/{max_retries})")
                time.sleep(delay)
    
    print(f"Failed to download {download_type} after {max_retries} attempts")
    raise last_error

def download_and_save_model(save_directory="./deepseek_model"):
    # Model identifier
    model_id = "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B"
    
    # Create directory if it doesn't exist
    os.makedirs(save_directory, exist_ok=True)
    
    print(f"Starting to download {model_id}...")

    try:
        # Download and save tokenizer with retries
        print("Downloading tokenizer...")
        tokenizer = download_with_retries(model_id, "tokenizer")
        print(f"Saving tokenizer to {save_directory}...")
        tokenizer.save_pretrained(save_directory)
        
        # Download and save model with retries
        print("Downloading model...")
        model = download_with_retries(model_id, "model")
        print(f"Saving model to {save_directory}...")
        model.save_pretrained(save_directory)
        
        print("Download and save complete!")
        return model, tokenizer
    
    except Exception as e:
        print(f"Critical error during download: {str(e)}")
        # Clean up potentially corrupted files
        if os.path.exists(save_directory):
            print("Cleaning up partial downloads...")
            for root, dirs, files in os.walk(save_directory, topdown=False):
                for name in files:
                    os.remove(os.path.join(root, name))
                for name in dirs:
                    os.rmdir(os.path.join(root, name))
            os.rmdir(save_directory)
        raise

def load_local_model(model_directory="./deepseek_model"):
    """
    Load the model and tokenizer from local directory with validation
    """
    print(f"Loading model and tokenizer from {model_directory}...")
    
    try:
        if not os.path.exists(model_directory):
            raise FileNotFoundError(f"Model directory {model_directory} not found")
            
        tokenizer = AutoTokenizer.from_pretrained(model_directory)
        model = AutoModelForCausalLM.from_pretrained(
            model_directory,
            torch_dtype=torch.float16,
            device_map="auto"
        )
        
        # Simple validation
        test_input = tokenizer("Hello, world!", return_tensors="pt")
        model.generate(**test_input, max_length=5)
        
        print("Local model loaded and validated successfully!")
        return model, tokenizer
    
    except Exception as e:
        print(f"Error loading local model: {str(e)}")
        print("The model files might be corrupted. Try re-downloading.")
        raise

if __name__ == "__main__":
    try:
        # First time: Download and save
        model, tokenizer = download_and_save_model()
        
        # Later: Load from local directory
        # model, tokenizer = load_local_model()
    
    except KeyboardInterrupt:
        print("\nDownload interrupted by user. Cleaning up...")
        exit(1)
    except Exception as e:
        print(f"Fatal error: {str(e)}")
        exit(1)