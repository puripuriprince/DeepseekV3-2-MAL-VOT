import os
import argparse
import requests
import json
import gzip
import shutil
from pathlib import Path
from tqdm import tqdm
import logging
from datasets import load_dataset
from PIL import Image
import io
import torch
import torchvision.transforms as transforms
from huggingface_hub import login
import getpass
from dotenv import load_dotenv
import time
import hashlib
from urllib.parse import urlparse
import tempfile
import concurrent.futures
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
from queue import Queue

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ProgressQueue:
    def __init__(self, total):
        self.queue = Queue()
        self.total = total
        self.current = 0
        self.lock = threading.Lock()
        
    def update(self, n=1):
        with self.lock:
            self.current += n
            self.queue.put(self.current)

class ProgressTracker:
    def __init__(self, total: int):
        self.total = total
        self.success = 0
        self.failed = 0
        self.lock = threading.Lock()
        self.download_bar = tqdm(total=total, desc="Downloading", position=0)
        self.failure_bar = tqdm(total=total, desc="Failures", position=1, bar_format='{desc}: {n_fmt}/{total_fmt} [{bar:30}] {percentage:3.1f}%')
        
    def update_success(self):
        with self.lock:
            self.success += 1
            self.download_bar.update(1)
            
    def update_failure(self):
        with self.lock:
            self.failed += 1
            self.failure_bar.update(1)
            
    def close(self):
        self.download_bar.close()
        self.failure_bar.close()

class ParallelImageDownloader:
    def __init__(self, cache_dir: Path, max_workers: int = 8):  # Reduced workers
        self.cache_dir = cache_dir
        self.max_workers = max_workers
        self.session = requests.Session()
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.lock = threading.Lock()
        
        # Add headers to mimic a browser
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
            'Accept': 'image/webp,image/apng,image/*,*/*;q=0.8',
            'Accept-Encoding': 'gzip, deflate, br',
            'Connection': 'keep-alive'
        })
        
        # Add rate limiting
        self.last_request = 0
        self.min_request_interval = 0.1  # 100ms between requests
    
    def download_single_image(self, sample: dict, idx: int, progress: ProgressTracker) -> tuple[int, Image.Image, bool]:
        """Download a single image with rate limiting"""
        image_path = self.cache_dir / f"{idx:05d}.pt"
        metadata_path = self.cache_dir / f"{idx:05d}.json"
        
        try:
            # If both image and metadata exist, skip
            if image_path.exists() and metadata_path.exists():
                progress.update_success()
                return idx, None, True
            
            # Rate limiting
            current_time = time.time()
            with self.lock:
                time_since_last = current_time - self.last_request
                if time_since_last < self.min_request_interval:
                    time.sleep(self.min_request_interval - time_since_last)
                self.last_request = time.time()
            
            # Download image
            response = self.session.get(sample['URL'], stream=True, timeout=10)
            response.raise_for_status()
            
            img = Image.open(io.BytesIO(response.content))
            
            # Save image
            transform = transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(256),
                transforms.ToTensor()
            ])
            image_tensor = transform(img)
            torch.save(image_tensor, image_path)
            
            # Save metadata
            metadata = {
                'caption': sample['TEXT'],
                'aesthetic_score': sample['aesthetic_score'],
                'url': sample['URL'],
                'image_file': f"{idx:05d}.pt"
            }
            with open(metadata_path, 'w', encoding='utf-8') as f:
                json.dump(metadata, f, ensure_ascii=False, indent=2)
            
            progress.update_success()
            return idx, img, True
            
        except Exception as e:
            logger.error(f"Error downloading/saving image {idx} from {sample['URL']}: {str(e)}")
            # Clean up any partially downloaded files
            if image_path.exists():
                image_path.unlink()
            if metadata_path.exists():
                metadata_path.unlink()
            progress.update_failure()
            return idx, None, False
    
    def download_images(self, samples: list[dict], batch_size: int = 100) -> dict[int, tuple[Image.Image, bool]]:
        """Download multiple images in parallel with batching"""
        results = {}
        total_samples = len(samples)
        progress = ProgressTracker(total_samples)
        
        print("\n")  # Add space for progress bars
        
        # Process in batches
        for start_idx in range(0, total_samples, batch_size):
            end_idx = min(start_idx + batch_size, total_samples)
            batch_samples = samples[start_idx:end_idx]
            
            with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                future_to_idx = {
                    executor.submit(
                        self.download_single_image, 
                        sample, 
                        idx + start_idx,
                        progress
                    ): idx + start_idx for idx, sample in enumerate(batch_samples)
                }
                
                for future in as_completed(future_to_idx):
                    idx = future_to_idx[future]
                    try:
                        idx, img, success = future.result()
                        results[idx] = (img, success)
                    except Exception as e:
                        logger.error(f"Error processing future for index {idx}: {str(e)}")
                        results[idx] = (None, False)
            
            # Clear memory after each batch
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
            
        progress.close()
        print("\n")  # Add space after progress bars
        return results

def authenticate_huggingface():
    """Authenticate with Hugging Face Hub"""
    try:
        # Load .env file
        load_dotenv()
        
        # Try to get token from environment variable
        token = os.environ.get('HF_TOKEN')
        if not token:
            logger.info("Please enter your Hugging Face token (from https://huggingface.co/settings/tokens):")
            token = getpass.getpass()
        
        # Login to Hugging Face
        login(token)
        logger.info("Successfully authenticated with Hugging Face!")
    except Exception as e:
        logger.error(f"Authentication failed: {e}")
        raise

def process_batch(batch_samples: list, output_dir: Path, start_idx: int):
    """Process a batch of samples"""
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(256),
        transforms.ToTensor()
    ])
    
    for i, sample in enumerate(batch_samples):
        try:
            # Use global index for file naming
            idx = start_idx + i
            
            # Save metadata with text description
            metadata_path = output_dir / f"{idx:05d}.json"
            if not metadata_path.exists():
                metadata = {
                    'caption': sample['TEXT'],
                    'aesthetic_score': sample['aesthetic_score'],
                    'url': sample['URL'],
                    'image_file': f"{idx:05d}.pt"
                }
                with open(metadata_path, 'w', encoding='utf-8') as f:
                    json.dump(metadata, f, ensure_ascii=False, indent=2)
            
        except Exception as e:
            logger.error(f"Error processing sample {idx}: {str(e)}")
            continue
    
    # Clear memory after batch
    torch.cuda.empty_cache() if torch.cuda.is_available() else None

def process_laion_subset(output_dir: Path, num_train: int = 10000, num_val: int = 1000):
    """Download subset of LAION Aesthetics V2 4.75+ dataset"""
    laion_dir = output_dir / "laion"
    train_dir = laion_dir / "train"
    val_dir = laion_dir / "val"
    
    # Create directories
    for dir_path in [train_dir, val_dir]:
        dir_path.mkdir(parents=True, exist_ok=True)
    
    # Initialize parallel downloaders with fewer workers
    train_downloader = ParallelImageDownloader(
        cache_dir=train_dir,
        max_workers=8
    )
    
    val_downloader = ParallelImageDownloader(
        cache_dir=val_dir,
        max_workers=8
    )
    
    # Load dataset with streaming
    logger.info("Loading LAION Aesthetics V2 4.75+ dataset...")
    dataset = load_dataset(
        "laion/aesthetics_v2_4.75",
        split="train",
        streaming=True,
        cache_dir=str(laion_dir / "hf_cache")
    )
    
    # Process samples in batches
    logger.info("Processing samples...")
    total_samples = num_train + num_val
    samples = []
    batch_size = 1000
    current_idx = 0
    
    with tqdm(total=total_samples, desc="Loading samples") as pbar:
        try:
            for i, sample in enumerate(dataset):
                if i >= total_samples:
                    break
                samples.append(sample)
                pbar.update(1)
                
                # Process batch if ready
                if len(samples) >= batch_size:
                    # Use appropriate downloader based on current index
                    downloader = train_downloader if current_idx < num_train else val_downloader
                    results = downloader.download_images(samples[:batch_size])
                    
                    current_idx += len(samples[:batch_size])
                    samples = samples[batch_size:]  # Keep remaining samples
                
        except KeyboardInterrupt:
            logger.warning("\nDownload interrupted by user. Saving progress...")
            if samples:
                downloader = train_downloader if current_idx < num_train else val_downloader
                downloader.download_images(samples)
        
        logger.info(f"Processed {current_idx} samples")

def process_c4_subset(output_dir: Path, max_size_gb: float = 10):
    """Download and prepare subset of C4 dataset"""
    c4_dir = output_dir / "c4"
    c4_dir.mkdir(parents=True, exist_ok=True)
    
    train_output = c4_dir / "train_small.jsonl"
    val_output = c4_dir / "val_small.jsonl"
    
    # Load C4 dataset
    logger.info("Loading C4 dataset...")
    train_dataset = load_dataset("allenai/c4", "en", split="train", streaming=True)
    val_dataset = load_dataset("allenai/c4", "en", split="validation", streaming=True)
    
    # Process training data
    if not train_output.exists():
        logger.info("Processing C4 training subset...")
        current_size = 0
        max_size = max_size_gb * 1024 * 1024 * 1024  # Convert to bytes
        
        with open(train_output, 'w') as f:
            for sample in tqdm(train_dataset):
                line = json.dumps(sample) + '\n'
                size = len(line.encode('utf-8'))
                if current_size + size > max_size:
                    break
                f.write(line)
                current_size += size
    
    # Process validation data
    if not val_output.exists():
        logger.info("Processing C4 validation subset...")
        current_size = 0
        max_size = (max_size_gb * 1024 * 1024 * 1024) / 10  # 1/10th of training size
        
        with open(val_output, 'w') as f:
            for sample in tqdm(val_dataset):
                line = json.dumps(sample) + '\n'
                size = len(line.encode('utf-8'))
                if current_size + size > max_size:
                    break
                f.write(line)
                current_size += size

def update_config(data_dir: Path):
    """Update model config with correct data paths"""
    config_path = Path("config/model_config.json")
    
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    # Update paths to point to processed files
    config["laion_path"] = str(data_dir / "laion/train")
    config["laion_val_path"] = str(data_dir / "laion/val")
    config["c4_path"] = str(data_dir / "c4/train_small.jsonl")
    config["c4_val_path"] = str(data_dir / "c4/val_small.jsonl")
    
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)

def main():
    parser = argparse.ArgumentParser(description='Download LAION Aesthetics V2 4.75+ and C4 datasets')
    parser.add_argument('--data_dir', type=str, required=True,
                      help='Directory to store the datasets')
    parser.add_argument('--num_train_images', type=int, default=10000,
                      help='Number of training images to download from LAION')
    parser.add_argument('--num_val_images', type=int, default=1000,
                      help='Number of validation images to download from LAION')
    parser.add_argument('--c4_size_gb', type=float, default=10,
                      help='Size of C4 training subset in GB')
    parser.add_argument('--token', type=str,
                      help='Hugging Face token. If not provided, will prompt or use HF_TOKEN environment variable')
    
    args = parser.parse_args()
    data_dir = Path(args.data_dir)
    
    # Create main data directory
    data_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        # Set token if provided
        if args.token:
            os.environ['HF_TOKEN'] = args.token
            
        # Authenticate with Hugging Face
        authenticate_huggingface()
        
        # Download and process datasets
        process_laion_subset(data_dir, args.num_train_images, args.num_val_images)
        process_c4_subset(data_dir, args.c4_size_gb)
        
        # Update config with correct paths
        update_config(data_dir)
        
        logger.info(f"""
Dataset download complete!
- LAION Aesthetics V2 4.75+ dataset: {data_dir}/laion/
  - Training samples: {args.num_train_images}
  - Validation samples: {args.num_val_images}
- C4 dataset: {data_dir}/c4/
  - Training size: {args.c4_size_gb}GB
  - Validation size: {args.c4_size_gb/10}GB
        """)
        
    except Exception as e:
        logger.error(f"Error during download: {e}")
        raise

if __name__ == "__main__":
    main() 