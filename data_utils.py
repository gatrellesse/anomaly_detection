"""Data loading utilities."""

import os
import tarfile
import urllib.request
from pathlib import Path
from types import MethodType, SimpleNamespace

from anomalib.data import MVTecAD
from torch.utils.data import Subset, DataLoader, Dataset
from torchvision import transforms
from PIL import Image
from typing import Optional, Tuple, Union, List
import torch


class BatchDict(dict):
    """Dict subclass that also supports attribute access."""
    def __getattr__(self, key):
        if key.startswith('_'):
            # Avoid issues with private attributes
            return object.__getattribute__(self, key)
        try:
            return self[key]
        except KeyError:
            raise AttributeError(f"No attribute '{key}'")
    
    def __setattr__(self, key, value):
        if key.startswith('_'):
            object.__setattr__(self, key, value)
        else:
            self[key] = value
    
    def __repr__(self):
        return f"BatchDict({dict.__repr__(self)})"


# MVTec AD dataset URL
MVTEC_URL = "https://www.mydrive.ch/shares/38536/3830184030e49fe74747669442f0f283/download/420938113-1629960298/mvtec_anomaly_detection.tar.xz"


class SimpleImageDataset(Dataset):
    """Simple dataset for loading images from a directory structure."""
    
    def __init__(self, image_files: List[Path], transform=None):
        self.image_files = image_files
        # Use ImageNet normalization as expected by anomalib models
        self.transform = transform or transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            ),
        ])
        # Add collate_fn attribute for anomalib compatibility
        self.collate_fn = self._collate_fn
    
    def _collate_fn(self, batch):
        """Custom collate function that converts dicts to batch objects with attributes."""
        # batch is a list of dicts from __getitem__
        if not batch:
            raise ValueError("Empty batch received in collate function")
            
        stacked_images = torch.stack([item["image"] for item in batch])
        
        # Move to GPU if available
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        stacked_images = stacked_images.to(device)
        
        # Collect image paths for the batch
        image_paths = [item["image_path"] for item in batch]
        batch_size = len(batch)
        img_height, img_width = stacked_images.shape[-2], stacked_images.shape[-1]
        
        # Create a batch dict that supports both dict methods (.update()) 
        # and attribute access (.image, .gt_mask, .image_path, .label, .anomaly_map)
        batch_dict = BatchDict(
            image=stacked_images,
            gt_mask=torch.zeros(batch_size, 1, img_height, img_width, device=device),
            image_path=image_paths,
            label=torch.zeros(batch_size, dtype=torch.long, device=device),  # Normal samples are labeled 0
            mask=torch.zeros(batch_size, 1, img_height, img_width, device=device),
            anomaly_map=None,  # Will be populated by model.test_step()
        )
        
        # Ensure batch_dict is not None and has all required fields
        assert batch_dict is not None, "batch_dict should not be None"
        assert 'image' in batch_dict, "batch_dict missing 'image' field"
        assert 'anomaly_map' in batch_dict, "batch_dict missing 'anomaly_map' field"
        
        return batch_dict
    
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        img_path = self.image_files[idx]
        img = Image.open(img_path).convert("RGB")
        
        if self.transform:
            img = self.transform(img)
        
        return {
            "image": img,
            "image_path": str(img_path),
        }

# MVTec AD dataset URL
MVTEC_URL = "https://www.mydrive.ch/shares/38536/3830184030e49fe74747669442f0f283/download/420938113-1629960298/mvtec_anomaly_detection.tar.xz"


def download_mvtec_ad(root: Union[str, Path], category: Optional[str] = None) -> Path:
    """Download MVTec AD dataset if not present.
    
    Args:
        root: Root directory where dataset should be stored
        category: Optional specific category to check (if None, checks for any category)
        
    Returns:
        Path to the dataset root directory
    """
    root = Path(root)
    
    # Check if dataset already exists
    if root.exists():
        # Check if it has the expected structure (at least one category folder)
        expected_categories = [
            "bottle", "cable", "capsule", "carpet", "grid", "hazelnut",
            "leather", "metal_nut", "pill", "screw", "tile", "toothbrush",
            "transistor", "wood", "zipper"
        ]
        
        if category:
            category_path = root / category
            if category_path.exists() and (category_path / "train").exists():
                print(f"MVTec AD category '{category}' already exists at {root}")
                return root
        else:
            for cat in expected_categories:
                cat_path = root / cat
                if cat_path.exists() and (cat_path / "train").exists():
                    print(f"MVTec AD dataset already exists at {root}")
                    return root
    
    # Create root directory
    root.mkdir(parents=True, exist_ok=True)
    
    # Download path
    tar_path = root.parent / "mvtec_anomaly_detection.tar.xz"
    
    print(f"MVTec AD dataset not found at {root}")
    print(f"Downloading MVTec AD dataset (~4.9GB)...")
    print(f"URL: {MVTEC_URL}")
    
    try:
        # Download with progress
        def report_progress(block_num, block_size, total_size):
            downloaded = block_num * block_size
            percent = min(100, downloaded * 100 / total_size)
            mb_downloaded = downloaded / (1024 * 1024)
            mb_total = total_size / (1024 * 1024)
            print(f"\rDownloading: {percent:.1f}% ({mb_downloaded:.1f}/{mb_total:.1f} MB)", end="", flush=True)
        
        urllib.request.urlretrieve(MVTEC_URL, tar_path, reporthook=report_progress)
        print("\nDownload complete!")

        # Extract directly into the target root directory
        print(f"Extracting to {root}...")
        with tarfile.open(tar_path, "r:xz") as tar:
            tar.extractall(path=root)
        
        # Clean up tar file
        if tar_path.exists():
            tar_path.unlink()

        print(f"MVTec AD dataset ready at {root}")
        return root
        
    except Exception as e:
        print(f"\nError downloading MVTec AD: {e}")
        print("\nYou can manually download the dataset from:")
        print("  https://www.mvtec.com/company/research/datasets/mvtec-ad")
        print(f"  Extract it to: {root}")
        raise


def load_mvtec_category(
    root: Union[str, Path],
    category: str,
    train_batch_size: int = 32,
    eval_batch_size: int = 32,
    limit_test_images: Optional[int] = None,
    auto_download: bool = True
) -> Tuple[MVTecAD, DataLoader, int]:
    """Load MVTec AD dataset for a specific category.
    
    Args:
        root: Path to MVTec AD dataset
        category: Category name
        train_batch_size: Batch size for training
        eval_batch_size: Batch size for evaluation
        limit_test_images: Optional limit on number of test images
        auto_download: Whether to automatically download if not found
        
    Returns:
        Tuple of (datamodule, test_dataloader, num_test_images)
    """
    root = Path(root)
    
    # Check and download if necessary
    if auto_download:
        root = download_mvtec_ad(root, category)
    
    category_path = root / category
    train_path = category_path / "train"
    test_path = category_path / "test"
    
    print(f"\n  Loading {category}...")
    
    # Manually load training images from train/good directory BEFORE creating datamodule
    # (The datamodule's training set is broken in anomalib 2.2.0)
    train_images = []
    if train_path.exists():
        train_good = train_path / "good"
        if train_good.exists():
            for ext in ['*.png', '*.jpg', '*.jpeg', '*.bmp']:
                train_images.extend(sorted(train_good.glob(ext)))
    
    num_train_images = len(train_images)
    print(f"    Found {num_train_images} training images")
    
    # Create MVTecAD datamodule 
    datamodule = MVTecAD(
        root=str(root),
        category=category,
        train_batch_size=train_batch_size,
        eval_batch_size=eval_batch_size,
    )
    
    # Skip setup() entirely since it breaks things - we have all data loaded manually
    # Don't call datamodule.setup()
    
    # Create custom train/val dataloaders
    if num_train_images > 0:
        train_dataset = SimpleImageDataset(train_images)
        datamodule.train_data = train_dataset
        datamodule.train_dataset = train_dataset
        # Empty validation set
        datamodule.val_data = Subset(train_dataset, [])
        datamodule.val_dataset = Subset(train_dataset, [])
    else:
        train_dataset = None
    
    def custom_train_dataloader():
        if train_dataset is not None:
            return DataLoader(
                train_dataset,
                batch_size=train_batch_size,
                shuffle=True,
                num_workers=0,
                collate_fn=train_dataset.collate_fn,
            )
        else:
            return DataLoader([], batch_size=train_batch_size)
    
    def custom_val_dataloader():
        return DataLoader([], batch_size=eval_batch_size)
    
    # Manually load test images from test directory
    test_images = []
    if test_path.exists():
        for subdir in sorted(test_path.iterdir()):
            if subdir.is_dir():
                for ext in ['*.png', '*.jpg', '*.jpeg', '*.bmp']:
                    test_images.extend(sorted(subdir.glob(ext)))
    
    num_test_images = len(test_images)
    print(f"    Found {num_test_images} test images")
    
    if num_test_images == 0:
        print(f"  Warning: No test images found for {category}")
        return datamodule, DataLoader([], batch_size=eval_batch_size), 0
    
    # Apply limit if requested
    if limit_test_images is not None and limit_test_images > 0:
        test_images = test_images[:limit_test_images]
        num_test_images = len(test_images)
        print(f"    Limited to {num_test_images} test images")
    
    # Create simple test dataset
    test_dataset = SimpleImageDataset(test_images)
    test_loader = DataLoader(
        test_dataset,
        batch_size=eval_batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=test_dataset.collate_fn,
    )
    
    # Override test_dataloader - we won't be using this but assign anyway for completeness
    # Engine will use the test_loader we pass explicitly instead
    def custom_test_dataloader():
        return test_loader
    
    # Create proper bound methods that don't require self
    # We'll create simple wrapper methods that look like they belong to the class
    def train_dataloader_wrapper(self):
        return custom_train_dataloader()
    
    def val_dataloader_wrapper(self):
        return custom_val_dataloader()
    
    def test_dataloader_wrapper(self):
        return custom_test_dataloader()
    
    # Bind these as methods of the datamodule instance
    datamodule.train_dataloader = MethodType(train_dataloader_wrapper, datamodule)
    datamodule.val_dataloader = MethodType(val_dataloader_wrapper, datamodule)
    datamodule.test_dataloader = MethodType(test_dataloader_wrapper, datamodule)
    
    return datamodule, test_loader, num_test_images
