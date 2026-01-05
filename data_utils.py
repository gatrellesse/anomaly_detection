"""Data loading utilities."""

import os
import tarfile
import urllib.request
from pathlib import Path

from anomalib.data import MVTecAD
from torch.utils.data import Subset, DataLoader
from typing import Optional, Tuple, Union

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
        
        # Extract
        print(f"Extracting to {root}...")
        with tarfile.open(tar_path, "r:xz") as tar:
            tar.extractall(path=root.parent)
        
        # The archive extracts to 'mvtec_anomaly_detection' folder
        extracted_path = root.parent / "mvtec_anomaly_detection"
        if extracted_path.exists() and extracted_path != root:
            # Rename to match expected root path
            if root.exists():
                import shutil
                shutil.rmtree(root)
            extracted_path.rename(root)
        
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
    
    # Load dataset
    datamodule = MVTecAD(
        root=str(root),
        category=category,
        train_batch_size=train_batch_size,
        eval_batch_size=eval_batch_size,
    )
    
    # Setup the datamodule
    datamodule.setup()
    
    # Get the test dataloader
    test_loader = datamodule.test_dataloader()
    
    # If we want to limit test images, create a custom loader
    if limit_test_images is not None:
        original_dataset = test_loader.dataset
        num_images = min(limit_test_images, len(original_dataset))
        
        # Create a subset
        limited_indices = range(num_images)
        test_dataset = Subset(original_dataset, limited_indices)
        
        # Create new dataloader with limited dataset
        test_loader = DataLoader(
            test_dataset,
            batch_size=test_loader.batch_size,
            shuffle=False,
            num_workers=0,  # Set to 0 to avoid multiprocessing issues
            collate_fn=test_loader.collate_fn  # Preserve anomalib's custom collate function
        )
        
        return datamodule, test_loader, num_images
    
    return datamodule, test_loader, len(test_loader.dataset)
