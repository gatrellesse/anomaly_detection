#!/usr/bin/env python3
"""Quick test to verify the fix works."""

import torch
from config import MVTEC_PATH, BATCH_SIZE_TRAIN, BATCH_SIZE_EVAL
from data_utils import load_mvtec_category

# Test loading a single category
print("Testing data loading for 'bottle' category...")
try:
    datamodule, test_loader, num_test = load_mvtec_category(
        root=MVTEC_PATH,
        category="bottle",
        train_batch_size=BATCH_SIZE_TRAIN,
        eval_batch_size=BATCH_SIZE_EVAL,
        limit_test_images=5
    )
    
    print(f"✓ Data loaded successfully")
    print(f"  Test images: {num_test}")
    
    # Check if datamodule has test_dataloader
    print(f"  Datamodule has test_dataloader: {hasattr(datamodule, 'test_dataloader')}")
    
    # Try to get a batch
    print("\nTesting batch retrieval...")
    batch = next(iter(test_loader))
    
    print(f"✓ Batch retrieved successfully")
    print(f"  Batch type: {type(batch)}")
    print(f"  Batch keys: {list(batch.keys()) if isinstance(batch, dict) else 'not a dict'}")
    
    if isinstance(batch, dict):
        for key in batch.keys():
            val = batch[key]
            if isinstance(val, torch.Tensor):
                print(f"    {key}: tensor {val.shape}, dtype: {val.dtype}")
            elif isinstance(val, list):
                print(f"    {key}: list of {len(val)} items")
            else:
                print(f"    {key}: {type(val)}")
    
    # Check anomaly_map
    if 'anomaly_map' in batch:
        print(f"\n✓ anomaly_map field exists: {batch['anomaly_map'] is not None}")
        if batch['anomaly_map'] is not None:
            print(f"  anomaly_map shape: {batch['anomaly_map'].shape}")
    
except Exception as e:
    print(f"✗ Error: {e}")
    import traceback
    traceback.print_exc()
