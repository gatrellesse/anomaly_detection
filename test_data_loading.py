#!/usr/bin/env python
"""Quick test to verify data loading works."""

import sys
from pathlib import Path
from data_utils import load_mvtec_category

def test_data_loading():
    """Test that data loading produces valid batches."""
    print("Testing data loading...")
    
    # Load bottle category with limited images  
    datamodule, test_loader, num_test_images = load_mvtec_category(
        root=Path("/home/rodrigo/ENSTA/Anomalies/anomaly_detection/MVTecAD"),
        category="bottle",
        train_batch_size=4,
        eval_batch_size=4,
        limit_test_images=10,
    )
    
    print(f"✓ Loaded {num_test_images} test images")
    print(f"✓ Test loader created with {len(test_loader)} batches")
    
    # Test train dataloader
    print("\nTesting train_dataloader()...")
    train_dl = datamodule.train_dataloader()
    train_batches = 0
    for batch in train_dl:
        train_batches += 1
        print(f"  Batch {train_batches}: image shape={batch['image'].shape}, has anomaly_map={'anomaly_map' in batch}")
        assert batch is not None, "Batch is None!"
        assert 'image' in batch, "Batch missing 'image' field"
        assert 'anomaly_map' in batch, "Batch missing 'anomaly_map' field"
        if train_batches >= 2:
            break
    print(f"✓ Train dataloader works ({train_batches} batches tested)")
    
    # Test test dataloader
    print("\nTesting test_dataloader (via engine)...")
    test_batches = 0
    for batch in test_loader:
        test_batches += 1
        print(f"  Batch {test_batches}: image shape={batch['image'].shape}, anomaly_map={batch['anomaly_map']}")
        assert batch is not None, "Batch is None!"
        assert 'image' in batch, "Batch missing 'image' field"
        assert 'anomaly_map' in batch, "Batch missing 'anomaly_map' field"
        if test_batches >= 3:
            break
    print(f"✓ Test dataloader works ({test_batches} batches tested)")
    
    print("\n✓ All data loading tests passed!")
    return True

if __name__ == "__main__":
    try:
        test_data_loading()
        sys.exit(0)
    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
