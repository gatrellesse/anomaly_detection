#!/usr/bin/env python3
"""Quick test to verify the model can train and test."""

import os
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

from anomalib.engine import Engine
from config import MVTEC_PATH, BATCH_SIZE_TRAIN, BATCH_SIZE_EVAL
from data_utils import load_mvtec_category
from models import get_model

print("=" * 80)
print("QUICK MODEL TEST")
print("=" * 80)

try:
    # Load data
    print("\n1. Loading data...")
    datamodule, test_loader, num_test = load_mvtec_category(
        root=MVTEC_PATH,
        category="bottle",
        train_batch_size=BATCH_SIZE_TRAIN,
        eval_batch_size=BATCH_SIZE_EVAL,
        limit_test_images=5
    )
    print(f"   ✓ Loaded {num_test} test images")
    
    # Get model
    print("\n2. Initializing model...")
    model = get_model("patchcore")
    print(f"   ✓ Model initialized: {type(model).__name__}")
    
    # Create engine
    print("\n3. Creating engine...")
    engine = Engine(default_root_dir="./results", accelerator="auto")
    print(f"   ✓ Engine created")
    
    # Train (briefly)
    print("\n4. Training model (this may take a minute)...")
    try:
        engine.fit(model, datamodule)
        print(f"   ✓ Training completed")
    except Exception as e:
        print(f"   ✗ Training failed: {e}")
        raise
    
    # Test
    print("\n5. Testing model...")
    try:
        raw_metrics = engine.test(model, datamodule=datamodule)
        print(f"   ✓ Testing completed")
        if raw_metrics:
            print(f"   Raw metrics type: {type(raw_metrics)}")
            if isinstance(raw_metrics, list) and len(raw_metrics) > 0:
                print(f"   Raw metrics[0] type: {type(raw_metrics[0])}")
                if isinstance(raw_metrics[0], dict):
                    print(f"   Raw metrics[0] keys: {list(raw_metrics[0].keys())}")
    except Exception as e:
        print(f"   ✗ Testing failed: {e}")
        import traceback
        traceback.print_exc()
        raise
    
    print("\n" + "=" * 80)
    print("✓ All tests passed!")
    print("=" * 80)
    
except Exception as e:
    print(f"\n✗ Test failed with error: {e}")
    print("=" * 80)
