"""
Anomaly Detection Testbench

This script benchmarks anomaly detection models on the MVTec AD dataset,
tracking performance metrics including:
- Image AUROC, Pixel AUROC, F1 Score
- Training time
- Inference time and FPS
- GPU and CPU memory utilization

Usage:
    python testbench.py                          # Run all categories and models
    python testbench.py --category bottle        # Run only 'bottle' category
    python testbench.py --model patchcore        # Run only 'patchcore' model
    python testbench.py --category bottle --model padim  # Run specific combination
    python testbench.py --list                   # List available categories and models
"""

import os
import sys
import gc
import shutil
import traceback
import argparse
import torch
from anomalib.engine import Engine

from config import (
    MVTEC_PATH, CATEGORIES, MODEL_NAMES,
    LIMIT_TEST_IMAGES, BATCH_SIZE_TRAIN, BATCH_SIZE_EVAL, CSV_OUTPUT
)
from models import get_model
from data_utils import load_mvtec_category
from metrics_utils import (
    PerformanceTracker, extract_model_metrics,
    format_time, format_memory
)
from results import BenchmarkResult, ResultsCollector


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Anomaly Detection Testbench for MVTec AD dataset",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python testbench.py                              # Run all
    python testbench.py --category bottle            # Single category
    python testbench.py --model patchcore            # Single model
    python testbench.py -c bottle -m padim           # Specific combination
    python testbench.py --category bottle capsule    # Multiple categories
    python testbench.py --list                       # Show available options
        """
    )
    
    parser.add_argument(
        "-c", "--category",
        nargs="+",
        choices=CATEGORIES,
        help=f"Category(ies) to run. Available: {', '.join(CATEGORIES)}"
    )
    
    parser.add_argument(
        "-m", "--model",
        nargs="+",
        choices=MODEL_NAMES,
        help=f"Model(s) to run. Available: {', '.join(MODEL_NAMES)}"
    )
    
    parser.add_argument(
        "-n", "--num-images",
        type=int,
        default=LIMIT_TEST_IMAGES,
        help=f"Number of test images per category (default: {LIMIT_TEST_IMAGES or 'ALL'})"
    )
    
    parser.add_argument(
        "-o", "--output",
        type=str,
        default=CSV_OUTPUT,
        help=f"Output CSV file path (default: {CSV_OUTPUT})"
    )
    
    parser.add_argument(
        "--list",
        action="store_true",
        help="List available categories and models"
    )
    
    parser.add_argument(
        "--append",
        action="store_true",
        help="Append results to existing CSV instead of overwriting"
    )
    
    return parser.parse_args()


def list_options():
    """Print available categories and models."""
    print("\nAvailable Categories:")
    for cat in CATEGORIES:
        print(f"  - {cat}")
    
    print("\nAvailable Models:")
    for model in MODEL_NAMES:
        print(f"  - {model}")
    
    print(f"\nDefault test images: {LIMIT_TEST_IMAGES or 'ALL'}")
    print(f"Default output file: {CSV_OUTPUT}")


def patch_windows_symlink():
    """Patch os.symlink on Windows to copy directory instead of creating symlink."""
    if sys.platform == "win32":
        original_symlink = os.symlink
        
        def symlink_or_copy(src, dst, target_is_directory=False):
            try:
                original_symlink(src, dst, target_is_directory=target_is_directory)
            except OSError:
                # Symlink failed (no privileges), use copy instead
                if target_is_directory:
                    if os.path.exists(dst):
                        shutil.rmtree(dst)
                    shutil.copytree(src, dst)
                else:
                    shutil.copy2(src, dst)
        
        os.symlink = symlink_or_copy
        print("Windows detected: patched symlink to use copy fallback")


def cleanup_gpu():
    """Clean up GPU memory between runs to prevent CUDA errors."""
    gc.collect()
    if torch.cuda.is_available():
        try:
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
        except Exception as e:
            print(f"  Warning: GPU cleanup failed ({e})")


def run_single_benchmark(
    model_name: str,
    datamodule,
    test_loader,
    num_test_images: int,
    tracker: PerformanceTracker
) -> BenchmarkResult:
    """Run a single model benchmark.
    
    Args:
        model_name: Name of the model to benchmark
        datamodule: The data module for training
        test_loader: DataLoader for testing
        num_test_images: Number of test images
        tracker: Performance tracker instance
        
    Returns:
        BenchmarkResult containing all metrics
    """
    # Clean up GPU memory before starting
    cleanup_gpu()
    
    tracker.reset()
    
    # Create fresh engine and model for each run
    engine = Engine(default_root_dir="./results")
    model = get_model(model_name)
    
    # Train model with timing
    print(f"  Training {model_name}...")
    tracker.start_training()
    engine.fit(model, datamodule)
    tracker.end_training()
    
    print(f"  Training completed in {format_time(tracker.get_train_time())}")
    
    # Evaluate on test set with timing
    print(f"  Testing {model_name} on {num_test_images} images...")
    tracker.start_inference()
    raw_metrics = engine.test(model, test_loader)
    tracker.end_inference(num_test_images)
    
    # Extract model metrics
    model_metrics = extract_model_metrics(raw_metrics)
    
    # Get performance metrics
    perf_metrics = tracker.get_metrics()
    
    # Print results
    print(f"\n  {model_name} Results:")
    print(f"    Image AUROC: {model_metrics['image_auroc']:.4f}" if model_metrics['image_auroc'] else "    Image AUROC: N/A")
    print(f"    Pixel AUROC: {model_metrics['pixel_auroc']:.4f}" if model_metrics['pixel_auroc'] else "    Pixel AUROC: N/A")
    print(f"    F1 Score: {model_metrics['f1_score']:.4f}" if model_metrics['f1_score'] else "    F1 Score: N/A")
    print(f"    Training Time: {format_time(perf_metrics.train_time_seconds)}")
    print(f"    Inference Time: {format_time(perf_metrics.inference_time_seconds)}")
    print(f"    Inference FPS: {perf_metrics.inference_fps:.2f}")
    print(f"    Peak GPU Memory: {format_memory(perf_metrics.peak_gpu_memory_mb)}")
    print(f"    Peak CPU Memory: {format_memory(perf_metrics.peak_cpu_memory_mb)}")
    
    # Build result before cleanup
    result = BenchmarkResult(
        category="",  # Will be set by caller
        model=model_name,
        image_AUROC=model_metrics['image_auroc'],
        pixel_AUROC=model_metrics['pixel_auroc'],
        F1_Score=model_metrics['f1_score'],
        train_time_sec=perf_metrics.train_time_seconds,
        inference_time_sec=perf_metrics.inference_time_seconds,
        inference_fps=perf_metrics.inference_fps,
        peak_gpu_memory_mb=perf_metrics.peak_gpu_memory_mb,
        peak_cpu_memory_mb=perf_metrics.peak_cpu_memory_mb,
        num_test_images=num_test_images,
    )
    
    # Clean up model and engine to free GPU memory
    del model
    del engine
    cleanup_gpu()
    
    return result


def run_testbench(
    categories: list,
    models: list,
    num_images: int,
    output_file: str,
    append: bool = False
):
    """Run the testbench with specified categories and models.
    
    Args:
        categories: List of categories to test
        models: List of models to test
        num_images: Number of test images (None for all)
        output_file: Path to output CSV file
        append: Whether to append to existing CSV
    """
    results_collector = ResultsCollector()
    tracker = PerformanceTracker()
    
    print("=" * 80)
    print("ANOMALY DETECTION TESTBENCH")
    print("=" * 80)
    print(f"Categories: {', '.join(categories)}")
    print(f"Models: {', '.join(models)}")
    print(f"Test images per category: {num_images or 'ALL'}")
    print(f"Output file: {output_file}")
    print(f"Mode: {'Append' if append else 'Overwrite'}")
    print("=" * 80)
    
    for category in categories:
        print(f"\n{'=' * 40}")
        print(f"CATEGORY: {category}")
        print(f"{'=' * 40}")
        
        try:
            # Load dataset
            datamodule, test_loader, actual_num_images = load_mvtec_category(
                root=MVTEC_PATH,
                category=category,
                train_batch_size=BATCH_SIZE_TRAIN,
                eval_batch_size=BATCH_SIZE_EVAL,
                limit_test_images=num_images
            )
            print(f"Loaded {actual_num_images} test images")
            
        except Exception as e:
            print(f"Error loading dataset for {category}: {e}")
            traceback.print_exc()
            # Add error results for all models
            for model_name in models:
                results_collector.add_result(BenchmarkResult(
                    category=category,
                    model=model_name,
                ))
            continue
        
        for model_name in models:
            print(f"\n--- Model: {model_name} ---")
            
            try:
                result = run_single_benchmark(
                    model_name=model_name,
                    datamodule=datamodule,
                    test_loader=test_loader,
                    num_test_images=actual_num_images,
                    tracker=tracker
                )
                result.category = category
                results_collector.add_result(result)
                
            except Exception as e:
                print(f"Error running {model_name} on {category}: {e}")
                traceback.print_exc()
                results_collector.add_result(BenchmarkResult(
                    category=category,
                    model=model_name,
                ))
    
    # Print summary and save results
    results_collector.print_summary()
    results_collector.save_csv(output_file, append=append)


def main():
    """Main entry point."""
    args = parse_args()
    
    # Handle --list flag
    if args.list:
        list_options()
        return
    
    # Apply Windows symlink patch
    patch_windows_symlink()
    
    # Determine categories and models to run
    categories = args.category if args.category else CATEGORIES
    models = args.model if args.model else MODEL_NAMES
    
    # Run testbench
    run_testbench(
        categories=categories,
        models=models,
        num_images=args.num_images,
        output_file=args.output,
        append=args.append
    )


if __name__ == "__main__":
    main()