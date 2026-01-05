"""
Anomaly Detection Testbench

This script benchmarks anomaly detection models on the MVTec AD dataset,
tracking performance metrics including:
- Image AUROC, Pixel AUROC, F1 Score
- Training time
- Inference time and FPS
- GPU and CPU memory utilization
"""

import os
import sys
import shutil
import traceback
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
    
    return BenchmarkResult(
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


def run_testbench():
    """Run the complete testbench across all categories and models."""
    results_collector = ResultsCollector()
    tracker = PerformanceTracker()
    
    print("=" * 80)
    print("ANOMALY DETECTION TESTBENCH")
    print("=" * 80)
    print(f"Categories: {', '.join(CATEGORIES)}")
    print(f"Models: {', '.join(MODEL_NAMES)}")
    print(f"Test images per category: {LIMIT_TEST_IMAGES or 'ALL'}")
    print("=" * 80)
    
    for category in CATEGORIES:
        print(f"\n{'=' * 40}")
        print(f"CATEGORY: {category}")
        print(f"{'=' * 40}")
        
        try:
            # Load dataset
            datamodule, test_loader, num_test_images = load_mvtec_category(
                root=MVTEC_PATH,
                category=category,
                train_batch_size=BATCH_SIZE_TRAIN,
                eval_batch_size=BATCH_SIZE_EVAL,
                limit_test_images=LIMIT_TEST_IMAGES
            )
            print(f"Loaded {num_test_images} test images")
            
        except Exception as e:
            print(f"Error loading dataset for {category}: {e}")
            traceback.print_exc()
            # Add error results for all models
            for model_name in MODEL_NAMES:
                results_collector.add_result(BenchmarkResult(
                    category=category,
                    model=model_name,
                ))
            continue
        
        for model_name in MODEL_NAMES:
            print(f"\n--- Model: {model_name} ---")
            
            try:
                result = run_single_benchmark(
                    model_name=model_name,
                    datamodule=datamodule,
                    test_loader=test_loader,
                    num_test_images=num_test_images,
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
    results_collector.save_csv(CSV_OUTPUT)


if __name__ == "__main__":
    patch_windows_symlink()
    run_testbench()