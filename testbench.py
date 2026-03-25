"""
Anomaly Detection Testbench

This script benchmarks anomaly detection models on the MVTec AD dataset,
tracking performance metrics including:
- Image AUROC, Pixel AUROC, F1 Score
- Training time
- Inference time and FPS
- GPU and CPU memory utilization

- Default mode (no flag): trains/tests each category separately (original behavior).
- Train-together mode (--train-together): trains each model ONCE on concatenated TRAIN set
  of selected categories, then tests per category.

Output behavior:
- If --train-together is set AND user did not explicitly set --output,
  output file defaults to: mvtec_results_train_together.csv
- Otherwise uses --output as provided

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
    LIMIT_TEST_IMAGES, BATCH_SIZE_TRAIN, BATCH_SIZE_EVAL, CSV_OUTPUT,
    MODEL_BATCH_SIZES, MODEL_EPOCHS
)
from utils.models import get_model
from utils.data_utils import load_mvtec_category
from utils.data_utils_multi import MultiCategoryMVTecAD
from utils.metrics_utils import (
    PerformanceTracker, extract_model_metrics,
    format_time, format_memory
)
from results.results import BenchmarkResult, ResultsCollector

torch.backends.cuda.matmul.allow_tf32 = True
torch.set_float32_matmul_precision("high")


def parse_args():
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

    parser.add_argument(
        "--cpu",
        action="store_true",
        help="Force CPU-only mode (useful for CUDA compatibility issues)"
    )

    parser.add_argument(
        "--train-together",
        action="store_true",
        help="Train each model once on concatenated TRAIN set of selected categories, then test per category."
    )

    parser.add_argument(
        "--limit-train-per-cat",
        type=int,
        default=None,
        help="(Optional, train-together only) Limit TRAIN images per category to speed up debugging."
    )

    return parser.parse_args()


def list_options():
    print("\nAvailable Categories:")
    for cat in CATEGORIES:
        print(f"  - {cat}")

    print("\nAvailable Models:")
    for model in MODEL_NAMES:
        print(f"  - {model}")

    print(f"\nDefault test images: {LIMIT_TEST_IMAGES or 'ALL'}")
    print(f"Default output file: {CSV_OUTPUT}")


def patch_windows_symlink():
    if sys.platform == "win32":
        original_symlink = os.symlink

        def symlink_or_copy(src, dst, target_is_directory=False):
            try:
                original_symlink(src, dst, target_is_directory=target_is_directory)
            except OSError:
                if target_is_directory:
                    if os.path.exists(dst):
                        shutil.rmtree(dst)
                    shutil.copytree(src, dst)
                else:
                    shutil.copy2(src, dst)

        os.symlink = symlink_or_copy
        print("Windows detected: patched symlink to use copy fallback")


def cleanup_gpu():
    gc.collect()
    if torch.cuda.is_available():
        try:
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
        except Exception as e:
            print(f"  Warning: GPU cleanup failed ({e})")


def build_engine(model_name: str, accelerator: str) -> Engine:
    if model_name.lower() in ["vlmad", "winclip"]:
        return Engine(default_root_dir="./results", accelerator=accelerator, max_epochs=2, limit_val_batches=0)
    if model_name.lower() in ["draem", "efficientad"]:
        epochs = MODEL_EPOCHS.get(model_name, 1)
        return Engine(default_root_dir="./results", accelerator=accelerator, max_epochs=epochs)
    return Engine(default_root_dir="./results", accelerator=accelerator, max_epochs=2)


def run_single_benchmark(
    model_name: str,
    datamodule,
    test_loader,
    num_test_images: int,
    tracker: PerformanceTracker,
    accelerator: str = "auto"
) -> BenchmarkResult:
    cleanup_gpu()
    tracker.reset()

    engine = build_engine(model_name, accelerator)
    model = get_model(model_name)

    print(f"  Training {model_name}...")
    tracker.start_training()
    engine.fit(model, datamodule)
    tracker.end_training()
    print(f"  Training completed in {format_time(tracker.get_train_time())}")

    print(f"  Testing {model_name} on {num_test_images} images...")
    tracker.start_inference()
    raw_metrics = engine.test(model, test_loader)
    tracker.end_inference(num_test_images)

    model_metrics = extract_model_metrics(raw_metrics)
    perf_metrics = tracker.get_metrics()

    print(f"\n  {model_name} Results:")
    print(f"    Image AUROC: {model_metrics['image_auroc']:.4f}" if model_metrics['image_auroc'] else "    Image AUROC: N/A")
    print(f"    Pixel AUROC: {model_metrics['pixel_auroc']:.4f}" if model_metrics['pixel_auroc'] else "    Pixel AUROC: N/A")
    print(f"    F1 Score: {model_metrics['f1_score']:.4f}" if model_metrics['f1_score'] else "    F1 Score: N/A")
    print(f"    Training Time: {format_time(perf_metrics.train_time_seconds)}")
    print(f"    Inference Time: {format_time(perf_metrics.inference_time_seconds)}")
    print(f"    Inference FPS: {perf_metrics.inference_fps:.2f}")
    print(f"    Peak GPU Memory: {format_memory(perf_metrics.peak_gpu_memory_mb)}")
    print(f"    Peak CPU Memory: {format_memory(perf_metrics.peak_cpu_memory_mb)}")

    result = BenchmarkResult(
        category="",
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

    del model
    del engine
    cleanup_gpu()
    return result


def run_testbench_separate(
    categories: list,
    models: list,
    num_images: int,
    output_file: str,
    append: bool,
    accelerator: str
):
    results_collector = ResultsCollector()
    tracker = PerformanceTracker()

    print("=" * 80)
    print("ANOMALY DETECTION TESTBENCH (SEPARATE TRAINING)")
    print("=" * 80)
    print(f"Categories: {', '.join(categories)}")
    print(f"Models: {', '.join(models)}")
    print(f"Test images per category: {num_images or 'ALL'}")
    print(f"Output file: {output_file}")
    print(f"Mode: {'Append' if append else 'Overwrite'}")
    print(f"Accelerator: {accelerator}")
    print("=" * 80)

    for category in categories:
        print(f"\n{'=' * 40}")
        print(f"CATEGORY: {category}")
        print(f"{'=' * 40}")

        for model_name in models:
            print(f"\n--- Model: {model_name} ---")

            try:
                model_config = MODEL_BATCH_SIZES.get(model_name.lower(), {})
                train_batch = model_config.get("train", BATCH_SIZE_TRAIN)
                eval_batch = model_config.get("eval", BATCH_SIZE_EVAL)
                if model_config:
                    print(f"  Using model-specific batch sizes: train={train_batch}, eval={eval_batch}")

                datamodule, test_loader, actual_num_images = load_mvtec_category(
                    root=MVTEC_PATH,
                    category=category,
                    train_batch_size=train_batch,
                    eval_batch_size=eval_batch,
                    limit_test_images=num_images
                )
                print(f"  Loaded {actual_num_images} test images")

                result = run_single_benchmark(
                    model_name=model_name,
                    datamodule=datamodule,
                    test_loader=test_loader,
                    num_test_images=actual_num_images,
                    tracker=tracker,
                    accelerator=accelerator
                )
                result.category = category
                results_collector.add_result(result)

                del datamodule
                del test_loader
                cleanup_gpu()

            except Exception as e:
                print(f"Error running {model_name} on {category}: {e}")
                traceback.print_exc()
                results_collector.add_result(BenchmarkResult(category=category, model=model_name))

    results_collector.print_summary()
    results_collector.save_csv(output_file, append=append)


def run_testbench_train_together(
    categories: list,
    models: list,
    num_images: int,
    output_file: str,
    append: bool,
    accelerator: str,
    limit_train_per_cat: int | None,
):
    results_collector = ResultsCollector()
    tracker = PerformanceTracker()

    print("=" * 80)
    print("ANOMALY DETECTION TESTBENCH (TRAIN TOGETHER)")
    print("=" * 80)
    print(f"Train-together categories: {', '.join(categories)}")
    print(f"Models: {', '.join(models)}")
    print(f"Test images per category: {num_images or 'ALL'}")
    print(f"Limit train images per category: {limit_train_per_cat or 'NONE'}")
    print(f"Output file: {output_file}")
    print(f"Mode: {'Append' if append else 'Overwrite'}")
    print(f"Accelerator: {accelerator}")
    print("=" * 80)

    for model_name in models:
        print(f"\n{'=' * 40}")
        print(f"MODEL (train once): {model_name}")
        print(f"{'=' * 40}")

        try:
            model_config = MODEL_BATCH_SIZES.get(model_name.lower(), {})
            train_batch = model_config.get("train", BATCH_SIZE_TRAIN)
            eval_batch = model_config.get("eval", BATCH_SIZE_EVAL)
            if model_config:
                print(f"  Using model-specific batch sizes: train={train_batch}, eval={eval_batch}")

            cleanup_gpu()
            tracker.reset()

            engine = build_engine(model_name, accelerator)
            model = get_model(model_name)

            multi_dm = MultiCategoryMVTecAD(
                root=MVTEC_PATH,
                categories=categories,
                train_batch_size=train_batch,
                eval_batch_size=eval_batch,
                limit_train_images_per_category=limit_train_per_cat,
                auto_download=True,
                num_workers=0,
            )
            multi_dm.setup()

            print(f"  Training {model_name} on concatenated TRAIN set...")
            tracker.start_training()
            engine.fit(
                model=model,
                train_dataloaders=multi_dm.train_dataloader(),
                val_dataloaders=multi_dm.val_dataloader(),  # pode ser None
            )
            tracker.end_training()
            train_time_sec = tracker.get_train_time()
            print(f"  Training completed in {format_time(train_time_sec)}")

            for category in categories: 
                print(f"\n--- Test category: {category} ---")
                try:
                    dm_cat, test_loader, actual_num_images = load_mvtec_category(
                        root=MVTEC_PATH,
                        category=category,
                        train_batch_size=train_batch,
                        eval_batch_size=eval_batch,
                        limit_test_images=num_images
                    )

                    tracker.reset()
                    tracker.start_inference()
                    raw_metrics = engine.test(model, test_loader)
                    tracker.end_inference(actual_num_images)

                    model_metrics = extract_model_metrics(raw_metrics)
                    perf_metrics = tracker.get_metrics()

                    results_collector.add_result(BenchmarkResult(
                        category=category,
                        model=model_name,
                        image_AUROC=model_metrics["image_auroc"],
                        pixel_AUROC=model_metrics["pixel_auroc"],
                        F1_Score=model_metrics["f1_score"],
                        train_time_sec=train_time_sec,
                        inference_time_sec=perf_metrics.inference_time_seconds,
                        inference_fps=perf_metrics.inference_fps,
                        peak_gpu_memory_mb=perf_metrics.peak_gpu_memory_mb,
                        peak_cpu_memory_mb=perf_metrics.peak_cpu_memory_mb,
                        num_test_images=actual_num_images,
                    ))

                    del dm_cat
                    del test_loader
                    cleanup_gpu()

                except Exception as e:
                    print(f"Error testing {model_name} on {category}: {e}")
                    traceback.print_exc()
                    results_collector.add_result(BenchmarkResult(category=category, model=model_name))

            del model
            del engine
            cleanup_gpu()

        except Exception as e:
            print(f"Error training {model_name} in train-together mode: {e}")
            traceback.print_exc()
            for category in categories:
                results_collector.add_result(BenchmarkResult(category=category, model=model_name))

    results_collector.print_summary()
    results_collector.save_csv(output_file, append=append)


def main():
    args = parse_args()

    if args.list:
        list_options()
        return

    patch_windows_symlink()

    categories = args.category if args.category else CATEGORIES
    models = args.model if args.model else MODEL_NAMES
    accelerator = "cpu" if args.cpu else "auto"

    # if train-together and user didn't override output, use a different default file
    output_file = args.output
    if args.train_together and args.output == CSV_OUTPUT:
        # default from config would overwrite, so we swap it
        output_file = "mvtec_results_train_together.csv"

    if args.train_together:
        run_testbench_train_together(
            categories=categories,
            models=models,
            num_images=args.num_images,
            output_file=output_file,
            append=args.append,
            accelerator=accelerator,
            limit_train_per_cat=args.limit_train_per_cat,
        )
    else:
        run_testbench_separate(
            categories=categories,
            models=models,
            num_images=args.num_images,
            output_file=output_file,
            append=args.append,
            accelerator=accelerator,
        )


if __name__ == "__main__":
    main()