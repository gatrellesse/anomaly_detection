"""Results storage and CSV export."""

import csv
from dataclasses import dataclass, asdict
from typing import List, Optional, Dict, Any
from metrics_utils import format_time, format_memory


@dataclass
class BenchmarkResult:
    """Container for a single benchmark result."""
    category: str
    model: str
    image_AUROC: Optional[float] = None
    pixel_AUROC: Optional[float] = None
    F1_Score: Optional[float] = None
    train_time_sec: Optional[float] = None
    inference_time_sec: Optional[float] = None
    inference_fps: Optional[float] = None
    peak_gpu_memory_mb: Optional[float] = None
    peak_cpu_memory_mb: Optional[float] = None
    num_test_images: Optional[int] = None


class ResultsCollector:
    """Collect and export benchmark results."""
    
    def __init__(self):
        self.results: List[BenchmarkResult] = []
    
    def add_result(self, result: BenchmarkResult):
        """Add a benchmark result."""
        self.results.append(result)
    
    def compute_averages(self) -> BenchmarkResult:
        """Compute average metrics across all results."""
        def safe_average(key: str) -> Optional[float]:
            values = [
                getattr(r, key) 
                for r in self.results 
                if getattr(r, key) is not None and r.category != "AVERAGE"
            ]
            return sum(values) / len(values) if values else None
        
        return BenchmarkResult(
            category="AVERAGE",
            model="ALL",
            image_AUROC=safe_average("image_AUROC"),
            pixel_AUROC=safe_average("pixel_AUROC"),
            F1_Score=safe_average("F1_Score"),
            train_time_sec=safe_average("train_time_sec"),
            inference_time_sec=safe_average("inference_time_sec"),
            inference_fps=safe_average("inference_fps"),
            peak_gpu_memory_mb=safe_average("peak_gpu_memory_mb"),
            peak_cpu_memory_mb=safe_average("peak_cpu_memory_mb"),
            num_test_images=None,  # Don't average this
        )
    
    def print_summary(self):
        """Print a summary of all results."""
        print("\n" + "=" * 80)
        print("BENCHMARK SUMMARY")
        print("=" * 80)
        
        for result in self.results:
            if result.category == "AVERAGE":
                continue
            
            print(f"\n{result.category} - {result.model}:")
            print(f"  Image AUROC: {result.image_AUROC:.4f}" if result.image_AUROC else "  Image AUROC: N/A")
            print(f"  Pixel AUROC: {result.pixel_AUROC:.4f}" if result.pixel_AUROC else "  Pixel AUROC: N/A")
            print(f"  F1 Score: {result.F1_Score:.4f}" if result.F1_Score else "  F1 Score: N/A")
            print(f"  Train Time: {format_time(result.train_time_sec)}" if result.train_time_sec else "  Train Time: N/A")
            print(f"  Inference Time: {format_time(result.inference_time_sec)}" if result.inference_time_sec else "  Inference Time: N/A")
            print(f"  Inference FPS: {result.inference_fps:.2f}" if result.inference_fps else "  Inference FPS: N/A")
            print(f"  Peak GPU Memory: {format_memory(result.peak_gpu_memory_mb)}")
            print(f"  Peak CPU Memory: {format_memory(result.peak_cpu_memory_mb)}")
        
        # Print averages
        avg = self.compute_averages()
        print("\n" + "=" * 80)
        print("GLOBAL AVERAGES")
        print("=" * 80)
        print(f"  Average Image AUROC: {avg.image_AUROC:.4f}" if avg.image_AUROC else "  Average Image AUROC: N/A")
        print(f"  Average Pixel AUROC: {avg.pixel_AUROC:.4f}" if avg.pixel_AUROC else "  Average Pixel AUROC: N/A")
        print(f"  Average F1 Score: {avg.F1_Score:.4f}" if avg.F1_Score else "  Average F1 Score: N/A")
        print(f"  Average Train Time: {format_time(avg.train_time_sec)}" if avg.train_time_sec else "  Average Train Time: N/A")
        print(f"  Average Inference Time: {format_time(avg.inference_time_sec)}" if avg.inference_time_sec else "  Average Inference Time: N/A")
        print(f"  Average Inference FPS: {avg.inference_fps:.2f}" if avg.inference_fps else "  Average Inference FPS: N/A")
        print(f"  Average Peak GPU Memory: {format_memory(avg.peak_gpu_memory_mb)}")
        print(f"  Average Peak CPU Memory: {format_memory(avg.peak_cpu_memory_mb)}")
    
    def save_csv(self, filepath: str):
        """Save results to CSV file."""
        # Add averages to results for CSV
        all_results = self.results + [self.compute_averages()]
        
        fieldnames = [
            "category", "model", "image_AUROC", "pixel_AUROC", "F1_Score",
            "train_time_sec", "inference_time_sec", "inference_fps",
            "peak_gpu_memory_mb", "peak_cpu_memory_mb", "num_test_images"
        ]
        
        with open(filepath, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for result in all_results:
                writer.writerow(asdict(result))
        
        print(f"\nResults saved to: {filepath}")
