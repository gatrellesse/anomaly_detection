"""Utility functions for metrics extraction and performance tracking."""

import time
import torch
import psutil
from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List


@dataclass
class PerformanceMetrics:
    """Container for performance metrics."""
    train_time_seconds: float = 0.0
    inference_time_seconds: float = 0.0
    inference_fps: float = 0.0
    num_test_images: int = 0
    
    # Memory metrics (in MB)
    peak_gpu_memory_mb: Optional[float] = None
    peak_cpu_memory_mb: Optional[float] = None
    
    # Model metrics
    image_auroc: Optional[float] = None
    pixel_auroc: Optional[float] = None
    f1_score: Optional[float] = None


class PerformanceTracker:
    """Track performance metrics during training and inference."""
    
    def __init__(self):
        self.reset()
        self._check_gpu_available()
    
    def _check_gpu_available(self):
        """Check if CUDA is available."""
        self.gpu_available = torch.cuda.is_available()
        if self.gpu_available:
            print(f"GPU detected: {torch.cuda.get_device_name(0)}")
        else:
            print("No GPU detected, tracking CPU memory only")
    
    def reset(self):
        """Reset all tracked metrics."""
        self.train_start_time = None
        self.train_end_time = None
        self.inference_start_time = None
        self.inference_end_time = None
        self.num_images_processed = 0
        self.peak_gpu_memory = 0
        self.peak_cpu_memory = 0
        self._initial_gpu_memory = 0
        self._initial_cpu_memory = 0
    
    def _get_gpu_memory_mb(self) -> float:
        """Get current GPU memory usage in MB."""
        if self.gpu_available:
            return torch.cuda.memory_allocated() / (1024 * 1024)
        return 0.0
    
    def _get_gpu_max_memory_mb(self) -> float:
        """Get peak GPU memory usage in MB."""
        if self.gpu_available:
            return torch.cuda.max_memory_allocated() / (1024 * 1024)
        return 0.0
    
    def _get_cpu_memory_mb(self) -> float:
        """Get current CPU memory usage in MB."""
        process = psutil.Process()
        return process.memory_info().rss / (1024 * 1024)
    
    def start_training(self):
        """Mark the start of training."""
        if self.gpu_available:
            torch.cuda.reset_peak_memory_stats()
            self._initial_gpu_memory = self._get_gpu_memory_mb()
        self._initial_cpu_memory = self._get_cpu_memory_mb()
        self.train_start_time = time.perf_counter()
    
    def end_training(self):
        """Mark the end of training."""
        self.train_end_time = time.perf_counter()
        self._update_peak_memory()
    
    def start_inference(self):
        """Mark the start of inference."""
        if self.gpu_available:
            torch.cuda.reset_peak_memory_stats()
        self.inference_start_time = time.perf_counter()
    
    def end_inference(self, num_images: int):
        """Mark the end of inference."""
        self.inference_end_time = time.perf_counter()
        self.num_images_processed = num_images
        self._update_peak_memory()
    
    def _update_peak_memory(self):
        """Update peak memory tracking."""
        if self.gpu_available:
            current_peak_gpu = self._get_gpu_max_memory_mb()
            self.peak_gpu_memory = max(self.peak_gpu_memory, current_peak_gpu)
        
        current_cpu = self._get_cpu_memory_mb()
        self.peak_cpu_memory = max(self.peak_cpu_memory, current_cpu)
    
    def get_train_time(self) -> float:
        """Get training time in seconds."""
        if self.train_start_time and self.train_end_time:
            return self.train_end_time - self.train_start_time
        return 0.0
    
    def get_inference_time(self) -> float:
        """Get inference time in seconds."""
        if self.inference_start_time and self.inference_end_time:
            return self.inference_end_time - self.inference_start_time
        return 0.0
    
    def get_fps(self) -> float:
        """Calculate frames per second during inference."""
        inference_time = self.get_inference_time()
        if inference_time > 0 and self.num_images_processed > 0:
            return self.num_images_processed / inference_time
        return 0.0
    
    def get_metrics(self) -> PerformanceMetrics:
        """Get all performance metrics."""
        return PerformanceMetrics(
            train_time_seconds=self.get_train_time(),
            inference_time_seconds=self.get_inference_time(),
            inference_fps=self.get_fps(),
            num_test_images=self.num_images_processed,
            peak_gpu_memory_mb=self.peak_gpu_memory if self.gpu_available else None,
            peak_cpu_memory_mb=self.peak_cpu_memory,
        )


def extract_metric(metrics: Dict[str, Any], possible_keys: List[str]) -> Optional[float]:
    """Try multiple possible metric keys and return the first found."""
    for key in possible_keys:
        if key in metrics:
            value = metrics[key]
            # Handle tensor values
            if hasattr(value, 'item'):
                return value.item()
            return value
    return None


def extract_model_metrics(raw_metrics: Any) -> Dict[str, Optional[float]]:
    """Extract model performance metrics from raw results."""
    # Handle metrics being returned as a list
    if isinstance(raw_metrics, list) and len(raw_metrics) > 0:
        raw_metrics = raw_metrics[0]
    
    return {
        "image_auroc": extract_metric(raw_metrics, ["image_AUROC", "image/AUROC", "AUROC"]),
        "pixel_auroc": extract_metric(raw_metrics, ["pixel_AUROC", "pixel/AUROC"]),
        "f1_score": extract_metric(raw_metrics, ["image_F1Score", "image/F1Score", "F1Score", "image_F1"]),
    }


def format_time(seconds: float) -> str:
    """Format seconds into a human-readable string."""
    if seconds < 60:
        return f"{seconds:.2f}s"
    elif seconds < 3600:
        minutes = int(seconds // 60)
        secs = seconds % 60
        return f"{minutes}m {secs:.2f}s"
    else:
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = seconds % 60
        return f"{hours}h {minutes}m {secs:.2f}s"


def format_memory(mb: Optional[float]) -> str:
    """Format memory in MB to a human-readable string."""
    if mb is None:
        return "N/A"
    if mb < 1024:
        return f"{mb:.2f} MB"
    return f"{mb / 1024:.2f} GB"
