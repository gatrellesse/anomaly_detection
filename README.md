# Anomaly Detection Benchmarking

A comprehensive benchmarking framework for evaluating various anomaly detection models on industrial datasets using the [Anomalib](https://github.com/openalchemist/anomalib) library.

## Overview

This project provides a standardized evaluation pipeline for multiple anomaly detection architectures across different datasets. It benchmarks both accuracy metrics (AUROC, AUPRO, F1-Score) and performance metrics (training/inference time, memory usage).

### Supported Models

- **Representation-based**: Patchcore, Padim
- **Reconstruction-based**: DRAEM, EfficientAD
- **Knowledge Distillation**: Dinomaly
- **Zero-Shot**: WinClip

### Supported Datasets

- **MVTecAD**: Industrial benchmark dataset with 15 categories (10 objects, 5 textures)
- **Custom Jeans Dataset**: Private dataset for jeans defect detection

## Installation

### Prerequisites
- Python 3.8+
- CUDA 12.x (optional, for GPU acceleration)

### Setup

1. Clone or navigate to the repository

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   pip install -r anomalib/requirements.txt
   ```

## Usage

### Running Benchmarks

#### MVTecAD Dataset

```bash
cd anomalib/src
python run_benchmark_mvtec_ad.py
```

This will:
- Benchmark all 6 models across 15 MVTecAD categories
- Generate results in `results/mvtec_results.csv`

#### Custom Jeans Dataset

```bash
cd anomalib/src
python run_benchmark_dataset_jeans.py
```

This will:
- Benchmark all 6 models on the jeans defect detection task
- Generate results in `results/results_benchmark_dataset_{category}_{model}.csv`

### Dataset Preparation

For custom datasets, use the scraping utilities:

```bash
cd scraping/src

# Remove backgrounds from images
python remove_background_images.py

# Annotate defects
python annotate.py

# Extract images from CSV format
python extract_images_from_csv.py

# Split dataset for training/validation/testing
python split_har.py

# Analyze dataset statistics
jupyter notebook analyse_dataset.ipynb
jupyter notebook compute_dataset_stats.ipynb
```

## Evaluation Metrics

### Accuracy Metrics
- **Image-AUROC**: Classification performance at image level
- **Pixel-AUROC**: Localization performance (anomaly segmentation)
- **AUPRO**: Area Under the Per-Region Overlap (robust segmentation metric)
- **Image-F1Score**: Harmonic mean of precision and recall

### Performance Metrics
- **Training Time**: Total time to train the model (seconds)
- **Inference Time**: Time to run predictions on the test set (seconds)
- **Inference FPS**: Throughput in frames per second
- **Peak GPU Memory**: Maximum GPU memory allocated (MB)
- **Peak CPU Memory**: Maximum CPU memory allocated (MB)

## Results Analysis

After benchmarking, visualize and format results:

```bash
cd anomalib/src
jupyter notebook dataviz_results_benchmark.ipynb
jupyter notebook mise_en_forme_results_benchmark.ipynb
```
