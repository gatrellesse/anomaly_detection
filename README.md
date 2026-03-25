# Anomaly Detection Benchmark Testbench

This repository contains a script to benchmark anomaly detection models on the MVTec AD dataset.

## ?? Overview

- `testbench.py`: main entrypoint.
- `config.py`: default dataset path, categories, models, and training settings.
- `utils/`: model loading, dataloaderutils, metrics tracking.
- `results/`: result classes and plotting helpers.

## ?? Requirements

Install dependencies:

```bash
pip install -r requirements.txt
```

## ?? Usage

Run all categories/models:

```bash
python testbench.py
```

Filter by category:

```bash
python testbench.py --category bottle
```

Filter by model:

```bash
python testbench.py --model patchcore
```

Filter by both:

```bash
python testbench.py --category bottle --model padim
```

List available options:

```bash
python testbench.py --list
```

Options:
- `-c/--category` multiple categories allowed.
- `-m/--model` multiple models allowed.
- `-n/--num-images` test image count per category.
- `-o/--output` output CSV file path.
- `--append` keep appending to CSV.
- `--cpu` force CPU-only run.

## ?? Output

Results are written to CSV (default path from `config.py`), may include:
- Image AUROC, pixel AUROC, F1
- train/inference time, FPS
- memory usage

## ?? References

- MVTec dataset: https://www.mvtec.com/company/research/datasets/mvtec-ad/
- anomalib: https://github.com/openvinotoolkit/anomalib
