"""Configuration constants for the anomaly detection testbench."""

from pathlib import Path

MVTEC_PATH = Path(__file__).parent / "MVTecAD"

# Categories to evaluate
CATEGORIES = [
    "bottle",
    "capsule",
    "cable",
    "wood"
]

# Models to benchmark
MODEL_NAMES = [
    # CNN methods
    "patchcore",
    "padim",
    "fastflow",
    # Transformer methods
    "dinomaly",
    "vlmad",
    "winclip",
]

# Test configuration
LIMIT_TEST_IMAGES = 50
BATCH_SIZE_TRAIN = 32
BATCH_SIZE_EVAL = 32

# Output file
CSV_OUTPUT = "mvtec_results.csv"
