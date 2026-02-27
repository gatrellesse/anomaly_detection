"""Configuration constants for the anomaly detection testbench."""

from pathlib import Path

MVTEC_PATH = Path(__file__).parent / "MVTecAD"

# Categories to evaluate
CATEGORIES = [
    "bottle",
    "cable",
    "capsule",
    "carpet",
    "grid",
    "hazelnut",
    "leather",
    "metal_nut",
    "pill",
    "screw",
    "tile",
    "toothbrush",
    "transistor",
    "wood",
    "zipper"
]

# Models to benchmark
MODEL_NAMES = [
    # CNN methods
    "patchcore",
    # "padim",
    "fastflow",
    # Transformer methods
    "dinomaly",
    # "vlmad",
    # "winclip",
    "draem",
    "efficientad",
]

# Test configuration
LIMIT_TEST_IMAGES = None
BATCH_SIZE_TRAIN = 32
BATCH_SIZE_EVAL = 16

# Model-specific batch sizes for memory-intensive models
MODEL_BATCH_SIZES = {
    "dinomaly": {"train": 4, "eval": 8},  # Large transformer model needs smaller batches
    "vlmad": {"train": 8, "eval": 16},
    "winclip": {"train": 8, "eval": 16},
    "draem": {"train": 2, "eval": 2},
    "efficientad": {"train": 1, "eval":16},
}

# Output file
CSV_OUTPUT = "mvtec_results.csv"

MODEL_EPOCHS = {
    "draem": 5,
    "efficientad": 5
}

