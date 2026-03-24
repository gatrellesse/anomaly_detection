from anomalib.metrics import Evaluator, AUROC, F1Score, AUPRO
from lightning.pytorch.loggers import CSVLogger
from anomalib.engine import Engine
import time
import torch
import sys
from utils import get_cpu_memory

def run_testbench(datamodule, category, model_class, model_name, max_epochs):
    """
    Executes a standardized benchmark for anomaly detection models using the Anomalib framework.

    This function automates the lifecycle of a model evaluation: from training 
    (fit) to inference (predict), while simultaneously tracking accuracy metrics 
    and hardware performance constraints.

    # Supported Models
    The function is designed to handle various Anomalib architectures, including:
    - Representation-based: Padim, Patchcore
    - Reconstruction-based: DRAEM, EfficientAD
    - Knowledge Distillation: Dinomaly
    - Zero-Shot: WinCLIP

    # Calculated Metrics
    The benchmark computes and returns two types of data:
    1.  Accuracy Metrics:
        - `Image-AUROC`: Overall classification performance.
        - `Pixel-AUROC`: Localization performance (anomaly segmentation).
        - `AUPRO`: Area Under the Per-Region Overlap (robust metric for segmentation).
        - `Image-F1Score`: Harmonic mean of precision and recall at image level.
    2.  Performance Metrics:
        - Training time (seconds), Inference time (seconds), and Throughput (FPS).
        - Peak Hardware Usage: Maximum GPU memory and CPU memory allocated.

    Args:
        datamodule (Folder): An instantiated Anomalib datamodule containing 
            train, validation, and test splits.
        model_class (type): The class of the model to be benchmarked (e.g., `Patchcore`).
        max_epochs (int): The maximum number of training epochs allowed.

    Returns:
        dict: A dictionary containing rounded performance results and raw metrics:
            {
                "train_time_sec": float,
                "inference_time_sec": float,
                "inference_fps": float,
                "peak_gpu_memory_mb": float,
                "peak_cpu_memory_mb": float,
                "raw_metrics": {
                    "Image-AUROC": Tensor,
                    "Pixel-AUROC": Tensor,
                    "Image-F1Score": Tensor,
                    "AUPRO": Tensor
                }
            }

    Notes:
        - Hardware measurements: GPU memory is tracked via `torch.cuda.max_memory_allocated`.
        - Logic Branching: If `model_class` is `WinClip`, training is skipped (0 sec) 
          and the model is initialized with `class_name=category`.
        - Logging: During training, Image-AUROC and AUPRO are computed every epoch and saved via `CSVLogger` in the `logs/` directory.
    """
    print(f"Running model {str(model_class)} ...")
    
    logger = CSVLogger(save_dir="logs", name=model_name, version=category)

    img_auroc = AUROC(fields=["pred_score", "gt_label"])
    aupro = AUPRO(fields=["anomaly_map", "gt_mask"])    
    evaluator = Evaluator(val_metrics=[aupro, img_auroc],
                          compute_on_cpu=False)

    # Setup Datamodule
    datamodule.setup()
    
    # Nombre total d'images de test pour le calcul du FPS
    num_test_images = len(datamodule.test_data)
    
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.empty_cache()

    test_metrics = {
        "Image-AUROC": AUROC(fields=["pred_score", "gt_label"]),
        "Pixel-AUROC": AUROC(fields=["anomaly_map", "gt_mask"]),
        "Image-F1Score": F1Score(fields=["pred_label", "gt_label"]),
        "AUPRO": AUPRO(fields=["anomaly_map", "gt_mask"])
    }

    engine = Engine(
        logger=logger,
        max_epochs=max_epochs,
        log_every_n_steps=sys.maxint,#pour que le logging ne s'effectue que à la fin d'une epoch, et jamais automatiquement après un certain nombre de step
        check_val_every_n_epoch=1
        )
    
    if model_name.lower() == "winclip":
        model = model_class(class_name=category, visualizer=False)
        train_time = 0  # No training for winclip, and need to specify class name
    # elif model_class == VlmAd:
    #     model = model_class(visualizer=False)
    #     train_time = 0  # No training for vlmad
    else:
        model = model_class(visualizer=False, evaluator=evaluator)
        start_train = time.time()
        engine.fit(model, datamodule=datamodule)
        train_time = time.time() - start_train

    start_inf = time.time()

    metrics = {}
    predictions = engine.predict(model, datamodule=datamodule)

    inference_time = time.time() - start_inf

    print("Computing the metrics :", test_metrics, " ...")
    for metric_name, metric in test_metrics.items():
        for batch in predictions:
            metric.update(batch)
        metrics[metric_name] = metric.compute()
    print("Metrics computed :", metrics)
    
    fps = num_test_images / inference_time if inference_time > 0 else 0
    
    peak_gpu = torch.cuda.max_memory_allocated() / (1024 * 1024) if torch.cuda.is_available() else 0
    peak_cpu = get_cpu_memory()

    performance_metrics = {
        "train_time_sec": round(train_time, 2),
        "inference_time_sec": round(inference_time, 2),
        "inference_fps": round(fps, 2),
        "peak_gpu_memory_mb": round(peak_gpu, 2),
        "peak_cpu_memory_mb": round(peak_cpu, 2),
        "raw_metrics": metrics
    }
    
    return performance_metrics