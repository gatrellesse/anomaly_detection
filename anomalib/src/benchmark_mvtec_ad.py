import os
import csv
import time
import torch
import psutil
from anomalib.data import MVTecAD
from anomalib.engine import Engine
from anomalib.models import Patchcore, Padim, EfficientAd

import get_cpu_memory, extract_metric from utils.py

def run_testbench(category, model_class, batch_size, max_epochs, mvtec_path):
    # Setup Datamodule
    datamodule = MVTecAD(
        root=mvtec_path,
        category=category,
        train_batch_size=batch_size,
        eval_batch_size=batch_size,
    )
    datamodule.setup()
    
    # Nombre total d'images de test pour le calcul du FPS
    num_test_images = len(datamodule.test_data)
    
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.empty_cache()

    engine = Engine(max_epochs=max_epochs)
    model = model_class()

    start_train = time.time()
    engine.fit(model, datamodule)
    train_time = time.time() - start_train

    start_inf = time.time()
    metrics = engine.test(model, datamodule)
    inference_time = time.time() - start_inf
    
    fps = num_test_images / inference_time if inference_time > 0 else 0
    
    peak_gpu = torch.cuda.max_memory_allocated() / (1024 * 1024) if torch.cuda.is_available() else 0
    peak_cpu = get_cpu_memory()

    if isinstance(metrics, list) and len(metrics) > 0:
        metrics = metrics[0]

    performance_metrics = {
        "train_time_sec": round(train_time, 2),
        "inference_time_sec": round(inference_time, 2),
        "inference_fps": round(fps, 2),
        "peak_gpu_memory_mb": round(peak_gpu, 2),
        "peak_cpu_memory_mb": round(peak_cpu, 2),
        "raw_metrics": metrics
        }
    
    return performance_metrics

if __name__ == "__main__":
    MVTEC_PATH = "./" 
    CSV_OUTPUT = "results/mvtec_results.csv"
    os.makedirs("results", exist_ok=True)

    CATEGORIES = [
        "bottle",
        "capsule",
        "hazelnut",
        "leather",
        "metal_nut"
        ]
    MODELS = {
        "patchcore": Patchcore,
        "padim": Padim,
        "efficientad": EfficientAd
    }

    MAX_EPOCHS = [
        1,
        1,
        10
    ]

    BATCH_SIZES = [
        32,
        32,
        1
    ]
    
    rows = []

    fieldnames = [
        "category", "model", "image_AUROC", "pixel_AUROC", "F1_Score",
        "train_time_sec", "inference_time_sec", "inference_fps", 
        "peak_gpu_memory_mb", "peak_cpu_memory_mb"
    ]

    for category in CATEGORIES:
        for model_name, model_class, batch_size, max_epochs in zip(MODELS.keys(), MODELS.values(), BATCH_SIZES, MAX_EPOCHS):
            print(f"\n>>> Processing: {category} | {model_name}")

            try:
                result_data = run_testbench(category, model_class, batch_size, max_epochs, MVTEC_PATH)
                metrics = result_data["raw_metrics"]

                image_auc = extract_metric(metrics, ["image_AUROC", "image/AUROC", "AUROC"])
                pixel_auc = extract_metric(metrics, ["pixel_AUROC", "pixel/AUROC"])
                f1_score = extract_metric(metrics, ["image_F1Score", "image/F1Score", "F1Score"])

                row = {
                    "category": category,
                    "model": model_name,
                    "image_AUROC": image_auc,
                    "pixel_AUROC": pixel_auc,
                    "F1_Score": f1_score,
                    "train_time_sec": result_data["train_time_sec"],
                    "inference_time_sec": result_data["inference_time_sec"],
                    "inference_fps": result_data["inference_fps"],
                    "peak_gpu_memory_mb": result_data["peak_gpu_memory_mb"],
                    "peak_cpu_memory_mb": result_data["peak_cpu_memory_mb"]
                }
                rows.append(row)
                print(f"Done. FPS: {row['inference_fps']} | GPU: {row['peak_gpu_memory_mb']} MB")

            except Exception as e:
                print(f"Error for {model_name}: {e}")
                import traceback
                traceback.print_exc()

    # Sauvegarde CSV
    with open(CSV_OUTPUT, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    
    print(f"\nResults saved to: {CSV_OUTPUT}")