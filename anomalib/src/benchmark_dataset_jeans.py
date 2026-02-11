import os
import csv
import time
import torch
import psutil
from anomalib.data import Folder
from anomalib.engine import Engine
from anomalib.models import Patchcore, Padim, EfficientAd

from utils import get_cpu_memory, extract_metric

def run_testbench(model_class, batch_size, max_epochs, dataset_path):
    # Setup Datamodule
    datamodule = Folder(
        name="jeans_troues_test",
        root=dataset_path,
        normal_dir="pas_troues",
        abnormal_dir="troues",
        mask_dir="mask/troues",
        normal_split_ratio=0.2,#Ratio to split normal training images and add to the test set in case test set doesn’t contain any normal images.
        test_split_ratio = 0.2,#Fraction of images from the train set that will be reserved for testing.
        train_batch_size=batch_size,
        eval_batch_size=batch_size# Validation, test and predict batch size
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
    DATASET_PATH = "/home/vince/ENSTA/4A/projet/scraping/datasets/jeans_troues" 
    CSV_OUTPUT = "results/results_benchmark_dataset_jeans.csv"
    os.makedirs("results", exist_ok=True)

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
        "model", "image_AUROC", "pixel_AUROC", "F1_Score",
        "train_time_sec", "inference_time_sec", "inference_fps", 
        "peak_gpu_memory_mb", "peak_cpu_memory_mb"
    ]

    for model_name, model_class, batch_size, max_epochs in zip(MODELS.keys(), MODELS.values(), BATCH_SIZES, MAX_EPOCHS):

        try:
            result_data = run_testbench(model_class, batch_size, max_epochs, DATASET_PATH)
            metrics = result_data["raw_metrics"]

            image_auc = extract_metric(metrics, ["image_AUROC", "image/AUROC", "AUROC"])
            pixel_auc = extract_metric(metrics, ["pixel_AUROC", "pixel/AUROC"])
            f1_score = extract_metric(metrics, ["image_F1Score", "image/F1Score", "F1Score"])

            row = {
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