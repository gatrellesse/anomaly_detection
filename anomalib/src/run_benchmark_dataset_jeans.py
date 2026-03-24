import os
import csv
from anomalib.models import Patchcore, Padim, EfficientAd, WinClip, Dinomaly, Draem
#from anomalib.models import VlmAd, Fastflow
from utils import extract_metric
from benchmark import run_testbench
from anomalib.data import Folder

if __name__ == "__main__":
    DATASET_PATH = "/home/ensta/ensta-bories/datasets/dataset_jeans" 
    os.makedirs("results", exist_ok=True)

    CATEGORIES = [
        "jeans",
        "jeans_detoures"
    ]

    MODELS = {
        "dinomaly": {"class": Dinomaly, "batch_size": 8, "epochs": 20},
        "winclip": {"class": WinClip, "batch_size": 8, "epochs": 0},
        "patchcore": {"class": Patchcore, "batch_size": 8, "epochs": 1},
        "padim": {"class": Padim, "batch_size": 8, "epochs": 1},
        "efficientad": {"class": EfficientAd, "batch_size": 1, "epochs": 40},
        "draem": {"class": Draem, "batch_size": 32, "epochs": 40}#700 epochs dans le papier
    }
    
    rows = []

    fieldnames = [
        "model", "image_AUROC", "pixel_AUROC", "AUPRO", "F1_Score",
        "train_time_sec", "inference_time_sec", "inference_fps", 
        "peak_gpu_memory_mb", "peak_cpu_memory_mb"
    ]
    for category in CATEGORIES:
        for model_name, config in MODELS.items():
            CSV_OUTPUT = f"results/results_benchmark_dataset_{category}_{model_name}.csv"
            try:
                datamodule = Folder(
                    name=category,
                    root=DATASET_PATH,
                    normal_dir=f"pas_troues/{category}",
                    abnormal_dir=f"troues/{category}",
                    mask_dir="masks",
                    normal_split_ratio=0.2,#Ratio to split normal training images and add to the test set in case test set doesn’t contain any normal images.
                    test_split_ratio = 0.2,#Fraction of images from the train set that will be reserved for testing.
                    train_batch_size=config["batch_size"],
                    eval_batch_size=config["batch_size"]# Validation, test and predict batch size
                )
                result_data = run_testbench(
                    datamodule,
                    category,
                    config["class"],
                    model_name,
                    config["epochs"],
                    )
                metrics = result_data["raw_metrics"]

                image_auc = extract_metric(metrics, ["Image-AUROC"])
                pixel_auc = extract_metric(metrics, ["Pixel-AUROC"])
                f1_score = extract_metric(metrics, ["Image-F1Score"])
                aupro = extract_metric(metrics, ["AUPRO"])

                row = {
                    "model": model_name,
                    "image_AUROC": image_auc,
                    "pixel_AUROC": pixel_auc,
                    "AUPRO": aupro,
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