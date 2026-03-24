import os
import csv
from anomalib.data import MVTecAD
from anomalib.models import Patchcore, Padim, EfficientAd, WinClip, Dinomaly, Draem, 
#from anomalib.models import VlmAd, Fastflow
from utils import extract_metric
from benchmark import run_testbench

if __name__ == "__main__":
    MVTEC_PATH = "./datasets/mvtec_ad" 
    CSV_OUTPUT = "results/mvtec_results.csv"
    os.makedirs("results", exist_ok=True)

    CATEGORIES = [
        # Objets (10)
        "bottle",
        "capsule",
        "hazelnut",
        "metal_nut",
        "pill",
        "screw",
        "toothbrush",
        "transistor",
        "zipper",
        "cable",
        
        # Textures (5)
        "carpet",
        "grid",
        "leather",
        "tile",
        "wood"
    ]
    
    MODELS = {
        "dinomaly": {"class": Dinomaly, "batch_size": 8, "epochs": 20},
        "winclip": {"class": WinClip, "batch_size": 8, "epochs": 0},
        "patchcore": {"class": Patchcore, "batch_size": 8, "epochs": 1},
        "padim": {"class": Padim, "batch_size": 8, "epochs": 1},
        "efficientad": {"class": EfficientAd, "batch_size": 1, "epochs": 40},
        "draem": {"class": Draem, "batch_size": 8, "epochs": 40}
    }
    
    rows = []

    fieldnames = [
        "category", "model", "image_AUROC", "pixel_AUROC", "AUPRO", "F1_Score",
        "train_time_sec", "inference_time_sec", "inference_fps", 
        "peak_gpu_memory_mb", "peak_cpu_memory_mb"
    ]

    for category in CATEGORIES:
        for model_name, config in MODELS.items():
            
            print(f"\n>>> Processing: {category} | {model_name}")
            
            try:

                datamodule = MVTecAD(
                    root=MVTEC_PATH,
                    category=category,
                    train_batch_size=config["batch_size"],
                    eval_batch_size=config["batch_size"],
                )

                result_data = run_testbench(
                    datamodule,
                    category,
                    config["class"],
                    model_name,
                    config["epochs"]
                )

                metrics = result_data["raw_metrics"]

                image_auc = extract_metric(metrics, ["Image-AUROC"])
                pixel_auc = extract_metric(metrics, ["Pixel-AUROC"])
                f1_score = extract_metric(metrics, ["Image-F1Score"])
                aupro = extract_metric(metrics, ["AUPRO"])
                row = {
                    "category": category,
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