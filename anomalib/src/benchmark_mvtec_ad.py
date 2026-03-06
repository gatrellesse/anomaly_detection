import os
import csv
import time
import torch
import psutil
from anomalib.data import MVTecAD
from anomalib.engine import Engine
from anomalib.models import Patchcore, Padim, EfficientAd, WinClip, Dinomaly, Draem, VlmAd, Fastflow
from utils import get_cpu_memory, extract_metric
from anomalib.metrics import Evaluator, AUROC, F1Score, AUPRO
from lightning.pytorch.loggers import CSVLogger

def run_testbench(category, model_class, batch_size, max_epochs, mvtec_path):
    
    #les logs seront sauvegardés dans le dossier logs/{model_name}/{category}
    logger = CSVLogger(save_dir="logs", name=model_name, version=category)

    img_auroc = AUROC(fields=["pred_score", "gt_label"])
    aupro = AUPRO(fields=["anomaly_map", "gt_mask"])    
    evaluator = Evaluator(val_metrics=[aupro, img_auroc],
                          compute_on_cpu=False)
    
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

    test_metrics = {
        "Image-AUROC": AUROC(fields=["pred_score", "gt_label"]),
        "Pixel-AUROC": AUROC(fields=["anomaly_map", "gt_mask"]),
        "Image-F1Score": F1Score(fields=["pred_label", "gt_label"]),
        "AUPRO": AUPRO(fields=["anomaly_map", "gt_mask"])
    }

    # 2. Créer l'engine en spécifiant ces métriques
    engine = Engine(
        logger=logger,
        max_epochs=max_epochs,
        log_every_n_steps=999999999,
        check_val_every_n_epoch=1
        )

    if model_class == WinClip:
        model = model_class(class_name=category, visualizer=False)
        train_time = 0  # No training for winclip
    elif model_class == VlmAd:
        model = model_class()
        train_time = 0  # No training for vlmad
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
        # "dinomaly": {"class": Dinomaly, "batch_size": 8, "epochs": 1}
        # "winclip": {"class": WinClip, "batch_size": 8, "epochs": 0},
        # "patchcore": {"class": Patchcore, "batch_size": 8, "epochs": 1},
        # "padim": {"class": Padim, "batch_size": 8, "epochs": 1},
        # "efficientad": {"class": EfficientAd, "batch_size": 1, "epochs": 40}
        # "draem": {"class": Draem, "batch_size": 8, "epochs": 700},
        "vlmad": {"class": VlmAd, "batch_size": 8, "epochs": 0}
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
                result_data = run_testbench(
                    category,
                    config["class"],
                    config["batch_size"],
                    config["epochs"],
                    MVTEC_PATH
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