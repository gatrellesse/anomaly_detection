import os
import csv
from anomalib.data import MVTecAD
from anomalib.engine import Engine
from anomalib.models import Patchcore, Padim, Fastflow
from torch.utils.data import Subset, DataLoader

MVTEC_PATH = "/home/gabriel/ensta_3/anomaly_detection/MVTecAD"

CATEGORIES = [
    "bottle",
    "capsule",
    "cable",
    "wood"
]

MODELS = {
    "patchcore": Patchcore,
    "padim": Padim,
}

LIMIT_TEST_IMAGES = 50
CSV_OUTPUT = "mvtec_results.csv"


def extract_metric(metrics, possible_keys):
    """Try multiple possible metric keys and return the first found."""
    for key in possible_keys:
        if key in metrics:
            value = metrics[key]
            # Handle tensor values
            if hasattr(value, 'item'):
                return value.item()
            return value
    return None


def run_testbench():
    rows = []

    for category in CATEGORIES:
        print(f"\n=== CATEGORY: {category} ===")

        try:
            # Load dataset
            datamodule = MVTecAD(
                root=MVTEC_PATH,
                category=category,
                train_batch_size=32,
                eval_batch_size=32,
            )
            
            # Setup the datamodule (important!)
            datamodule.setup()

            # Get the test dataloader
            test_loader = datamodule.test_dataloader()
            
            # If we want to limit test images, we need to create a custom loader
            if LIMIT_TEST_IMAGES is not None:
                # Access the underlying dataset from the dataloader
                original_dataset = test_loader.dataset
                
                # Create a subset
                limited_indices = range(min(LIMIT_TEST_IMAGES, len(original_dataset)))
                test_dataset = Subset(original_dataset, limited_indices)
                
                # Create new dataloader with limited dataset, preserving collate_fn
                test_loader = DataLoader(
                    test_dataset, 
                    batch_size=test_loader.batch_size,
                    shuffle=False,
                    num_workers=0,  # Set to 0 to avoid multiprocessing issues
                    collate_fn=test_loader.collate_fn  # Preserve anomalib's custom collate function
                )
                
                print(f"Limited test set to {len(test_dataset)} images")

        except Exception as e:
            print(f"Error loading dataset for {category}: {e}")
            import traceback
            traceback.print_exc()
            continue

        for model_name, model_class in MODELS.items():
            print(f"\nRunning model: {model_name}")

            try:
                # Create fresh engine for each model
                engine = Engine()
                model = model_class()

                # Train model
                print(f"  Training {model_name}...")
                engine.fit(model, datamodule)

                # Evaluate on test set (limited or full)
                print(f"  Testing {model_name}...")
                metrics = engine.test(model, test_loader)

                print(f"  Raw metrics: {metrics}")  # Debug: see what keys are available

                # Handle metrics being returned as a list
                if isinstance(metrics, list) and len(metrics) > 0:
                    metrics = metrics[0]
                
                # Extract metrics with multiple possible keys
                image_auc = extract_metric(metrics, ["image_AUROC", "image/AUROC", "AUROC"])
                pixel_auc = extract_metric(metrics, ["pixel_AUROC", "pixel/AUROC"])
                pro = extract_metric(metrics, ["PRO", "pixel/PRO"])

                rows.append({
                    "category": category,
                    "model": model_name,
                    "image_AUROC": image_auc,
                    "pixel_AUROC": pixel_auc,
                    "PRO": pro,
                })

                print(f"{model_name} results:")
                print(f"  Image AUROC: {image_auc}")
                print(f"  Pixel AUROC: {pixel_auc}")
                print(f"  PRO:         {pro}")

            except Exception as e:
                print(f"Error running {model_name} on {category}: {e}")
                import traceback
                traceback.print_exc()
                rows.append({
                    "category": category,
                    "model": model_name,
                    "image_AUROC": None,
                    "pixel_AUROC": None,
                    "PRO": None,
                })
                continue

    # Compute global averages (excluding None values)
    def safe_average(key):
        values = [r[key] for r in rows if r[key] is not None and r["category"] != "AVERAGE"]
        return sum(values) / len(values) if values else None

    avg_image_auc = safe_average("image_AUROC")
    avg_pixel_auc = safe_average("pixel_AUROC")
    avg_pro = safe_average("PRO")

    print("\n==========================")
    print("GLOBAL AVERAGE METRICS")
    print("==========================")
    if avg_image_auc:
        print(f"Average Image AUROC: {avg_image_auc:.4f}")
    else:
        print("Average Image AUROC: N/A")
    
    if avg_pixel_auc:
        print(f"Average Pixel AUROC: {avg_pixel_auc:.4f}")
    else:
        print("Average Pixel AUROC: N/A")
    
    if avg_pro:
        print(f"Average PRO:         {avg_pro:.4f}")
    else:
        print("Average PRO:         N/A")

    # Append average row to CSV
    rows.append({
        "category": "AVERAGE",
        "model": "ALL",
        "image_AUROC": avg_image_auc,
        "pixel_AUROC": avg_pixel_auc,
        "PRO": avg_pro,
    })

    # Save CSV
    with open(CSV_OUTPUT, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["category", "model", "image_AUROC", "pixel_AUROC", "PRO"])
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nResults saved to: {CSV_OUTPUT}")


if __name__ == "__main__":
    run_testbench()