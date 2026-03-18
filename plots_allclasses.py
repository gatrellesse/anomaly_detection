import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# --------------------------------------------------
# Setup
# --------------------------------------------------
os.makedirs("plots_train_together", exist_ok=True)
sns.set_theme(style="whitegrid")

FILES = {
    "baseline": "mvtec_results.csv",
    "textures": "mvtec_tt_textures.csv",
    "objects": "mvtec_tt_objects.csv",
    "mixed5": "mvtec_tt_mixed5.csv",
    # "classes3": "mvtec_tt_3classes.csv",
    "allclasses": "mvtec_results_allclasses.csv",
}

# --------------------------------------------------
# Load and merge
# --------------------------------------------------
dfs = []
for exp_name, path in FILES.items():
    if os.path.exists(path):
        df = pd.read_csv(path)
        df = df[df["model"] != "ALL"].copy() if "model" in df.columns else df.copy()
        df["experiment"] = exp_name
        dfs.append(df)
    else:
        print(f"[WARNING] File not found: {path}")

if not dfs:
    raise ValueError("No CSV files found.")

df = pd.concat(dfs, ignore_index=True)

# --------------------------------------------------
# Filter models (important!)
# --------------------------------------------------
MODELS_TO_KEEP = ["draem", "fastflow", "patchcore"]
df = df[df["model"].isin(MODELS_TO_KEEP)].copy()

# Optional: ensure category exists
if "category" not in df.columns:
    df["category"] = "unknown"

print("Experiments found:", sorted(df["experiment"].unique()))
print("Models found:", sorted(df["model"].dropna().unique()))

# --------------------------------------------------
# Helpers
# --------------------------------------------------
def save_plot(name):
    path = os.path.join("plots_train_together", f"{name}.png")
    plt.tight_layout()
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()
    print("Saved:", path)

def mean_plot(metric, title, ylabel=None):
    data = df.dropna(subset=[metric]).copy()
    if data.empty:
        return

    plt.figure(figsize=(11, 6))
    sns.barplot(data=data, x="experiment", y=metric, hue="model", errorbar="sd")
    plt.title(title)
    plt.xlabel("Experiment")
    plt.ylabel(ylabel or metric)
    plt.xticks(rotation=25)
    save_plot(f"{metric}_by_experiment_model")

def heatmap_for_model(metric, model_name):
    data = df[(df["model"] == model_name)].dropna(subset=[metric]).copy()
    if data.empty:
        return

    pivot = data.pivot_table(
        index="category",
        columns="experiment",
        values=metric,
        aggfunc="mean"
    )

    plt.figure(figsize=(10, 7))
    sns.heatmap(pivot, annot=True, fmt=".3f", cmap="viridis")
    plt.title(f"{metric} heatmap — {model_name}")
    plt.xlabel("Experiment")
    plt.ylabel("Category")
    save_plot(f"heatmap_{metric}_{model_name}")

def scatter_plot(x, y, title):
    data = df.dropna(subset=[x, y]).copy()
    if data.empty:
        return

    plt.figure(figsize=(9, 6))
    sns.scatterplot(data=data, x=x, y=y, hue="model", style="experiment", s=120)

    for _, row in data.iterrows():
        label = f"{row['model']}-{row['experiment']}"
        plt.annotate(label, (row[x], row[y]), fontsize=7, alpha=0.8)

    plt.title(title)
    save_plot(f"scatter_{x}_vs_{y}")

def category_barplot(metric, model_name):
    data = df[(df["model"] == model_name)].dropna(subset=[metric]).copy()
    if data.empty:
        return

    plt.figure(figsize=(14, 6))
    sns.barplot(data=data, x="category", y=metric, hue="experiment")
    plt.title(f"{metric} by category — {model_name}")
    plt.xlabel("Category")
    plt.ylabel(metric)
    plt.xticks(rotation=45)
    save_plot(f"{metric}_by_category_{model_name}")

# --------------------------------------------------
# Summary table
# --------------------------------------------------
summary = (
    df.groupby(["experiment", "model"], as_index=False)
      .agg({
          "image_AUROC": "mean",
          "pixel_AUROC": "mean",
          "F1_Score": "mean",
          "train_time_sec": "mean",
          "inference_fps": "mean",
          "peak_gpu_memory_mb": "mean"
      })
)

summary.to_csv("plots_train_together/summary_by_experiment_model.csv", index=False)
print("Saved: plots_train_together/summary_by_experiment_model.csv")

# --------------------------------------------------
# Main plots
# --------------------------------------------------
mean_plot("image_AUROC", "Mean Image AUROC by experiment and model")
mean_plot("pixel_AUROC", "Mean Pixel AUROC by experiment and model")
mean_plot("F1_Score", "Mean F1 Score by experiment and model")
mean_plot("train_time_sec", "Mean training time by experiment and model", "Train time (s)")
mean_plot("inference_fps", "Mean inference FPS by experiment and model", "FPS")
mean_plot("peak_gpu_memory_mb", "Mean GPU memory by experiment and model", "GPU memory (MB)")

# --------------------------------------------------
# Per-model detailed plots
# --------------------------------------------------
for model_name in sorted(df["model"].dropna().unique()):
    heatmap_for_model("image_AUROC", model_name)
    heatmap_for_model("pixel_AUROC", model_name)
    category_barplot("image_AUROC", model_name)
    category_barplot("pixel_AUROC", model_name)

# --------------------------------------------------
# Trade-off plots
# --------------------------------------------------
scatter_plot("train_time_sec", "image_AUROC", "Image AUROC vs training time")
scatter_plot("inference_fps", "image_AUROC", "Image AUROC vs inference FPS")
scatter_plot("peak_gpu_memory_mb", "image_AUROC", "Image AUROC vs GPU memory")