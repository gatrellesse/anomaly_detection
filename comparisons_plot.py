import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# --------------------------------------------------
# Setup
# --------------------------------------------------
os.makedirs("plots", exist_ok=True)

sns.set_theme(style="whitegrid")

# --------------------------------------------------
# Load CSV
# --------------------------------------------------
df = pd.read_csv("mvtec_results.csv")

# Work on a copy to avoid pandas warnings
df_clean = df.copy()

print("Models found:", sorted(df_clean["model"].unique()))

# --------------------------------------------------
# Helper function
# --------------------------------------------------
def save_plot(name):
    path = f"plots/{name}.png"
    plt.tight_layout()
    plt.savefig(path, dpi=300)
    plt.close()
    print("Saved:", path)


# --------------------------------------------------
# Image AUROC
# --------------------------------------------------
data = df_clean.dropna(subset=["image_AUROC"])

if not data.empty:
    plt.figure(figsize=(10,6))
    sns.barplot(data=data, x="model", y="image_AUROC")
    plt.title("Image-level AUROC by Model")
    plt.xticks(rotation=45)
    save_plot("image_auroc_by_model")


# --------------------------------------------------
# Pixel AUROC
# --------------------------------------------------
data = df_clean.dropna(subset=["pixel_AUROC"])

if not data.empty:
    plt.figure(figsize=(10,6))
    sns.barplot(data=data, x="model", y="pixel_AUROC")
    plt.title("Pixel-level AUROC by Model")
    plt.xticks(rotation=45)
    save_plot("pixel_auroc_by_model")


# --------------------------------------------------
# F1 Score
# --------------------------------------------------
data = df_clean.dropna(subset=["F1_Score"])

if not data.empty:
    plt.figure(figsize=(10,6))
    sns.barplot(data=data, x="model", y="F1_Score")
    plt.title("F1 Score by Model")
    plt.xticks(rotation=45)
    save_plot("f1_score_by_model")


# --------------------------------------------------
# Speed vs Accuracy
# --------------------------------------------------
data = df_clean.dropna(subset=["inference_fps", "image_AUROC"])

if not data.empty:
    plt.figure(figsize=(8,6))

    sns.scatterplot(
        data=data,
        x="inference_fps",
        y="image_AUROC",
        hue="model",
        s=120
    )

    for _, row in data.iterrows():
        plt.text(row["inference_fps"], row["image_AUROC"], row["model"])

    plt.title("Speed vs Accuracy")
    save_plot("speed_vs_accuracy")


# --------------------------------------------------
# Training Time
# --------------------------------------------------
data = df_clean.dropna(subset=["train_time_sec"])

if not data.empty:
    plt.figure(figsize=(10,6))
    sns.barplot(data=data, x="model", y="train_time_sec")
    plt.title("Training Time by Model")
    plt.xticks(rotation=45)
    save_plot("training_time")


# --------------------------------------------------
# GPU Memory
# --------------------------------------------------
data = df_clean.dropna(subset=["peak_gpu_memory_mb"])

if not data.empty:
    plt.figure(figsize=(10,6))
    sns.barplot(data=data, x="model", y="peak_gpu_memory_mb")
    plt.title("Peak GPU Memory Usage")
    plt.xticks(rotation=45)
    save_plot("gpu_memory")


# --------------------------------------------------
# Efficiency (only when possible)
# --------------------------------------------------
eff = df_clean.dropna(subset=["image_AUROC", "peak_gpu_memory_mb"]).copy()

if not eff.empty:
    eff["efficiency"] = eff["image_AUROC"] / eff["peak_gpu_memory_mb"]

    plt.figure(figsize=(10,6))
    sns.barplot(data=eff, x="model", y="efficiency")
    plt.title("Efficiency (AUROC per MB of GPU)")
    plt.xticks(rotation=45)
    save_plot("efficiency")


# --------------------------------------------------
# Accuracy vs Memory
# --------------------------------------------------
data = df_clean.dropna(subset=["peak_gpu_memory_mb", "image_AUROC"])

if not data.empty:
    plt.figure(figsize=(8,6))

    sns.scatterplot(
        data=data,
        x="peak_gpu_memory_mb",
        y="image_AUROC",
        hue="model",
        s=120
    )

    for _, row in data.iterrows():
        plt.text(row["peak_gpu_memory_mb"], row["image_AUROC"], row["model"])

    plt.title("Accuracy vs GPU Memory")
    save_plot("accuracy_vs_memory")


# --------------------------------------------------
# Correlation Heatmap
# --------------------------------------------------
numeric = df_clean.select_dtypes(include="number")

if not numeric.empty:
    plt.figure(figsize=(8,6))
    sns.heatmap(numeric.corr(), annot=True, cmap="coolwarm")
    plt.title("Metric Correlation")
    save_plot("metric_correlation")


# --------------------------------------------------
# MODEL LEADERBOARD (average per model)
# --------------------------------------------------
leaderboard = (
    df_clean
    .groupby("model")
    .mean(numeric_only=True)
    .sort_values("image_AUROC", ascending=False)
)

print("\nMODEL LEADERBOARD (mean across categories)")
print(leaderboard[[
    "image_AUROC",
    "pixel_AUROC",
    "F1_Score",
    "inference_fps"
]].to_string())


# --------------------------------------------------
# SPEED LEADERBOARD
# --------------------------------------------------
speed_board = (
    df_clean
    .groupby("model")
    .mean(numeric_only=True)
    .sort_values("inference_fps", ascending=False)
)

print("\nFASTEST MODELS")
print(speed_board[["inference_fps"]].to_string())

