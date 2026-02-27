#!/usr/bin/env bash
#SBATCH --job-name=mvtec
#SBATCH --partition=ENSTA-l40s        # ou ENSTA-h100
#SBATCH --nodelist=ensta-l40s02.r2.enst.fr   # opcional: especifica nó específico
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --time=04:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

# (opcional) cria pasta de logs
mkdir -p logs

echo "==== SLURM JOB INFO ===="
echo "JobID: ${SLURM_JOB_ID}"
echo "Node: ${SLURM_NODELIST}"
echo "GPUs: ${SLURM_GPUS:-N/A}"
echo "========================"

# 1) vá para a pasta do projeto (AJUSTE!)
cd "$HOME/anomaly_detection"

# 2) ativar venv (AJUSTE o caminho se necessário)
source venv/bin/activate

# # 3) info rápido
# python -V
# nvidia-smi || true

# 4) rodar o testbench
# ---- EXEMPLO: train-together com 3 categorias e 1 modelo ----
echo "Running testbench with train-together, with all categories and all models"
python testbench.py \
 --train-together 
  # --category bottle cable capsule \ colocando todas as categorias, não precisa especificar
  # --model patchcore 
  # --num-images 20 \
  # --limit-train-per-cat 200

echo "Done."