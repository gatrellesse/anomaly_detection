import psutil
import torch
import os
from scipy.ndimage import label
import numpy as np

def get_cpu_memory():
    """Retourne la mémoire CPU utilisée par le processus actuel en MB."""
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / (1024 * 1024)

def extract_metric(metrics, possible_keys):
    """Try multiple possible metric keys and return the first found."""
    for key in possible_keys:
        if key in metrics:
            value = metrics[key]
            if hasattr(value, 'item'):
                return value.item()
            return value
    return None

def calculate_pro_score(pred_mask, gt_mask):
    """
    Calcule le score PRO pour une image donnée.
    pred_mask : matrice binaire (0,1) des prédictions
    gt_mask : matrice binaire (0,1) de la vérité terrain
    """
    # 1. Identifier les régions séparées dans le masque réel
    labeled_gt, num_regions = label(gt_mask)
    
    if num_regions == 0:
        # Si l'image est saine, le PRO n'est pas défini par région.
        return np.nan

    region_pro_scores = []

    # 2. Pour chaque région réelle, calculer le taux de recouvrement
    for i in range(1, num_regions + 1):
        # Créer un masque pour la région i uniquement
        region_mask = (labeled_gt == i)
        
        # Calcul de l'intersection : pixels prédits corrects dans cette région
        intersection = np.logical_and(pred_mask, region_mask).sum()
        
        # Taille de la région i (nombre de pixels)
        region_size = region_mask.sum()
        
        # Score de cette région
        region_pro_scores.append(intersection / region_size)

    # 3. Le PRO final est la moyenne des scores de toutes les régions
    return np.mean(region_pro_scores)