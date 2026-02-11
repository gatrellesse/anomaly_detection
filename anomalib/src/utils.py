import psutil
import torch
import os

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