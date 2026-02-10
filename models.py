"""Model factory and registration."""

from anomalib.models import Patchcore, Padim, Fastflow, Dinomaly, VlmAd, WinClip
from typing import Dict, Type, Any


# Registry of available models
MODEL_REGISTRY: Dict[str, Type] = {
    # CNN methods
    "patchcore": Patchcore,
    "padim": Padim,
    "fastflow": Fastflow,
    # Transformer methods
    "dinomaly": Dinomaly,
    "vlmad": VlmAd,
    "winclip": WinClip,
}


def get_model(model_name: str) -> Any:
    """Get a model instance by name.
    
    Args:
        model_name: Name of the model (case-insensitive)
        
    Returns:
        An instance of the requested model
        
    Raises:
        ValueError: If the model name is not registered
    """
    model_name = model_name.lower()
    if model_name not in MODEL_REGISTRY:
        available = ", ".join(MODEL_REGISTRY.keys())
        raise ValueError(f"Unknown model: {model_name}. Available models: {available}")
    
    return MODEL_REGISTRY[model_name]()


def list_available_models() -> list:
    """Get a list of available model names."""
    return list(MODEL_REGISTRY.keys())
