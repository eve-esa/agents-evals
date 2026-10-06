from __future__ import annotations

from typing import Any, Dict, Optional, Tuple
import threading

_SAM3_LOCK = threading.Lock()
_SAM3_CACHE: Dict[str, Tuple[Any, Any]] = {}
_INFERENCE_LOCKS_GUARD = threading.Lock()
_SAM3_INFERENCE_LOCKS: Dict[str, threading.Lock] = {}


def _resolve_device(device: Optional[str]) -> "torch.device":
    import torch

    if isinstance(device, torch.device):
        return device
    requested = (device or "").strip()
    if not requested:
        requested = "cuda" if torch.cuda.is_available() else "cpu"
    if requested.startswith("cuda") and not torch.cuda.is_available():
        requested = "cpu"
    return torch.device(requested)


def get_sam3_components(device: Optional[str] = None) -> Tuple[Any, Any]:
    from transformers import Sam3Model, Sam3Processor
    from eo_gym.runtime.core import config_loader

    torch_device = _resolve_device(device)
    cache_key = str(torch_device)
    cached = _SAM3_CACHE.get(cache_key)
    if cached:
        return cached
    with _SAM3_LOCK:
        cached = _SAM3_CACHE.get(cache_key)
        if cached:
            return cached
        sam3_model_path = getattr(config_loader, "SAM3_MODEL_ID", "facebook/sam3")
        print(f"[model_cache] Loading SAM3 model from: {sam3_model_path} (device={torch_device})")
        model = Sam3Model.from_pretrained(sam3_model_path).to(torch_device)
        print(f"[model_cache] Loading SAM3 processor from: {sam3_model_path}")
        processor = Sam3Processor.from_pretrained(sam3_model_path)
        _SAM3_CACHE[cache_key] = (model, processor)
        return model, processor


def get_sam3_inference_lock(device: Optional[str] = None) -> threading.Lock:
    torch_device = _resolve_device(device)
    cache_key = str(torch_device)
    with _INFERENCE_LOCKS_GUARD:
        lock = _SAM3_INFERENCE_LOCKS.get(cache_key)
        if lock is None:
            lock = threading.Lock()
            _SAM3_INFERENCE_LOCKS[cache_key] = lock
        return lock


def get_grounding_dino_components(device: Optional[str] = None) -> Tuple[Any, Any]:
    # Backward-compat alias for callers not yet migrated to SAM3 naming.
    return get_sam3_components(device)


def get_grounding_dino_inference_lock(device: Optional[str] = None) -> threading.Lock:
    # Backward-compat alias for callers not yet migrated to SAM3 naming.
    return get_sam3_inference_lock(device)


def preload_sam3_components(device: Optional[str] = None) -> None:
    get_sam3_components(device)


def preload_grounding_dino_components(device: Optional[str] = None) -> None:
    preload_sam3_components(device)
