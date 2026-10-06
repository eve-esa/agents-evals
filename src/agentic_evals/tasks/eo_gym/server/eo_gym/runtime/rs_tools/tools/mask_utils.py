"""
Helpers for saving binary masks to the local image cache with structured logging.
"""

from __future__ import annotations

import re
import uuid
from io import BytesIO
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Union

import numpy as np
from PIL import Image
import requests

from eo_gym.runtime.rs_tools.tools.image_cache_paths import cache_local_path
from eo_gym.runtime.rs_tools.tools.image_cache_logger import log_image_event
import eo_gym.runtime.core.config_loader as config_loader


def _sanitize_label(label: str) -> str:
    cleaned = re.sub(r"[^a-z0-9]+", "_", (label or "object").lower()).strip("_")
    return cleaned or "object"


def bbox_to_mask(
    bbox: Sequence[float],
    width: int,
    height: int,
) -> np.ndarray:
    """
    Convert a [x, y, w, h] bbox into a binary mask with ones inside the box.
    """
    if len(bbox) != 4:
        raise ValueError("bbox must have four values [x, y, w, h].")
    x, y, w, h = [float(v) for v in bbox]
    x0 = max(0, int(round(x)))
    y0 = max(0, int(round(y)))
    x1 = min(width, int(round(x + w)))
    y1 = min(height, int(round(y + h)))
    if x1 <= x0 or y1 <= y0:
        return np.zeros((height, width), dtype=np.uint8)
    mask = np.zeros((height, width), dtype=np.uint8)
    mask[y0:y1, x0:x1] = 1
    return mask


def mask_to_bbox(mask: np.ndarray) -> Optional[List[int]]:
    """
    Compute a tight bounding box [x0, y0, w, h] from a binary mask.
    Returns None when the mask is empty.
    """
    if mask is None or mask.size == 0:
        return None
    coords = np.argwhere(mask > 0)
    if coords.size == 0:
        return None
    y_min, x_min = coords.min(axis=0)
    y_max, x_max = coords.max(axis=0)
    return [
        int(x_min),
        int(y_min),
        int(x_max - x_min + 1),
        int(y_max - y_min + 1),
    ]


def save_mask_image(
    mask: np.ndarray,
    *,
    source_image_url: str,
    label: str,
    method: str,
    metadata: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Persist a binary mask as a PNG in the local cache and return metadata.
    """
    if mask.ndim != 2:
        raise ValueError("mask must be a 2D array.")
    config_loader.IMAGE_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    safe_label = _sanitize_label(label)
    source_stem = Path(source_image_url.split("?")[0]).stem or "image"
    file_name = f"mask_{safe_label}_{source_stem}_{uuid.uuid4().hex}.png"
    out_path = config_loader.IMAGE_CACHE_DIR / file_name
    mask_uint8 = (mask > 0).astype(np.uint8) * 255
    Image.fromarray(mask_uint8).save(out_path, format="PNG")

    mask_path = cache_local_path(out_path)
    size = [int(mask.shape[0]), int(mask.shape[1])]
    bbox = mask_to_bbox(mask)
    record: Dict[str, Any] = {
        "label": label,
        "mask_url": mask_path,
        "source_image_url": source_image_url,
        "size": size,
        "bbox": bbox,
        "method": method,
    }
    if metadata:
        record.update(metadata)

    log_image_event(
        "mask_generated",
        {
            **record,
            "local_file": file_name,
        },
    )
    return record


def load_mask_from_source(mask_input: Union[str, np.ndarray, List[List[int]], List[List[bool]]]) -> np.ndarray:
    """
    Load a binary mask from a URL/path or accept an existing array/list.
    Returns a boolean numpy array.
    """
    if isinstance(mask_input, np.ndarray):
        return mask_input.astype(bool)
    if isinstance(mask_input, (list, tuple)):
        return np.asarray(mask_input, dtype=bool)
    if isinstance(mask_input, str):
        raw = mask_input
        if config_loader.RUN_ON_LOCAL:
            raw = config_loader.to_local_path(raw)
        path = Path(raw)
        try:
            if raw.startswith("http://") or raw.startswith("https://"):
                resp = requests.get(raw, timeout=20)
                resp.raise_for_status()
                img = Image.open(BytesIO(resp.content))
            else:
                img = Image.open(path)
            return np.array(img.convert("L")) > 0
        except Exception as exc:
            raise ValueError(f"Failed to load mask from {mask_input}: {exc}")
    raise ValueError("mask_input must be a URL, path, numpy array, or nested list.")


__all__ = ["bbox_to_mask", "mask_to_bbox", "save_mask_image", "load_mask_from_source"]
