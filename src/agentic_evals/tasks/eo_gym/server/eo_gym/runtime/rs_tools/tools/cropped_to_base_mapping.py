"""
Helpers for persisting and reading cropped-image -> base-image mappings.
"""

from __future__ import annotations

import fcntl
import json
import os
from pathlib import Path
from typing import Dict, Optional, Tuple

import eo_gym.runtime.core.config_loader as config_loader
from eo_gym.runtime.rs_tools.tools.ground_truth_cache import (
    BASE_IMAGE_GT_MAP_PATH,
    _canonicalize_url_key,
    _ensure_ground_truth_size,
    load_image_gt_index,
    resolve_ground_truth_for_image,
)

_CROPPED_TO_BASE_URL_CACHE: Optional[Dict[str, Dict[str, object]]] = None
_CROPPED_TO_BASE_BASENAME_CACHE: Optional[Dict[str, Dict[str, object]]] = None


def _reset_cropped_to_base_cache() -> None:
    global _CROPPED_TO_BASE_URL_CACHE, _CROPPED_TO_BASE_BASENAME_CACHE
    _CROPPED_TO_BASE_URL_CACHE = None
    _CROPPED_TO_BASE_BASENAME_CACHE = None


def _load_cropped_to_base_index(
    path: Optional[Path] = None,
) -> Tuple[Dict[str, Dict[str, object]], Dict[str, Dict[str, object]]]:
    global _CROPPED_TO_BASE_URL_CACHE, _CROPPED_TO_BASE_BASENAME_CACHE
    if path is None:
        path = config_loader.CROPPED_TO_BASE_PATH
    if (
        path == config_loader.CROPPED_TO_BASE_PATH
        and _CROPPED_TO_BASE_URL_CACHE is not None
        and _CROPPED_TO_BASE_BASENAME_CACHE is not None
    ):
        return _CROPPED_TO_BASE_URL_CACHE, _CROPPED_TO_BASE_BASENAME_CACHE
    url_map: Dict[str, Dict[str, object]] = {}
    basename_map: Dict[str, Dict[str, object]] = {}
    if path.exists():
        with path.open("r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                for obj in _iter_json_objects(line):
                    if not isinstance(obj, dict):
                        continue
                    crop_url = (
                        obj.get("cropped_image_url")
                        or obj.get("cropped_image_path")
                        or obj.get("image_path")
                        or obj.get("image_url")
                    )
                    if not crop_url:
                        continue
                    key = _canonicalize_url_key(str(crop_url))
                    url_map[key] = obj
                    name = Path(key).name
                    if name and name not in basename_map:
                        basename_map[name] = obj
    if path == config_loader.CROPPED_TO_BASE_PATH:
        _CROPPED_TO_BASE_URL_CACHE = url_map
        _CROPPED_TO_BASE_BASENAME_CACHE = basename_map
    return url_map, basename_map


def _lookup_cropped_entry(cropped_url: Optional[str]) -> Optional[Dict[str, object]]:
    if not cropped_url:
        return None
    key = _canonicalize_url_key(str(cropped_url))
    url_map, basename_map = _load_cropped_to_base_index()
    return url_map.get(key) or basename_map.get(Path(key).name)

def _iter_json_objects(line: str):
    decoder = json.JSONDecoder()
    idx = 0
    text = line.strip()
    while idx < len(text):
        obj, end = decoder.raw_decode(text, idx)
        yield obj
        idx = end
        while idx < len(text) and text[idx].isspace():
            idx += 1


def append_cropped_to_base_mapping(
    cropped_url: str,
    base_url: str,
    *,
    crop_rect: Optional[Dict[str, int]] = None,
    base_image_size: Optional[Dict[str, int]] = None,
) -> None:
    """Persist cropped->base mapping so navigation tools can resolve origins."""
    if not cropped_url or not base_url:
        return
    try:
        path = config_loader.CROPPED_TO_BASE_PATH
        path.parent.mkdir(parents=True, exist_ok=True)
        payload: Dict[str, object] = {"cropped_image_url": str(cropped_url), "base_image_url": str(base_url)}
        if crop_rect and all(k in crop_rect for k in ("x", "y", "width", "height")):
            payload["crop_rect"] = {
                "x": int(crop_rect["x"]),
                "y": int(crop_rect["y"]),
                "width": int(crop_rect["width"]),
                "height": int(crop_rect["height"]),
            }
        if base_image_size and all(k in base_image_size for k in ("width", "height")):
            payload["base_image_size"] = {
                "width": int(base_image_size["width"]),
                "height": int(base_image_size["height"]),
            }
        fh = None
        try:
            fh = path.open("a+", encoding="utf-8")
            fcntl.flock(fh.fileno(), fcntl.LOCK_EX)
            fh.seek(0, os.SEEK_END)
            fh.write(json.dumps(payload, ensure_ascii=False) + "\n")
            fh.flush()
            os.fsync(fh.fileno())
        finally:
            if fh is not None:
                try:
                    fcntl.flock(fh.fileno(), fcntl.LOCK_UN)
                except Exception:
                    pass
                try:
                    fh.close()
                except Exception:
                    pass
        _reset_cropped_to_base_cache()
    except Exception:
        # Best effort: failure to persist should not block navigation.
        return

def lookup_base_image_url(cropped_url: Optional[str]) -> Optional[str]:
    """Map a cropped image URL/path back to its original image URL using the cropped samples file."""
    entry = _lookup_cropped_entry(cropped_url)
    if not entry:
        return None
    return entry.get("base_image_url") or entry.get("original_image_path") or entry.get("source_image_url")


def resolve_base_ground_truth(image_url: str) -> Optional[Tuple[dict, str]]:
    """
    Attempt to resolve the base image ground truth from image_ground_truth_map.jsonl.
    Returns (ground_truth, base_image_url) if found.
    """
    base_url = lookup_base_image_url(image_url) or image_url
    url_map, basename_map = load_image_gt_index(BASE_IMAGE_GT_MAP_PATH)
    key = _canonicalize_url_key(base_url)
    gt = url_map.get(key) or basename_map.get(Path(key).name)
    if gt:
        return gt, base_url
    return None


def _parse_crop_rect(obj: Dict[str, object]) -> Optional[Dict[str, int]]:
    rect = obj.get("crop_rect")
    if not isinstance(rect, dict):
        return None
    x = rect.get("x")
    y = rect.get("y")
    w = rect.get("width")
    h = rect.get("height")
    if not all(isinstance(v, (int, float)) for v in (x, y, w, h)):
        return None
    return {"x": int(round(x)), "y": int(round(y)), "width": int(round(w)), "height": int(round(h))}


def _parse_base_size(obj: Dict[str, object]) -> Optional[Dict[str, int]]:
    size = obj.get("base_image_size")
    if not isinstance(size, dict):
        return None
    w = size.get("width")
    h = size.get("height")
    if not all(isinstance(v, (int, float)) for v in (w, h)):
        return None
    return {"width": int(round(w)), "height": int(round(h))}


def resolve_base_info(
    image_url: str, image_size: Optional[Tuple[int, int]] = None
) -> Dict[str, Optional[Dict[str, int] | str]]:
    """
    Resolve base image URL, crop_rect (relative to base), and base_image_size for a cropped image.
    If no mapping exists, treat the image as its own base; if image_size is provided, use it for defaults.
    """
    base_url: Optional[str] = None
    crop_rect: Optional[Dict[str, int]] = None
    base_image_size: Optional[Dict[str, int]] = None

    entry = _lookup_cropped_entry(image_url)
    if entry:
        base_url = entry.get("base_image_url") or entry.get("original_image_path") or entry.get("source_image_url")
        crop_rect = _parse_crop_rect(entry) or crop_rect
        base_image_size = _parse_base_size(entry) or base_image_size
    else:
        base_url = lookup_base_image_url(image_url)

    if base_url is None:
        base_url = image_url
    if config_loader.RUN_ON_LOCAL and base_url:
        base_url = config_loader.to_local_path(base_url)

    if not base_image_size and image_size:
        base_image_size = {"width": int(image_size[0]), "height": int(image_size[1])}

    if not crop_rect and base_image_size:
        crop_rect = {
            "x": 0,
            "y": 0,
            "width": int(base_image_size["width"]),
            "height": int(base_image_size["height"]),
        }

    return {
        "base_image_url": base_url,
        "crop_rect": crop_rect,
        "base_image_size": base_image_size,
    }


def resolve_base_context(
    image_url: str,
    image_size: Optional[Tuple[int, int]] = None,
    *,
    require_ground_truth: bool = True,
    default_source_id: Optional[str] = None,
) -> Dict[str, object]:
    """
    Resolve base image URL, crop rect, base size, optional ground truth, and a source_record_id.
    If no mapping exists, treat the image itself as the base with a full-image crop_rect.
    """
    base_info = resolve_base_info(image_url, image_size)
    base_url = base_info.get("base_image_url") or image_url
    base_size = base_info.get("base_image_size")
    if not base_size and image_size:
        base_size = {"width": int(image_size[0]), "height": int(image_size[1])}
    crop_rect = base_info.get("crop_rect")
    if not crop_rect and base_size:
        crop_rect = {
            "x": 0,
            "y": 0,
            "width": int(base_size.get("width", 0) or 0),
            "height": int(base_size.get("height", 0) or 0),
        }
    base_gt: Optional[Dict[str, object]] = None
    if require_ground_truth:
        base_gt_tuple = resolve_base_ground_truth(base_url)
        base_gt = base_gt_tuple[0] if base_gt_tuple else None
        if base_gt is None:
            url_map, basename_map = load_image_gt_index(BASE_IMAGE_GT_MAP_PATH)
            key = _canonicalize_url_key(base_url)
            base_gt = url_map.get(key) or basename_map.get(Path(key).name)
        if base_gt is None:
            try:
                base_gt, _ = resolve_ground_truth_for_image(base_url)
            except Exception:
                base_gt = None
        if base_gt:
            base_gt = _ensure_ground_truth_size(base_gt, base_url)
    source_record_id = default_source_id or Path(_canonicalize_url_key(base_url)).stem
    return {
        "base_image_url": base_url,
        "crop_rect": crop_rect,
        "base_image_size": base_size,
        "base_ground_truth": base_gt,
        "source_record_id": source_record_id,
    }


def reset_cropped_to_base_cache() -> None:
    _reset_cropped_to_base_cache()
