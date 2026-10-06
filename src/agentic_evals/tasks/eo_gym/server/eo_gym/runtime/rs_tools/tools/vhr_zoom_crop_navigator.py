from __future__ import annotations

import uuid
import csv
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import requests
from PIL import Image

from eo_gym.runtime.rs_tools.tools.cropped_to_base_mapping import (
    lookup_base_image_url,
    append_cropped_to_base_mapping,
    resolve_base_context,
)
from eo_gym.runtime.rs_tools.tools.ground_truth_cache import (
    _canonicalize_url_key,
    append_vhr_crop_mapping,
    clip_ground_truth_to_aoi,
    load_vhr_crop_index,
    record_crop_ground_truth,
)
from eo_gym.runtime.rs_tools.tools.image_cache_logger import log_image_event
from eo_gym.runtime.rs_tools.utils import check_file_reference
from eo_gym.runtime.rs_tools.tools.image_cache_paths import cache_local_path

import eo_gym.runtime.core.config_loader as config_loader

DEFAULT_GSD = 0.1
UPDATE_CROP_GT = False


@lru_cache(maxsize=1)
def _load_masking_url_map() -> Dict[str, str]:
    if not config_loader.RUN_WITH_MASKING:
        return {}
    path = config_loader.VQA_MASK_MAPPING_DIR / "masking_source_to_masking_url.csv"
    if not path.exists():
        return {}
    mapping: Dict[str, str] = {}
    with path.open("r", encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            source_url = (row.get("source_url") or "").strip()
            masking_url = (row.get("masking_url") or "").strip()
            if source_url and masking_url:
                mapping[source_url] = masking_url
    return mapping


def _resolve_masking_url(url: str) -> str:
    if not config_loader.RUN_WITH_MASKING:
        return url
    mapping = _load_masking_url_map()
    if not mapping:
        return url
    normalized = config_loader.to_remote_url(url).split("?", 1)[0]
    return mapping.get(normalized, url)

def _find_entry_for_image(image_url: str) -> Dict[str, Any]:
    key = _canonicalize_url_key(str(image_url).split("?")[0])
    url_map, basename_map = load_vhr_crop_index()
    entry = url_map.get(key) or basename_map.get(Path(key).name)
    if entry:
        return entry
    raise ValueError(
        f"No VHR_Zoom crop mapping found for {image_url!r}; ensure it exists in {config_loader.CROPPED_TO_CROPPED_GT_PATH} "
        "or generate it with the cropping tools."
    )


def _resolve_base_context(
    image_url: str,
    entry: Optional[Dict[str, Any]] = None,
) -> Tuple[str, Tuple[int, int, int, int], Tuple[int, int], Dict[str, Any], Dict[str, Any]]:
    """Resolve base image URL, current crop rect (relative to base), base size, base GT, and metadata."""
    entry_base_url = entry.get("base_image_url") if isinstance(entry, dict) else None
    entry_source_id = entry.get("source_record_id") if isinstance(entry, dict) else None
    lookup_url = entry_base_url or image_url
    ctx = resolve_base_context(
        lookup_url,
        require_ground_truth=UPDATE_CROP_GT,
        default_source_id=entry_source_id,
    )
    base_url = entry_base_url or ctx.get("base_image_url") or lookup_url
    crop_rect = ctx.get("crop_rect") or {"x": 0, "y": 0, "width": 0, "height": 0}
    base_size = ctx.get("base_image_size") or {"width": 0, "height": 0}
    base_gt = ctx.get("base_ground_truth") if UPDATE_CROP_GT else None
    meta = {"source_record_id": entry_source_id or ctx.get("source_record_id")}
    return (
        base_url,
        (
            int(crop_rect.get("x", 0)),
            int(crop_rect.get("y", 0)),
            int(crop_rect.get("width", 0)),
            int(crop_rect.get("height", 0)),
        ),
        (int(base_size.get("width", 0)), int(base_size.get("height", 0))),
        base_gt,
        meta,
    )


def _ensure_base_image_local(entry: Dict[str, Any]) -> Path:
    """
    Resolve a local base image file for navigation. Prefer base_image_abspath; if missing,
    download base_image_url into the cache directory.
    """
    candidates = [
        entry.get("base_image_abspath"),
        entry.get("base_image_path"),
        entry.get("cropped_image_path"),
    ]
    for candidate in candidates:
        if not candidate:
            continue
        path = Path(candidate)
        if not path.is_absolute():
            path = config_loader.REPO_ROOT / path
        if path.exists():
            return path
        
    base_url = entry.get("base_image_url") or lookup_base_image_url(
        entry.get("image_url") or entry.get("cropped_image_url") or entry.get("cropped_image_path")
    )
    if not base_url:
        raise FileNotFoundError("No base image location available for navigation.")
    base_url = str(base_url)
    if config_loader.RUN_ON_LOCAL:
        base_url = config_loader.to_local_path(base_url)
    # If base_url is a local path, return it directly when it exists.
    local_candidate = Path(base_url)
    if local_candidate.exists():
        return local_candidate
    if not local_candidate.is_absolute():
        repo_candidate = config_loader.REPO_ROOT / local_candidate
        if repo_candidate.exists():
            return repo_candidate

    if not str(base_url).startswith(("http://", "https://")):
        raise FileNotFoundError(f"Base image not found locally: {base_url}")

    # Fallback to download when given an HTTP(S) URL.
    cache_dir = config_loader.IMAGE_CACHE_DIR
    cache_dir.mkdir(parents=True, exist_ok=True)
    dest = cache_dir / f"base_{Path(base_url).name}"
    if not dest.exists():
        resp = requests.get(base_url, timeout=20)
        dest.write_bytes(resp.content)
    return dest


def _compute_crop_rect(
    base_width: int,
    base_height: int,
    crop_rect: Tuple[int, int, int, int],
    action: str,
    value: int,
) -> Tuple[Tuple[int, int, int, int], List[str]]:
    x, y, w, h = crop_rect
    warnings: List[str] = []
    if action == "zoom_out":
        padding = max(0, int(value))
        if padding:
            if x - padding < 0:
                warnings.append("left of the map service boundary")
            if y - padding < 0:
                warnings.append("above the map service boundary")
            if x + w + padding > base_width:
                warnings.append("right of the map service boundary")
            if y + h + padding > base_height:
                warnings.append("below the map service boundary")
        new_x = max(0, x - padding)
        new_y = max(0, y - padding)
        new_w = min(base_width - new_x, w + padding * 2)
        new_h = min(base_height - new_y, h + padding * 2)
    elif action == "move_up":
        new_x = x
        if y - value < 0:
            warnings.append("above the map service boundary")
        new_y = max(0, y - value)
        new_w = w
        new_h = h
    elif action == "move_down":
        new_x = x
        if y + value > base_height - h:
            warnings.append("below the map service boundary")
        new_y = min(base_height - h, y + value)
        new_w = w
        new_h = h
    elif action == "move_left":
        if x - value < 0:
            warnings.append("left of the map service boundary")
        new_x = max(0, x - value)
        new_y = y
        new_w = w
        new_h = h
    elif action == "move_right":
        if x + value > base_width - w:
            warnings.append("right of the map service boundary")
        new_x = min(base_width - w, x + value)
        new_y = y
        new_w = w
        new_h = h
    else:
        raise ValueError("Unknown navigation action.")
    new_w = max(1, int(round(new_w)))
    new_h = max(1, int(round(new_h)))
    new_x = min(max(0, int(round(new_x))), base_width - new_w)
    new_y = min(max(0, int(round(new_y))), base_height - new_h)
    return (new_x, new_y, new_w, new_h), warnings


def _crop_base_image(base_path: Path, rect: Tuple[int, int, int, int], record_id: str) -> Tuple[Path, Tuple[int, int, int, int]]:
    x0, y0, w, h = rect
    with Image.open(base_path) as img:
        img = img.convert("RGB")
        crop_box = (x0, y0, x0 + w, y0 + h)
        cropped = img.crop(crop_box)
    config_loader.IMAGE_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    suffix = f"{record_id}_{uuid.uuid4().hex[:6]}"
    crop_name = f"{suffix}_{x0}_{y0}_{w}x{h}.png"
    dest_path = config_loader.IMAGE_CACHE_DIR / crop_name
    cropped.save(dest_path, format="PNG")
    return dest_path, (x0, y0, w, h)


def _cache_cropped_image(dest_path: Path, source_url: str, action: str, value: int) -> str:
    cached_path = cache_local_path(dest_path)
    log_image_event(
        "vhr_crop_navigation",
        {
            "source_image_url": source_url,
            "cropped_path": str(dest_path),
            "cropped_url": cached_path,
            "action": action,
            "action_value": value,
        },
    )
    return cached_path


def _persist_crop_ground_truth(
    *,
    base_gt: Dict[str, Any],
    new_rect: Tuple[int, int, int, int],
    base_width: int,
    base_height: int,
    cached_path: str,
    dest_path: Path,
) -> Dict[str, Any]:
    if not base_gt or base_width <= 0 or base_height <= 0:
        raise ValueError("Missing base ground truth or base image size for navigation crop.")
    x, y, w, h = new_rect
    aoi_norm = [
        x / float(base_width),
        y / float(base_height),
        (x + w) / float(base_width),
        (y + h) / float(base_height),
    ]
    clipped = clip_ground_truth_to_aoi(base_gt, aoi_norm)
    if clipped is None:
        raise ValueError("Failed to clip ground truth for navigation crop.")
    primary_key = cached_path or dest_path.resolve().as_uri() or str(dest_path)
    record_crop_ground_truth(
        crop_path=str(primary_key),
        ground_truth=clipped,
    )
    return clipped


def _navigate_crop(optical_image_url: str, action: str, value: int) -> Dict[str, Any]:
    error = check_file_reference(optical_image_url, label="image", strict=True)
    if error:
        return {"message": error}
    try:
        entry = _find_entry_for_image(optical_image_url)
    except Exception:
        entry = {"image_url": optical_image_url}
    base_image_url, rect, base_size, base_gt, meta = _resolve_base_context(optical_image_url, entry)
    if config_loader.RUN_WITH_MASKING:
        masked_base_url = _resolve_masking_url(base_image_url)
        if masked_base_url and masked_base_url != base_image_url:
            base_image_url = masked_base_url
    entry["base_image_url"] = base_image_url
    entry.setdefault("crop_rect", {"x": rect[0], "y": rect[1], "width": rect[2], "height": rect[3]})
    entry.setdefault("base_image_size", {"width": base_size[0], "height": base_size[1]})
    base_abspath = _ensure_base_image_local(entry)
    base_width, base_height = base_size
    if base_width <= 0 or base_height <= 0:
        with Image.open(base_abspath) as img:
            base_width, base_height = img.width, img.height
    if rect[2] <= 0 or rect[3] <= 0:
        rect = (0, 0, base_width, base_height)
    if UPDATE_CROP_GT and base_gt is None:
        raise ValueError(f"Missing base ground truth for {base_image_url}")
    new_rect, warnings = _compute_crop_rect(base_width, base_height, rect, action, value)
    record_id = entry.get("record_id") or Path(_canonicalize_url_key(optical_image_url)).stem
    if config_loader.RUN_WITH_MASKING:
        record_id = Path(_canonicalize_url_key(optical_image_url)).stem or record_id
    dest_path, crop_rect = _crop_base_image(base_abspath, new_rect, record_id)
    cached_path = _cache_cropped_image(
        dest_path,
        base_image_url,
        action,
        value,
    )
    append_cropped_to_base_mapping(
        cached_path,
        base_image_url,
        crop_rect={"x": int(new_rect[0]), "y": int(new_rect[1]), "width": int(new_rect[2]), "height": int(new_rect[3])},
        base_image_size={"width": int(base_width), "height": int(base_height)},
    )
    clipped_gt = None
    if UPDATE_CROP_GT and base_gt:
        clipped_gt = _persist_crop_ground_truth(
            base_gt=base_gt,
            new_rect=new_rect,
            base_width=base_width,
            base_height=base_height,
            cached_path=cached_path,
            dest_path=dest_path,
        )
    base_source_id = entry.get("source_record_id") or meta.get("source_record_id") or Path(base_image_url).stem
    mapping_entry = {
        "record_id": f"{base_source_id}_nav_{uuid.uuid4().hex[:6]}",
        "source_record_id": base_source_id,
        "dataset": "VHR_Zoom",
        "image_url": cached_path,
        "ground_truth": clipped_gt,
        "base_image_url": base_image_url,
        "crop_rect": {"x": int(new_rect[0]), "y": int(new_rect[1]), "width": int(new_rect[2]), "height": int(new_rect[3])},
        "base_image_size": {"width": int(base_width), "height": int(base_height)},
    }
    if UPDATE_CROP_GT and clipped_gt is not None:
        append_vhr_crop_mapping(mapping_entry)

    payload: Dict[str, Any] = {
        "new_image_url": mapping_entry["image_url"],
        "image_width": crop_rect[2],
        "image_height": crop_rect[3],
        "gsd": DEFAULT_GSD,
    }
    if warnings:
        payload["warning"] = "; ".join(warnings)
    return payload


class ZoomOutCropNavigator:
    """
    Expand the visible area from a cropped VHR image by padding `zoom_out_pixel` around the current crop.
    Input: optical_image_url (str) pointing to an existing crop URL and zoom_out_pixel (int) giving padding in base pixels.
    Returns a dict with `new_image_url`, `image_width`, `image_height`, `gsd`.
    """
    def __init__(self, device: str = "cpu"):
        self.device = device

    def __call__(self, optical_image_url: str, zoom_out_pixel: int) -> Dict[str, Any]:
        return _navigate_crop(optical_image_url, "zoom_out", max(0, int(zoom_out_pixel)))


class MoveUpCropNavigator:
    """
    Shift the crop upward by `move_pixel` within the base image while keeping width/height constant.
    Input: optical_image_url (str) and move_pixel (int).
    Returns a dict with `new_image_url`, `image_width`, `image_height`, `gsd`.
    """
    def __init__(self, device: str = "cpu"):
        self.device = device

    def __call__(self, optical_image_url: str, move_pixel: int) -> Dict[str, Any]:
        return _navigate_crop(optical_image_url, "move_up", max(0, int(move_pixel)))


class MoveDownCropNavigator:
    """
    Shift the crop downward by `move_pixel` while preserving width/height.
    Input: optical_image_url (str) and move_pixel (int).
    Returns a dict with `new_image_url`, `image_width`, `image_height`, `gsd`.
    """
    def __init__(self, device: str = "cpu"):
        self.device = device

    def __call__(self, optical_image_url: str, move_pixel: int) -> Dict[str, Any]:
        return _navigate_crop(optical_image_url, "move_down", max(0, int(move_pixel)))


class MoveLeftCropNavigator:
    """
    Shift the crop leftward by `move_pixel` while preserving width/height.
    Input: optical_image_url (str) and move_pixel (int). 
    Returns a dict with `new_image_url`, `image_width`, `image_height`, `gsd`.
    """
    def __init__(self, device: str = "cpu"):
        self.device = device

    def __call__(self, optical_image_url: str, move_pixel: int) -> Dict[str, Any]:
        return _navigate_crop(optical_image_url, "move_left", max(0, int(move_pixel)))


class MoveRightCropNavigator:
    """
    Shift the crop rightward by `move_pixel` while preserving width/height.
    Input: optical_image_url (str) and move_pixel (int). 
    Returns a dict with `new_image_url`, `image_width`, `image_height`, `gsd`.
    """
    def __init__(self, device: str = "cpu"):
        self.device = device

    def __call__(self, optical_image_url: str, move_pixel: int) -> Dict[str, Any]:
        return _navigate_crop(optical_image_url, "move_right", max(0, int(move_pixel)))
