"""
Utilities for clipping ground truth to crops and persisting crop->ground truth mappings.
Also provides helpers to look up ground truth for images and VHR crop mapping utilities shared by
cropping tools.
"""

from __future__ import annotations

import fcntl
import json
import os
from pathlib import Path
import tempfile
import urllib.request
from typing import Any, Dict, List, Optional

from PIL import Image, ImageDraw, ImageFont

import eo_gym.runtime.core.config_loader as config_loader
from eo_gym.runtime.rs_tools.tools.image_metadata import describe_image_url

REPO_ROOT = config_loader.REPO_ROOT
BASE_IMAGE_GT_MAP_PATH = config_loader.IMAGE_GROUND_TRUTH_MAP_PATH
CROP_BASE_IMAGE_GT_MAP_PATH = config_loader.CROPPED_TO_CROPPED_GT_PATH

_IMAGE_GT_URL_CACHE: Optional[Dict[str, Dict[str, Any]]] = None
_IMAGE_GT_BASENAME_CACHE: Optional[Dict[str, Dict[str, Any]]] = None
_VHR_CROP_URL_CACHE: Optional[Dict[str, Dict[str, Any]]] = None
_VHR_CROP_BASENAME_CACHE: Optional[Dict[str, Dict[str, Any]]] = None
_CROPPED_SIZE_URL_CACHE: Optional[Dict[str, Dict[str, int]]] = None
_CROPPED_SIZE_BASENAME_CACHE: Optional[Dict[str, Dict[str, int]]] = None
_BASE_SIZE_URL_CACHE: Optional[Dict[str, Dict[str, int]]] = None
_BASE_SIZE_BASENAME_CACHE: Optional[Dict[str, Dict[str, int]]] = None


def _iter_json_objects(line: str) -> List[Dict[str, Any]]:
    decoder = json.JSONDecoder()
    idx = 0
    text = line.strip()
    objects: List[Dict[str, Any]] = []
    while idx < len(text):
        obj, end = decoder.raw_decode(text, idx)
        if isinstance(obj, dict):
            objects.append(obj)
        idx = end
        while idx < len(text) and text[idx].isspace():
            idx += 1
    return objects


def _resolve_local_image_path(crop_path: str) -> Optional[Path]:
    # Direct file or file://
    try:
        candidate = Path(crop_path.replace("file://", ""))
        if candidate.exists():
            return candidate
    except Exception:
        pass
    # Try repo-relative path from URL structure.
    try:
        # Accept URLs that include the repo-relative segment after the domain.
        for marker in ("/projects/dea-reasoning/", "/dea-vlm-reasoning/"):
            if marker in crop_path:
                rel = crop_path.split(marker, 1)[1].lstrip("/")
                candidate = REPO_ROOT / rel
                if candidate.exists():
                    return candidate
        parsed_path = Path(crop_path)
        if not parsed_path.is_absolute():
            candidate = REPO_ROOT / parsed_path
            if candidate.exists():
                return candidate
    except Exception:
        return None
    return None


def _download_temp_image(crop_path: str) -> Optional[Path]:
    """
    Best-effort download of a remote crop image so we can render overlays.
    Only supports http/https URLs. Caller is responsible for cleanup.
    """
    if not isinstance(crop_path, str):
        return None
    if config_loader.RUN_ON_LOCAL:
        return None
    lower = crop_path.lower()
    if not (lower.startswith("http://") or lower.startswith("https://")):
        return None
    try:
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=Path(crop_path).suffix or ".png")
        tmp_path = Path(tmp.name)
        tmp.close()
        urllib.request.urlretrieve(crop_path, tmp_path)
        return tmp_path
    except Exception:
        try:
            if "tmp_path" in locals() and tmp_path.exists():
                tmp_path.unlink()
        except Exception:
            pass
    return None


def _canonicalize_url_key(url: str) -> str:
    normalized = config_loader.to_remote_url(url)
    return str(normalized).split("?")[0]


def _ensure_ground_truth_size(ground_truth: Dict[str, Any], image_url: str) -> Dict[str, Any]:
    if not isinstance(ground_truth, dict):
        return ground_truth
    width = ground_truth.get("image_width") or ground_truth.get("width")
    height = ground_truth.get("image_height") or ground_truth.get("height")
    if isinstance(width, (int, float)) and isinstance(height, (int, float)):
        return ground_truth
    meta = describe_image_url(image_url)
    image_size = meta.get("image_size")
    if isinstance(image_size, (list, tuple)) and len(image_size) == 2:
        w, h = image_size
        if isinstance(w, (int, float)) and isinstance(h, (int, float)):
            updated = dict(ground_truth)
            updated.setdefault("image_width", int(round(w)))
            updated.setdefault("image_height", int(round(h)))
            return updated
    alt_size = _lookup_size_from_cropped_mapping(image_url)
    if not alt_size:
        return ground_truth
    w, h = alt_size
    updated = dict(ground_truth)
    updated.setdefault("image_width", int(round(w)))
    updated.setdefault("image_height", int(round(h)))
    return updated


def _parse_size_payload(payload: Any) -> Optional[Dict[str, int]]:
    if not isinstance(payload, dict):
        return None
    w = payload.get("width")
    h = payload.get("height")
    if not (isinstance(w, (int, float)) and isinstance(h, (int, float))):
        return None
    if w <= 0 or h <= 0:
        return None
    return {"width": int(round(w)), "height": int(round(h))}


def _load_cropped_size_index(
    path: Optional[Path] = None,
) -> tuple[Dict[str, Dict[str, int]], Dict[str, Dict[str, int]], Dict[str, Dict[str, int]], Dict[str, Dict[str, int]]]:
    global _CROPPED_SIZE_URL_CACHE, _CROPPED_SIZE_BASENAME_CACHE
    global _BASE_SIZE_URL_CACHE, _BASE_SIZE_BASENAME_CACHE
    if path is None:
        path = config_loader.CROPPED_TO_BASE_PATH
    if (
        path == config_loader.CROPPED_TO_BASE_PATH
        and _CROPPED_SIZE_URL_CACHE is not None
        and _CROPPED_SIZE_BASENAME_CACHE is not None
        and _BASE_SIZE_URL_CACHE is not None
        and _BASE_SIZE_BASENAME_CACHE is not None
    ):
        return (
            _CROPPED_SIZE_URL_CACHE,
            _CROPPED_SIZE_BASENAME_CACHE,
            _BASE_SIZE_URL_CACHE,
            _BASE_SIZE_BASENAME_CACHE,
        )
    cropped_url_map: Dict[str, Dict[str, int]] = {}
    cropped_basename_map: Dict[str, Dict[str, int]] = {}
    base_url_map: Dict[str, Dict[str, int]] = {}
    base_basename_map: Dict[str, Dict[str, int]] = {}
    if path.exists():
        with path.open("r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    entries = _iter_json_objects(line)
                except Exception:
                    continue
                for obj in entries:
                    if not isinstance(obj, dict):
                        continue
                    crop_url = (
                        obj.get("cropped_image_url")
                        or obj.get("cropped_image_path")
                        or obj.get("image_path")
                        or obj.get("image_url")
                    )
                    base_url = obj.get("base_image_url") or obj.get("original_image_path") or obj.get("source_image_url")
                    crop_rect = obj.get("crop_rect") or {}
                    base_size = _parse_size_payload(obj.get("base_image_size") or {})
                    crop_size = _parse_size_payload(crop_rect)
                    if crop_url and crop_size:
                        key = _canonicalize_url_key(str(crop_url))
                        cropped_url_map.setdefault(key, crop_size)
                        name = Path(key).name
                        if name and name not in cropped_basename_map:
                            cropped_basename_map[name] = crop_size
                    if base_url and base_size:
                        key = _canonicalize_url_key(str(base_url))
                        base_url_map.setdefault(key, base_size)
                        name = Path(key).name
                        if name and name not in base_basename_map:
                            base_basename_map[name] = base_size
    if path == config_loader.CROPPED_TO_BASE_PATH:
        _CROPPED_SIZE_URL_CACHE = cropped_url_map
        _CROPPED_SIZE_BASENAME_CACHE = cropped_basename_map
        _BASE_SIZE_URL_CACHE = base_url_map
        _BASE_SIZE_BASENAME_CACHE = base_basename_map
    return cropped_url_map, cropped_basename_map, base_url_map, base_basename_map


def _lookup_size_from_cropped_mapping(image_url: str) -> Optional[tuple[int, int]]:
    if not image_url:
        return None
    cropped_url_map, cropped_basename_map, base_url_map, base_basename_map = _load_cropped_size_index()
    key = _canonicalize_url_key(image_url)
    size = cropped_url_map.get(key)
    if not size:
        size = cropped_basename_map.get(Path(key).name)
    if not size:
        size = base_url_map.get(key)
    if not size:
        size = base_basename_map.get(Path(key).name)
    if not size:
        return None
    return size.get("width"), size.get("height")


def _validate_coco_ground_truth(gt: Dict[str, Any]) -> None:
    if not isinstance(gt, dict):
        raise ValueError("Ground truth must be a dict.")
    if "objects" not in gt or not isinstance(gt.get("objects"), list):
        raise ValueError("Ground truth must contain an 'objects' list.")
    width = gt.get("image_width") or gt.get("width")
    height = gt.get("image_height") or gt.get("height")
    if not (isinstance(width, (int, float)) and isinstance(height, (int, float))):
        raise ValueError("Ground truth must include image_width and image_height.")
    _validate_ground_truth_bounds(_normalize_crop_ground_truth(gt))


def _render_crop_ground_truth(crop_path: str, ground_truth: Dict[str, Any]) -> None:
    if not (ground_truth.get("objects") or []):
        return
    local_path = _resolve_local_image_path(crop_path)
    temp_download: Optional[Path] = None
    if not local_path or not local_path.exists():
        temp_download = _download_temp_image(crop_path)
        local_path = temp_download
    if not local_path or not local_path.exists():
        raise FileNotFoundError(f"Could not locate crop image for rendering: {crop_path}")
    try:
        img = Image.open(local_path).convert("RGB")
    except Exception:
        if temp_download and temp_download.exists():
            temp_download.unlink(missing_ok=True)
        raise

    gt_w = ground_truth.get("image_width")
    gt_h = ground_truth.get("image_height")
    if isinstance(gt_w, (int, float)) and isinstance(gt_h, (int, float)):
        if abs(img.size[0] - int(gt_w)) > 1 or abs(img.size[1] - int(gt_h)) > 1:
            raise ValueError(f"GT size ({gt_w},{gt_h}) does not match image size {img.size} for {crop_path}")
    draw = ImageDraw.Draw(img)

    font = None
    try:
        font = ImageFont.load_default()
    except Exception:
        font = None

    for obj in ground_truth.get("objects") or []:
        bbox = obj.get("bbox") if isinstance(obj, dict) else None
        if not (isinstance(bbox, (list, tuple)) and len(bbox) == 4):
            raise ValueError(f"Invalid bbox in GT for {crop_path}: {bbox}")
        x, y, w, h = bbox
        if w <= 0 or h <= 0:
            raise ValueError(f"Non-positive bbox in GT for {crop_path}: {bbox}")
        x1, y1 = x + w, y + h
        if x < -1e-3 or y < -1e-3 or x1 > img.size[0] + 1e-3 or y1 > img.size[1] + 1e-3:
            raise ValueError(f"Bbox {bbox} spills outside image size {img.size} for {crop_path}")
        color = (255, 0, 0)
        draw.rectangle([(x, y), (x1, y1)], outline=color, width=2)
        label = str(obj.get("category_name") or obj.get("category_id") or "")
        if label and font:
            tx, ty = x + 2, max(0, y - 14)
            try:
                bbox = draw.textbbox((tx, ty), label, font=font)
                if bbox:
                    tw = bbox[2] - bbox[0]
                    th = bbox[3] - bbox[1]
                    draw.rectangle(
                        [(tx - 2, ty - 2), (tx + tw + 2, ty + th + 2)],
                        fill=(0, 0, 0, 160),
                    )
            except Exception:
                pass
            draw.text((tx, ty), label, fill=(255, 255, 255), font=font)

    out_dir = config_loader.IMAGE_CACHE_DIR / "crop_gt_vis"
    out_dir.mkdir(parents=True, exist_ok=True)
    safe_name = Path(crop_path).stem or "crop"
    out_path = out_dir / f"{safe_name}.png"
    img.save(out_path)
    if temp_download and temp_download.exists():
        temp_download.unlink(missing_ok=True)


def _validate_ground_truth_bounds(ground_truth: Dict[str, Any]) -> None:
    w = ground_truth.get("image_width")
    h = ground_truth.get("image_height")
    if not (isinstance(w, (int, float)) and isinstance(h, (int, float)) and w > 0 and h > 0):
        raise ValueError(f"Invalid image dimensions in ground truth: {w}x{h}")
    width = float(w)
    height = float(h)
    for obj in ground_truth.get("objects") or []:
        bbox = obj.get("bbox") if isinstance(obj, dict) else None
        if not (isinstance(bbox, (list, tuple)) and len(bbox) == 4 and all(isinstance(v, (int, float)) for v in bbox)):
            raise ValueError(f"Invalid bbox format in ground truth: {bbox}")
        x, y, bw, bh = bbox
        if bw <= 0 or bh <= 0:
            raise ValueError(f"Non-positive bbox in ground truth: {bbox}")
        if x < -1e-3 or y < -1e-3 or (x + bw) > width + 1e-3 or (y + bh) > height + 1e-3:
            raise ValueError(f"Bbox {bbox} exceeds image bounds {width}x{height}")


def _normalize_crop_ground_truth(ground_truth: Dict[str, Any]) -> Dict[str, Any]:
    """
    If ground_truth carries a crop_rect but bbox coordinates are in the
    source-image frame, translate + clip them to the crop-local frame and set
    image_width/height to the crop size.
    """
    if not isinstance(ground_truth, dict):
        return ground_truth
    crop_rect = ground_truth.get("crop_rect") or {}
    if not isinstance(crop_rect, dict):
        return ground_truth
    cx = crop_rect.get("x")
    cy = crop_rect.get("y")
    cw = crop_rect.get("width")
    ch = crop_rect.get("height")
    if not all(isinstance(v, (int, float)) for v in (cx, cy, cw, ch)):
        return ground_truth
    cx = float(cx)
    cy = float(cy)
    cw = float(cw)
    ch = float(ch)
    if cw <= 0 or ch <= 0:
        return ground_truth

    # Heuristic: if any bbox lies outside the crop box or image_width != crop width,
    # treat coordinates as source-frame and convert to crop-local.
    needs_transform = False
    iw = ground_truth.get("image_width")
    ih = ground_truth.get("image_height")
    if not (isinstance(iw, (int, float)) and isinstance(ih, (int, float))):
        needs_transform = True
    elif abs(float(iw) - cw) > 1e-3 or abs(float(ih) - ch) > 1e-3:
        needs_transform = True

    objects: List[Dict[str, Any]] = []
    for obj in ground_truth.get("objects") or []:
        if not isinstance(obj, dict):
            continue
        bbox = obj.get("bbox")
        if not (isinstance(bbox, (list, tuple)) and len(bbox) == 4):
            continue
        x, y, w, h = [float(v) for v in bbox]
        if w <= 0 or h <= 0:
            continue
        if not needs_transform:
            objects.append(obj)
            continue
        x1 = x + w
        y1 = y + h
        lx0 = max(cx, x)
        ly0 = max(cy, y)
        lx1 = min(cx + cw, x1)
        ly1 = min(cy + ch, y1)
        if lx1 <= lx0 or ly1 <= ly0:
            continue
        partial_sides: List[str] = []
        if x < cx:
            partial_sides.append("left")
        if y < cy:
            partial_sides.append("top")
        if x1 > cx + cw:
            partial_sides.append("right")
        if y1 > cy + ch:
            partial_sides.append("bottom")
        updated = dict(obj)
        updated["bbox"] = [lx0 - cx, ly0 - cy, lx1 - lx0, ly1 - ly0]
        updated["is_partial"] = bool(partial_sides)
        updated["partial_sides"] = partial_sides
        objects.append(updated)

    if not needs_transform:
        return {**ground_truth, "objects": objects}

    return {
        **ground_truth,
        "image_width": int(round(cw)),
        "image_height": int(round(ch)),
        "objects": objects,
    }


def extract_ground_truth_for_url(ground_truth: Any, image_url: str) -> Optional[Dict[str, Any]]:
    """
    Support both single-image ground truth payloads and mappings keyed by URL.
    """
    if not image_url:
        return None
    if isinstance(ground_truth, dict):
        if "objects" in ground_truth and ("image_width" in ground_truth or "width" in ground_truth):
            return ground_truth
        candidate = ground_truth.get(image_url)
        if isinstance(candidate, dict):
            return candidate
    return None


def clip_ground_truth_to_aoi(gt: Any, aoi: List[float]) -> Optional[Dict[str, Any]]:
    """
    Clip ground-truth objects to a normalized AOI [x0,y0,x1,y1] and adjust bboxes to crop coordinates.
    """
    if not isinstance(gt, dict):
        return None
    if not isinstance(aoi, (list, tuple)) or len(aoi) != 4:
        return None
    try:
        x0_rel, y0_rel, x1_rel, y1_rel = [float(v) for v in aoi]
    except Exception:
        return None
    width = gt.get("image_width") or gt.get("width")
    height = gt.get("image_height") or gt.get("height")
    if not (isinstance(width, (int, float)) and isinstance(height, (int, float))):
        return None
    W = float(width)
    H = float(height)
    x0_px, y0_px = x0_rel * W, y0_rel * H
    x1_px, y1_px = x1_rel * W, y1_rel * H
    crop_w = max(1.0, x1_px - x0_px)
    crop_h = max(1.0, y1_px - y0_px)
    objs = gt.get("objects") or []
    new_objs: List[Dict[str, Any]] = []
    for obj in objs:
        bbox = obj.get("bbox") if isinstance(obj, dict) else None
        if not (isinstance(bbox, (list, tuple)) and len(bbox) == 4):
            continue
        ox, oy, ow, oh = bbox
        ox1, oy1 = ox + ow, oy + oh
        ix0 = max(x0_px, ox)
        iy0 = max(y0_px, oy)
        ix1 = min(x1_px, ox1)
        iy1 = min(y1_px, oy1)
        if ix1 <= ix0 or iy1 <= iy0:
            continue
        clipped = {
            **obj,
            "bbox": [
                max(0.0, ix0 - x0_px),
                max(0.0, iy0 - y0_px),
                max(1.0, ix1 - ix0),
                max(1.0, iy1 - iy0),
            ],
        }
        new_objs.append(clipped)
    return {
        "image_width": int(round(crop_w)),
        "image_height": int(round(crop_h)),
        "objects": new_objs,
    }


def record_crop_ground_truth(
    *,
    crop_path: str,
    ground_truth: Dict[str, Any],
) -> None:
    """
    Persist a single mapping {crop_path: clipped_ground_truth} to the shared JSONL file.
    """
    if not crop_path or not ground_truth:
        return
    ground_truth = _normalize_crop_ground_truth(ground_truth)
    if not (ground_truth.get("objects") or []):
        return  # Skip empty crops
    _validate_ground_truth_bounds(ground_truth)
    # Render for debugging; will raise if GT and image sizes are inconsistent.
    _render_crop_ground_truth(crop_path, ground_truth)


def reset_image_gt_cache() -> None:
    global _IMAGE_GT_URL_CACHE, _IMAGE_GT_BASENAME_CACHE
    global _CROPPED_SIZE_URL_CACHE, _CROPPED_SIZE_BASENAME_CACHE
    global _BASE_SIZE_URL_CACHE, _BASE_SIZE_BASENAME_CACHE
    _IMAGE_GT_URL_CACHE = None
    _IMAGE_GT_BASENAME_CACHE = None
    _CROPPED_SIZE_URL_CACHE = None
    _CROPPED_SIZE_BASENAME_CACHE = None
    _BASE_SIZE_URL_CACHE = None
    _BASE_SIZE_BASENAME_CACHE = None


def reset_vhr_crop_cache() -> None:
    global _VHR_CROP_URL_CACHE, _VHR_CROP_BASENAME_CACHE
    _VHR_CROP_URL_CACHE = None
    _VHR_CROP_BASENAME_CACHE = None


def reset_crop_mapping_cache() -> None:
    """
    Clear caches derived from cropped-image mapping files without resetting base GT caches.
    """
    global _VHR_CROP_URL_CACHE, _VHR_CROP_BASENAME_CACHE
    global _CROPPED_SIZE_URL_CACHE, _CROPPED_SIZE_BASENAME_CACHE
    global _BASE_SIZE_URL_CACHE, _BASE_SIZE_BASENAME_CACHE
    _VHR_CROP_URL_CACHE = None
    _VHR_CROP_BASENAME_CACHE = None
    _CROPPED_SIZE_URL_CACHE = None
    _CROPPED_SIZE_BASENAME_CACHE = None
    _BASE_SIZE_URL_CACHE = None
    _BASE_SIZE_BASENAME_CACHE = None


def load_image_gt_index(
    path: Optional[Path] = None,
) -> tuple[Dict[str, Dict[str, Any]], Dict[str, Dict[str, Any]]]:
    global _IMAGE_GT_URL_CACHE, _IMAGE_GT_BASENAME_CACHE
    if path is None:
        path = config_loader.IMAGE_GROUND_TRUTH_MAP_PATH
    if path == BASE_IMAGE_GT_MAP_PATH and _IMAGE_GT_URL_CACHE is not None and _IMAGE_GT_BASENAME_CACHE is not None:
        return _IMAGE_GT_URL_CACHE, _IMAGE_GT_BASENAME_CACHE
    url_map: Dict[str, Dict[str, Any]] = {}
    basename_map: Dict[str, Dict[str, Any]] = {}
    if path.exists():
        with path.open("r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    entries = _iter_json_objects(line)
                except Exception:
                    continue
                for obj in entries:
                    gt = obj.get("ground_truth")
                    image_url = obj.get("image_url") or obj.get("image_path")
                    if not (isinstance(gt, dict) and isinstance(image_url, str) and image_url.strip()):
                        continue
                    key = _canonicalize_url_key(image_url)
                    url_map[key] = gt
                    name = Path(key).name
                    if name and name not in basename_map:
                        basename_map[name] = gt
    if path == BASE_IMAGE_GT_MAP_PATH:
        _IMAGE_GT_URL_CACHE = url_map
        _IMAGE_GT_BASENAME_CACHE = basename_map
    return url_map, basename_map


def load_vhr_crop_index(
    path: Optional[Path] = None,
) -> tuple[Dict[str, Dict[str, Any]], Dict[str, Dict[str, Any]]]:
    global _VHR_CROP_URL_CACHE, _VHR_CROP_BASENAME_CACHE
    if path is None:
        path = config_loader.CROPPED_TO_CROPPED_GT_PATH
    if (
        path == CROP_BASE_IMAGE_GT_MAP_PATH
        and _VHR_CROP_URL_CACHE is not None
        and _VHR_CROP_BASENAME_CACHE is not None
    ):
        return _VHR_CROP_URL_CACHE, _VHR_CROP_BASENAME_CACHE
    url_map: Dict[str, Dict[str, Any]] = {}
    basename_map: Dict[str, Dict[str, Any]] = {}
    if path.exists():
        with path.open("r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    entries = _iter_json_objects(line)
                except Exception:
                    continue
                for entry in entries:
                    crop_url = entry.get("image_url") or entry.get("cropped_image_url") or entry.get("cropped_image_path")
                    if not (isinstance(crop_url, str) and crop_url.strip()):
                        continue
                    key = _canonicalize_url_key(crop_url)
                    url_map[key] = entry
                    name = Path(key).name
                    if name and name not in basename_map:
                        basename_map[name] = entry
    if path == CROP_BASE_IMAGE_GT_MAP_PATH:
        _VHR_CROP_URL_CACHE = url_map
        _VHR_CROP_BASENAME_CACHE = basename_map
    return url_map, basename_map


def append_vhr_crop_mapping(entry: Dict[str, Any], path: Optional[Path] = None) -> None:
    """
    Append a mapping entry to the shared VHR crop mapping file with an exclusive lock.
    """
    if path is None:
        path = config_loader.CROPPED_TO_CROPPED_GT_PATH
    # Validate any embedded ground truth blocks to ensure they are COCO-style.
    for key in ("cropped_ground_truth", "ground_truth"):
        if key in entry and entry[key]:
            _validate_coco_ground_truth(entry[key])
    path.parent.mkdir(parents=True, exist_ok=True)
    fh = None
    try:
        fh = path.open("a+", encoding="utf-8")
        fcntl.flock(fh.fileno(), fcntl.LOCK_EX)
        fh.seek(0, os.SEEK_END)
        fh.write(json.dumps(entry, ensure_ascii=False) + "\n")
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
    if path == CROP_BASE_IMAGE_GT_MAP_PATH:
        reset_vhr_crop_cache()


def resolve_ground_truth_for_image(
    image_url: str,
    *,
    BASE_IMAGE_GT_MAP_PATH: Optional[Path] = None,
    CROP_BASE_IMAGE_GT_MAP_PATH: Optional[Path] = None,
) -> tuple[Dict[str, Any], Dict[str, Any]]:
    """
    Locate ground truth for a given image URL from the image GT map or crop mapping.
    Returns (ground_truth, meta) where meta includes base image context.
    Raises ValueError if not found or inconsistent.
    """
    if not image_url:
        raise ValueError("image_url is required.")
    key = _canonicalize_url_key(image_url)
    base_map_path = BASE_IMAGE_GT_MAP_PATH or config_loader.IMAGE_GROUND_TRUTH_MAP_PATH
    crop_map_path = CROP_BASE_IMAGE_GT_MAP_PATH or config_loader.CROPPED_TO_CROPPED_GT_PATH
    url_map, basename_map = load_image_gt_index(base_map_path)
    gt: Optional[Dict[str, Any]] = None
    if key in url_map:
        gt = url_map[key]
    else:
        name = Path(key).name
        gt = basename_map.get(name)
    if gt:
        gt = _ensure_ground_truth_size(gt, image_url)
        width = gt.get("image_width") or gt.get("width")
        height = gt.get("image_height") or gt.get("height")
        if not (isinstance(width, (int, float)) and isinstance(height, (int, float))):
            raise ValueError(f"Ground truth missing size for {image_url}")
        return gt, {
            "base_image_url": image_url,
            "base_image_width": int(width),
            "base_image_height": int(height),
            "offset_x": 0,
            "offset_y": 0,
            "source_record_id": Path(key).stem,
            "base_annotation_count": len(gt.get("objects") or []),
            "partial_candidate_indexes": [],
            "total_partial_candidates": 0,
            "base_image_path": None,
            "base_image_abspath": None,
        }
    crop_url_map, crop_basename_map = load_vhr_crop_index(crop_map_path)
    entry = crop_url_map.get(key) or crop_basename_map.get(Path(key).name)
    if not entry:
        raise ValueError(
            f"No ground truth found for {image_url}. Ensure it exists in {base_map_path} "
            f"or {crop_map_path}."
        )
    # If the entry already stores the cropped ground truth directly, return it.
    direct_gt = entry.get("ground_truth") or entry.get("cropped_ground_truth")
    if direct_gt:
        direct_gt = _ensure_ground_truth_size(direct_gt, image_url)
        _validate_coco_ground_truth(direct_gt)
        width = direct_gt.get("image_width") or direct_gt.get("width")
        height = direct_gt.get("image_height") or direct_gt.get("height")
        if not (isinstance(width, (int, float)) and isinstance(height, (int, float))):
            raise ValueError(f"Ground truth missing size for {image_url}")
        return direct_gt, {
            "base_image_url": entry.get("base_image_url") or entry.get("image_url") or image_url,
            "base_image_width": int(width),
            "base_image_height": int(height),
            "offset_x": 0,
            "offset_y": 0,
            "source_record_id": entry.get("source_record_id") or entry.get("record_id") or Path(key).stem,
            "base_annotation_count": len(direct_gt.get("objects") or []),
            "partial_candidate_indexes": entry.get("partial_candidate_indexes") or [],
            "total_partial_candidates": entry.get("total_partial_candidates") or 0,
            "base_image_path": entry.get("base_image_path"),
            "base_image_abspath": entry.get("base_image_abspath"),
        }
    base_url = entry.get("base_image_url")
    crop_rect = entry.get("crop_rect") or {}
    base_size = entry.get("base_image_size") or {}
    if not (base_url and crop_rect):
        raise ValueError(f"Incomplete VHR crop mapping for {image_url}")
    base_gt, _ = resolve_ground_truth_for_image(
        base_url,
        BASE_IMAGE_GT_MAP_PATH=base_map_path,
        CROP_BASE_IMAGE_GT_MAP_PATH=crop_map_path,
    )
    base_w = base_size.get("width") or base_gt.get("image_width") or base_gt.get("width")
    base_h = base_size.get("height") or base_gt.get("image_height") or base_gt.get("height")
    if not (isinstance(base_w, (int, float)) and isinstance(base_h, (int, float))):
        raise ValueError(f"Base image size missing for {base_url}")
    x = crop_rect.get("x")
    y = crop_rect.get("y")
    w = crop_rect.get("width")
    h = crop_rect.get("height")
    if not all(isinstance(v, (int, float)) for v in (x, y, w, h)):
        raise ValueError(f"Invalid crop_rect in mapping for {image_url}: {crop_rect}")
    aoi_norm = [x / float(base_w), y / float(base_h), (x + w) / float(base_w), (y + h) / float(base_h)]
    clipped = clip_ground_truth_to_aoi(base_gt, aoi_norm)
    if clipped is None:
        raise ValueError(f"Failed to clip base ground truth for {image_url}")
    meta = {
        "base_image_url": base_url,
        "base_image_width": int(round(base_w)),
        "base_image_height": int(round(base_h)),
        "offset_x": int(round(x)),
        "offset_y": int(round(y)),
        "source_record_id": entry.get("source_record_id") or entry.get("record_id") or Path(key).stem,
        "base_annotation_count": entry.get("annotation_count") or len(base_gt.get("objects") or []),
        "partial_candidate_indexes": entry.get("partial_candidate_indexes") or [],
        "total_partial_candidates": entry.get("total_partial_candidates") or 0,
        "base_image_path": entry.get("base_image_path"),
        "base_image_abspath": entry.get("base_image_abspath"),
    }
    return clipped, meta


def get_crop_ground_truth(crop_path: str) -> Optional[Dict[str, Any]]:
    return None


__all__ = [
    "BASE_IMAGE_GT_MAP_PATH",
    "CROP_BASE_IMAGE_GT_MAP_PATH",
    "clip_ground_truth_to_aoi",
    "get_crop_ground_truth",
    "load_image_gt_index",
    "load_vhr_crop_index",
    "append_vhr_crop_mapping",
    "resolve_ground_truth_for_image",
    "reset_crop_mapping_cache",
    "reset_image_gt_cache",
    "reset_vhr_crop_cache",
    "record_crop_ground_truth",
]
