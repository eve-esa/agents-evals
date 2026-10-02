"""
Download an image (optical or SAR), crop by a relative AOI, and save the cropped file.

Inputs:
  - image_url (str): HTTP/HTTPS URL to an optical/SAR image.
  - aoi (List[float]): [x0, y0, x1, y1] normalized to [0,1]; x1>x0, y1>y0.

Output:
  - Dict with keys:
      - cropped_path (str): local path to the cropped image.
      - width (int), height (int): dimensions of the cropped image.
      - bbox_px (List[int]): pixel bbox used for cropping.
      - aoi_norm (List[float]): the normalized AOI used.
      - input_image (Dict): {image_url, image_size, gsd} metadata for the source image.
"""

from __future__ import annotations

import uuid
import csv
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Optional

import requests
from PIL import Image
from urllib.parse import urlparse

from eo_gym.runtime.rs_tools.tools.image_cache_paths import cache_local_path
from eo_gym.runtime.rs_tools.tools.image_cache_logger import log_image_event
from eo_gym.runtime.rs_tools.tools.cropped_to_base_mapping import append_cropped_to_base_mapping, resolve_base_context
from eo_gym.runtime.rs_tools.tools.ground_truth_cache import (
    append_vhr_crop_mapping,
    clip_ground_truth_to_aoi,
    record_crop_ground_truth,
)
from eo_gym.runtime.rs_tools.tools.image_metadata import describe_image_url
from eo_gym.runtime.rs_tools.utils import check_file_reference

import eo_gym.runtime.core.config_loader as config_loader

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


def _relative_to_repo(path: Path) -> str:
    try:
        return str(path.relative_to(config_loader.REPO_ROOT))
    except ValueError:
        return str(path)


def _resolve_input_image(image_url: str) -> tuple[Path, bool, str]:
    """
    Resolve input image URL/path to a local file.

    Returns (local_path, cleanup_needed, source_name).
    """
    raw = str(image_url)
    if config_loader.RUN_ON_LOCAL:
        raw = config_loader.to_local_path(raw)
    source_name = Path(raw.split("?")[0].split("/")[-1]).stem or "image"
    if raw.startswith(("http://", "https://")):
        resp = requests.get(raw, timeout=20)
        config_loader.IMAGE_CACHE_DIR.mkdir(parents=True, exist_ok=True)
        tmp_in = config_loader.IMAGE_CACHE_DIR / f"tmp_{source_name}_{uuid.uuid4().hex}"
        tmp_in.write_bytes(resp.content)
        return tmp_in, True, source_name

    candidate = Path(raw.replace("file://", ""))
    if not candidate.is_absolute():
        candidate = Path(config_loader.to_local_path(str(candidate)))
    if not candidate.exists():
        raise FileNotFoundError(f"Input image not found: {image_url}")
    return candidate, False, source_name


class CropOpticalOrSarImage:
    """
    Crop an optical or SAR image by a normalized AOI and return the saved path.
    Downloads and crops an optical or SAR image to the AOI; returns local crop path.
    Inputs: image_url (str), aoi (List[float] normalized 0-1 [x0, y0, x1, y1]).
    Output: cropped_path (str) pointing to the saved PNG on disk.
    """

    def __init__(self, device: str = "cpu"):
        self.device = device

    def __call__(self, image_url: str, aoi: List[float]) -> Dict:
        if not image_url:
            raise ValueError("image_url must be a non-empty string or path.")
        image_url = str(image_url)
        error = check_file_reference(image_url, label="image", strict=True)
        if error:
            return {"message": error}
        if config_loader.RUN_ON_LOCAL:
            image_url = config_loader.to_local_path(image_url)

        # Accept stringified AOIs (e.g., "[0.1,0.2,0.9,0.9]") and coerce to list.
        if isinstance(aoi, str):
            parsed = None
            try:
                import json

                parsed = json.loads(aoi)
            except Exception:
                try:
                    import ast

                    parsed = ast.literal_eval(aoi)
                except Exception:
                    parsed = None
            if isinstance(parsed, (list, tuple)):
                aoi = list(parsed)

        if not isinstance(aoi, (list, tuple)) or len(aoi) != 4:
            raise ValueError("aoi must be a list of four relative coordinates [x0, y0, x1, y1].")
        try:
            x0_rel, y0_rel, x1_rel, y1_rel = [float(v) for v in aoi]
        except Exception:
            raise ValueError("aoi values must be numeric.")
        if not (0 <= x0_rel < 1 and 0 <= y0_rel < 1 and 0 <= x1_rel <= 1 and 0 <= y1_rel <= 1):
            raise ValueError("aoi values must be in [0,1].")
        if x1_rel <= x0_rel or y1_rel <= y0_rel:
            raise ValueError("aoi must satisfy x1>x0 and y1>y0.")

        tmp_in, cleanup_tmp, source_name = _resolve_input_image(image_url)
        keep_base_local = "VHR_Zoom" in str(image_url)

        cropped_bbox_px: Optional[List[int]] = None
        clipped_gt: Optional[Dict[str, Any]] = None
        try:
            with Image.open(tmp_in) as img:
                w, h = img.size
                base_ctx = resolve_base_context(
                    image_url,
                    (w, h),
                    require_ground_truth=UPDATE_CROP_GT,
                    default_source_id=source_name,
                )
                base_image_url = base_ctx.get("base_image_url") or image_url
                if config_loader.RUN_WITH_MASKING:
                    base_image_url = _resolve_masking_url(base_image_url)
                base_rect = base_ctx.get("crop_rect") or {"x": 0, "y": 0, "width": w, "height": h}
                base_size = base_ctx.get("base_image_size") or {"width": w, "height": h}
                base_gt = base_ctx.get("base_ground_truth") if UPDATE_CROP_GT else None
                base_w = base_size.get("width") or w
                base_h = base_size.get("height") or h

                x0_base = max(
                    0,
                    min(int(round(base_rect.get("x", 0) + x0_rel * base_rect.get("width", w))), int(base_w) - 1),
                )
                y0_base = max(
                    0,
                    min(int(round(base_rect.get("y", 0) + y0_rel * base_rect.get("height", h))), int(base_h) - 1),
                )
                x1_base = max(
                    x0_base + 1,
                    min(int(round(base_rect.get("x", 0) + x1_rel * base_rect.get("width", w))), int(base_w)),
                )
                y1_base = max(
                    y0_base + 1,
                    min(int(round(base_rect.get("y", 0) + y1_rel * base_rect.get("height", h))), int(base_h)),
                )

                x0 = max(0, min(int(round(x0_rel * w)), w - 1))
                y0 = max(0, min(int(round(y0_rel * h)), h - 1))
                x1 = max(x0 + 1, min(int(round(x1_rel * w)), w))
                y1 = max(y0 + 1, min(int(round(y1_rel * h)), h))
                cropped = img.crop((x0, y0, x1, y1))
                # Prefer source image format; fall back to PNG. Avoid using the builtin `format` function.
                out_format = (img.format or "PNG")
                out_format = out_format.upper()
                suffix = f".{out_format.lower()}"
                out_path = config_loader.IMAGE_CACHE_DIR / f"crop_{source_name}_{uuid.uuid4().hex}{suffix}"
                cropped.save(out_path, format=out_format)
                cropped_bbox_px = [x0, y0, x1, y1]
                if UPDATE_CROP_GT and base_gt:
                    aoi_norm_base = [
                        x0_base / float(base_w),
                        y0_base / float(base_h),
                        x1_base / float(base_w),
                        y1_base / float(base_h),
                    ]
                    clipped_gt = clip_ground_truth_to_aoi(base_gt, aoi_norm_base)
                    if clipped_gt is None:
                        raise ValueError(f"Failed to clip ground truth for {image_url}")
                crop_w = x1 - x0
                crop_h = y1 - y0
                if clipped_gt is not None:
                    cg_w = clipped_gt.get("image_width") or clipped_gt.get("width")
                    cg_h = clipped_gt.get("image_height") or clipped_gt.get("height")
                    if isinstance(cg_w, (int, float)) and isinstance(cg_h, (int, float)):
                        if int(round(cg_w)) != crop_w or int(round(cg_h)) != crop_h:
                            raise ValueError(
                                f"Cropped GT size ({cg_w},{cg_h}) does not match crop size ({crop_w},{crop_h}) for {image_url}"
                            )
                    else:
                        clipped_gt["image_width"] = crop_w
                        clipped_gt["image_height"] = crop_h
        finally:
            if cleanup_tmp and not keep_base_local:
                try:
                    tmp_in.unlink()
                except Exception:
                    pass

        cached_path = cache_local_path(out_path)
        cropped_local_path = None
        if keep_base_local and Path(out_path).exists():
            cropped_local_path = _relative_to_repo(out_path)
        cropped_path_field = cropped_local_path or str(cached_path)
        log_image_event(
            "crop_saved",
            {
                "source_image_url": image_url,
                "cropped_path": cached_path,
                "aoi_norm": [x0_rel, y0_rel, x1_rel, y1_rel],
                "crop_bbox_px": cropped_bbox_px or [],
                "width": int((cropped_bbox_px or [0, 0, 0, 0])[2] - (cropped_bbox_px or [0, 0, 0, 0])[0]),
                "height": int((cropped_bbox_px or [0, 0, 0, 0])[3] - (cropped_bbox_px or [0, 0, 0, 0])[1]),
            },
        )
        crop_path_for_gt = cropped_path_field
        if UPDATE_CROP_GT and clipped_gt is not None:
            record_crop_ground_truth(
                crop_path=crop_path_for_gt,
                ground_truth=clipped_gt,
            )

        append_cropped_to_base_mapping(
            str(cached_path),
            str(base_image_url),
            crop_rect={"x": int(x0_base), "y": int(y0_base), "width": int(x1_base - x0_base), "height": int(y1_base - y0_base)},
            base_image_size={"width": int(base_w), "height": int(base_h)},
        )

        # Persist mapping entry to enable navigation tools.
        if UPDATE_CROP_GT and clipped_gt is not None and cropped_bbox_px:
            x0_px, y0_px, x1_px, y1_px = cropped_bbox_px
            crop_w = x1_px - x0_px
            crop_h = y1_px - y0_px
            base_image_width = base_w
            base_image_height = base_h
            if base_image_width <= 0 or base_image_height <= 0:
                raise ValueError(f"Missing base image size for {base_image_url}")

            source_record_id = Path(urlparse(base_image_url).path).stem or source_name
            record_id = f"{source_record_id}_crop_{uuid.uuid4().hex[:6]}"
            entry = {
                "dataset": "cropped_image",
                "record_id": record_id,
                "image_url": str(cached_path),
                "ground_truth": clipped_gt,
                "base_image_url": str(base_image_url),
                "crop_rect": {"x": int(x0_base), "y": int(y0_base), "width": int(x1_base - x0_base), "height": int(y1_base - y0_base)},
                "base_image_size": {"width": int(base_image_width), "height": int(base_image_height)},
            }
            append_vhr_crop_mapping(entry)
        return {
            "cropped_path": cached_path,
            "width": int((cropped_bbox_px or [0, 0, 0, 0])[2] - (cropped_bbox_px or [0, 0, 0, 0])[0]),
            "height": int((cropped_bbox_px or [0, 0, 0, 0])[3] - (cropped_bbox_px or [0, 0, 0, 0])[1]),
            "bbox_px": cropped_bbox_px or [],
            "aoi_norm": [x0_rel, y0_rel, x1_rel, y1_rel],
            "input_image": describe_image_url(image_url),
        }
