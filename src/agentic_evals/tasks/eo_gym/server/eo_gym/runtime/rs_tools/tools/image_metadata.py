from __future__ import annotations

import csv
import json
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

from eo_gym.runtime.core.config_loader import EO_GYM_FMOW_CSV, EO_GYM_M4_SAR_CSV, IMAGE_SIZE_AND_GSD_MAPPING_PATH, to_remote_url

@lru_cache(maxsize=1)
def _load_mapping() -> Dict[str, Dict[str, Any]]:
    if not IMAGE_SIZE_AND_GSD_MAPPING_PATH.exists():
        return {}
    try:
        with IMAGE_SIZE_AND_GSD_MAPPING_PATH.open("r", encoding="utf-8") as fh:
            return json.load(fh)
    except Exception:
        return {}


def _coerce_int(value: Any) -> Optional[int]:
    if value is None:
        return None
    try:
        if isinstance(value, (int, float)):
            return int(round(float(value)))
        text = str(value).strip()
        if not text:
            return None
        return int(round(float(text)))
    except Exception:
        return None


def _coerce_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        if isinstance(value, (int, float)):
            return float(value)
        text = str(value).strip()
        if not text:
            return None
        return float(text)
    except Exception:
        return None


@lru_cache(maxsize=1)
def _load_eo_gym_size_index() -> Dict[str, Dict[str, Any]]:
    mapping: Dict[str, Dict[str, Any]] = {}
    mapping_dir = IMAGE_SIZE_AND_GSD_MAPPING_PATH.parent
    source_map = mapping_dir / "masking_source_to_masking_url.csv"
    csv_paths = [EO_GYM_M4_SAR_CSV, EO_GYM_FMOW_CSV, source_map]
    for path in csv_paths:
        if not path.exists():
            continue
        try:
            with path.open("r", encoding="utf-8", newline="") as fh:
                reader = csv.DictReader(fh)
                for row in reader:
                    if not row:
                        continue
                    url = (
                        row.get("image_path")
                        or row.get("masking_url")
                        or row.get("image_url")
                        or row.get("url")
                    )
                    if not url:
                        continue
                    width = _coerce_int(row.get("image_width") or row.get("width"))
                    height = _coerce_int(row.get("image_height") or row.get("height"))
                    gsd = _coerce_float(row.get("gsd"))
                    entry: Dict[str, Any] = {}
                    if width is not None and height is not None:
                        entry["width"] = width
                        entry["height"] = height
                    if gsd is not None:
                        entry["gsd"] = gsd
                    if not entry:
                        continue
                    key = to_remote_url(url)
                    mapping.setdefault(key, entry)
                    name = Path(str(key).split("?", 1)[0]).name
                    if name and name not in mapping:
                        mapping[name] = entry
        except Exception:
            continue
    return mapping


def _find_entry(url: str, mapping: Dict[str, Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    if not url:
        return None
    entry = mapping.get(url)
    if entry:
        return entry
    normalized = to_remote_url(url)
    entry = mapping.get(normalized)
    if entry:
        return entry
    key = Path(str(normalized).split("?")[0]).name
    return mapping.get(key)


def describe_image_url(url: str) -> Dict[str, Any]:
    mapping = _load_mapping()
    entry = _find_entry(url, mapping)
    if not entry:
        entry = _find_entry(url, _load_eo_gym_size_index())
    image_size: Optional[Tuple[int, int]] = None
    if entry:
        width = entry.get("width")
        height = entry.get("height")
        if isinstance(width, int) and isinstance(height, int):
            image_size = (width, height)
    gsd = entry.get("gsd") if entry else None
    return {
        "image_url": url,
        "image_size": image_size,
        "gsd": gsd,
    }
