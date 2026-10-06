"""
Helpers for optical image lookups.

These utilities use the eo_gym_fmow.csv index to avoid repeatedly crawling
`DATA/fmow`. Capture dates/times are read directly from the CSV.
"""

from __future__ import annotations

from datetime import date, datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

from eo_gym.runtime.core.config_loader import EO_GYM_FMOW_CSV, RUN_ON_LOCAL, to_local_path, to_remote_url
from eo_gym.runtime.rs_tools.tools.image_metadata import describe_image_url
from eo_gym.runtime.rs_tools.utils import check_file_reference

DateLike = str
OpticalImageEntry = Dict[str, Any]

_FMOW_INDEX_CACHE: Optional[Dict[str, Any]] = None


def _annotate_image_with_metadata(image_url: str) -> Dict[str, Any]:
    resolved_url = to_remote_url(image_url)
    output_url = to_local_path(resolved_url) if RUN_ON_LOCAL else image_url
    entry: Dict[str, Any] = {"image_url": output_url}
    size_from_index: Optional[Tuple[int, int]] = None
    capture_time: Optional[str] = None
    try:
        index = _load_fmow_index()
    except FileNotFoundError:
        index = None
    if index:
        size_from_index = index.get("by_url_size", {}).get(resolved_url)
        if not size_from_index:
            size_from_index = index.get("by_basename_size", {}).get(Path(resolved_url).name)
        capture_time = index.get("by_url_capture_time", {}).get(resolved_url)
        if not capture_time:
            capture_time = index.get("by_basename_capture_time", {}).get(Path(resolved_url).name)
    metadata = describe_image_url(resolved_url)
    if size_from_index:
        entry["image_size"] = [size_from_index[0], size_from_index[1]]
        entry["image_width"] = size_from_index[0]
        entry["image_height"] = size_from_index[1]
    else:
        image_size = metadata.get("image_size")
        if image_size:
            try:
                width = int(image_size[0])
                height = int(image_size[1])
                entry["image_size"] = [width, height]
                entry["image_width"] = width
                entry["image_height"] = height
            except (TypeError, ValueError, IndexError):
                pass
    gsd = metadata.get("gsd")
    if gsd is not None:
        try:
            entry["gsd"] = float(gsd)
        except (TypeError, ValueError):
            pass
    if capture_time:
        entry["capture_time"] = str(capture_time)
    return entry


def _parse_date(value: DateLike) -> date:
    if not isinstance(value, str):
        raise ValueError(f"Invalid date type: {type(value).__name__}. Expected YYYY-MM-DD string.")
    value_str = value.strip()
    try:
        return datetime.strptime(value_str, "%Y-%m-%d").date()
    except ValueError as exc:
        raise ValueError(f"Invalid date format: {value!r}. Expected YYYY-MM-DD.") from exc


def _load_fmow_index() -> Dict[str, Any]:
    global _FMOW_INDEX_CACHE
    if _FMOW_INDEX_CACHE is not None:
        return _FMOW_INDEX_CACHE
    if not EO_GYM_FMOW_CSV.exists():
        raise FileNotFoundError("Optical image index file not found.")

    by_url: Dict[str, str] = {}
    by_basename: Dict[str, str] = {}
    by_region: Dict[str, List[Tuple[date, str]]] = {}
    by_url_size: Dict[str, Tuple[int, int]] = {}
    by_basename_size: Dict[str, Tuple[int, int]] = {}
    by_url_capture_time: Dict[str, str] = {}
    by_basename_capture_time: Dict[str, str] = {}

    df = pd.read_csv(EO_GYM_FMOW_CSV, dtype=str)
    df = df.dropna(subset=["image_path", "region_name", "capture_time"]).copy()
    df["image_path"] = df["image_path"].astype(str).str.strip()
    df["region_name"] = df["region_name"].astype(str).str.strip()
    df["capture_time"] = df["capture_time"].astype(str).str.strip()
    df = df[(df["image_path"] != "") & (df["region_name"] != "") & (df["capture_time"] != "")]
    df["capture_date"] = pd.to_datetime(df["capture_time"], errors="coerce").dt.date
    df = df[df["capture_date"].notna()]
    if "image_width" in df.columns and "image_height" in df.columns:
        df["image_width"] = pd.to_numeric(df["image_width"], errors="coerce")
        df["image_height"] = pd.to_numeric(df["image_height"], errors="coerce")
    else:
        df["image_width"] = pd.NA
        df["image_height"] = pd.NA

    for _, row in df.iterrows():
        image_url = row["image_path"]
        region_name = row["region_name"]
        capture_date = row["capture_date"]
        capture_time = row["capture_time"]
        width_raw = row.get("image_width")
        height_raw = row.get("image_height")
        width = int(width_raw) if pd.notna(width_raw) else None
        height = int(height_raw) if pd.notna(height_raw) else None
        by_url[image_url] = region_name
        by_basename[Path(image_url).name] = region_name
        by_region.setdefault(region_name, []).append((capture_date, image_url))
        by_url_capture_time[image_url] = capture_time
        by_basename_capture_time[Path(image_url).name] = capture_time
        if width is not None and height is not None:
            by_url_size[image_url] = (width, height)
            by_basename_size[Path(image_url).name] = (width, height)

    for entries in by_region.values():
        entries.sort(key=lambda item: (item[0], item[1]))

    _FMOW_INDEX_CACHE = {
        "by_url": by_url,
        "by_basename": by_basename,
        "by_region": by_region,
        "by_url_size": by_url_size,
        "by_basename_size": by_basename_size,
        "by_url_capture_time": by_url_capture_time,
        "by_basename_capture_time": by_basename_capture_time,
    }
    return _FMOW_INDEX_CACHE


def _resolve_region_name(optical_image_path: str) -> Optional[str]:
    index = _load_fmow_index()
    key = to_remote_url(optical_image_path)
    region = index["by_url"].get(key)
    if not region:
        region = index["by_basename"].get(Path(key).name)
    return region


def get_optical_image_list(
    optical_image_path: str,
    *,
    start_date: DateLike,
    end_date: DateLike,
) -> List[OpticalImageEntry]:
    """
    Return all RGB images in the same region as the provided image, along
    with capture time and spatial resolution metadata when available.
    """
    error = check_file_reference(optical_image_path, label="image", strict=True)
    if error:
        return [{"message": error}]
    try:
        region_name = _resolve_region_name(optical_image_path)
    except FileNotFoundError as exc:
        return [{"message": str(exc)}]
    if not region_name:
        return [{"message": f"No index entry found for {optical_image_path!s}."}]
    try:
        start = _parse_date(start_date)
        end = _parse_date(end_date)
    except ValueError as exc:
        return [{"message": str(exc)}]
    if start > end:
        return [{"message": f"start_date must be <= end_date (got {start_date!r}..{end_date!r})."}]
    entries = _load_fmow_index()["by_region"].get(region_name, [])
    if not entries:
        return [{"message": f"No images found for region {region_name}."}]
    filtered = [u for dt, u in entries if start <= dt <= end]
    if not filtered:
        return [
            {
                "message": (
                    "No images found for "
                    f"{optical_image_path} between {start_date} and {end_date}. Please use get_previous_optical_image or get_next_optical_image if want to access the relative image earlier or later than the given image."
                )
            }
        ]
    return [_annotate_image_with_metadata(u) for u in filtered]


def _find_neighbor(
    optical_image_path: str,
    reference_date: DateLike,
    *,
    direction: str,
) -> Optional[OpticalImageEntry]:
    error = check_file_reference(optical_image_path, label="image", strict=True)
    if error:
        return {"message": error}
    try:
        region_name = _resolve_region_name(optical_image_path)
    except FileNotFoundError as exc:
        return {"message": str(exc)}
    if not region_name:
        return {"message": f"No index entry found for {optical_image_path!s}."}
    entries = _load_fmow_index()["by_region"].get(region_name, [])
    if not entries:
        return {"message": f"No images found for region {region_name}."}
    try:
        ref_date = _parse_date(reference_date)
    except ValueError as exc:
        return {"message": str(exc)}
    if direction == "previous":
        prior = [u for dt, u in entries if dt < ref_date]
        if not prior:
            return {
                "message": (
                    "No previous image found for "
                    f"{optical_image_path} before {reference_date}."
                )
            }
        return _annotate_image_with_metadata(prior[-1])
    later = [u for dt, u in entries if dt > ref_date]
    if not later:
        return {
            "message": (
                "No next image found for "
                f"{optical_image_path} after {reference_date}."
            )
        }
    return _annotate_image_with_metadata(later[0])


def get_previous_optical_image(optical_image_path: str, reference_date: DateLike) -> Optional[OpticalImageEntry]:
    """
    Return the closest earlier RGB image URL to the given image, including capture time
    and optional spatial metadata when available.
    """
    return _find_neighbor(optical_image_path, reference_date, direction="previous")


def get_next_optical_image(optical_image_path: str, reference_date: DateLike) -> Optional[OpticalImageEntry]:
    """
    Return the closest later RGB image URL to the given image, including capture time
    and optional spatial metadata when available.
    """
    return _find_neighbor(optical_image_path, reference_date, direction="next")


class GetOpticalImageList:
    """
    List RGB images co-located with the provided frame, ordered by timestamp or frame index. Only return lists optical images without analyze content.
    Inputs: optical_image_path (str/Path), start_date/end_date (required YYYY-MM-DD).
    Output: List of entries containing each image URL, capture time, and optional size/GSD metadata.
    """

    def __init__(self, device: str = "cpu"):
        self.device = device

    def __call__(self, optical_image_path: str, start_date: str, end_date: str) -> List[OpticalImageEntry]:
        return get_optical_image_list(optical_image_path, start_date=start_date, end_date=end_date)


class GetPreviousOpticalImage:
    """
    Return the closest earlier RGB image URL to the given image.
    Inputs: optical_image_path (str/Path), reference_date (YYYY-MM-DD).
    Output: Optional entry with image URL, capture time, and size/GSD metadata.
    """

    def __init__(self, device: str = "cpu"):
        self.device = device

    def __call__(self, optical_image_path: str, reference_date: str) -> Optional[OpticalImageEntry]:
        return get_previous_optical_image(optical_image_path, reference_date)


class GetNextOpticalImage:
    """
    Return the closest later RGB image URL to the given image.
    Inputs: optical_image_path (str/Path), reference_date (YYYY-MM-DD).
    Output: Optional entry with image URL, capture time, and size/GSD metadata.
    """

    def __init__(self, device: str = "cpu"):
        self.device = device

    def __call__(self, optical_image_path: str, reference_date: str) -> Optional[OpticalImageEntry]:
        return get_next_optical_image(optical_image_path, reference_date)


__all__ = [
    "GetOpticalImageList",
    "GetPreviousOpticalImage",
    "GetNextOpticalImage",
    "get_optical_image_list",
    "get_previous_optical_image",
    "get_next_optical_image",
]
