"""
Helpers for FAIR1M multispectral lookups.

These utilities use a cached CSV index of Sentinel-2 and Landsat
acquisitions instead of rescanning the filesystem on every call.
Capture times are read directly from the CSV.
"""

from __future__ import annotations

from datetime import date, datetime
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import pandas as pd

from eo_gym.runtime.core.config_loader import EO_GYM_FAIR1M_MS_CSV, to_local_path, to_remote_url

DateLike = str
MultispectralEntry = Dict[str, Any]

def _normalize_sensor(sensor: str) -> str:
    sensor_lower = (sensor or "").strip().lower()
    if sensor_lower not in {"s2", "ls"}:
        raise ValueError("Unknown sensor. Use one of: s2, ls.")
    return sensor_lower


def _parse_date(value: DateLike) -> date:
    value_str = str(value).strip()
    try:
        return datetime.strptime(value_str, "%Y-%m-%d").date()
    except ValueError as exc:
        raise ValueError(f"Invalid date format: {value!r}. Expected YYYY-MM-DD.") from exc


def _normalize_date_range(date_range: Optional[Sequence[Optional[DateLike]]]) -> Tuple[Optional[date], Optional[date]]:
    if not date_range:
        return None, None
    start_raw = date_range[0] if len(date_range) > 0 else None
    end_raw = date_range[1] if len(date_range) > 1 else None
    start = _parse_date(start_raw) if start_raw else None
    end = _parse_date(end_raw) if end_raw else None
    return start, end


@lru_cache(maxsize=1)
def _load_index() -> pd.DataFrame:
    if not EO_GYM_FAIR1M_MS_CSV.exists():
        raise FileNotFoundError("Multispectral index file not found.")
    return pd.read_csv(EO_GYM_FAIR1M_MS_CSV)


def _candidate_ms_folder_keys(ms_folder: Union[str, Path]) -> List[str]:
    raw = str(ms_folder or "").strip()
    if raw.endswith("/"):
        raw = raw.rstrip("/")
    if not raw:
        return []
    candidates = {raw}
    if raw.startswith("file://"):
        candidates.add(raw[len("file://") :].rstrip("/"))
    candidates.add(to_remote_url(raw).rstrip("/"))
    candidates.add(to_local_path(raw).rstrip("/"))
    return [c for c in candidates if c]


def _collect_ms_entries(ms_folder: Union[str, Path], sensor: str) -> List[Tuple[date, str, str]]:
    """
    Resolve the region + sensor from the reference ms_folder, then collect all matching entries.
    """
    sensor_key = _normalize_sensor(sensor)
    try:
        entries = _load_index().copy()
    except FileNotFoundError:
        raise ValueError("Multispectral index file not found. Please check your setup.")
    if "folder_path" not in entries.columns:
        raise ValueError("Multispectral index file is missing folder_path entries.")
    filtered: List[Tuple[date, str, str]] = []
    candidates = _candidate_ms_folder_keys(ms_folder)
    if not candidates:
        raise ValueError("Missing multispectral folder reference. Please check the provided parameters.")
    matches = entries[entries["folder_path"].isin(candidates)]
    if matches.empty:
        raise ValueError("Multispectral folder not found. Please check the provided folder path.")
    region_name = str(matches.iloc[0]["region_name"]).strip().lower()
    entries["region_name"] = entries["region_name"].astype(str).str.lower()
    entries["sensor_name"] = entries["sensor_name"].astype(str).str.lower()
    entries["sensor_key"] = entries["sensor_name"].str[:2]
    entries["capture_date"] = pd.to_datetime(entries["capture_time"], errors="coerce").dt.date
    subset = entries[
        (entries["sensor_key"] == sensor_key)
        & (entries["region_name"] == region_name)
    ]
    for _, row in subset.iterrows():
        dt = row["capture_date"]
        if pd.isna(dt):
            continue
        path_str = str(row.get("folder_path") or "")
        marker = "EO_GYM_DATA/"
        if marker in path_str:
            path_str = marker + path_str.split(marker, 1)[1]
        capture_time = str(row.get("capture_time") or "")
        filtered.append((dt, path_str, capture_time))

    filtered.sort(key=lambda item: item[0])
    return filtered


def rebuild_fair1m_ms_index() -> pd.DataFrame:
    """
    Refresh and return FAIR1M multispectral index entries from CSV.
    """
    _load_index.cache_clear()
    return _load_index()


def get_multispectral_image_list(
    ms_folder: Union[str, Path],
    *,
    start_date: DateLike,
    end_date: DateLike,
    sensor: str = "s2",
) -> List[MultispectralEntry]:
    """
    Return all multispectral folders for the given FAIR1M scene (identified by ms_folder) within a date window,
    including capture_time metadata.
    Returns a descriptive message when no folders fall within the date range.
    """
    start, end = _normalize_date_range((start_date, end_date))
    try:
        candidates = _collect_ms_entries(ms_folder, sensor)
    except ValueError as exc:
        return [{"message": str(exc)}]
    filtered = [
        {"ms_folder": path_str, "capture_time": capture_time}
        for dt, path_str, capture_time in candidates
        if (start is None or dt >= start) and (end is None or dt <= end)
    ]
    if not filtered:
        return [{
            "message": (
                "No multispectral folders found for "
                f"{ms_folder!s} between {start_date} and {end_date} (sensor={sensor})."
            )
        }]
    return filtered


def get_previous_multispectral(
    ms_folder: Union[str, Path],
    *,
    reference_date: DateLike,
    sensor: str = "s2",
) -> Optional[MultispectralEntry]:
    """
    Return the closest previous multispectral folder (before reference_date), including capture_time metadata.
    Returns a descriptive message when no prior folder is found.
    """
    ref_dt = _parse_date(reference_date)
    try:
        candidates = _collect_ms_entries(ms_folder, sensor)
    except ValueError as exc:
        return {"message": str(exc)}
    prior = [(dt, path, capture_time) for dt, path, capture_time in candidates if dt < ref_dt]
    if not prior:
        return {
            "message": (
                "No previous multispectral folder found for "
                f"{ms_folder!s} before {reference_date} (sensor={sensor})."
            )
        }
    _, path, capture_time = prior[-1]
    return {"ms_folder": path, "capture_time": capture_time}


def get_next_multispectral(
    ms_folder: Union[str, Path],
    *,
    reference_date: DateLike,
    sensor: str = "s2",
) -> Optional[MultispectralEntry]:
    """
    Return the closest next multispectral folder (after reference_date), including capture_time metadata.
    Returns a descriptive message when no later folder is found.
    """
    ref_dt = _parse_date(reference_date)
    try:
        candidates = _collect_ms_entries(ms_folder, sensor)
    except ValueError as exc:
        return {"message": str(exc)}
    later = [(dt, path, capture_time) for dt, path, capture_time in candidates if dt > ref_dt]
    if not later:
        return {
            "message": (
                "No next multispectral folder found for "
                f"{ms_folder!s} after {reference_date} (sensor={sensor})."
            )
        }
    _, path, capture_time = later[0]
    return {"ms_folder": path, "capture_time": capture_time}


class GetMultispectralList:
    """
    Get several multispectral dataset locations (e.g., Sentinel-2/Landsat) for a given multispectral dataset location.
    Lists multispectral dataset folders covering the scene identified by a reference ms_folder.
    Inputs: ms_folder (str/Path), sensor (str: s2 or ls), start_date/end_date (required str/ISO date).
    Output: List[dict] of multispectral entries with ms_folder and capture_time.
    """

    def __init__(self, device: str = "cpu"):
        self.device = device

    def __call__(
        self,
        ms_folder: str,
        sensor: str,
        start_date: str,
        end_date: str,
    ) -> List[MultispectralEntry]:
        return get_multispectral_image_list(
            ms_folder,
            start_date=start_date,
            end_date=end_date,
            sensor=sensor,
        )


class GetPreviousMultispectral:
    """
    Return the most recent multispectral folder before a reference date.
    Inputs: ms_folder (str/Path), sensor (str: s2 or ls), reference_date (str/ISO date).
    Output: Optional[dict] entry with ms_folder and capture_time from the cached index.
    """

    def __init__(self, device: str = "cpu"):
        self.device = device

    def __call__(self, ms_folder: str, sensor: str, reference_date: str) -> Optional[MultispectralEntry]:
        return get_previous_multispectral(
            ms_folder,
            reference_date=reference_date,
            sensor=sensor,
        )


class GetNextMultispectral:
    """
    Return the next multispectral folder after a reference date.
    Inputs: ms_folder (str/Path), sensor (str: s2 or ls), reference_date (str/ISO date).
    Output: Optional[dict] entry with ms_folder and capture_time from the cached index.
    """

    def __init__(self, device: str = "cpu"):
        self.device = device

    def __call__(self, ms_folder: str, sensor: str, reference_date: str) -> Optional[MultispectralEntry]:
        return get_next_multispectral(
            ms_folder,
            reference_date=reference_date,
            sensor=sensor,
        )


__all__ = [
    "GetMultispectralList",
    "GetPreviousMultispectral",
    "GetNextMultispectral",
    "get_multispectral_image_list",
    "get_previous_multispectral",
    "get_next_multispectral",
    "rebuild_fair1m_ms_index",
]
