"""
Shared helpers for multispectral band loading and AOI handling.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import List, Optional, Sequence, Union

import numpy as np
import rasterio
import xarray as xr

from eo_gym.runtime.core.config_loader import PUBLIC_DATA_BASE_URL, REPO_ROOT, RUN_ON_LOCAL, to_local_path


def validate_aoi(aoi: List[float]) -> List[float]:
    if not isinstance(aoi, (list, tuple)) or len(aoi) != 4:
        raise ValueError("aoi must be a list of four normalized coordinates [x0, y0, x1, y1].")
    vals = [float(v) for v in aoi]
    x0, y0, x1, y1 = vals
    if not (0 <= x0 < 1 and 0 <= y0 < 1 and 0 <= x1 <= 1 and 0 <= y1 <= 1):
        raise ValueError("aoi values must lie within [0,1].")
    if x1 <= x0 or y1 <= y0:
        raise ValueError("aoi must satisfy x1 > x0 and y1 > y0.")
    return vals


def band_variants(band_name: str) -> List[str]:
    variants = [band_name]
    if band_name.startswith("B"):
        suffix = band_name[1:]
        if suffix.isdigit():
            variants.append(f"B{int(suffix):02d}")
    return list(dict.fromkeys(variants))


def band_file_candidates(band_name: str, sensor_key: Optional[str] = None) -> List[str]:
    if sensor_key == "ls":
        prefixes = ["SR_", ""]
    elif sensor_key == "s2":
        prefixes = [""]
    else:
        prefixes = ["SR_", ""]
    candidates: List[str] = []
    for variant in band_variants(band_name):
        for prefix in prefixes:
            candidates.append(f"{prefix}{variant}.tif")
    return candidates


def resolve_ms_folder(ms_folder: Union[str, Path]) -> Path:
    folder_str = str(ms_folder)
    if RUN_ON_LOCAL:
        folder_str = to_local_path(folder_str)
    if folder_str.startswith("http"):
        base = (PUBLIC_DATA_BASE_URL or "").rstrip("/")
        if base and folder_str.startswith(base):
            remainder = folder_str[len(base) :].lstrip("/")
            return (REPO_ROOT / remainder).resolve()
    return Path(folder_str).expanduser().resolve()


def is_netcdf_path(path_str: str) -> bool:
    lowered = path_str.lower()
    return lowered.endswith((".nc", ".nc4", ".netcdf"))


def select_band_from_dataset(ds: xr.Dataset, band_name: str) -> np.ndarray:
    if "ms_crop" not in ds or "band" not in ds:
        raise ValueError("NetCDF missing required variables (ms_crop/band).")
    if band_name not in ds["band"].values:
        raise ValueError(f"Band '{band_name}' not found in NetCDF; ensure it was included during cropping.")
    arr = ds["ms_crop"].sel(band=band_name).values
    return np.asarray(arr, dtype=np.float32)


def select_band_from_dataset_candidates(ds: xr.Dataset, candidates: Sequence[str]) -> Optional[np.ndarray]:
    if "ms_crop" not in ds or "band" not in ds:
        raise ValueError("NetCDF missing required variables (ms_crop/band).")
    for name in candidates:
        if name in ds["band"].values:
            arr = ds["ms_crop"].sel(band=name).values
            return np.asarray(arr, dtype=np.float32)
    return None


@lru_cache(maxsize=128)
def _load_band_from_folder_cached(
    ms_folder_key: str,
    band_name: str,
    sensor_key: Optional[str],
) -> Optional[np.ndarray]:
    ms_dir = resolve_ms_folder(ms_folder_key)
    if not ms_dir.exists():
        return None
    for name in band_file_candidates(band_name, sensor_key):
        candidate = ms_dir / name
        if candidate.exists():
            with rasterio.open(candidate) as ds:
                arr = ds.read(1)
            return np.asarray(arr, dtype=np.float32)
    return None


def load_band_from_folder(
    ms_folder: Union[str, Path],
    band_name: str,
    sensor_key: Optional[str] = None,
    *,
    raise_missing: bool = True,
) -> Optional[np.ndarray]:
    ms_folder_key = str(ms_folder)
    arr = _load_band_from_folder_cached(ms_folder_key, band_name, sensor_key)
    if arr is not None:
        # Return a copy so callers cannot mutate cached tensors.
        return np.array(arr, dtype=np.float32, copy=True)

    ms_dir = resolve_ms_folder(ms_folder_key)
    if not ms_dir.exists():
        if raise_missing:
            raise FileNotFoundError(f"Multispectral folder not found: {ms_dir}")
        return None
    if raise_missing:
        raise FileNotFoundError(f"Band '{band_name}' not found under {ms_dir}")
    return None


def select_band_from_folder(
    ms_folder: Union[str, Path],
    candidates: Sequence[str],
    sensor_key: Optional[str] = None,
) -> Optional[np.ndarray]:
    for band_name in candidates:
        arr = load_band_from_folder(ms_folder, band_name, sensor_key, raise_missing=False)
        if arr is not None:
            return arr
    return None


__all__ = [
    "band_file_candidates",
    "band_variants",
    "is_netcdf_path",
    "load_band_from_folder",
    "resolve_ms_folder",
    "select_band_from_dataset",
    "select_band_from_dataset_candidates",
    "select_band_from_folder",
    "validate_aoi",
]
