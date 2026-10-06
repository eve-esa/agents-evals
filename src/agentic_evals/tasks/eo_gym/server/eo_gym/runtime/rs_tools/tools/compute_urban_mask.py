"""
Compute an urban/built-up mask from a multispectral crop NetCDF or band folder using NDBI.

Inputs:
  - multispectral_image_path (str/Path): NetCDF saved by load_multispectral_image or a folder URL/path.
  - sensor (str): "ls" for Landsat 8/9 or "s2" for Sentinel-2.

Output:
  - str: local path to the saved PNG mask.
"""

from __future__ import annotations

import uuid
from pathlib import Path
from typing import Union

import numpy as np
import xarray as xr
from PIL import Image

from eo_gym.runtime.rs_tools.tools.image_cache_paths import cache_local_path
from eo_gym.runtime.rs_tools.utils import (
    check_file_reference,
    check_multispectral_folder_reference,
    download_file,
)
from eo_gym.runtime.rs_tools.tools.multispectral_shared import (
    is_netcdf_path,
    load_band_from_folder,
    select_band_from_dataset,
)
import eo_gym.runtime.core.config_loader as config_loader

# Common NDBI thresholds for urban/built-up detection by sensor family.
NDBI_THRESHOLDS = {
    "ls": 0.1,  # Landsat 8/9
    "s2": 0.1,  # Sentinel-2
}

# Band selection per sensor: (SWIR1, NIR).
SENSOR_BANDS = {
    "ls": ("B6", "B5"),
    "s2": ("B11", "B8"),
}

def compute_urban_mask_by_multispectral(multispectral_image_path: Union[str, Path], sensor: str) -> str:
    """
    Compute a binary urban/built-up mask from a multispectral crop NetCDF or band folder using NDBI.
    Inputs: multispectral_image_path (str/Path folder or NetCDF), sensor (ls|s2).
    Output: path (str) to PNG mask 
    """
    sensor_key = (sensor or "").strip().lower()
    if sensor_key not in SENSOR_BANDS:
        raise ValueError("sensor must be 'ls' (Landsat 8/9) or 's2' (Sentinel-2).")

    swir1_band, nir_band = SENSOR_BANDS[sensor_key]
    path_str = str(multispectral_image_path)
    if is_netcdf_path(path_str):
        error = check_file_reference(path_str, label="file", strict=True)
        if error:
            return error
        local_path = download_file(path_str, config_loader.IMAGE_CACHE_DIR)
        local_str = str(local_path)
        if local_str.startswith(("http://", "https://")):
            raise FileNotFoundError(
                f"NetCDF download failed or file unavailable: {local_str}"
            )
        nc_file = Path(local_str).expanduser()
        if not nc_file.exists():
            raise FileNotFoundError(f"NetCDF file not found: {nc_file}")
        with xr.open_dataset(nc_file) as ds:
            ds = ds.load()
        swir1 = select_band_from_dataset(ds, swir1_band)
        nir = select_band_from_dataset(ds, nir_band)
        name_stem = nc_file.stem
    else:
        folder_error = check_multispectral_folder_reference(path_str)
        if folder_error:
            return folder_error
        swir1 = load_band_from_folder(multispectral_image_path, swir1_band, sensor_key)
        nir = load_band_from_folder(multispectral_image_path, nir_band, sensor_key)
        name_stem = Path(str(multispectral_image_path).rstrip("/")).name

    ndbi = (swir1 - nir) / (swir1 + nir + 1e-6)
    ndbi = np.nan_to_num(ndbi, nan=-1.0, posinf=-1.0, neginf=-1.0)

    threshold = NDBI_THRESHOLDS[sensor_key]
    urban_mask = (ndbi > threshold).astype(np.uint8) * 255

    mask_img = Image.fromarray(urban_mask)
    config_loader.IMAGE_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    out_name = f"urban_mask_{name_stem}_{sensor_key}_{uuid.uuid4().hex}.png"
    out_path = config_loader.IMAGE_CACHE_DIR / out_name
    mask_img.save(out_path, format="PNG")

    return cache_local_path(out_path)


class ComputeUrbanMaskByMultispectral:
    """
    Compute an urban mask from a multispectral crop NetCDF using NDBI and sensor thresholds.
    Derives an urban mask from NDBI thresholds; returns mask metadata/arrays.
    Inputs: multispectral_image_path (str/Path), sensor (ls|s2).
    Output: mask_path (str) to PNG 
    """

    def __init__(self, device: str = "cpu"):
        self.device = device

    def __call__(self, multispectral_image_path: Union[str, Path], sensor: str) -> str:
        return compute_urban_mask_by_multispectral(multispectral_image_path, sensor)


__all__ = [
    "compute_urban_mask_by_multispectral",
    "ComputeUrbanMaskByMultispectral",
]
