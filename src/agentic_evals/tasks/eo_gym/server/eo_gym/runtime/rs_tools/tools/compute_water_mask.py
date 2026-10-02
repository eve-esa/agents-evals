"""
Compute a water mask from a multispectral crop NetCDF or a multispectral band folder using NDWI.

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

# Common NDWI thresholds for water detection by sensor family.
NDWI_THRESHOLDS = {
    "ls": 0.3,  # Landsat 8/9
    "s2": 0.2,  # Sentinel-2
}

# Band selection per sensor: (Green, NIR).
SENSOR_BANDS = {
    "ls": ("B3", "B5"),
    "s2": ("B3", "B8"),
}


def compute_water_mask_by_multispectral(multispectral_image_path: Union[str, Path], sensor: str) -> str:
    """
    Compute a binary water mask from a multispectral crop NetCDF or band folder using NDWI.
    Inputs: multispectral_image_path (str/Path folder or NetCDF), sensor (ls|s2).
    Output: path (str) to PNG mask 
    """
    sensor_key = (sensor or "").strip().lower()
    if sensor_key not in SENSOR_BANDS:
        raise ValueError("sensor must be 'ls' (Landsat 8/9) or 's2' (Sentinel-2).")

    green_band, nir_band = SENSOR_BANDS[sensor_key]
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
        green = select_band_from_dataset(ds, green_band)
        nir = select_band_from_dataset(ds, nir_band)
        name_stem = nc_file.stem
    else:
        folder_error = check_multispectral_folder_reference(path_str)
        if folder_error:
            return folder_error
        green = load_band_from_folder(multispectral_image_path, green_band, sensor_key)
        nir = load_band_from_folder(multispectral_image_path, nir_band, sensor_key)
        name_stem = Path(str(multispectral_image_path).rstrip("/")).name

    ndwi = (green - nir) / (green + nir + 1e-6)
    ndwi = np.nan_to_num(ndwi, nan=-1.0, posinf=-1.0, neginf=-1.0)

    threshold = NDWI_THRESHOLDS[sensor_key]
    water_mask = (ndwi > threshold).astype(np.uint8) * 255

    mask_img = Image.fromarray(water_mask)
    config_loader.IMAGE_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    out_name = f"water_mask_{name_stem}_{sensor_key}_{uuid.uuid4().hex}.png"
    out_path = config_loader.IMAGE_CACHE_DIR / out_name
    mask_img.save(out_path, format="PNG")

    return cache_local_path(out_path)


class ComputeWaterMaskByMultispectral:
    """
    Compute a water mask from a multispectral crop NetCDF using NDWI and sensor thresholds.
    Derives a water mask from NDWI thresholds; returns mask metadata/arrays.
    Inputs: multispectral_image_path (str/Path), sensor (ls|s2).
    Output: mask_path (str) to PNG 
    """

    def __init__(self, device: str = "cpu"):
        self.device = device

    def __call__(self, multispectral_image_path: Union[str, Path], sensor: str) -> str:
        return compute_water_mask_by_multispectral(multispectral_image_path, sensor)


__all__ = [
    "compute_water_mask_by_multispectral",
    "ComputeWaterMaskByMultispectral",
]
