"""
Compute spectral index zonal statistics (NDVI/NDWI/NDBI/NDSI) from a NetCDF crop or band folder.

Inputs:
  - multispectral_image_path (str/Path): NetCDF path or multispectral band folder URL/path.
  - index_name (str): one of {"ndvi", "ndwi", "ndbi", "ndsi"}.

Output:
  - dict with index_name, mask_shape, and distribution stats (min, max, mean, std, percentiles).
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict

import numpy as np
import xarray as xr

from eo_gym.runtime.rs_tools.utils import (
    check_file_reference,
    check_multispectral_folder_reference,
    download_file,
)
import eo_gym.runtime.core.config_loader as config_loader
from eo_gym.runtime.rs_tools.tools.multispectral_shared import (
    is_netcdf_path,
    select_band_from_dataset_candidates,
    select_band_from_folder,
)


def _compute_index(ds: xr.Dataset, index_name: str) -> np.ndarray:
    idx = index_name.lower()
    if idx == "ndvi":
        nir = select_band_from_dataset_candidates(ds, ("B8", "B5"))
        red = select_band_from_dataset_candidates(ds, ("B4",))
        if nir is None or red is None:
            raise ValueError("NDVI requires NIR and Red bands (B8/B5 and B4).")
        return (nir - red) / (nir + red + 1e-8)
    if idx == "ndwi":
        nir = select_band_from_dataset_candidates(ds, ("B8", "B5"))
        green = select_band_from_dataset_candidates(ds, ("B3",))
        if nir is None or green is None:
            raise ValueError("NDWI requires Green and NIR bands (B3 and B8/B5).")
        return (green - nir) / (green + nir + 1e-8)
    if idx == "ndbi":
        nir = select_band_from_dataset_candidates(ds, ("B8", "B5"))
        swir1 = select_band_from_dataset_candidates(ds, ("B11", "B6"))
        if nir is None or swir1 is None:
            raise ValueError("NDBI requires SWIR1 and NIR bands (B11/B6 and B8/B5).")
        return (swir1 - nir) / (swir1 + nir + 1e-8)
    if idx == "ndsi":
        green = select_band_from_dataset_candidates(ds, ("B3",))
        swir1 = select_band_from_dataset_candidates(ds, ("B11", "B6"))
        if green is None or swir1 is None:
            raise ValueError("NDSI requires Green and SWIR1 bands (B3 and B11/B6).")
        return (green - swir1) / (green + swir1 + 1e-8)
    raise ValueError("index_name must be one of: ndvi, ndwi, ndbi, ndsi.")


def _compute_index_from_folder(ms_folder: str, index_name: str) -> np.ndarray:
    idx = index_name.lower()
    if idx == "ndvi":
        nir = select_band_from_folder(ms_folder, ("B8", "B5"))
        red = select_band_from_folder(ms_folder, ("B4",))
        if nir is None or red is None:
            raise ValueError("NDVI requires NIR and Red bands (B8/B5 and B4).")
        return (nir - red) / (nir + red + 1e-8)
    if idx == "ndwi":
        nir = select_band_from_folder(ms_folder, ("B8", "B5"))
        green = select_band_from_folder(ms_folder, ("B3",))
        if nir is None or green is None:
            raise ValueError("NDWI requires Green and NIR bands (B3 and B8/B5).")
        return (green - nir) / (green + nir + 1e-8)
    if idx == "ndbi":
        nir = select_band_from_folder(ms_folder, ("B8", "B5"))
        swir1 = select_band_from_folder(ms_folder, ("B11", "B6"))
        if nir is None or swir1 is None:
            raise ValueError("NDBI requires SWIR1 and NIR bands (B11/B6 and B8/B5).")
        return (swir1 - nir) / (swir1 + nir + 1e-8)
    if idx == "ndsi":
        green = select_band_from_folder(ms_folder, ("B3",))
        swir1 = select_band_from_folder(ms_folder, ("B11", "B6"))
        if green is None or swir1 is None:
            raise ValueError("NDSI requires Green and SWIR1 bands (B3 and B11/B6).")
        return (green - swir1) / (green + swir1 + 1e-8)
    raise ValueError("index_name must be one of: ndvi, ndwi, ndbi, ndsi.")


def compute_index_zonal_stats(multispectral_image_path: str, index_name: str) -> Dict:
    """
    Compute NDVI/NDWI/NDBI/NDSI from a NetCDF crop or band folder and return stats.
    Inputs: multispectral_image_path (str/Path folder or NetCDF), index_name (ndvi|ndwi|ndbi|ndsi).
    Output: dict with mask_shape and distribution stats (min/max/mean/std/percentiles).
    """
    path_str = str(multispectral_image_path)
    if is_netcdf_path(path_str):
        error = check_file_reference(path_str, label="file", strict=True)
        if error:
            return {"message": error}
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
        if "ms_crop" not in ds or "band" not in ds:
            raise ValueError("NetCDF missing required variables (ms_crop/band).")
        idx_arr = _compute_index(ds, index_name)
    else:
        folder_error = check_multispectral_folder_reference(path_str)
        if folder_error:
            return {"message": folder_error}
        idx_arr = _compute_index_from_folder(path_str, index_name)
    mask = np.asarray(idx_arr, dtype=np.float32)
    mask = np.nan_to_num(mask, nan=0.0, posinf=0.0, neginf=0.0)

    percentiles = np.percentile(mask, [5, 25, 50, 75, 95]).tolist()

    return {
        "index_name": index_name.lower(),
        "mask_shape": list(mask.shape),
        "min": float(mask.min()),
        "max": float(mask.max()),
        "mean": float(mask.mean()),
        "std": float(mask.std()),
        "percentiles": {
            "p05": float(percentiles[0]),
            "p25": float(percentiles[1]),
            "p50": float(percentiles[2]),
            "p75": float(percentiles[3]),
            "p95": float(percentiles[4]),
        },
    }


class ComputeIndexZonalStats:
    """
    Compute NDVI/NDWI/NDBI/NDSI from a multispectral crop NetCDF and return distribution stats (no mask payload).
    Computes stats for NDVI/NDWI/NDBI/NDSI over a multispectral crop; returns numeric distributions.
    Inputs: multispectral_image_path (str/Path), index_name (ndvi|ndwi|ndbi|ndsi).
    Output: dict with index_name, mask_shape (List[int]), min/max/mean/std (float), percentiles.
    """

    def __init__(self, device: str = "cpu"):
        self.device = device

    def __call__(self, multispectral_image_path: str, index_name: str) -> Dict:
        return compute_index_zonal_stats(multispectral_image_path, index_name)


class ComputeNDVIByMultispectral(ComputeIndexZonalStats):
    """
    Shortcut to compute NDVI stats from a multispectral NetCDF crop or band folder.
    """

    def __call__(self, multispectral_image_path: str) -> Dict:
        return super().__call__(multispectral_image_path, "ndvi")


class ComputeNDWIByMultispectral(ComputeIndexZonalStats):
    """
    Shortcut to compute NDWI stats from a multispectral NetCDF crop or band folder.
    """

    def __call__(self, multispectral_image_path: str) -> Dict:
        return super().__call__(multispectral_image_path, "ndwi")


class ComputeNDBIByMultispectral(ComputeIndexZonalStats):
    """
    Shortcut to compute NDBI stats from a multispectral NetCDF crop or band folder.
    """

    def __call__(self, multispectral_image_path: str) -> Dict:
        return super().__call__(multispectral_image_path, "ndbi")


class ComputeNDSIByMultispectral(ComputeIndexZonalStats):
    """
    Shortcut to compute NDSI (snow) stats from a multispectral NetCDF crop or band folder.
    """

    def __call__(self, multispectral_image_path: str) -> Dict:
        return super().__call__(multispectral_image_path, "ndsi")


__all__ = [
    "ComputeIndexZonalStats",
    "ComputeNDVIByMultispectral",
    "ComputeNDWIByMultispectral",
    "ComputeNDBIByMultispectral",
    "ComputeNDSIByMultispectral",
    "compute_index_zonal_stats",
]
