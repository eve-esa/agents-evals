"""
Load multispectral bands, crop by normalized AOI, and save to image_cache.

Inputs:
  - ms_folders (List[str/Path]): list of multispectral folders (each contains band GeoTIFFs).
  - aoi (List[float]): normalized [x0, y0, x1, y1] on the multispectral image (0-1, x1>x0, y1>y0).
  - bands (List[str]): bands to load (e.g., ["B1","B2","B3","B4"]).

Output:
  - List[str]: local paths to saved NetCDF crops.
"""

from __future__ import annotations

import uuid
from pathlib import Path
from typing import Dict, List, Union

import numpy as np
import rasterio
import xarray as xr
from rasterio.windows import Window

from eo_gym.runtime.rs_tools.tools.image_cache_paths import cache_local_path
from eo_gym.runtime.rs_tools.tools.multispectral_shared import resolve_ms_folder, validate_aoi
from eo_gym.runtime.rs_tools.utils import check_multispectral_folder_reference
import eo_gym.runtime.core.config_loader as config_loader


def _load_window(path: Path, window: Window) -> np.ndarray:
    with rasterio.open(path) as ds:
        return ds.read(1, window=window)


def _build_window(width: int, height: int, aoi: List[float]) -> Window:
    x0_rel, y0_rel, x1_rel, y1_rel = aoi
    x0 = max(0, min(int(round(x0_rel * width)), width - 1))
    y0 = max(0, min(int(round(y0_rel * height)), height - 1))
    x1 = max(x0 + 1, min(int(round(x1_rel * width)), width))
    y1 = max(y0 + 1, min(int(round(y1_rel * height)), height))
    w = x1 - x0
    h = y1 - y0
    return Window(col_off=x0, row_off=y0, width=w, height=h), [x0, y0, x1, y1]
def _normalize_band_name(name: str) -> str:
    candidate = name.upper()
    if candidate.startswith("SR_"):
        candidate = candidate[3:]
    return candidate


def _discover_band_files(ms_dir: Path) -> List[tuple[str, Path]]:
    band_map: Dict[str, Path] = {}
    for candidate in sorted(ms_dir.glob("*.tif")):
        band_name = _normalize_band_name(candidate.stem)
        if not band_name or not band_name.startswith("B"):
            continue
        band_map[band_name] = candidate
    return sorted(band_map.items())


def crop_multispectral_image(
    ms_folders: List[Union[str, Path]],
    aoi: List[float],
) -> List[dict]:
    """
    Load all available bands from FAIR1M multispectral folders, crop by AOI, and save to NetCDF.
    Inputs: ms_folders (List[str/Path]) and aoi (List[float] normalized [x0,y0,x1,y1]).
    Output: List[dict] of mapping from ms_folder to saved NetCDF paths.
    """
    aoi_vals = validate_aoi(aoi)
    if not ms_folders:
        return [{"message": "Missing multispectral folder reference. Please check the provided parameters."}]
    for folder in ms_folders:
        folder_error = check_multispectral_folder_reference(str(folder))
        if folder_error:
            return [{"message": folder_error}]
    return_mappings = []
    config_loader.IMAGE_CACHE_DIR.mkdir(parents=True, exist_ok=True)

    for folder in ms_folders:
        return_mapping = {}
        ms_dir = resolve_ms_folder(folder)
        band_files = _discover_band_files(ms_dir)
        if not band_files:
            continue
        base_path = band_files[0][1]
        with rasterio.open(base_path) as ds:
            width, height = ds.width, ds.height
        window, _ = _build_window(width, height, aoi_vals)

        band_arrays: List[np.ndarray] = []
        loaded_bands: List[str] = []
        for band_name, band_path in band_files:
            arr = _load_window(band_path, window)
            band_arrays.append(arr)
            loaded_bands.append(band_name)
        if not band_arrays:
            continue

        data = np.stack(band_arrays, axis=0)
        # y,x dimensions derived from window height/width.
        h = band_arrays[0].shape[0]
        w = band_arrays[0].shape[1]
        da = xr.DataArray(
            data,
            dims=("band", "y", "x"),
            coords={"band": loaded_bands, "y": np.arange(h), "x": np.arange(w)},
            name="ms_crop",
        )
        ds_out = xr.Dataset({"ms_crop": da})

        suffix = "_".join(loaded_bands) if loaded_bands else "bands"
        out_name = f"ms_crop_{ms_dir.name}_{suffix}_{uuid.uuid4().hex}.nc"
        out_path = config_loader.IMAGE_CACHE_DIR / out_name
        ds_out.to_netcdf(out_path)

        cached_path = cache_local_path(out_path)
        return_mapping["original_ms_folder"] = folder
        return_mapping["cropped_ms_file"] = cached_path
        return_mappings.append(return_mapping)

    return return_mappings


class CropMultispectralImage:
    """
    Load all available FAIR1M multispectral bands, crop to AOI, and save NetCDF outputs.
    Inputs: ms_folders (List[str/Path]) and aoi (List[float] normalized [x0,y0,x1,y1]).
    Output: List[dict] of mapping from ms_folder to saved NetCDF paths.
    """

    def __init__(self, device: str = "cpu"):
        self.device = device

    def __call__(self, ms_folders: List[str], aoi: List[float]) -> List[str]:
        return crop_multispectral_image(ms_folders, aoi)


__all__ = [
    "crop_multispectral_image",
    "CropMultispectralImage",
]
