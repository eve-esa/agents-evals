"""Raster helpers for the Earth-Bench kits.

Ported from Earth-Agent ``agent/tools/utils.py`` (OpenDataLab, MIT). ``osgeo.gdal``
is replaced with rasterio so the kits run on Python 3.12 without a system GDAL.
Array layout matches GDAL's ``ReadAsArray``: one band stays 2-D, several bands
come back as height, width, band. ``GetGeoTransform``'s six-tuple is rasterio's
``(c, a, b, f, d, e)``.
"""

import numpy as np
import rasterio


def read_image(file_path: str) -> np.ndarray:
    with rasterio.open(file_path) as ds:
        img = ds.read()
    if img.shape[0] == 1:
        return img[0]
    return np.transpose(img, (1, 2, 0))


def read_image_uint8(file_path: str) -> np.ndarray:
    img = read_image(file_path).astype(np.float32)
    min_val = np.min(img)
    max_val = np.max(img)
    if max_val > min_val:
        img = (img - min_val) / (max_val - min_val) * 255
    else:
        img = np.zeros_like(img)
    return img.astype(np.uint8)


def write_like(output_path, array: np.ndarray, template_path: str) -> None:
    """Write ``array`` as a GeoTIFF, copying georeference from ``template_path``.

    A 2-D array is one band. A 3-D array is height, width, band, the layout
    ``read_image`` returns for multi-band files.
    """
    from pathlib import Path

    values = np.asarray(array)
    if values.ndim == 2:
        data = values[np.newaxis, ...]
    elif values.ndim == 3:
        data = np.transpose(values, (2, 0, 1))
    else:
        raise ValueError(f"expected a 2-D or 3-D array, got shape {values.shape}")
    with rasterio.open(template_path) as src:
        profile = src.profile.copy()
    profile.update(
        driver="GTiff",
        height=data.shape[1],
        width=data.shape[2],
        count=data.shape[0],
        dtype=str(data.dtype),
    )
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(output_path, "w", **profile) as dst:
        dst.write(data)


def get_geotransform(file_path) -> tuple:
    with rasterio.open(file_path) as ds:
        transform = ds.transform
        geo = (transform.c, transform.a, transform.b, transform.f, transform.d, transform.e)
        proj = ds.crs.to_wkt() if ds.crs is not None else ""
    if geo == (0, 1.0, 0, 0, 0, 1.0):
        return None, None
    return geo, proj
