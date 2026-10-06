"""
Lookup recommended bands and index expressions for high-level themes (vegetation, water, urban).

Inputs:
  - theme (str): one of {"vegetation", "water", "urban"}.
  - sensor (str): "ls" (Landsat 8/9) or "s2" (Sentinel-2).

Output:
  - dict with keys: theme, sensor, index, description, expression, bands (List[str]).
"""

from __future__ import annotations

from typing import Dict


THEME_MAP: Dict[str, Dict[str, str]] = {
    "vegetation": {
        "index": "NDVI",
        "description": "Normalized Difference Vegetation Index",
        "formula": "(NIR - Red) / (NIR + Red)",
        "ls": "(B5 - B4) / (B5 + B4)",
        "s2": "(B8 - B4) / (B8 + B4)",
        "bands_ls": ["B5", "B4"],
        "bands_s2": ["B8", "B4"],
    },
    "water": {
        "index": "NDWI",
        "description": "Normalized Difference Water Index",
        "formula": "(Green - NIR) / (Green + NIR)",
        "ls": "(B3 - B5) / (B3 + B5)",
        "s2": "(B3 - B8) / (B3 + B8)",
        "bands_ls": ["B3", "B5"],
        "bands_s2": ["B3", "B8"],
    },
    "urban": {
        "index": "NDBI",
        "description": "Normalized Difference Built-Up Index",
        "formula": "(SWIR1 - NIR) / (SWIR1 + NIR)",
        "ls": "(B6 - B5) / (B6 + B5)",
        "s2": "(B11 - B8) / (B11 + B8)",
        "bands_ls": ["B6", "B5"],
        "bands_s2": ["B11", "B8"],
    },
    "snow": {
        "index": "NDSI",
        "description": "Normalized Difference Snow Index",
        "formula": "(Green - SWIR1) / (Green + SWIR1)",
        "ls": "(B3 - B6) / (B3 + B6)",
        "s2": "(B3 - B11) / (B3 + B11)",
        "bands_ls": ["B3", "B6"],
        "bands_s2": ["B3", "B11"],
    },
}


class ThemeIndexLookup:
    """
    Return recommended index expression and bands for a theme (vegetation/water/urban) and sensor (ls/s2).
    Returns recommended spectral index formula/bands for a theme; does not process imagery.
    Inputs: theme (str), sensor (str: ls or s2).
    Output: dict with theme, sensor, index, description, expression, bands (List[str]).
    """

    def __init__(self, device: str = "cpu"):
        self.device = device

    def __call__(self, theme: str, sensor: str) -> Dict:
        key = (theme or "").strip().lower()
        if key not in THEME_MAP:
            raise ValueError("theme must be one of: vegetation, water, urban, snow.")
        sensor_key = (sensor or "").strip().lower()
        if sensor_key not in {"ls", "s2"}:
            raise ValueError("sensor must be 'ls' (Landsat 8/9) or 's2' (Sentinel-2).")

        spec = THEME_MAP[key]
        expr = spec["ls"] if sensor_key == "ls" else spec["s2"]
        bands = spec["bands_ls"] if sensor_key == "ls" else spec["bands_s2"]
        return {
            "theme": key,
            "sensor": sensor_key,
            "index": spec["index"],
            "description": spec["description"],
            "expression": expr,
            "bands": bands,
        }


__all__ = ["ThemeIndexLookup"]
