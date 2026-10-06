"""
Helpers for M4-SAR cross-modality lookups (optical <-> SAR).

Uses eo_gym_m4_sar.csv as the source of truth for optical/SAR pairing.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Dict, List, Optional, Union

import pandas as pd

from eo_gym.runtime.core.config_loader import EO_GYM_M4_SAR_CSV, RUN_ON_LOCAL, to_local_path, to_remote_url
from eo_gym.runtime.rs_tools.utils import check_file_reference


@lru_cache(maxsize=1)
def _load_m4_sar_df() -> pd.DataFrame:
    if not EO_GYM_M4_SAR_CSV.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(EO_GYM_M4_SAR_CSV, dtype=str, keep_default_na=False)
    except Exception:
        return pd.DataFrame()


def _resolve_pair(image_path: Union[str, Path]) -> tuple[Optional[str], Optional[str]]:
    key = to_remote_url(image_path)
    df = _load_m4_sar_df()
    if df.empty:
        return None, "Image pairing index not found or empty."
    if "image_path" not in df.columns:
        return None, "Image pairing index missing image_path column."
    matches = df.loc[df["image_path"] == key]
    if matches.empty:
        return None, "Pair not found for the provided image. Please make sure you pass the correct SAR or optical image path to expected function."
    row = matches.iloc[0]
    pair = row.get("pair_image_path") or ""
    if RUN_ON_LOCAL and pair:
        pair = to_local_path(pair)
    return str(pair), None


def get_sar_image_from_optical(optical_image_path: Union[str, Path]) -> Optional[str]:
    """
    Given an optical image URL/path, return the paired SAR URL.
    """

    df = _load_m4_sar_df()
    if df.empty:
        return "Image pairing index not found or empty."
    if "image_path" not in df.columns:
        return "Image pairing index missing image_path column."
    matches = df.loc[df["image_path"] == optical_image_path]
    if matches.empty:
        return f"There are {len(matches)} matches for the provided optical image path. Please make sure you pass the correct SAR or optical image path to expected function."

    row = matches.iloc[0]
    pair = row.get("pair_image_path") or ""
    return pair


def get_optical_image_from_sar(sar_image_path: Union[str, Path]) -> Optional[str]:
    """
    Given a SAR image URL/path, return the paired optical URL.
    """

    df = _load_m4_sar_df()
    if df.empty:
        return "Image pairing index not found or empty."
    if "image_path" not in df.columns:
        return "Image pairing index missing image_path column."
    matches = df.loc[df["image_path"] == sar_image_path]
    if matches.empty:
        return f"There are {len(matches)} matches for the provided SAR image path. Please make sure you pass the correct SAR or optical image path to expected function."

    row = matches.iloc[0]
    pair = row.get("pair_image_path") or ""
    return pair


def rebuild_m4_sar_index() -> Dict[str, List[str]]:
    """
    Compatibility shim: return a CSV-derived map of image_path -> [image_path, pair_image_path].
    """
    df = _load_m4_sar_df()
    if df.empty:
        return {}
    if "image_path" not in df.columns or "pair_image_path" not in df.columns:
        return {}
    url_map: Dict[str, List[str]] = {}
    for _, row in df.iterrows():
        image_path = row.get("image_path") or ""
        pair_path = row.get("pair_image_path") or ""
        if image_path and pair_path:
            if RUN_ON_LOCAL:
                local_image = to_local_path(image_path)
                local_pair = to_local_path(pair_path)
                url_map[local_image] = [local_image, local_pair]
            else:
                url_map[str(image_path)] = [str(image_path), str(pair_path)]
    return url_map


class GetSarFromOptical:
    """
    Return the SAR companion for an optical input.
    Returns paired SAR URL/path for the given optical image; no detection or analysis.
    Inputs: optical_image_path (str/Path).
    Output: Optional[str] relative SAR image path.
    """

    def __init__(self, device: str = "cpu"):
        self.device = device

    def __call__(self, optical_image_path: str) -> Optional[str]:
        return get_sar_image_from_optical(optical_image_path)


class GetOpticalFromSar:
    """
    Return the optical companion for a SAR input.
    Returns paired optical URL/path for the given SAR image; no detection or analysis.
    Inputs: sar_image_path (str/Path).
    Output: Optional[str] relative optical image path.
    """

    def __init__(self, device: str = "cpu"):
        self.device = device

    def __call__(self, sar_image_path: str) -> Optional[str]:
        return get_optical_image_from_sar(sar_image_path)


__all__ = [
    "get_sar_image_from_optical",
    "get_optical_image_from_sar",
    "rebuild_m4_sar_index",
    "GetSarFromOptical",
    "GetOpticalFromSar",
]
