"""
Legacy entrypoint that aggregates the concrete tool implementations spread
across the rs_tools package. Importing from this module preserves the original
API used by the dataset builders and agents while allowing the actual tool
logic to live in maintainable submodules.
"""

from __future__ import annotations

import os
from typing import List, Optional

from eo_gym.runtime.rs_tools.datasets.fair1m_ms import (
    GetMultispectralList,
    GetNextMultispectral,
    GetPreviousMultispectral,
)
from eo_gym.runtime.rs_tools.datasets.m4_sar import (
    GetOpticalFromSar,
    GetSarFromOptical,
)
from eo_gym.runtime.rs_tools.datasets.fmow import (
    GetNextOpticalImage,
    GetOpticalImageList,
    GetPreviousOpticalImage,
)
from eo_gym.runtime.rs_tools.tools.load_image_with_crop import CropOpticalOrSarImage
from eo_gym.runtime.rs_tools.tools.load_multispectral_image_with_crop import CropMultispectralImage
from eo_gym.runtime.rs_tools.tools.compute_index_zonal_stats import (
    ComputeNDVIByMultispectral,
    ComputeNDWIByMultispectral,
    ComputeNDBIByMultispectral,
    ComputeNDSIByMultispectral,
)
from eo_gym.runtime.rs_tools.tools.compute_water_mask import ComputeWaterMaskByMultispectral
from eo_gym.runtime.rs_tools.tools.compute_vegetation_mask import ComputeVegetationMaskByMultispectral
from eo_gym.runtime.rs_tools.tools.compute_urban_mask import ComputeUrbanMaskByMultispectral
from eo_gym.runtime.rs_tools.tools.sam3_text_bbox import (
    GetObjectBboxByOpticalImageSam3,
    GetObjectBboxByOpticalImageSam3 as GetObjectBboxByOpticalImage,
)
from eo_gym.runtime.rs_tools.tools.sam3_text_mask import (
    GetObjectMaskByOpticalImage,
    GetRoadMaskByOpticalImage,
    GetBuildingMaskByOpticalImage,
)
from eo_gym.runtime.rs_tools.tools.get_object_bbox_sar import GetObjectBboxBySARImage
from eo_gym.runtime.rs_tools.tools.get_object_bbox_optical_sar import GetObjectBboxByOpticalSarImage
from eo_gym.runtime.rs_tools.tools.basic_python_tools import (
    CountObjectsByCategory,
    GetBboxGeospatialRelationship,
    BasicCalculator,
    GetMaskGeospatialRelationship,
    NormalizeBoundingBoxes,
    SummarizeTemporalMaskChanges,
    SummarizeTemporalObjectChanges,
)
from eo_gym.runtime.rs_tools.tools.theme_index_lookup import ThemeIndexLookup
from eo_gym.runtime.rs_tools.tools.ai_optical_analysis import (
    AnalyzeOpticalScene,
    DescribeOpticalObject,
)
from eo_gym.runtime.rs_tools.tools.vhr_zoom_crop_navigator import (
    ZoomOutCropNavigator,
    MoveUpCropNavigator,
    MoveDownCropNavigator,
    MoveLeftCropNavigator,
    MoveRightCropNavigator,
)

DEVICE = os.getenv("TOOLS_DEVICE", "cuda")

# --------------------------------------------------------------------------------------
# --- Tool registry ---
# --------------------------------------------------------------------------------------
AVAILABLE_TOOL_CLASSES = {
    "get_multispectral_list": GetMultispectralList,
    "get_previous_multispectral": GetPreviousMultispectral,
    "get_next_multispectral": GetNextMultispectral,
    "get_optical_image_list": GetOpticalImageList,
    "get_previous_optical_image": GetPreviousOpticalImage,
    "get_next_optical_image": GetNextOpticalImage,
    "get_sar_from_optical": GetSarFromOptical,
    "get_optical_from_sar": GetOpticalFromSar,
    "crop_optical_or_sar_image": CropOpticalOrSarImage,
    "crop_multispectral_image": CropMultispectralImage,
    "get_object_bbox_by_optical_image": GetObjectBboxByOpticalImageSam3,
    "get_object_mask_by_optical_image": GetObjectMaskByOpticalImage,
    "get_road_mask_by_optical_image": GetRoadMaskByOpticalImage,
    "get_building_mask_by_optical_image": GetBuildingMaskByOpticalImage,
    "get_object_bbox_by_sar_image": GetObjectBboxBySARImage,
    "get_object_bbox_by_optical_sar_image": GetObjectBboxByOpticalSarImage,
    "get_bbox_geospatial_relationship": GetBboxGeospatialRelationship,
    "basic_calculator": BasicCalculator,
    "get_mask_geospatial_relationship": GetMaskGeospatialRelationship,
    "normalize_bounding_boxes": NormalizeBoundingBoxes,
    "theme_index_lookup": ThemeIndexLookup,
    "compute_ndvi_by_multispectral": ComputeNDVIByMultispectral,
    "compute_ndwi_by_multispectral": ComputeNDWIByMultispectral,
    "compute_ndbi_by_multispectral": ComputeNDBIByMultispectral,
    "compute_ndsi_by_multispectral": ComputeNDSIByMultispectral,
    "compute_water_mask_by_multispectral": ComputeWaterMaskByMultispectral,
    "compute_vegetation_mask_by_multispectral": ComputeVegetationMaskByMultispectral,
    "compute_urban_mask_by_multispectral": ComputeUrbanMaskByMultispectral,
    "analyze_optical_scene": AnalyzeOpticalScene,
    "describe_optical_object": DescribeOpticalObject,
    "zoom_out_optical_image": ZoomOutCropNavigator,
    "move_up_optical_image": MoveUpCropNavigator,
    "move_down_optical_image": MoveDownCropNavigator,
    "move_left_optical_image": MoveLeftCropNavigator,
    "move_right_optical_image": MoveRightCropNavigator,
}

DATASET_TOOL_ALIASES = {
    "fair1m": "fair1m_ms",
    "fair1m_full": "fair1m_ms",
    "hichtala_dota": "dota",
    "hichtala/dota": "dota",
    "hichtala_xview": "xview",
    "hichtala/xview": "xview",
    "dior": "dior",
    "m4-sar": "m4-sar",
    "m4sar": "m4-sar",
    "m4_sar": "m4-sar",
    "sardet_100k": "sardet_100k",
    "sardet-100k": "sardet_100k",
    "fmow": "fmow",
    "xbd": "xbd",
    "vhr_zoom": "fair1_vhr",
}

# Curated tool subsets to bias agent trajectories toward dataset-relevant tools.
DATASET_TOOL_RECOMMENDATIONS = {
    "fair1m_ms": [
        "get_multispectral_list",
        "get_previous_multispectral",
        "get_next_multispectral",
        "crop_multispectral_image",
        "compute_ndvi_by_multispectral",
        "compute_ndwi_by_multispectral",
        "compute_ndbi_by_multispectral",
        "compute_ndsi_by_multispectral",
        "compute_water_mask_by_multispectral",
        "compute_vegetation_mask_by_multispectral",
        "compute_urban_mask_by_multispectral",
        "get_bbox_geospatial_relationship",
        "basic_calculator",
        "get_mask_geospatial_relationship",
        "theme_index_lookup",
    ],
    "fmow": [
        "get_optical_image_list",
        "get_previous_optical_image",
        "get_next_optical_image",
        "crop_optical_or_sar_image",
        "get_object_bbox_by_optical_image",
        "get_object_mask_by_optical_image",
        "get_road_mask_by_optical_image",
        "get_building_mask_by_optical_image",
        "get_bbox_geospatial_relationship",
        "basic_calculator",
        "get_mask_geospatial_relationship",
        "normalize_bounding_boxes",
    ],
    "dior": [
        "crop_optical_or_sar_image",
        "get_object_bbox_by_optical_image",
        "get_object_mask_by_optical_image",
        "get_road_mask_by_optical_image",
        "get_building_mask_by_optical_image",
        "get_bbox_geospatial_relationship",
        "basic_calculator",
        "get_mask_geospatial_relationship",
        "analyze_optical_scene",
        "describe_optical_object",
    ],
    "dota": [
        "crop_optical_or_sar_image",
        "get_object_bbox_by_optical_image",
        "get_object_mask_by_optical_image",
        "get_road_mask_by_optical_image",
        "get_building_mask_by_optical_image",
        "get_bbox_geospatial_relationship",
        "basic_calculator",
        "get_mask_geospatial_relationship",
        "analyze_optical_scene",
        "describe_optical_object",
    ],
    "xview": [
        "crop_optical_or_sar_image",
        "get_object_bbox_by_optical_image",
        "get_object_mask_by_optical_image",
        "get_road_mask_by_optical_image",
        "get_building_mask_by_optical_image",
        "get_bbox_geospatial_relationship",
        "basic_calculator",
        "get_mask_geospatial_relationship",
        "analyze_optical_scene",
        "describe_optical_object",
    ],
    "xbd": [
        "crop_optical_or_sar_image",
        "get_object_bbox_by_optical_image",
        "get_object_mask_by_optical_image",
        "get_building_mask_by_optical_image",
        "get_bbox_geospatial_relationship",
        "basic_calculator",
        "get_mask_geospatial_relationship",
        "analyze_optical_scene",
        "describe_optical_object",
    ],
    "fair1_vhr": [
        "crop_optical_or_sar_image",
        "zoom_out_optical_image",
        "move_up_optical_image",
        "move_down_optical_image",
        "move_left_optical_image",
        "move_right_optical_image",
        "basic_calculator",
        "get_object_bbox_by_optical_image",
        "get_bbox_geospatial_relationship",
        "get_mask_geospatial_relationship",
        "get_object_mask_by_optical_image",
    ],
    "m4-sar": [
        "get_object_bbox_by_optical_sar_image",
        "get_sar_from_optical",
        "get_optical_from_sar",
        "crop_optical_or_sar_image",
        "get_bbox_geospatial_relationship",
        "basic_calculator",
        "get_mask_geospatial_relationship",
    ],
    "sardet_100k": [
        "crop_optical_or_sar_image",
        "get_object_bbox_by_sar_image",
        "get_bbox_geospatial_relationship",
        "basic_calculator",
        "get_mask_geospatial_relationship",
    ],
}


def resolve_tool_class(tool_name: str):
    """Return the registered tool class for the provided name."""
    return AVAILABLE_TOOL_CLASSES.get(tool_name)


def get_recommended_tools_for_dataset(
    dataset: Optional[str],
    *,
    allow_fallback: bool = True,
) -> List[str]:
    """
    Return the ordered list of tool names recommended for the provided dataset.

    When ``allow_fallback`` is True (default), unknown datasets return the full
    tool registry. When False, unknown datasets return an empty list.
    """
    key = (dataset or "").strip().lower()
    canonical = DATASET_TOOL_ALIASES.get(key, key)
    tools = DATASET_TOOL_RECOMMENDATIONS.get(canonical)
    if tools:
        return list(tools)
    if allow_fallback:
        return list(AVAILABLE_TOOL_CLASSES.keys())
    return []


def load_custom_tool(tool_name: str, device: str = DEVICE):
    tool_class = resolve_tool_class(tool_name)
    if tool_class is None:
        raise KeyError(f"Custom tool '{tool_name}' not found in AVAILABLE_TOOL_CLASSES.")
    return tool_class(device=device)
