"""
Shared tool implementations.
"""

__all__ = [
    "crop_optical_or_sar_image",
    "crop_multispectral_image",
    "get_object_bbox_by_optical_image",
    "get_object_bbox_by_optical_image_sam3",
    "get_object_mask_by_optical_image",
    "get_road_mask_by_optical_image",
    "get_building_mask_by_optical_image",
    "get_object_bbox_by_sar_image",
    "get_object_bbox_by_optical_sar_image",
    "count_objects_by_category",
    "get_bbox_geospatial_relationship",
    "basic_calculator",
    "get_mask_geospatial_relationship",
    "summarize_temporal_mask_changes",
    "summarize_temporal_object_changes",
    "theme_index_lookup",
    "compute_index_zonal_stats",
    "compute_water_mask",
    "compute_vegetation_mask",
    "compute_urban_mask",
    "analyze_optical_scene",
    "describe_optical_object",
    "zoom_out_crop_image",
    "move_up_crop_image",
    "move_down_crop_image",
    "move_left_crop_image",
    "move_right_crop_image",
    "multispectral_shared",
]

# Re-export tool classes for convenience
from eo_gym.runtime.rs_tools.tools.load_image_with_crop import CropOpticalOrSarImage  # noqa: F401
from eo_gym.runtime.rs_tools.tools.load_multispectral_image_with_crop import CropMultispectralImage  # noqa: F401
from eo_gym.runtime.rs_tools.tools.sam3_text_bbox import (  # noqa: F401
    GetObjectBboxByOpticalImageSam3,
    GetObjectBboxByOpticalImageSam3 as GetObjectBboxByOpticalImage,
)
from eo_gym.runtime.rs_tools.tools.sam3_text_mask import (
    GetObjectMaskByOpticalImage,
    GetRoadMaskByOpticalImage,
    GetBuildingMaskByOpticalImage,
)  # noqa: F401
from eo_gym.runtime.rs_tools.tools.get_object_bbox_sar import GetObjectBboxBySARImage  # noqa: F401
from eo_gym.runtime.rs_tools.tools.get_object_bbox_optical_sar import GetObjectBboxByOpticalSarImage  # noqa: F401
from eo_gym.runtime.rs_tools.tools.basic_python_tools import (
    CountObjectsByCategory,
    GetBboxGeospatialRelationship,
    BasicCalculator,
    GetMaskGeospatialRelationship,
    SummarizeTemporalMaskChanges,
    SummarizeTemporalObjectChanges,
)  # noqa: F401
from eo_gym.runtime.rs_tools.tools.theme_index_lookup import ThemeIndexLookup  # noqa: F401
from eo_gym.runtime.rs_tools.tools.compute_index_zonal_stats import (  # noqa: F401
    ComputeIndexZonalStats,
    ComputeNDVIByMultispectral,
    ComputeNDWIByMultispectral,
    ComputeNDBIByMultispectral,
)
from eo_gym.runtime.rs_tools.tools.compute_water_mask import ComputeWaterMaskByMultispectral  # noqa: F401
from eo_gym.runtime.rs_tools.tools.compute_vegetation_mask import ComputeVegetationMaskByMultispectral  # noqa: F401
from eo_gym.runtime.rs_tools.tools.compute_urban_mask import ComputeUrbanMaskByMultispectral  # noqa: F401
from eo_gym.runtime.rs_tools.tools.ai_optical_analysis import (
    AnalyzeOpticalScene,
    DescribeOpticalObject,
)  # noqa: F401
from eo_gym.runtime.rs_tools.tools.vhr_zoom_crop_navigator import (  # noqa: F401
    MoveDownCropNavigator,
    MoveLeftCropNavigator,
    MoveRightCropNavigator,
    MoveUpCropNavigator,
    ZoomOutCropNavigator,
)
