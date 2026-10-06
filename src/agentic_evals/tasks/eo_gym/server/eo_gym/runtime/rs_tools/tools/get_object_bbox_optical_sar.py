"""
Text-prompted bbox selection for paired optical and SAR images using cached ground truth.

This tool uses the LightweightBboxSimulator solution to filter ground-truth
objects with a tiny local matcher model, then emits bbox detections from the
selected objects.

Inputs:
  - optical_image_path (str/Path): path or URL to the optical image.
  - sar_image_path (str/Path): path or URL to the SAR image.
  - text_prompt (str): target description used to select objects.

Output:
  - List[dict] of detections with keys: ``bbox`` (xyxy), ``score`` and ``label``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

from eo_gym.runtime.core.config_loader import to_remote_url
from eo_gym.runtime.rs_tools.tools.ground_truth_cache import resolve_ground_truth_for_image
from eo_gym.runtime.rs_tools.utils import check_file_reference
from eo_gym.runtime.rs_tools.tools.lightweight_bbox_simulator import (
    DEFAULT_BBOX_MATCH_MODEL,
    LightweightBboxSimulator,
)


def _normalize_image_key(image_path: Union[str, Path]) -> str:
    raw = str(image_path or "").strip()
    if not raw:
        return ""
    if raw.startswith("file://"):
        raw = raw[len("file://") :]
    try:
        normalized = to_remote_url(raw)
    except Exception:
        normalized = raw
    return str(normalized).split("?", 1)[0]


def _resolve_ground_truth(
    image_path: Union[str, Path],
) -> Tuple[Optional[Dict[str, Any]], Optional[Dict[str, Any]]]:
    try:
        gt, meta = resolve_ground_truth_for_image(str(image_path))
        return gt, meta
    except Exception:
        return None, None


class GetObjectBboxByOpticalSarImage:
    """
    Object bbox selection using an optical+SAR pair and a text prompt.
    """

    def __init__(
        self,
        device: str = "cpu",
        model_id: str = "bbox_tool_names_ground_truth",
        confidence_threshold: float = 0.25,
        llm_model_name: Optional[str] = DEFAULT_BBOX_MATCH_MODEL,
    ):
        # Kept for backward compatibility with existing call sites/configs.
        self.device = device
        self.model_id = model_id
        self.confidence_threshold = confidence_threshold
        self.llm_model_name = llm_model_name
        self._bbox_simulator = LightweightBboxSimulator(model_name=self.llm_model_name)

    def __call__(
        self,
        optical_image_path: Union[str, Path],
        sar_image_path: Union[str, Path],
        text_prompt: str,
    ) -> List[Dict]:
        if not text_prompt or not isinstance(text_prompt, str):
            raise ValueError("text_prompt must be a non-empty string.")
        if not optical_image_path:
            raise ValueError("optical_image_path is required.")
        if not sar_image_path:
            raise ValueError("sar_image_path is required.")

        error = check_file_reference(str(optical_image_path), label="image", strict=True)
        if error:
            return [{"error": error}]
        error = check_file_reference(str(sar_image_path), label="image", strict=True)
        if error:
            return [{"error": error}]

        optical_key = _normalize_image_key(optical_image_path)
        sar_key = _normalize_image_key(sar_image_path)
        if optical_key == "":
            raise ValueError("optical_image_path is invalid or empty.")
        if sar_key == "":
            raise ValueError("sar_image_path is invalid or empty.")
        if optical_key == sar_key:
            raise ValueError("optical_image_path and sar_image_path must refer to different images.")

        prompt = text_prompt.strip()
        ground_truth, gt_meta = _resolve_ground_truth(optical_image_path)
        context = optical_image_path
        if not ground_truth:
            ground_truth, gt_meta = _resolve_ground_truth(sar_image_path)
            context = sar_image_path
        if not ground_truth:
            return []
        return self._bbox_simulator.build_detections_from_ground_truth(
            ground_truth,
            prompt,
            context=str(context),
            target_image_ref=str(context),
            meta=gt_meta,
        )


def preload_optical_sar_detector_components(
    device: str = "cpu",
    model_id: Optional[str] = None,
    confidence_threshold: float = 0.25,
) -> None:
    """
    Compatibility shim: no detector checkpoints are loaded in bbox-tool mode.
    """
    _ = (device, model_id, confidence_threshold)
    return None


__all__ = ["GetObjectBboxByOpticalSarImage", "preload_optical_sar_detector_components"]
