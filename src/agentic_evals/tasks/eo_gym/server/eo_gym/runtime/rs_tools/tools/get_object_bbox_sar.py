"""
SAR text-prompt bbox selection backed by cached ground-truth metadata.

This tool uses the LightweightBboxSimulator solution to filter ground-truth
objects with a tiny local matcher model, then emits bbox detections from the
selected objects.

Inputs:
  - sar_image_urls (List[str]): URLs or local paths to SAR images.
  - text_prompt (str): target description used to select objects.

Output:
  - Dict[str, List[dict]] mapping each image URL to a list of detections.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

from eo_gym.runtime.rs_tools.utils import check_file_reference
from eo_gym.runtime.rs_tools.tools.ground_truth_cache import resolve_ground_truth_for_image
from eo_gym.runtime.rs_tools.tools.lightweight_bbox_simulator import (
    DEFAULT_BBOX_MATCH_MODEL,
    LightweightBboxSimulator,
)


def _resolve_ground_truth(image_path: str) -> Tuple[Optional[Dict[str, Any]], Optional[Dict[str, Any]]]:
    try:
        gt, meta = resolve_ground_truth_for_image(str(image_path))
        return gt, meta
    except Exception:
        return None, None


class GetObjectBboxBySARImage:
    """
    SAR text-prompt bbox tool using cached ground truth + optional LLM label filtering.

    Inputs: sar_image_urls (List[str]), text_prompt (str).
    Output: Dict[str, List[dict]].
    """

    def __init__(
        self,
        device: str = "cpu",
        tacmt_config: Optional[str] = None,
        tacmt_checkpoint: Optional[str] = None,
        score_threshold: float = 0.1,
        mask_threshold: float = 0.1,
        llm_model_name: Optional[str] = DEFAULT_BBOX_MATCH_MODEL,
    ):
        # Kept for backward compatibility with existing call sites/configs.
        self.device = device
        self.tacmt_config = tacmt_config
        self.tacmt_checkpoint = tacmt_checkpoint
        self.score_threshold = score_threshold
        self.mask_threshold = mask_threshold
        self.llm_model_name = llm_model_name
        self._bbox_simulator = LightweightBboxSimulator(model_name=self.llm_model_name)

    def __call__(self, sar_image_urls: List[str], text_prompt: str) -> Dict[str, List[dict]]:
        if not sar_image_urls or not isinstance(sar_image_urls, list):
            raise ValueError("sar_image_urls must be a non-empty list.")
        if not text_prompt or not isinstance(text_prompt, str):
            raise ValueError("text_prompt must be a non-empty string.")

        prompt = text_prompt.strip()
        results: Dict[str, List[dict]] = {}
        for url in sar_image_urls:
            error = check_file_reference(url, label="image", strict=True)
            if error:
                results[str(url)] = [{"error": error}]
                continue
            ground_truth, gt_meta = _resolve_ground_truth(str(url))
            if not ground_truth:
                results[str(url)] = []
                continue
            results[str(url)] = self._bbox_simulator.build_detections_from_ground_truth(
                ground_truth,
                prompt,
                context=str(url),
                target_image_ref=str(url),
                meta=gt_meta,
            )
        return results


__all__ = ["GetObjectBboxBySARImage"]
