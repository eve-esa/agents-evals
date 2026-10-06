"""
Run text-prompted segmentation on remote images and return binary mask URLs.

Inputs:
  - optical_image_urls (List[str]): HTTP/HTTPS URLs to optical images.
  - text_prompt (str): text prompt to segment.

Output:
  - Dict[str, dict] mapping optical_image_url -> {
      "image": {image_url, image_size, gsd},
      "masks": List[dict]{label, mask_url, bbox, size, score, method}
    }.
"""

from __future__ import annotations

import os
from io import BytesIO
from typing import Any, Dict, List, Optional

import numpy as np
import torch
import requests
from PIL import Image
from transformers import Sam3Model, Sam3Processor

from eo_gym.runtime.core.config_loader import RUN_ON_LOCAL, to_local_path
from eo_gym.runtime.rs_tools.tools.image_cache_logger import log_image_event
from eo_gym.runtime.rs_tools.tools.image_metadata import describe_image_url
from eo_gym.runtime.rs_tools.tools.mask_utils import mask_to_bbox, save_mask_image
from eo_gym.runtime.rs_tools.utils import check_file_reference
from eo_gym.runtime.rs_tools.tools.model_cache import get_sam3_components, get_sam3_inference_lock

def _resolve_device(device: Optional[str]) -> torch.device:
    """
    Prefer the requested device, but fall back to GPU when available and CPU otherwise.
    """
    if isinstance(device, torch.device):
        return device
    requested = (device or "").strip()
    if not requested:
        requested = "cuda" if torch.cuda.is_available() else "cpu"
    if requested.startswith("cuda") and not torch.cuda.is_available():
        requested = "cpu"
    return torch.device(requested)


class GetObjectMaskByOpticalImage:
    """
    Segment optical images using a text prompt and return binary mask URLs.
    Text-prompt segmentation; returns per-URL list of mask metadata pointing to saved PNGs.
    Inputs: optical_image_urls (List[str]), text_prompt (str).
    Output: Dict[str, dict] mapping optical_image_url -> {
        "image": {image_url, image_size, gsd},
        "masks": List[dict]{label, mask_url, bbox, size, score, method},
    }.
    """

    def __init__(self, device: Optional[str] = None):
        self.device = _resolve_device(device)
        self.model: Optional[Sam3Model] = None
        self.processor: Optional[Sam3Processor] = None
        self.mask_threshold = 0.3
        self.score_threshold = 0.3
        # Sam3Predictor/Sam3Processor can accumulate internal vocab; reset regularly.
        self.reset_interval = int(os.getenv("SAM3_PREDICTOR_RESET_INTERVAL", "50"))
        self._images_since_reset = 0
        self._inference_lock = get_sam3_inference_lock(self.device)

    def _reset_predictor(self) -> None:
        self.model, self.processor = get_sam3_components(self.device)
        self._images_since_reset = 0

    def _ensure_model(self):
        if self.model is None or self.processor is None:
            self._reset_predictor()

    def __call__(
        self,
        optical_image_urls: List[str],
        text_prompt: str,
    ) -> Dict[str, Dict[str, Any]]:
        if not optical_image_urls or not isinstance(optical_image_urls, list):
            raise ValueError("optical_image_urls must be a non-empty list of URLs.")
        if not text_prompt or not isinstance(text_prompt, str):
            raise ValueError("text_prompt must be a non-empty string.")

        self._ensure_model()
        assert self.model is not None and self.processor is not None

        results: Dict[str, Dict[str, Any]] = {}
        for url in optical_image_urls:
            error = check_file_reference(url, label="image", strict=True)
            if error:
                results[url] = [{"error": error}]
                continue
            if RUN_ON_LOCAL:
                url = to_local_path(url)
            try:
                if url.startswith(("http://", "https://")):
                    response = requests.get(url, timeout=20)
                    response.raise_for_status()
                    with Image.open(BytesIO(response.content)) as opened:
                        image = opened.convert("RGB")
                else:
                    with Image.open(url) as opened:
                        image = opened.convert("RGB")
            except Exception as exc:
                results[url] = [{"error": f"failed_to_load_image: {exc}"}]
                continue

            with self._inference_lock:
                if self._images_since_reset >= max(1, self.reset_interval):
                    self._reset_predictor()
                self._images_since_reset += 1
                inputs = self.processor(images=image, text=text_prompt, return_tensors="pt").to(self.device)

                with torch.no_grad():
                    outputs = self.model(**inputs)

                processed = self.processor.post_process_instance_segmentation(
                    outputs,
                    threshold=self.score_threshold,
                    mask_threshold=self.mask_threshold,
                    target_sizes=inputs.get("original_sizes").tolist(),
                )[0]

            masks = processed.get("masks")
            scores = processed.get("scores")
            entries: List[Dict] = []
            if masks is not None:
                masks_np = masks.detach().cpu().numpy()
                scores_np = scores.detach().cpu().numpy() if scores is not None else np.ones(masks_np.shape[0])
                if masks_np.size > 0:
                    composite = None
                    valid_scores: List[float] = []
                    for mask_arr, score in zip(masks_np, scores_np):
                        if score is not None and float(score) < self.score_threshold:
                            continue
                        bin_mask = (mask_arr > self.mask_threshold).astype(np.uint8)
                        if bin_mask.sum() == 0:
                            continue
                        valid_scores.append(float(score))
                        composite = bin_mask if composite is None else np.maximum(composite, bin_mask)
                    if composite is not None:
                        saved = save_mask_image(
                            composite,
                            source_image_url=url,
                            label=text_prompt,
                            method="sam3",
                            metadata={
                                "score_max": max(valid_scores) if valid_scores else None,
                                "mask_count": len(valid_scores),
                            },
                        )
                        if not saved.get("bbox"):
                            saved["bbox"] = mask_to_bbox(composite)
                        entries.append(saved)
            log_image_event(
                "mask_batch_completed",
                {
                    "source_image_url": url,
                    "count": len(entries),
                    "label": text_prompt,
                    "method": "sam3",
                },
            )
            results[url] = {"image": describe_image_url(url), "masks": entries}
        return results


__all__ = ["GetObjectMaskByOpticalImage"]


class GetRoadMaskByOpticalImage(GetObjectMaskByOpticalImage):
    """
    Run object detection model to extract road masks for the provided image URLs.
    Road extraction shortcut; returns binary road masks per URL.
    Inputs: optical_image_urls (List[str]).
    Output: Dict[str, List[dict]] with mask_url metadata per image URL.
    """

    def __call__(
        self,
        optical_image_urls: List[str],
    ) -> Dict[str, List[Dict]]:
        return super().__call__(
            optical_image_urls=optical_image_urls,
            text_prompt="road",
        )


class GetBuildingMaskByOpticalImage(GetObjectMaskByOpticalImage):
    """
    Run object detection model to extract building masks for the provided image URLs.
    Building extraction shortcut; returns binary building masks per URL.
    Inputs: optical_image_urls (List[str]).
    Output: Dict[str, List[dict]] with mask_url metadata per image URL.
    """

    def __call__(
        self,
        optical_image_urls: List[str],
    ) -> Dict[str, List[Dict]]:
        return super().__call__(
            optical_image_urls=optical_image_urls,
            text_prompt="building",
        )


__all__ += ["GetRoadMaskByOpticalImage", "GetBuildingMaskByOpticalImage"]
