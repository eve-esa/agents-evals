"""
Run text-prompted SAM3 instance segmentation and return bounding boxes.

Inputs:
  - optical_image_urls (List[str]): HTTP/HTTPS URLs to optical images.
  - text_prompt (str): detection prompt (e.g., "ship", "chimney").

Output:
  - If one URL: {"image": {image_url, image_size, gsd}, "detections": List[dict]{bbox, score, label}}.
  - If multiple URLs: List of {"image": {...}, "detections": [...]} entries in input order.
"""

from __future__ import annotations

import os
from io import BytesIO
from typing import Any, Dict, List, Optional, Tuple

import torch
import requests
from PIL import Image
from transformers import Sam3Model, Sam3Processor

from eo_gym.runtime.core.config_loader import RUN_ON_LOCAL, to_local_path
from eo_gym.runtime.rs_tools.tools.ground_truth_cache import resolve_ground_truth_for_image
from eo_gym.runtime.rs_tools.tools.image_cache_logger import log_image_event
from eo_gym.runtime.rs_tools.tools.image_metadata import describe_image_url
from eo_gym.runtime.rs_tools.tools.lightweight_bbox_simulator import (
    DEFAULT_BBOX_MATCH_MODEL,
    LightweightBboxSimulator,
)
from eo_gym.runtime.rs_tools.utils import check_file_reference
from eo_gym.runtime.rs_tools.tools.model_cache import get_sam3_components, get_sam3_inference_lock


# Change this constant to "ground_truth" to compare cached GT detections against SAM3.
OPTICAL_BBOX_BACKEND = "ground_truth"
_VALID_OPTICAL_BBOX_BACKENDS = {"sam3", "ground_truth"}

if OPTICAL_BBOX_BACKEND not in _VALID_OPTICAL_BBOX_BACKENDS:
    raise ValueError(
        f"Unsupported OPTICAL_BBOX_BACKEND={OPTICAL_BBOX_BACKEND!r}. "
        f"Expected one of {sorted(_VALID_OPTICAL_BBOX_BACKENDS)}."
    )


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


def _resolve_ground_truth(image_path: str) -> Tuple[Optional[Dict[str, Any]], Optional[Dict[str, Any]]]:
    try:
        ground_truth, meta = resolve_ground_truth_for_image(str(image_path))
        return ground_truth, meta
    except Exception:
        return None, None


def _xywh_to_xyxy(bbox_xywh: Any) -> Optional[List[float]]:
    if not (isinstance(bbox_xywh, (list, tuple)) and len(bbox_xywh) == 4):
        return None
    x, y, w, h = [float(value) for value in bbox_xywh]
    if w <= 0 or h <= 0:
        return None
    return [x, y, x + w, y + h]


class GetObjectBboxByOpticalImageSam3:
    """
    Detect objects from optical images using text prompts and return bounding boxes. Text-prompt detection; returns per-URL metadata plus bounding boxes.
    Inputs: optical_image_urls (List[str]), text_prompt (str).
    Output:
      - If one URL: {"image": {image_url, image_size, gsd}, "detections": List[dict]{bbox, score, label}}.
      - If multiple URLs: List of {"image": {...}, "detections": [...]} entries in input order.
    """

    def __init__(self, device: Optional[str] = None, llm_model_name: Optional[str] = DEFAULT_BBOX_MATCH_MODEL):
        self.device = _resolve_device(device)
        self.model: Optional[Sam3Model] = None
        self.processor: Optional[Sam3Processor] = None
        self.mask_threshold = 0.3
        self.score_threshold = 0.3
        self.reset_interval = int(os.getenv("SAM3_PREDICTOR_RESET_INTERVAL", "50"))
        self._images_since_reset = 0
        self._inference_lock = get_sam3_inference_lock(self.device)
        self.llm_model_name = llm_model_name
        self._bbox_simulator: Optional[LightweightBboxSimulator] = None

    def _reset_predictor(self) -> None:
        self.model, self.processor = get_sam3_components(self.device)
        self._images_since_reset = 0

    def _ensure_model(self) -> None:
        if self.model is None or self.processor is None:
            self._reset_predictor()

    def _ensure_bbox_simulator(self) -> LightweightBboxSimulator:
        if self._bbox_simulator is None:
            self._bbox_simulator = LightweightBboxSimulator(model_name=self.llm_model_name)
        return self._bbox_simulator

    def _build_ground_truth_results(
        self,
        optical_image_urls: List[str],
        text_prompt: str,
    ) -> List[Dict[str, Any]]:
        prompt = text_prompt.strip()
        results: List[Dict[str, Any]] = []

        for raw_url in optical_image_urls:
            error = check_file_reference(raw_url, label="image", strict=True)
            if error:
                results.append(
                    {
                        "image": describe_image_url(raw_url),
                        "detections": [{"error": error}],
                    }
                )
                continue

            ground_truth, gt_meta = _resolve_ground_truth(raw_url)
            detections: List[Dict[str, Any]] = []
            if ground_truth:
                bbox_simulator = self._ensure_bbox_simulator()
                gt_detections = bbox_simulator.build_detections_from_ground_truth(
                    ground_truth,
                    prompt,
                    context=str(raw_url),
                    target_image_ref=str(raw_url),
                    meta=gt_meta,
                )
                for detection in gt_detections:
                    if not isinstance(detection, dict):
                        continue
                    bbox_xyxy = _xywh_to_xyxy(detection.get("bbox"))
                    if bbox_xyxy is None:
                        continue
                    score = detection.get("score")
                    if not isinstance(score, (int, float)):
                        score = 1.0
                    detections.append(
                        {
                            "bbox": bbox_xyxy,
                            "score": float(score),
                            "label": detection.get("label") or prompt,
                        }
                    )

            log_image_event(
                "bbox_batch_completed",
                {
                    "source_image_url": raw_url,
                    "count": len(detections),
                    "label": prompt,
                    "method": "ground_truth",
                },
            )
            results.append({"image": describe_image_url(raw_url), "detections": detections})

        return results

    def __call__(
        self,
        optical_image_urls: List[str],
        text_prompt: str,
    ) -> Any:
        if not optical_image_urls or not isinstance(optical_image_urls, list):
            raise ValueError("optical_image_urls must be a non-empty list of URLs.")
        if not text_prompt or not isinstance(text_prompt, str):
            raise ValueError("text_prompt must be a non-empty string.")

        if OPTICAL_BBOX_BACKEND == "ground_truth":
            results = self._build_ground_truth_results(optical_image_urls, text_prompt)
            if len(optical_image_urls) == 1:
                return results[0] if results else {"image": describe_image_url(optical_image_urls[0]), "detections": []}
            return results

        self._ensure_model()
        assert self.model is not None and self.processor is not None

        results: List[Dict[str, Any]] = []
        for url in optical_image_urls:
            error = check_file_reference(url, label="image", strict=True)
            if error:
                results.append(
                    {
                        "image": describe_image_url(url),
                        "detections": [{"error": error}],
                    }
                )
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
                results.append(
                    {
                        "image": describe_image_url(url),
                        "detections": [{"error": f"failed_to_load_image: {exc}"}],
                    }
                )
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

            boxes = processed.get("boxes")
            scores = processed.get("scores")
            detections: List[Dict[str, Any]] = []
            if boxes is not None:
                boxes_list = boxes.detach().cpu().tolist()
                scores_list = (
                    scores.detach().cpu().tolist() if scores is not None else [1.0] * len(boxes_list)
                )
                for box, score in zip(boxes_list, scores_list):
                    if not (isinstance(box, (list, tuple)) and len(box) == 4):
                        continue
                    x0, y0, x1, y1 = [float(v) for v in box]
                    detections.append(
                        {
                            "bbox": [x0, y0, x1, y1],
                            "score": float(score),
                            "label": text_prompt,
                        }
                    )
            log_image_event(
                "bbox_batch_completed",
                {
                    "source_image_url": url,
                    "count": len(detections),
                    "label": text_prompt,
                    "method": "sam3",
                },
            )
            results.append({"image": describe_image_url(url), "detections": detections})

        if len(optical_image_urls) == 1:
            return results[0] if results else {"image": describe_image_url(optical_image_urls[0]), "detections": []}
        return results


__all__ = ["GetObjectBboxByOpticalImageSam3"]
