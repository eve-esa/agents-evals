"""
LLM-based helpers for optical imagery (classification + object description).
Uses local Qwen2.5-VL backend.
"""

from __future__ import annotations

import hashlib
import os
import threading
from pathlib import Path
from typing import Dict

import requests
from eo_gym.runtime.core.llm_logger import log_interaction
import eo_gym.runtime.core.config_loader as config_loader
from eo_gym.runtime.rs_tools.utils import check_file_reference


def _describe_model_ref(model_id: str) -> str:
    raw = str(model_id or "").strip()
    if not raw:
        return raw
    candidate = Path(raw).expanduser()
    if candidate.exists():
        try:
            return str(candidate.resolve())
        except Exception:
            return str(candidate)
    return raw


DEFAULT_QWEN_MODEL_ID = "Qwen/Qwen2.5-VL-3B-Instruct"


def _resolve_qwen_model_id() -> str:
    for env_key in ("TOOL_LOCAL_VLM_MODEL_ID", "TOOL_LOCAL_VLM_MODEL_PATH", "QWEN_MODEL_ID"):
        value = str(os.getenv(env_key, "")).strip()
        if value:
            return value
    return str(getattr(config_loader, "LOCAL_VLM_MODEL_ID_OR_PATH", DEFAULT_QWEN_MODEL_ID) or DEFAULT_QWEN_MODEL_ID)


def _is_explicit_local_model_path(model_id: str) -> bool:
    raw = str(model_id or "").strip()
    if not raw:
        return False
    if raw.startswith(("~", "/", "./", "../")):
        return True
    return Path(raw).expanduser().exists()


QWEN_MODEL_ID = _resolve_qwen_model_id()
QWEN_ATTN_IMPLEMENTATION = "sdpa"
_QWEN_MODEL = None
_QWEN_PROCESSOR = None
_QWEN_LOCK = threading.Lock()
LOCAL_IMAGE_DOWNLOAD_TIMEOUT = float(os.getenv("TOOL_IMAGE_DOWNLOAD_TIMEOUT", "120"))
LOCAL_IMAGE_DOWNLOAD_CHUNK_SIZE = 1024 * 1024
_local_files_only_raw = os.getenv("TOOL_LOCAL_VLM_LOCAL_FILES_ONLY", "").strip()
if _local_files_only_raw:
    LOCAL_VLM_LOCAL_FILES_ONLY = _local_files_only_raw.lower() not in {"0", "false", "no", "off"}
else:
    LOCAL_VLM_LOCAL_FILES_ONLY = bool(getattr(config_loader, "LOCAL_VLM_LOCAL_FILES_ONLY", True))
if LOCAL_VLM_LOCAL_FILES_ONLY:
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

def _download_image_to_cache(url: str) -> str:
    url_str = str(url)
    cache_dir = config_loader.IMAGE_CACHE_DIR
    cache_dir.mkdir(parents=True, exist_ok=True)
    url_no_query = url_str.split("?", 1)[0]
    name = Path(url_no_query).name or "image"
    stem = Path(name).stem or "image"
    suffix = Path(name).suffix
    digest = hashlib.sha256(url_str.encode("utf-8")).hexdigest()[:12]
    filename = f"{stem}_{digest}{suffix}" if suffix else f"{stem}_{digest}"
    local_path = cache_dir / filename
    if local_path.exists() and local_path.stat().st_size > 0:
        return str(local_path.resolve())
    resp = requests.get(url_str, stream=True, timeout=LOCAL_IMAGE_DOWNLOAD_TIMEOUT)
    resp.raise_for_status()
    with local_path.open("wb") as fh:
        for chunk in resp.iter_content(chunk_size=LOCAL_IMAGE_DOWNLOAD_CHUNK_SIZE):
            if chunk:
                fh.write(chunk)
    return str(local_path.resolve())


def _ensure_local_image_path(optical_image_path: str) -> str:
    raw = str(optical_image_path or "")
    if not raw:
        raise ValueError("optical_image_path is required.")
    if raw.startswith("file://"):
        raw = raw[len("file://") :]

    if raw.startswith(("http://", "https://")):
        local_candidate = config_loader.to_local_path(raw)
        if not local_candidate.startswith(("http://", "https://")):
            candidate_path = Path(local_candidate).expanduser()
            if candidate_path.exists():
                return str(candidate_path.resolve())
        return _download_image_to_cache(raw)

    candidate_str = config_loader.to_local_path(raw)
    if candidate_str.startswith(("http://", "https://")):
        return _download_image_to_cache(candidate_str)
    candidate_path = Path(candidate_str).expanduser()
    if candidate_path.exists():
        return str(candidate_path.resolve())
    raise FileNotFoundError(f"Local image not found: {candidate_path}")


def _load_vlm_model_with_fallback(loader_cls, model_id: str):
    model_ref = _describe_model_ref(model_id)
    try:
        print(
            f"[ai_optical_analysis] Loading {loader_cls.__name__} from: {model_ref} "
            f"(attn_implementation={QWEN_ATTN_IMPLEMENTATION}, local_files_only={LOCAL_VLM_LOCAL_FILES_ONLY})"
        )
        return loader_cls.from_pretrained(
            model_id,
            torch_dtype="auto",
            device_map="auto",
            attn_implementation=QWEN_ATTN_IMPLEMENTATION,
            local_files_only=LOCAL_VLM_LOCAL_FILES_ONLY,
        ).eval()
    except Exception as exc:
        raise RuntimeError(
            f"Failed to load local VLM model '{model_id}' "
            f"with attn_implementation='{QWEN_ATTN_IMPLEMENTATION}': {exc}"
        ) from exc


def _load_local_vlm_components(model_id: str):
    resolved_model_id = str(model_id or "").strip()
    if not resolved_model_id:
        raise RuntimeError(
            "Local VLM model ID/path is empty. Set TOOL_LOCAL_VLM_MODEL_ID "
            f"or TOOL_LOCAL_VLM_MODEL_PATH (default: {DEFAULT_QWEN_MODEL_ID})."
        )
    if _is_explicit_local_model_path(resolved_model_id):
        model_path = Path(resolved_model_id).expanduser()
        if not model_path.exists():
            raise FileNotFoundError(f"Local VLM model path not found: {model_path}")
    model_ref = _describe_model_ref(resolved_model_id)
    try:
        from transformers import AutoModelForVision2Seq, AutoProcessor
    except Exception:
        AutoModelForVision2Seq = None
        AutoProcessor = None
    if AutoModelForVision2Seq is not None:
        model = _load_vlm_model_with_fallback(AutoModelForVision2Seq, resolved_model_id)
        print(
            f"[ai_optical_analysis] Loading AutoProcessor from: {model_ref} "
            f"(local_files_only={LOCAL_VLM_LOCAL_FILES_ONLY})"
        )
        processor = AutoProcessor.from_pretrained(
            resolved_model_id,
            local_files_only=LOCAL_VLM_LOCAL_FILES_ONLY,
        )
        return model, processor

    if AutoProcessor is None:
        from transformers import AutoProcessor
    from transformers import Qwen2_5_VLForConditionalGeneration

    model = _load_vlm_model_with_fallback(Qwen2_5_VLForConditionalGeneration, resolved_model_id)
    print(
        f"[ai_optical_analysis] Loading AutoProcessor from: {model_ref} "
        f"(local_files_only={LOCAL_VLM_LOCAL_FILES_ONLY})"
    )
    processor = AutoProcessor.from_pretrained(
        resolved_model_id,
        local_files_only=LOCAL_VLM_LOCAL_FILES_ONLY,
    )
    return model, processor


def _get_qwen_components():
    global _QWEN_MODEL, _QWEN_PROCESSOR
    if _QWEN_MODEL is None or _QWEN_PROCESSOR is None:
        with _QWEN_LOCK:
            if _QWEN_MODEL is None or _QWEN_PROCESSOR is None:
                _QWEN_MODEL, _QWEN_PROCESSOR = _load_local_vlm_components(QWEN_MODEL_ID)
    return _QWEN_MODEL, _QWEN_PROCESSOR


def preload_qwen_components() -> None:
    """
    Warm up the local Qwen2.5-VL model/processor so later tool calls reuse it.
    """
    _get_qwen_components()


def _run_local_qwen(optical_image_path: str, prompt: str, max_new_tokens: int = 128) -> str:
    model, processor = _get_qwen_components()
    if model is None or processor is None:
        raise RuntimeError("Qwen2.5-VL model is not available.")
    apply_chat_template = getattr(processor, "apply_chat_template", None)
    if not callable(apply_chat_template):
        raise RuntimeError("Local VLM processor.apply_chat_template is not callable.")
    if not callable(processor):
        raise RuntimeError("Local VLM processor is not callable.")
    generate_fn = getattr(model, "generate", None)
    if not callable(generate_fn):
        raise RuntimeError("Local VLM model.generate is not callable.")
    try:
        from qwen_vl_utils import process_vision_info
    except Exception as exc:
        raise RuntimeError(
            "qwen_vl_utils.process_vision_info is required for the local Qwen backend."
        ) from exc
    if not callable(process_vision_info):
        raise RuntimeError(
            "qwen_vl_utils.process_vision_info is not callable; check your qwen_vl_utils install."
        )

    image_ref = _ensure_local_image_path(optical_image_path)
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image_ref},
                {"type": "text", "text": prompt},
            ],
        }
    ]
    text = apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )
    device = getattr(model, "device", None)
    if device is None:
        device = next(model.parameters()).device
    inputs = inputs.to(device)
    import torch

    with torch.no_grad():
        generated_ids = generate_fn(**inputs, max_new_tokens=max_new_tokens)
    generated_ids_trimmed = [
        out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = processor.batch_decode(
        generated_ids_trimmed,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )
    return output_text[0].strip() if output_text else ""


class AnalyzeOpticalScene:
    """
    Classify a scene from an optical image using an LLM with vision support. Remember to mention if you pass a masking or SAR image in given prompt_text. LLM vision scene classification; returns a scene label string only (not for counting/detection).

    Inputs: optical_image_path (str), prompt_text (str).
    Output: classification result (str).
    """

    def __init__(self, device: str = "cpu"):
        self.device = device

    def __call__(self, optical_image_path: str, prompt_text: str) -> Dict:
        if not prompt_text or not isinstance(prompt_text, str):
            raise ValueError("prompt_text must be a non-empty string.")
        try:
            optical_image_path = str(optical_image_path)
            prompt_text = str(prompt_text)
            error = check_file_reference(optical_image_path, label="image", strict=True)
            if error:
                return {"error": error, "image_path": optical_image_path, "prompt_text": prompt_text}
            text = _run_local_qwen(
                optical_image_path,
                prompt_text,
            )
            log_interaction(
                source="ai_optical_analysis",
                interaction_type="optical_scene",
                input_payload={"optical_image_path": optical_image_path, "prompt_text": prompt_text},
                response=text,
                metadata={
                    "model": QWEN_MODEL_ID,
                    "local_fallback": False,
                    "local_error": None,
                },
            )
            return text
        except Exception as exc:
            return {
                "error": f"Failed to classify scene locally: {exc}",
                "image_path": optical_image_path,
                "prompt_text": prompt_text,
            }


class DescribeOpticalObject:
    """
    Describe object attributes (e.g., color) from an optical image via an LLM.
    LLM vision description of a named object’s attributes; not for detection or counting.
    Inputs: optical_image_path (str), object_name (str).
    Output: dict with `description` (str), or `error`.
    """

    def __init__(self, device: str = "cpu"):
        self.device = device

    def __call__(self, optical_image_path: str, object_name: str) -> Dict:
        if not object_name or not isinstance(object_name, str):
            raise ValueError("object_name must be a non-empty string.")
        try:
            optical_image_path = str(optical_image_path)
            error = check_file_reference(optical_image_path, label="image", strict=True)
            if error:
                return {"error": error, "image_path": optical_image_path, "object_name": object_name}
            prompt_text = f"Describe the {object_name} in this optical image. Mention color and notable attributes."
            text = _run_local_qwen(
                optical_image_path,
                prompt_text,
            )
            log_interaction(
                source="ai_optical_analysis",
                interaction_type="object_description",
                input_payload={"optical_image_path": optical_image_path, "object_name": object_name},
                response=text,
                metadata={
                    "model": QWEN_MODEL_ID,
                    "local_fallback": False,
                    "local_error": None,
                },
            )
            return {"object_name": object_name, "description": text}
        except Exception as exc:
            return {
                "error": f"Failed to describe object locally: {exc}",
                "image_path": optical_image_path,
                "object_name": object_name,
            }


__all__ = ["AnalyzeOpticalScene", "DescribeOpticalObject", "preload_qwen_components"]
