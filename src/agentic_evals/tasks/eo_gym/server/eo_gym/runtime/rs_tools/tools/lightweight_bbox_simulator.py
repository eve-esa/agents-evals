from __future__ import annotations

import os
import re
import threading
from functools import lru_cache
from io import BytesIO
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence

import requests

from eo_gym.runtime.core import config_loader
from eo_gym.runtime.core.config_loader import RUN_ON_LOCAL, to_local_path
from eo_gym.runtime.rs_tools.tools.image_metadata import describe_image_url
from eo_gym.runtime.rs_tools.tools.llm_ground_truth_filter import extract_ground_truth_label


def _resolve_default_bbox_match_model() -> str:
    configured = (os.getenv("BBOX_MATCH_MODEL") or "").strip()
    if configured:
        return configured
    configured_path = str(
        getattr(config_loader, "BBOX_MATCH_MODEL_ID_OR_PATH", "Qwen/Qwen2.5-0.5B-Instruct")
        or "Qwen/Qwen2.5-0.5B-Instruct"
    ).strip()
    return configured_path


DEFAULT_BBOX_MATCH_MODEL = _resolve_default_bbox_match_model()
_TINY_MATCHER_LOCK = threading.Lock()
_TINY_MATCHERS: Dict[str, "_TinyLLMMatcher"] = {}


def _normalize_label_text(label: str) -> str:
    return " ".join(str(label).lower().replace("_", " ").split())


def _split_prompt_targets(text_prompt: str) -> List[str]:
    raw = (text_prompt or "").strip()
    if not raw:
        return []
    cleaned = re.sub(r"[;/]+", ",", raw)
    parts = [p.strip() for p in re.split(r",|\band\b|\bor\b", cleaned, flags=re.IGNORECASE)]
    return [p for p in parts if p] or [raw]


_GENERIC_MATCH_TOKENS = {
    "a",
    "an",
    "all",
    "and",
    "any",
    "area",
    "areas",
    "based",
    "by",
    "compare",
    "count",
    "current",
    "detect",
    "determine",
    "equal",
    "find",
    "footprint",
    "footprints",
    "for",
    "from",
    "get",
    "given",
    "greater",
    "how",
    "identify",
    "image",
    "images",
    "in",
    "is",
    "less",
    "locate",
    "many",
    "median",
    "number",
    "object",
    "objects",
    "of",
    "on",
    "or",
    "part",
    "position",
    "primarily",
    "provided",
    "ratio",
    "relative",
    "scene",
    "scenes",
    "select",
    "selected",
    "show",
    "small",
    "than",
    "target",
    "targets",
    "the",
    "their",
    "this",
    "those",
    "visible",
    "what",
    "where",
    "with",
}


def _tokenize_label_words(text: str) -> List[str]:
    normalized = _normalize_label_text(text)
    normalized = re.sub(r"(?<=[a-z])(?=[0-9])|(?<=[0-9])(?=[a-z])", " ", normalized)
    return re.findall(r"[a-z0-9]+", normalized)


def _collapse_alnum(text: str) -> str:
    return "".join(_tokenize_label_words(text))


def _singularize_token(token: str) -> str:
    raw = str(token or "").strip().lower()
    if len(raw) <= 3:
        return raw
    if raw.endswith("ouses") and len(raw) > 6:
        return raw[:-1]
    if raw.endswith("ies") and len(raw) > 4:
        return raw[:-3] + "y"
    if raw.endswith("es") and len(raw) > 4:
        if raw.endswith(("ches", "shes", "sses", "xes", "zes")):
            return raw[:-2]
        if raw.endswith("ses"):
            return raw[:-1]
        return raw[:-1]
    if raw.endswith("s") and not raw.endswith("ss"):
        return raw[:-1]
    return raw


def _semantic_tokens(text: str) -> List[str]:
    raw_tokens = [_singularize_token(token) for token in _tokenize_label_words(text)]
    filtered = [token for token in raw_tokens if token and token not in _GENERIC_MATCH_TOKENS]
    return filtered or raw_tokens


def _candidate_match_rank(target_word: str, candidate: str) -> Optional[tuple[int, ...]]:
    norm_target = _normalize_label_text(target_word)
    norm_candidate = _normalize_label_text(candidate)
    if not norm_target or not norm_candidate:
        return None
    collapsed_target = _collapse_alnum(target_word)
    collapsed_candidate = _collapse_alnum(candidate)

    target_tokens = _semantic_tokens(target_word)
    candidate_tokens = _semantic_tokens(candidate)
    if not target_tokens or not candidate_tokens:
        return None

    target_set = set(target_tokens)
    candidate_set = set(candidate_tokens)
    overlap = len(target_set & candidate_set)
    exact = norm_target == norm_candidate
    collapsed_exact = collapsed_target == collapsed_candidate and bool(collapsed_target)
    candidate_covered = candidate_set.issubset(target_set)
    target_covered = target_set.issubset(candidate_set)
    phrase_overlap = norm_candidate in norm_target or norm_target in norm_candidate
    collapsed_overlap = (
        bool(collapsed_target)
        and bool(collapsed_candidate)
        and (collapsed_candidate in collapsed_target or collapsed_target in collapsed_candidate)
    )
    head_match = target_tokens[-1] == candidate_tokens[-1]
    accept = (
        exact
        or collapsed_exact
        or candidate_covered
        or target_covered
        or phrase_overlap
        or collapsed_overlap
        or overlap >= 2
        or (overlap == 1 and head_match)
    )
    if not accept:
        return None

    return (
        1 if exact else 0,
        1 if collapsed_exact else 0,
        overlap,
        1 if candidate_covered else 0,
        1 if target_covered else 0,
        1 if phrase_overlap else 0,
        1 if collapsed_overlap else 0,
        1 if head_match else 0,
        len(candidate_tokens) if target_covered else -abs(len(candidate_tokens) - len(target_tokens)),
        -abs(len(norm_candidate) - len(norm_target)),
    )


def _find_deterministic_matches(target_word: str, candidates: Sequence[str]) -> List[str]:
    best_rank: Optional[tuple[int, ...]] = None
    matches: List[str] = []
    for candidate in candidates:
        rank = _candidate_match_rank(target_word, candidate)
        if rank is None:
            continue
        if best_rank is None or rank > best_rank:
            best_rank = rank
            matches = [candidate]
            continue
        if rank == best_rank:
            matches.append(candidate)
    return matches


def _as_size_tuple(value: Any) -> Optional[tuple[float, float]]:
    if isinstance(value, (list, tuple)) and len(value) == 2:
        w, h = value
    elif isinstance(value, dict):
        w = value.get("width")
        h = value.get("height")
    else:
        return None
    if not (isinstance(w, (int, float)) and isinstance(h, (int, float))):
        return None
    if float(w) <= 0 or float(h) <= 0:
        return None
    return float(w), float(h)


def _as_rect(value: Any) -> Optional[tuple[float, float, float, float]]:
    if not isinstance(value, dict):
        return None
    x = value.get("x")
    y = value.get("y")
    w = value.get("width")
    h = value.get("height")
    if not all(isinstance(v, (int, float)) for v in (x, y, w, h)):
        return None
    if float(w) <= 0 or float(h) <= 0:
        return None
    return float(x), float(y), float(w), float(h)


def _resolve_gt_size(ground_truth: Optional[Dict[str, Any]]) -> Optional[tuple[float, float]]:
    if not isinstance(ground_truth, dict):
        return None
    return _as_size_tuple(
        (
            ground_truth.get("image_width") or ground_truth.get("width"),
            ground_truth.get("image_height") or ground_truth.get("height"),
        )
    )


@lru_cache(maxsize=4096)
def _load_image_size(image_ref: str) -> Optional[tuple[float, float]]:
    raw = str(image_ref or "").strip()
    if not raw:
        return None

    described = describe_image_url(raw).get("image_size")
    parsed = _as_size_tuple(described)
    if parsed is not None:
        return parsed

    local_candidate = to_local_path(raw) if RUN_ON_LOCAL else raw
    try:
        from PIL import Image
    except Exception:
        return None
    try:
        if str(local_candidate).startswith(("http://", "https://")):
            response = requests.get(str(local_candidate), timeout=20)
            response.raise_for_status()
            with Image.open(BytesIO(response.content)) as opened:
                w, h = opened.size
                return float(w), float(h)
        with Image.open(str(local_candidate)) as opened:
            w, h = opened.size
            return float(w), float(h)
    except Exception:
        return None


def _maybe_crop_to_location(
    bbox_xywh: tuple[float, float, float, float],
    *,
    ground_truth: Optional[Dict[str, Any]],
    meta: Optional[Dict[str, Any]],
    gt_size: Optional[tuple[float, float]],
) -> tuple[tuple[float, float, float, float], Optional[tuple[float, float]]]:
    x, y, w, h = bbox_xywh

    location_rect = _as_rect((ground_truth or {}).get("crop_rect"))
    if location_rect is None and isinstance(meta, dict):
        location_rect = _as_rect(meta.get("crop_rect"))
    if location_rect is None:
        return (x, y, w, h), gt_size

    lx, ly, lw, lh = location_rect
    gt_w = gt_size[0] if gt_size else None
    gt_h = gt_size[1] if gt_size else None
    looks_like_source_frame = True
    if isinstance(gt_w, (int, float)) and isinstance(gt_h, (int, float)):
        if abs(float(gt_w) - lw) <= 1e-3 and abs(float(gt_h) - lh) <= 1e-3:
            looks_like_source_frame = False

    if not looks_like_source_frame:
        return (x, y, w, h), (lw, lh)

    x1 = x + w
    y1 = y + h
    ix0 = max(lx, x)
    iy0 = max(ly, y)
    ix1 = min(lx + lw, x1)
    iy1 = min(ly + lh, y1)
    if ix1 <= ix0 or iy1 <= iy0:
        return (0.0, 0.0, -1.0, -1.0), (lw, lh)
    return (ix0 - lx, iy0 - ly, ix1 - ix0, iy1 - iy0), (lw, lh)


def _scale_bbox_to_target(
    bbox_xywh: tuple[float, float, float, float],
    *,
    source_size: Optional[tuple[float, float]],
    target_size: Optional[tuple[float, float]],
) -> Optional[List[float]]:
    x, y, w, h = bbox_xywh
    if w <= 0 or h <= 0:
        return None

    if source_size is not None and target_size is not None:
        src_w, src_h = source_size
        tgt_w, tgt_h = target_size
        if src_w > 0 and src_h > 0:
            x *= tgt_w / src_w
            y *= tgt_h / src_h
            w *= tgt_w / src_w
            h *= tgt_h / src_h

    if target_size is not None:
        tgt_w, tgt_h = target_size
        x = max(0.0, min(x, tgt_w))
        y = max(0.0, min(y, tgt_h))
        w = max(0.0, min(w, tgt_w - x))
        h = max(0.0, min(h, tgt_h - y))

    if w <= 0 or h <= 0:
        return None
    return [float(x), float(y), float(w), float(h)]


class _TinyLLMMatcher:
    def __init__(self, model_name: str) -> None:
        self.model_name = model_name
        self.model = None
        self.tokenizer = None
        self._torch = None
        self.ready = False
        self.error: Optional[str] = None

        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer

            self._torch = torch
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = self._load_model(AutoModelForCausalLM, model_name)
            self.ready = True
        except Exception as exc:
            self.error = str(exc)
            self.ready = False

    def _load_model(self, loader: Any, model_name: str) -> Any:
        errors: List[Exception] = []
        for impl in ("flash_attention_2", "sdpa", "eager"):
            try:
                return loader.from_pretrained(
                    model_name,
                    torch_dtype="auto",
                    device_map="auto",
                    attn_implementation=impl,
                )
            except TypeError as exc:
                if "attn_implementation" in str(exc):
                    break
                errors.append(exc)
            except Exception as exc:
                errors.append(exc)
        try:
            return loader.from_pretrained(
                model_name,
                torch_dtype="auto",
                device_map="auto",
            )
        except Exception as exc:
            errors.append(exc)
            detail = "; ".join(str(err) for err in errors if err)
            raise RuntimeError(detail or "Unable to load tiny matcher model.") from exc

    def _match_candidate(self, reply: str, candidates: Sequence[str]) -> Optional[str]:
        cleaned = (reply or "").strip().strip("\"'`")
        cleaned = cleaned.splitlines()[-1].strip() if cleaned else ""
        if not cleaned:
            return None
        lowered = cleaned.lower()
        if lowered in {"none", "null", "no match"}:
            return None
        norm_reply = _normalize_label_text(cleaned)
        norm_map = {_normalize_label_text(cand): cand for cand in candidates if cand}
        if norm_reply in norm_map:
            return norm_map[norm_reply]
        for norm, cand in norm_map.items():
            if norm_reply and (norm_reply in norm or norm in norm_reply):
                return cand
        return None

    def find_match(self, target_word: str, word_list: Sequence[str]) -> Optional[str]:
        if not target_word or not word_list:
            return None
        deterministic = _find_deterministic_matches(target_word, word_list)
        if deterministic:
            return deterministic[0]
        if not self.ready or not self.model or not self.tokenizer or not self._torch:
            return None

        prompt = (
            "\n"
            "Task: Identify the word from the candidate list that matches the target word.\n"
            f'Target Word: "{target_word}"\n'
            f"Candidate List: {list(word_list)}\n\n"
            "Instruction: If there is a synonym or highly similar word in the list, output ONLY that word.\n"
            'If there is no match, output "None".\n'
            "Answer:\n"
        )
        messages = [
            {"role": "system", "content": "You are a helpful assistant that matches similar words."},
            {"role": "user", "content": prompt},
        ]
        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        model_inputs = self.tokenizer([text], return_tensors="pt").to(self.model.device)
        with self._torch.no_grad():
            generated_ids = self.model.generate(
                **model_inputs,
                max_new_tokens=10,
                do_sample=False,
            )
        response = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
        reply = response.split("Answer:")[-1].strip() if "Answer:" in response else response.strip()
        return self._match_candidate(reply, word_list)


def _get_tiny_matcher(model_name: str) -> _TinyLLMMatcher:
    with _TINY_MATCHER_LOCK:
        matcher = _TINY_MATCHERS.get(model_name)
        if matcher is None:
            matcher = _TinyLLMMatcher(model_name)
            _TINY_MATCHERS[model_name] = matcher
        return matcher


class LightweightBboxSimulator:
    """Lightweight GT bbox selector driven by a tiny local text model."""

    def __init__(
        self,
        *,
        model_name: Optional[str] = None,
        warn_fn: Optional[Callable[[str], None]] = None,
    ) -> None:
        self.model_name = str(model_name or DEFAULT_BBOX_MATCH_MODEL).strip() or DEFAULT_BBOX_MATCH_MODEL
        self.matcher = _get_tiny_matcher(self.model_name)
        self.warn_fn = warn_fn
        self._fallback_warned = False

    def _warn(self, message: str) -> None:
        if self.warn_fn:
            try:
                self.warn_fn(message)
                return
            except Exception:
                pass
        print(message)

    def select_objects(
        self,
        objects: Sequence[Dict[str, Any]],
        text_prompt: str,
        *,
        context: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        del context
        labels = [
            extract_ground_truth_label(obj)
            for obj in objects
            if isinstance(obj, dict) and extract_ground_truth_label(obj)
        ]
        if not labels:
            return []
        candidates = sorted({label.strip() for label in labels if isinstance(label, str) and label.strip()})
        if not candidates:
            return []

        matched_norms = set()
        unresolved_targets: List[str] = []
        for target in _split_prompt_targets(text_prompt):
            deterministic = _find_deterministic_matches(target, candidates)
            if deterministic:
                matched_norms.update(_normalize_label_text(match) for match in deterministic)
                continue
            unresolved_targets.append(target)

        if unresolved_targets and (not self.matcher or not self.matcher.ready):
            if matched_norms:
                unresolved_targets = []
            else:
                if not self._fallback_warned:
                    detail = f" ({self.matcher.error})" if self.matcher and self.matcher.error else ""
                    self._warn(
                        f"[WARN] BBox simulator LLM unavailable{detail}; returning unfiltered ground truth."
                    )
                    self._fallback_warned = True
                return [obj for obj in objects if isinstance(obj, dict)]

        for target in unresolved_targets:
            matched = self.matcher.find_match(target, candidates)
            if matched:
                matched_norms.add(_normalize_label_text(matched))
        if not matched_norms:
            return []

        selected: List[Dict[str, Any]] = []
        for obj in objects:
            if not isinstance(obj, dict):
                continue
            label = extract_ground_truth_label(obj)
            if label and _normalize_label_text(label) in matched_norms:
                selected.append(obj)
        return selected

    def build_detections(
        self,
        objects: Sequence[Dict[str, Any]],
        *,
        text_prompt: str,
        ground_truth: Optional[Dict[str, Any]] = None,
        target_image_ref: Optional[str] = None,
        meta: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        detections: List[Dict[str, Any]] = []
        gt_size = _resolve_gt_size(ground_truth)
        target_size = _load_image_size(str(target_image_ref)) if target_image_ref else None
        for obj in objects:
            if not isinstance(obj, dict):
                continue
            bbox = obj.get("bbox")
            if not (isinstance(bbox, (list, tuple)) and len(bbox) == 4):
                continue
            x, y, w, h = [float(v) for v in bbox]
            remapped_bbox, source_size = _maybe_crop_to_location(
                (x, y, w, h),
                ground_truth=ground_truth,
                meta=meta,
                gt_size=gt_size,
            )
            mapped_bbox = _scale_bbox_to_target(
                remapped_bbox,
                source_size=source_size or gt_size,
                target_size=target_size,
            )
            if mapped_bbox is None:
                continue
            score = obj.get("score")
            if not isinstance(score, (int, float)):
                score = 1.0
            detections.append(
                {
                    "bbox": mapped_bbox,
                    "score": float(score),
                    "label": extract_ground_truth_label(obj) or text_prompt,
                }
            )
        return detections

    def build_detections_from_ground_truth(
        self,
        ground_truth: Optional[Dict[str, Any]],
        text_prompt: str,
        *,
        context: Optional[str] = None,
        target_image_ref: Optional[str] = None,
        meta: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        objects = [obj for obj in (ground_truth or {}).get("objects") or [] if isinstance(obj, dict)]
        selected = self.select_objects(objects, text_prompt, context=context)
        return self.build_detections(
            selected,
            text_prompt=text_prompt,
            ground_truth=ground_truth,
            target_image_ref=target_image_ref,
            meta=meta,
        )


__all__ = ["DEFAULT_BBOX_MATCH_MODEL", "LightweightBboxSimulator"]
