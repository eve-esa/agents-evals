from __future__ import annotations

import json
import os
import re
from typing import Any, Dict, List, Optional, Sequence

from eo_gym.runtime.openai_transport import ResponsesTransport, extract_output_text
from eo_gym.runtime.core.llm_logger import log_interaction

DEFAULT_GROUND_TRUTH_MODEL = (os.getenv("GROUND_TRUTH_FILTER_MODEL") or "gpt-4.1").strip() or "gpt-4.1"


def normalize_label(label: str) -> str:
    return " ".join(str(label).lower().replace("_", " ").split())


def extract_ground_truth_label(obj: Dict[str, Any]) -> str:
    for key in ("category_name", "label", "category", "name", "class_name", "class"):
        val = obj.get(key)
        if isinstance(val, str) and val.strip():
            return val.strip()
    category_id = obj.get("category_id")
    if category_id is not None:
        return str(category_id)
    return ""


def _safe_json_loads(raw: str) -> Optional[Any]:
    if not raw or not isinstance(raw, str):
        return None
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return None


def _parse_keep_indices(text: str, label_count: int) -> Optional[List[int]]:
    raw = (text or "").strip()
    if not raw:
        return None
    data = _safe_json_loads(raw)
    if data is None:
        match = re.search(r"\{.*\}", raw, re.S)
        if match:
            data = _safe_json_loads(match.group(0))
    if data is None:
        match = re.search(r"\[.*\]", raw, re.S)
        if match:
            data = _safe_json_loads(match.group(0))

    candidate = None
    if isinstance(data, dict):
        for key in ("keep_indices", "indices", "keep", "matches"):
            if key in data:
                candidate = data.get(key)
                break
    elif isinstance(data, list):
        candidate = data

    if candidate is None:
        lowered = raw.lower()
        if re.search(r"\bnone\b|\bno match\b|\bno matches\b", lowered):
            return []
        numbers = re.findall(r"\d+", raw)
        if not numbers:
            return None
        candidate = numbers

    indices: List[int] = []
    for item in candidate:
        try:
            idx = int(item)
        except (TypeError, ValueError):
            continue
        indices.append(idx)
    if not indices:
        return []
    if min(indices) == 0 and max(indices) < label_count:
        indices = [i + 1 for i in indices]
    valid = sorted({i - 1 for i in indices if 1 <= i <= label_count})
    return valid


def select_label_indices_with_llm(
    labels: Sequence[str],
    text_prompt: str,
    *,
    model_name: Optional[str] = None,
    transport: Optional[ResponsesTransport] = None,
    context: Optional[str] = None,
) -> Optional[List[int]]:
    if not labels:
        return []
    if not text_prompt or not isinstance(text_prompt, str):
        return []
    resolved_model = (model_name or DEFAULT_GROUND_TRUTH_MODEL or "").strip()
    if not resolved_model:
        return None
    if not os.getenv("OPENAI_API_KEY"):
        return None

    label_lines = "\n".join(f"{idx + 1}. {label}" for idx, label in enumerate(labels))
    system_prompt = (
        "You match ground-truth object labels to a user query. "
        "Select labels that refer to the same object category, using synonyms and "
        "hypernyms or hyponyms when appropriate. Ignore attributes like color or size. "
        "If the query includes multiple targets, match any of them."
    )
    user_prompt = (
        f"Query: {text_prompt}\n"
        f"Labels:\n{label_lines}\n"
        'Return JSON only as {"keep_indices":[...]} using the label indices above. '
        'If none match, return {"keep_indices": []}.'
    )
    request_json = {
        "model": resolved_model,
        "input": [
            {"role": "system", "content": [{"type": "input_text", "text": system_prompt}]},
            {"role": "user", "content": [{"type": "input_text", "text": user_prompt}]},
        ],
        "temperature": 0,
        "max_output_tokens": 128,
    }

    try:
        transport = transport or ResponsesTransport()
        payload = transport.send(request_json=request_json)
    except Exception:
        return None

    response_text = extract_output_text(payload).strip()
    log_interaction(
        source="llm_ground_truth_filter",
        interaction_type="label_match",
        input_payload=request_json,
        response=response_text,
        metadata={"model": resolved_model, "label_count": len(labels), "context": context},
    )
    return _parse_keep_indices(response_text, len(labels))


def select_object_indices_with_llm(
    objects: Sequence[Dict[str, Any]],
    text_prompt: str,
    *,
    model_name: Optional[str] = None,
    transport: Optional[ResponsesTransport] = None,
    context: Optional[str] = None,
) -> Optional[List[int]]:
    if not objects:
        return []
    label_order: List[str] = []
    label_display: Dict[str, str] = {}
    label_to_indices: Dict[str, List[int]] = {}

    for idx, obj in enumerate(objects):
        if not isinstance(obj, dict):
            continue
        label = extract_ground_truth_label(obj)
        if not label:
            continue
        norm = normalize_label(label)
        if not norm:
            continue
        if norm not in label_to_indices:
            label_to_indices[norm] = []
            label_order.append(norm)
            label_display[norm] = label.strip()
        label_to_indices[norm].append(idx)

    if not label_order:
        return []

    labels = [label_display[key] for key in label_order]
    selected_label_indices = select_label_indices_with_llm(
        labels,
        text_prompt,
        model_name=model_name,
        transport=transport,
        context=context,
    )
    if selected_label_indices is None:
        return None

    selected_object_indices: List[int] = []
    for label_idx in selected_label_indices:
        if 0 <= label_idx < len(label_order):
            selected_object_indices.extend(label_to_indices[label_order[label_idx]])
    return sorted(set(selected_object_indices))


__all__ = [
    "DEFAULT_GROUND_TRUTH_MODEL",
    "extract_ground_truth_label",
    "normalize_label",
    "select_label_indices_with_llm",
    "select_object_indices_with_llm",
]
