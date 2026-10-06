from __future__ import annotations

import re
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

QUESTION_TYPE_BUCKET_MAP_OVERRIDES: Mapping[str, str] = {
    "moving reasoning": "crop_roi",
    "cut reasoning": "crop_roi",
    "crop image reasoning": "crop_roi",
    "multispectral temporal analysis": "temporal_motion",
    "optical temporal analysis": "temporal_motion",
    "damage_severity": "damage_impact",
}
QUESTION_TYPE_BUCKET_DISPLAY_NAMES: Mapping[str, str] = {
    "crop_roi": "Spatial Navigation",
    "temporal_motion": "Temporal Reasoning",
    "damage_impact": "Disaster Impact",
    "counting_arithmetic": "Object Counting",
    "spatial_geometry": "Geospatial Reasoning",
    "visual_understanding": "Visual Understanding",
}

TAXONOMY_BUCKET_SPECS: Sequence[Mapping[str, Any]] = (
    {
        "key": "damage_impact",
        "question_types": (
            "damage_presence",
            "damage_count",
            "damage_majority",
            "damage_severity",
            "demange_change",
            "damage_change",
        ),
    },
    {
        "key": "temporal_motion",
        "question_types": (
            "multispectral temporal analysis",
            "optical temporal analysis",
        ),
    },
    {
        "key": "crop_roi",
        "question_types": (
            "cut reasoning",
            "moving reasoning",
            "crop image reasoning",
        ),
    },
    {
        "key": "visual_understanding",
        "question_types": (
            "scene understanding",
            "cross-modal analysis",
            "object attribute",
        ),
    },
    {
        "key": "counting_arithmetic",
        "question_types": (
            "counting",
            "arithmetic reasoning",
        ),
    },
    {
        "key": "spatial_geometry",
        "question_types": (
            "area dominance",
            "orientation",
            "overlap analysis",
            "spatial distribution",
            "spatial relationship",
        ),
    },
)

TAXONOMY_BUCKET_ORDER = [str(spec["key"]) for spec in TAXONOMY_BUCKET_SPECS]
QUESTION_TYPE_BUCKET_DISPLAY_TO_KEY = {
    str(display_name): bucket_key
    for bucket_key, display_name in QUESTION_TYPE_BUCKET_DISPLAY_NAMES.items()
}

DEFAULT_QUESTION_TYPE_TO_BUCKET: Dict[str, str] = {}
for bucket_spec in TAXONOMY_BUCKET_SPECS:
    bucket_key = str(bucket_spec["key"])
    for question_type in bucket_spec["question_types"]:
        DEFAULT_QUESTION_TYPE_TO_BUCKET[str(question_type)] = bucket_key


def normalize_question_type_key(question_type: Any) -> str:
    return re.sub(r"\s+", " ", str(question_type or "").replace("_", " ").strip().lower())


def normalize_data_source_key(data_source: Any) -> str:
    return re.sub(r"[^a-z0-9]+", "_", str(data_source or "").strip().lower()).strip("_")


def clean_question_type(
    question_type: Any,
    data_source: Any,
    *,
    apply_xbd_alias: bool = False,
) -> str:
    raw_question_type = str(question_type or "").strip()
    if not raw_question_type:
        return ""

    normalized_question_type = normalize_question_type_key(raw_question_type)
    normalized_data_source = normalize_data_source_key(data_source)

    if normalized_question_type == "temporal analysis":
        if normalized_data_source == "fair1m_ms":
            return "multispectral temporal analysis"
        if normalized_data_source == "fmow":
            return "optical temporal analysis"
    if normalized_question_type == "damage max severity":
        return "damage_severity"
    if normalized_question_type == "damage ratio bucket":
        return "damage_count"
    if normalized_question_type in {"crop reasoning", "crop image reasoning"}:
        return "crop image reasoning"
    if normalized_question_type == "visual interpretation":
        return "scene understanding"
    if normalized_question_type == "footprint distribution":
        return "spatial distribution"
    if normalized_question_type == "overall analysis":
        return "spatial distribution"
    if apply_xbd_alias and normalized_data_source == "xbd":
        return "demange_change"
    return raw_question_type


def build_question_type_bucket_map(
    extra_map: Optional[Mapping[str, str]] = None,
) -> Dict[str, str]:
    out = {
        normalize_question_type_key(question_type): bucket
        for question_type, bucket in DEFAULT_QUESTION_TYPE_TO_BUCKET.items()
    }
    if extra_map:
        for question_type, bucket in extra_map.items():
            out[normalize_question_type_key(question_type)] = str(bucket)
    for question_type, bucket in QUESTION_TYPE_BUCKET_MAP_OVERRIDES.items():
        out[normalize_question_type_key(question_type)] = bucket
    return out


def question_type_to_bucket(
    question_type: Any,
    *,
    data_source: Any = None,
    question_type_bucket_map: Optional[Mapping[str, str]] = None,
    apply_xbd_alias: bool = False,
    unknown_value: str = "unknown",
) -> str:
    cleaned_question_type = clean_question_type(
        question_type,
        data_source,
        apply_xbd_alias=apply_xbd_alias,
    )
    if not cleaned_question_type:
        return unknown_value
    mapping = build_question_type_bucket_map(question_type_bucket_map)
    return mapping.get(normalize_question_type_key(cleaned_question_type), cleaned_question_type)


def resolve_question_type_and_bucket(
    question_type: Any,
    data_source: Any,
    *,
    question_type_bucket_map: Optional[Mapping[str, str]] = None,
    apply_xbd_alias: bool = False,
    unknown_value: str = "unknown",
) -> Tuple[str, str]:
    cleaned_question_type = clean_question_type(
        question_type,
        data_source,
        apply_xbd_alias=apply_xbd_alias,
    )
    if not cleaned_question_type:
        return "", unknown_value
    return (
        cleaned_question_type,
        question_type_to_bucket(
            cleaned_question_type,
            data_source=data_source,
            question_type_bucket_map=question_type_bucket_map,
            apply_xbd_alias=apply_xbd_alias,
            unknown_value=unknown_value,
        ),
    )


def question_type_bucket_sort_key(bucket: Any) -> Tuple[int, str]:
    bucket_text = str(bucket or "").strip()
    bucket_key = QUESTION_TYPE_BUCKET_DISPLAY_TO_KEY.get(bucket_text, bucket_text)
    if bucket_key in TAXONOMY_BUCKET_ORDER:
        return (TAXONOMY_BUCKET_ORDER.index(bucket_key), bucket_text)
    return (len(TAXONOMY_BUCKET_ORDER), bucket_text.lower())


def question_type_bucket_display_name(bucket: Any) -> str:
    bucket_text = str(bucket or "").strip()
    if not bucket_text:
        return bucket_text
    bucket_key = QUESTION_TYPE_BUCKET_DISPLAY_TO_KEY.get(bucket_text, bucket_text)
    return str(QUESTION_TYPE_BUCKET_DISPLAY_NAMES.get(bucket_key, bucket_text))
