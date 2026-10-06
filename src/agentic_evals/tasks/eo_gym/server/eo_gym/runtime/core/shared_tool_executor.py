"""
Shared ToolExecutor implementation used by dataset generation scripts.

The logic validates tool parameters against the available tool signatures
defined in `Tools.py`, handles convenient coercions, and formats tool
outputs into strings suitable for logging or LLM observations.
"""

from __future__ import annotations

import ast
import inspect
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

from eo_gym.runtime.registry.Tools import AVAILABLE_TOOL_CLASSES, load_custom_tool


class ToolExecutor:
    """Loads tools on demand and executes them with consistent validation."""

    def __init__(self, device: str = "cuda", *, log_calls: bool = False):
        if load_custom_tool is None:
            raise ImportError(
                "`Tools.py` (and its dependencies) must be importable to use ToolExecutor."
            )
        self.device = device
        self.log_calls = log_calls
        self._tool_cache: Dict[str, Any] = {}

    def _load_tool(self, tool_name: str):
        if tool_name in self._tool_cache:
            return self._tool_cache[tool_name]

        tool = load_custom_tool(tool_name, device=self.device)
        if tool is None:
            raise ValueError(f"Tool '{tool_name}' could not be loaded.")
        self._tool_cache[tool_name] = tool
        return tool

    def execute(
        self,
        tool_name: str,
        params: Optional[Dict[str, Any]],
        image_path: Path,
    ) -> str:
        params = params or {}

        # Enforce registry-aligned tool names (snake_case) exactly as defined in Tools.AVAILABLE_TOOL_CLASSES.
        if tool_name not in AVAILABLE_TOOL_CLASSES:
            available = ", ".join(sorted(AVAILABLE_TOOL_CLASSES.keys()))
            return f"Error: Unknown tool '{tool_name}'. Valid tools: {available}."

        def _debug(message: str) -> None:
            if self.log_calls:
                print(message)

        _debug(f"[ToolExecutor] Running {tool_name} with params={params}")

        try:
            tool = self._load_tool(tool_name)
        except ValueError as exc:
            return f"Error: {exc}"

        image_path_str = str(image_path)

        def _coerce_literal(value: Any) -> Any:
            if isinstance(value, str):
                try:
                    return ast.literal_eval(value)
                except Exception:
                    return value
            return value

        def _to_int_value(
            raw: Any,
            param_name: str,
            *,
            required: bool = True,
        ) -> Tuple[Optional[int], Optional[str]]:
            if raw is None:
                if required:
                    return None, f"Error: Missing '{param_name}' parameter for {tool_name}."
                return None, None
            coerced = _coerce_literal(raw)
            try:
                return int(coerced), None
            except (TypeError, ValueError):
                return None, f"Error: '{param_name}' must be an integer for {tool_name}."

        def _to_float_value(
            raw: Any,
            param_name: str,
            *,
            required: bool = True,
        ) -> Tuple[Optional[float], Optional[str]]:
            if raw is None:
                if required:
                    return None, f"Error: Missing '{param_name}' parameter for {tool_name}."
                return None, None
            coerced = _coerce_literal(raw)
            try:
                return float(coerced), None
            except (TypeError, ValueError):
                return None, f"Error: '{param_name}' must be a number for {tool_name}."

        def _to_bool_value(raw: Any, param_name: str) -> Tuple[Optional[bool], Optional[str]]:
            if raw is None:
                return None, None
            if isinstance(raw, bool):
                return raw, None
            coerced = _coerce_literal(raw)
            if isinstance(coerced, bool):
                return coerced, None
            if isinstance(coerced, (int, float)):
                return bool(coerced), None
            if isinstance(coerced, str):
                lowered = coerced.strip().lower()
                if lowered in {"true", "1", "yes", "on"}:
                    return True, None
                if lowered in {"false", "0", "no", "off"}:
                    return False, None
            return None, f"Error: '{param_name}' must be a boolean for {tool_name}."

        def _ensure_sequence(
            raw: Any,
            param_name: str,
            *,
            required: bool = True,
        ) -> Tuple[Optional[List[Any]], Optional[str]]:
            if raw is None:
                if required:
                    return None, f"Error: Missing '{param_name}' parameter for {tool_name}."
                return None, None
            coerced = _coerce_literal(raw)
            if isinstance(coerced, (list, tuple)):
                sequence = list(coerced)
            elif isinstance(coerced, str):
                sequence = [item.strip() for item in coerced.split(",") if item.strip()]
            else:
                return None, f"Error: '{param_name}' must be a list or tuple for {tool_name}."
            if required and not sequence:
                return None, f"Error: '{param_name}' must contain at least one item for {tool_name}."
            return sequence, None

        def _describe_payload(payload: Any) -> str:
            if payload is None:
                return "None"
            cls_name = type(payload).__name__
            size_attr = getattr(payload, "size", None)
            mode_attr = getattr(payload, "mode", None)
            shape_attr = getattr(payload, "shape", None)
            dtype_attr = getattr(payload, "dtype", None)
            if size_attr and mode_attr:
                return f"{cls_name}(size={size_attr}, mode={mode_attr})"
            if shape_attr is not None and dtype_attr is not None:
                return f"{cls_name}(shape={shape_attr}, dtype={dtype_attr})"
            if isinstance(payload, (list, tuple)):
                return f"{cls_name}(len={len(payload)})"
            return cls_name

        def _format_result(result: Any) -> str:
            if isinstance(result, list):
                try:
                    return json.dumps(result, ensure_ascii=False)
                except TypeError:
                    sample = []
                    for item in result[:3]:
                        if isinstance(item, tuple) and len(item) == 2 and isinstance(item[1], dict):
                            sample.append(
                                {
                                    "data": _describe_payload(item[0]),
                                    "metadata": item[1],
                                }
                            )
                        else:
                            sample.append(_describe_payload(item))
                    suffix = "" if len(result) <= 3 else "..."
                    return f"List[{len(result)}] Sample: {json.dumps(sample, ensure_ascii=False)}{suffix}"
            if isinstance(result, dict):
                return json.dumps(result, ensure_ascii=False)
            if isinstance(result, tuple):
                if len(result) == 2 and isinstance(result[1], dict):
                    wrapped = {
                        "data": _describe_payload(result[0]),
                        "metadata": result[1],
                    }
                    return json.dumps(wrapped, ensure_ascii=False)
                try:
                    return json.dumps(result, ensure_ascii=False)
                except TypeError:
                    return str(result)
            if isinstance(result, (int, float, str)):
                return str(result)
            return f"Tool returned an unhandled type ({type(result).__name__}): {result}"

        try:
            if tool_name == "get_object_bbox_by_optical_image":
                image_urls = params.get("optical_image_urls") or params.get("image_urls")
                text_prompt = params.get("text_prompt") or params.get("text")
                urls = _ensure_sequence(image_urls, "optical_image_urls")[0]
                if not urls:
                    return "Error: get_object_bbox_by_optical_image requires 'optical_image_urls' (list)."
                if text_prompt is None or str(text_prompt).strip() == "":
                    return "Error: get_object_bbox_by_optical_image requires 'text_prompt'."
                _debug(f"  [DEBUG] Calling {tool_name}(urls={len(urls)}, text='{text_prompt}')")
                result = tool(urls, str(text_prompt))

            elif tool_name in {"get_object_mask_by_optical_image", "get_road_mask_by_optical_image", "get_building_mask_by_optical_image"}:
                image_urls = params.get("optical_image_urls") or params.get("image_urls")
                urls = _ensure_sequence(image_urls, "optical_image_urls")[0]
                if not urls:
                    return f"Error: {tool_name} requires 'optical_image_urls' (list)."
                text_prompt = params.get("text_prompt") or params.get("text")
                # Road/building subclasses ignore prompt; base class requires it.
                if tool_name == "get_object_mask_by_optical_image":
                    if text_prompt is None or str(text_prompt).strip() == "":
                        return "Error: get_object_mask_by_optical_image requires 'text_prompt'."
                    _debug(f"  [DEBUG] Calling {tool_name}(urls={len(urls)}, text='{text_prompt}')")
                    result = tool(urls, str(text_prompt))
                else:
                    _debug(f"  [DEBUG] Calling {tool_name}(urls={len(urls)})")
                    result = tool(urls)

            elif tool_name == "get_object_bbox_by_sar_image":
                image_urls = params.get("sar_image_urls") or params.get("image_urls")
                text_prompt = params.get("text_prompt") or params.get("text")
                urls = _ensure_sequence(image_urls, "sar_image_urls")[0]
                if not urls:
                    return "Error: get_object_bbox_by_sar_image requires 'sar_image_urls' (list)."
                if text_prompt is None or str(text_prompt).strip() == "":
                    return "Error: get_object_bbox_by_sar_image requires 'text_prompt'."
                _debug(f"  [DEBUG] Calling {tool_name}(urls={len(urls)}, text='{text_prompt}')")
                result = tool(urls, str(text_prompt))

            elif tool_name == "get_object_bbox_by_optical_sar_image":
                optical_path = params.get("optical_image_path") or params.get("optical_path") or str(image_path)
                sar_path = params.get("sar_image_path") or params.get("sar_path")
                text_prompt = params.get("text_prompt") or params.get("text")
                if not optical_path or not sar_path:
                    return "Error: get_object_bbox_by_optical_sar_image requires 'optical_image_path' and 'sar_image_path'."
                if text_prompt is None or str(text_prompt).strip() == "":
                    return "Error: get_object_bbox_by_optical_sar_image requires 'text_prompt'."
                _debug(
                    f"  [DEBUG] Calling {tool_name}(optical='{optical_path}', sar='{sar_path}', text='{text_prompt}')"
                )
                result = tool(str(optical_path), str(sar_path), str(text_prompt))

            elif tool_name == "count_objects_by_category":
                detections = params.get("objects_by_category") or params.get("objects") or []
                if not isinstance(detections, (list, tuple)):
                    return "Error: count_objects_by_category requires a list of detections with 'label'."
                _debug(f"  [DEBUG] Calling {tool_name}(detections={len(detections)})")
                result = tool(detections)

            elif tool_name == "get_bbox_geospatial_relationship":
                loc_a = _coerce_literal(params.get("loc_a"))
                loc_b = _coerce_literal(params.get("loc_b"))
                if loc_a is None or loc_b is None:
                    return "Error: get_bbox_geospatial_relationship requires 'loc_a' and 'loc_b'."
                _debug(f"  [DEBUG] Calling {tool_name}(loc_a={loc_a}, loc_b={loc_b})")
                result = tool(loc_a, loc_b)

            elif tool_name == "get_mask_geospatial_relationship":
                mask_a = params.get("mask_a")
                mask_b = params.get("mask_b")
                if mask_a is None or mask_b is None:
                    return "Error: get_mask_geospatial_relationship requires 'mask_a' and 'mask_b'."
                _debug(f"  [DEBUG] Calling {tool_name}(mask_a, mask_b)")
                result = tool(mask_a, mask_b)

            elif tool_name == "summarize_temporal_mask_changes":
                changes = params.get("changes")
                if changes is None:
                    return "Error: summarize_temporal_mask_changes requires 'changes'."
                _debug(f"  [DEBUG] Calling {tool_name}(entries={len(changes) if hasattr(changes,'__len__') else 'unknown'})")
                result = tool(changes)

            elif tool_name == "summarize_temporal_object_changes":
                changes = params.get("changes")
                if changes is None:
                    return "Error: summarize_temporal_object_changes requires 'changes'."
                _debug(f"  [DEBUG] Calling {tool_name}(entries={len(changes) if hasattr(changes,'__len__') else 'unknown'})")
                result = tool(changes)

            elif tool_name == "basic_calculator":
                expr = params.get("expression")
                if expr is None or str(expr).strip() == "":
                    return "Error: basic_calculator requires 'expression'."
                # Replace caret with Python power operator to avoid XOR surprises from LLM output.
                safe_expr = str(expr).replace("^", "**")
                _debug(f"  [DEBUG] Calling {tool_name}(expression='{safe_expr}')")
                result = tool(safe_expr)

            elif tool_name in {"analyze_optical_scene", "describe_optical_object"}:
                optical_path = params.get("optical_image_path") or params.get("image_path") or str(image_path)
                if not optical_path:
                    return f"Error: {tool_name} requires 'optical_image_path'."
                if tool_name == "describe_optical_object":
                    obj = params.get("object_name") or params.get("text_prompt") or params.get("text")
                    if obj is None or str(obj).strip() == "":
                        return "Error: describe_optical_object requires 'object_name'."
                    _debug(f"  [DEBUG] Calling {tool_name}(path='{optical_path}', object='{obj}')")
                    result = tool(str(optical_path), str(obj))
                else:
                    prompt_text = params.get("prompt_text") or params.get("text_prompt") or params.get("text")
                    if prompt_text is None or str(prompt_text).strip() == "":
                        return "Error: analyze_optical_scene requires 'prompt_text'."
                    _debug(
                        f"  [DEBUG] Calling {tool_name}(path='{optical_path}', prompt_text='{prompt_text}')"
                    )
                    result = tool(str(optical_path), str(prompt_text))

            else:
                call_params = dict(params)
                try:
                    tool_signature = inspect.signature(tool.__call__)
                    param_names = tool_signature.parameters
                except (TypeError, ValueError):
                    param_names = {}

                if "image_path" in param_names and "image_path" not in call_params:
                    call_params["image_path"] = image_path_str

                _debug(f"  [DEBUG] Calling {tool_name} with inferred params {call_params}...")
                result = tool(**call_params)

            return _format_result(result)

        except Exception as exc:
            return f"Error executing {tool_name}: {type(exc).__name__} - {exc}"
