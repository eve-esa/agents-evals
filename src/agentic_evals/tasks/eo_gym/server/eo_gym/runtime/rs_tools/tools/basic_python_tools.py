"""
Pure-Python helper tools (no model dependencies).

Tools:
  - CountObjectsByCategory: counts objects per category given a dict of bboxes.
  - ComputeRelativeLocation: infers relative direction between two locations.
  - BasicCalculator: evaluates simple arithmetic expressions safely.
"""

from __future__ import annotations

import ast
import operator
import re
import math
import numpy as np
from typing import Any, Dict, List, Optional, Tuple, Union

from PIL import Image as PILImage
from eo_gym.runtime.rs_tools.utils import check_file_reference

class CountObjectsByCategory:
    """
    Count objects per category from a list of detections.
    Counts provided detections per label; does not run detection.
    Aims to summary the time-series of detected objects from prior steps. Not use it on single timepoints summaries.
    Input: flat list of detections (each dict must contain 'label', 'bbox', and 'score').
    Guidance: pass in the bboxes produced by the prior "get object bbox relative" step (or any upstream detector),
    not by asking the model to visually re-derive them. This tool only tallies provided boxes/labels.
    Output: dict[str, int] counts per category.
    """

    def __init__(self, device: str = "cpu"):
        self.device = device

    def __call__(self, objects_by_category: Any) -> Dict[str, int]:
        if not isinstance(objects_by_category, (list, tuple)):
            raise ValueError("objects_by_category must be a list of detections with 'label'.")
        counts: Dict[str, int] = {}
        for entry in objects_by_category:
            if not isinstance(entry, dict):
                continue
            label = entry.get("label")
            bbox = entry.get("bbox")
            if label is None:
                continue
            key = str(label)
            if bbox is None:
                continue
            elif len(bbox) > 0: # only count if bbox is non-empty
                counts[key] = counts.get(key, 0) + 1
            else:
                counts[key] = counts.get(key, 0) + 0
        return counts


class GetBboxGeospatialRelationship:
    """
    Compute a coarse relative geospatial relationship between two locations.
    Computes relative geospatial relation (e.g., north/south/east/west) between boxes/points.
    Inputs: loc_a (List[float]) and loc_b (List[float]) as [x, y] or [x0,y0,x1,y1] (centroid used).
    Output: dict with direction (str) and dx, dy (float).
    """

    def __init__(self, device: str = "cpu"):
        self.device = device

    @staticmethod
    def _center(loc: Union[List[float], tuple]) -> List[float]:
        vals = list(loc)
        if len(vals) == 2:
            return [float(vals[0]), float(vals[1])]
        if len(vals) == 4:
            x0, y0, x1, y1 = [float(v) for v in vals]
            return [(x0 + x1) / 2.0, (y0 + y1) / 2.0]
        raise ValueError("Location must be length-2 or length-4 sequence.")

    def __call__(self, loc_a: Union[List[float], tuple], loc_b: Union[List[float], tuple]) -> Dict[str, Union[str, float]]:
        ax, ay = self._center(loc_a)
        bx, by = self._center(loc_b)
        dx = bx - ax
        dy = by - ay
        direction_parts: List[str] = []
        if abs(dy) > 0:
            direction_parts.append("north" if dy < 0 else "south")
        if abs(dx) > 0:
            direction_parts.append("east" if dx > 0 else "west")
        direction = "-".join(direction_parts) if direction_parts else "same-location"
        return {"direction": direction, "dx": dx, "dy": dy}


class BasicCalculator:
    """
    Evaluate a simple arithmetic expression safely.
    Evaluates a simple arithmetic expression; no vision or spatial reasoning.
    Input: expression (str) supporting +, -, *, /, //, **, ^ (as power), parentheses, numbers,
           and math funcs: abs, round, min, max, sum, sqrt, sin, cos, tan, atan, atan2, log, exp. Division by zero is not allowed. For round, ndigits is coerced to int.
           Constants: pi, e.
    Output: dict with result (float).
    """

    def __init__(self, device: str = "cpu"):
        self.device = device

    def __call__(self, expression: str) -> Dict[str, Union[float, str]]:
        if not isinstance(expression, str) or not expression.strip():
            return {"error": "expression must be a non-empty string."}
        normalized = expression.strip()
        if re.search(r"\bmonth(s)?\b", normalized, re.IGNORECASE):
            return {
                "error": "basic_calculator only accepts numeric math expressions; convert months to numeric offsets before calling."
            }
        if re.search(r"\bminus\b", normalized, re.IGNORECASE):
            return {
                "error": "basic_calculator only accepts numeric math expressions; use '-' instead of the word 'minus'."
            }
        if re.search(r"\b\d{4}-\d{2}-\d{2}\b", normalized):
            return {
                "error": "basic_calculator does not accept date-like literals such as '2017-04-05'; translate dates into numeric offsets first."
            }
        try:
            node = ast.parse(normalized, mode="eval")
            result = self._eval_node(node.body)
        except (SyntaxError, ValueError) as exc:
            return {"error": str(exc)}
        return {"result": result}

    def _eval_node(self, node):
        allowed_funcs = {
            "abs": abs,
            "round": round,
            "min": min,
            "max": max,
            "sum": sum,
            "sqrt": math.sqrt,
            "sin": math.sin,
            "cos": math.cos,
            "tan": math.tan,
            "atan": math.atan,
            "atan2": math.atan2,
            "log": math.log,
            "exp": math.exp,
        }
        allowed_func_list = ", ".join(sorted(allowed_funcs.keys()))
        allowed_constants = {"pi": math.pi, "e": math.e}
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            return float(node.value)
        if isinstance(node, ast.Name):
            if node.id in allowed_constants:
                return float(allowed_constants[node.id])
            raise ValueError("Unsupported variable name.")
        if isinstance(node, (ast.List, ast.Tuple)):
            return [self._eval_node(elt) for elt in node.elts]
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
            operand = self._eval_node(node.operand)
            return +operand if isinstance(node.op, ast.UAdd) else -operand
        if isinstance(node, ast.Compare):
            if len(node.ops) != 1 or len(node.comparators) != 1:
                raise ValueError("Only single comparisons are supported.")
            left = self._eval_node(node.left)
            right = self._eval_node(node.comparators[0])
            op = node.ops[0]
            cmp_ops = {
                ast.Lt: operator.lt,
                ast.LtE: operator.le,
                ast.Gt: operator.gt,
                ast.GtE: operator.ge,
                ast.Eq: operator.eq,
                ast.NotEq: operator.ne,
            }
            if type(op) not in cmp_ops:
                raise ValueError("Unsupported comparison operator.")
            return bool(cmp_ops[type(op)](left, right))
        if isinstance(node, ast.BinOp) and isinstance(
            node.op, (ast.Add, ast.Sub, ast.Mult, ast.Div, ast.FloorDiv, ast.Pow, ast.BitXor)
        ):
            left = self._eval_node(node.left)
            right = self._eval_node(node.right)
            if isinstance(node.op, (ast.Div, ast.FloorDiv)) and float(right) == 0.0:
                raise ValueError("Division by zero is not allowed. Adjust the expression.")
            ops = {
                ast.Add: operator.add,
                ast.Sub: operator.sub,
                ast.Mult: operator.mul,
                ast.Div: operator.truediv,
                ast.FloorDiv: operator.floordiv,
                ast.Pow: operator.pow,
                ast.BitXor: operator.pow,  # treat ^ as exponent for convenience
            }
            return ops[type(node.op)](left, right)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            func_name = node.func.id
            if func_name not in allowed_funcs:
                raise ValueError(f"Unsupported function. Allowed: {allowed_func_list}.")
            if node.keywords:
                raise ValueError("Keyword arguments are not allowed.")
            args = [self._eval_node(arg) for arg in node.args]
            # round should accept up to 2 arguments; ndigits is coerced to int if provided.
            if func_name == "round" and len(args) >= 2:
                try:
                    args[1] = int(args[1])
                except Exception:
                    raise ValueError("round ndigits must be an integer.")
            return float(allowed_funcs[func_name](*args))
        raise ValueError(
            f"Unsupported expression. Allowed: numbers, + - * / ** ^, parentheses, and functions {allowed_func_list}."
        )


class GetMaskGeospatialRelationship:
    """
    Compute basic geospatial relationships between two binary masks.
    Computes overlap/IoU and coverage stats between two binary masks.
    Inputs: mask_a, mask_b as binary mask URLs/paths (PNG read if URL/path).
    Output: dict with intersection, union, iou, and coverage stats.
    """

    def __init__(self, device: str = "cpu"):
        self.device = device

    def __call__(self, mask_a, mask_b) -> Dict[str, float]:
        import numpy as np  # local import to avoid hard dependency when unused
        from PIL import Image
        from eo_gym.runtime.rs_tools.tools.mask_utils import load_mask_from_source

        if isinstance(mask_a, str):
            error = check_file_reference(mask_a, label="mask", strict=True)
            if error:
                return {"message": error}
        if isinstance(mask_b, str):
            error = check_file_reference(mask_b, label="mask", strict=True)
            if error:
                return {"message": error}
        a = load_mask_from_source(mask_a)
        b = load_mask_from_source(mask_b)
        if a.shape != b.shape:
            # Resize smaller mask(s) to the largest height/width using nearest-neighbor
            target_h = max(a.shape[0], b.shape[0])
            target_w = max(a.shape[1], b.shape[1])

            def _resize(mask: np.ndarray) -> np.ndarray:
                if mask.shape == (target_h, target_w):
                    return mask
                img = Image.fromarray(mask.astype(np.uint8) * 255)
                resized = img.resize((target_w, target_h), resample=Image.NEAREST)
                return (np.array(resized) > 0).astype(mask.dtype)

            a = _resize(a)
            b = _resize(b)
        inter = np.logical_and(a, b).sum()
        union = np.logical_or(a, b).sum()
        area_a = a.sum()
        area_b = b.sum()
        iou = float(inter) / float(union) if union > 0 else 0.0
        return {
            "intersection": int(inter),
            "union": int(union),
            "iou": iou,
            "coverage_a": float(inter) / float(area_a) if area_a > 0 else 0.0,
            "coverage_b": float(inter) / float(area_b) if area_b > 0 else 0.0,
        }


__all__ = [
    "CountObjectsByCategory",
    "GetBboxGeospatialRelationship",
    "BasicCalculator",
    "GetMaskGeospatialRelationship",
    "NormalizeBoundingBoxes",
    "SummarizeTemporalObjectChanges",
    "SummarizeTemporalMaskChanges",
]


def _as_positive_float(value: Any) -> Optional[float]:
    try:
        num = float(value)
    except Exception:
        return None
    return num if num > 0 else None


def _extract_image_dimensions(payload: Dict[str, Any]) -> Tuple[Optional[float], Optional[float]]:
    width = _as_positive_float(
        payload.get("image_width")
        or payload.get("width")
        or payload.get("img_width")
        or payload.get("w")
    )
    height = _as_positive_float(
        payload.get("image_height")
        or payload.get("height")
        or payload.get("img_height")
        or payload.get("h")
    )
    if (width is None or height is None) and isinstance(payload.get("image_size"), (list, tuple)):
        dims = payload["image_size"]
        if len(dims) >= 2:
            if width is None:
                width = _as_positive_float(dims[0])
            if height is None:
                height = _as_positive_float(dims[1])
    if (width is None or height is None) and isinstance(payload.get("size"), (list, tuple)):
        dims = payload["size"]
        if len(dims) >= 2:
            width = width or _as_positive_float(dims[0])
            height = height or _as_positive_float(dims[1])
    return width, height


def _normalize_bbox(bbox: Any, width: Optional[float], height: Optional[float]) -> Tuple[Any, ...]:
    if not isinstance(bbox, (list, tuple)) or not bbox:
        return tuple(bbox) if isinstance(bbox, (list, tuple)) else (bbox,)
    normalized = []
    for idx, value in enumerate(bbox):
        try:
            val = float(value)
        except Exception:
            normalized.append(value)
            continue
        denom = width if idx % 2 == 0 else height
        if denom and denom > 0:
            normalized.append(val / denom)
        else:
            normalized.append(val)
    return tuple(normalized)


def _dimensions_from_record(record: Dict[str, Any]) -> Tuple[Optional[float], Optional[float]]:
    width = _as_positive_float(record.get("image_width"))
    height = _as_positive_float(record.get("image_height"))
    return width, height


class NormalizeBoundingBoxes:
    """
    Normalize bounding boxes using a shared reference size (largest image in the batch).

    Accepts a list of records, each containing bbox coordinates plus image
    dimensions (`image_width` and `image_height`) along with `image_url` for
    reference. Computes the maximum width/height across the batch and scales
    all bboxes to that common size (i.e., values / max_width or max_height).
    Returns the records augmented with `normalized_bbox`.
    """

    def __init__(self, device: str = "cpu"):
        self.device = device

    def __call__(self, records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        if not isinstance(records, list):
            raise ValueError(
                "records must be a list of bounding box entries (not a JSON string). "
                "Each entry should include at least bbox and, when available, image_width and image_height."
            )
        # Determine shared reference size (max width/height across all records).
        max_w = None
        max_h = None
        for entry in records:
            if not isinstance(entry, dict):
                continue
            w, h = _dimensions_from_record(entry)
            if w is None or h is None:
                raise ValueError("normalize_bounding_boxes requires image_width and image_height for every record.")
            max_w = w if max_w is None else max(max_w, w)
            max_h = h if max_h is None else max(max_h, h)
        if max_w is None or max_h is None:
            raise ValueError("normalize_bounding_boxes requires at least one valid record with image_width and image_height.")

        normalized_records: List[Dict[str, Any]] = []
        for entry in records:
            if not isinstance(entry, dict):
                continue
            bbox = entry.get("bbox")
            normalized_bbox: Optional[List[Any]] = None
            if bbox is not None:
                scaled = _normalize_bbox(bbox, max_w, max_h)
                normalized_bbox = [float(v) if isinstance(v, (int, float)) else v for v in scaled]
            enriched = dict(entry)
            enriched["normalized_bbox"] = normalized_bbox
            normalized_records.append(enriched)
        return normalized_records


class SummarizeTemporalObjectChanges:
    """
    Summarize object appearance/disappearance over time.
    Summarizes object appearance/disappearance timelines; returns change metrics only.
    Input: changes (list[dict]) each with {"timestamp": str, "objects": List[dict{category,bbox}], "image_width": int, "image_height": int}
    Output: dict with ordered timeline plus category counts of new and lost objects.
    """

    def __init__(self, device: str = "cpu"):
        self.device = device

    def __call__(self, changes: list) -> dict:
        if not isinstance(changes, (list, tuple)):
            raise ValueError("changes must be a list of {'timestamp', 'objects'} entries.")
        entries = []
        for idx, payload in enumerate(changes):
            if not isinstance(payload, dict):
                continue
            ts = payload.get("timestamp")
            objs = payload.get("objects") or []
            if not isinstance(objs, (list, tuple)):
                raise ValueError("objects must be a list.")
            width, height = _extract_image_dimensions(payload)
            entries.append((str(ts), f"t{idx}", width, height, objs))
        # Sort by timestamp string (lexicographic ISO assumed)
        entries.sort(key=lambda x: x[0])
        prev_map = {}
        timeline = []
        new_counts = {}
        lost_counts = {}
        for ts, img_id, width, height, objs in entries:
            curr_map = {}
            for obj in objs:
                key = (obj.get("category"), _normalize_bbox(obj.get("bbox", []), width, height))
                curr_map[key] = obj
            new_keys = set(curr_map.keys()) - set(prev_map.keys())
            lost_keys = set(prev_map.keys()) - set(curr_map.keys())
            new_objs = [curr_map[k] for k in new_keys]
            lost_objs = [prev_map[k] for k in lost_keys]
            for obj in objs:
                cat = obj.get("category")
                if cat and (obj in new_objs):
                    new_counts[cat] = new_counts.get(cat, 0) + 1
            for obj in lost_objs:
                cat = obj.get("category")
                if cat:
                    lost_counts[cat] = lost_counts.get(cat, 0) + 1
            timeline.append(
                {
                    "image_id": img_id,
                    "timestamp": ts,
                    "new_objects": new_objs,
                    "lost_objects": lost_objs,
                }
            )
            prev_map = curr_map
        return {
            "timeline": timeline,
            "new_object_counts": new_counts,
            "lost_object_counts": lost_counts,
        }


def _resample_mask_to_shape(mask: Any, target_shape: Tuple[int, int]) -> "np.ndarray":
    import numpy as np

    arr = np.asarray(mask, dtype=np.uint8)
    bool_mask = (arr > 0).astype(np.uint8)
    if bool_mask.shape == target_shape:
        return bool_mask.astype(bool)
    img = PILImage.fromarray(bool_mask * 255)
    width, height = target_shape[1], target_shape[0]
    resized = img.resize((width, height), resample=PILImage.NEAREST)
    return (np.asarray(resized) > 0)


class SummarizeTemporalMaskChanges:
    """
    Summarize mask change areas over time (e.g., growth/loss between images).
    Summarizes mask change areas over time; returns change metrics only.
    Input: changes (list[dict]) each with {"timestamp": str, "mask_url": str}
    Output: dict with ordered timeline entries: added_area, removed_area, intersection, union, iou.
    """

    def __init__(self, device: str = "cpu"):
        self.device = device

    def __call__(self, changes: list) -> dict:
        if not isinstance(changes, (list, tuple)):
            raise ValueError("changes must be a list of {'timestamp', 'mask_url'} entries.")
        import numpy as np  # local import to avoid hard dependency when unused
        from eo_gym.runtime.rs_tools.tools.mask_utils import load_mask_from_source

        entries = []
        for idx, payload in enumerate(changes):
            if not isinstance(payload, dict):
                continue
            ts = payload.get("timestamp")
            mask_url = payload.get("mask_url") or payload.get("mask")
            if ts is None or mask_url is None:
                continue
            if not isinstance(mask_url, str):
                raise ValueError("mask_url must be a string path or URL.")
            error = check_file_reference(mask_url, label="mask", strict=True)
            if error:
                return {"message": error}
            mask = load_mask_from_source(mask_url)
            entries.append((str(ts), f"t{idx}", mask))
        entries.sort(key=lambda x: x[0])

        timeline = []
        prev_mask = None
        for ts, img_id, mask in entries:
            mask = np.asarray(mask, dtype=bool)
            if prev_mask is None:
                timeline.append(
                    {
                        "image_id": img_id,
                        "timestamp": ts,
                        "added_area": int(mask.sum()),
                        "removed_area": 0,
                        "intersection": int(mask.sum()),
                        "union": int(mask.sum()),
                        "iou": 1.0,
                    }
                )
            else:
                target_shape = prev_mask.shape if prev_mask.size <= mask.size else mask.shape
                prev_aligned = _resample_mask_to_shape(prev_mask, target_shape)
                curr_aligned = _resample_mask_to_shape(mask, target_shape)
                added = np.logical_and(curr_aligned, ~prev_aligned).sum()
                removed = np.logical_and(prev_aligned, ~curr_aligned).sum()
                inter = np.logical_and(curr_aligned, prev_aligned).sum()
                union = np.logical_or(curr_aligned, prev_aligned).sum()
                iou = float(inter) / float(union) if union > 0 else 0.0
                timeline.append(
                    {
                        "image_id": img_id,
                        "timestamp": ts,
                        "added_area": int(added),
                        "removed_area": int(removed),
                        "intersection": int(inter),
                "union": int(union),
                        "iou": iou,
                    }
                )
            prev_mask = mask

        return {"timeline": timeline}
