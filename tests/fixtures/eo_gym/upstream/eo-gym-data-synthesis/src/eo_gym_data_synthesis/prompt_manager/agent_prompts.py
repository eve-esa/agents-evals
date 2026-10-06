from __future__ import annotations
__all__ = [
    "build_function_call_prompt",
    "get_tools_prompt_text",
    "VHR_ZOOM_VIEW_ADJUSTMENT_APPENDIX",
    "TRAJECTORY_LOGIC_AUDIT_SYSTEM_PROMPT",
    "FAILED_TRAJECTORY_IMPROVEMENT_SYSTEM_PROMPT",
]

from typing import List, Any, Sequence

import inspect

# Import from legacy shim to preserve registry access.
from eo_gym.runtime.registry.Tools import AVAILABLE_TOOL_CLASSES


def _format_param(param: inspect.Parameter) -> str:
    def _type_label(p: inspect.Parameter) -> str:
        if p.annotation is not inspect._empty:
            ann = p.annotation
            if isinstance(ann, str):
                return ann
            if getattr(ann, "__module__", "") == "typing":
                return str(ann).replace("typing.", "")
            if hasattr(ann, "__name__"):
                return ann.__name__
            return str(ann)
        if p.default is not inspect._empty:
            return type(p.default).__name__
        return "any"

    if param.kind in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD):
        return param.name
    type_str = _type_label(param)
    if param.default is inspect._empty:
        return f"{param.name}: {type_str}"
    if isinstance(param.default, (int, float, str, bool)):
        return f"{param.name}: {type_str}={param.default!r}"
    return f"{param.name}: {type_str}"


def _format_signature(cls: type) -> str:
    try:
        sig = inspect.signature(cls.__call__)
    except (TypeError, ValueError):
        return ""
    params: List[str] = []
    for p in list(sig.parameters.values())[1:]:  # skip self
        params.append(_format_param(p))
    return ", ".join(params)


def get_tools_prompt_text(tool_names: Sequence[str] | None = None) -> str:
    """
    Build a bullet list of available tools from the current registry.
    """
    def _clean_return(text: str) -> str:
        import re
        return re.sub(r"^(?:[\w\[\]\.,<>/ +:-]+)\s*[:\-]\s*", "", text or "").strip()

    def _extract_return_info(doc: str) -> str:
        if not doc:
            return ""
        lines = [line.strip() for line in doc.splitlines()]
        for idx, line in enumerate(lines):
            lower = line.lower()
            if lower.startswith(("output", "outputs", "return", "returns")):
                content = line.split(":", 1)[1].strip() if ":" in line else line
                collected = [_clean_return(content)] if content else []
                j = idx + 1
                while j < len(lines) and lines[j]:
                    collected.append(_clean_return(lines[j]))
                    j += 1
                collected = [c for c in collected if c]
                return " ".join(collected).strip()
        return ""
    names: List[str] = []
    if tool_names:
        seen = set()
        for name in tool_names:
            if name in AVAILABLE_TOOL_CLASSES and name not in seen:
                names.append(name)
                seen.add(name)
    else:
        names = sorted(AVAILABLE_TOOL_CLASSES.keys())

    lines: List[str] = []
    for name in names:
        cls = AVAILABLE_TOOL_CLASSES[name]
        doc = (cls.__doc__ or "").strip()
        desc = doc.splitlines()[0] if doc else "Tool implementation."
        ret = _extract_return_info(doc)
        if ret:
            desc = f"{desc} Returns: {ret}"
        sig = _format_signature(cls)
        sig_part = f"({sig})" if sig else "()"
        lines.append(f"- {name}{sig_part}: {desc}")
    return "\n".join(lines)


def build_function_call_prompt() -> str:
    return """
You are an expert remote-sensing analyst.

To each task, briefly state relevant information from the image (if provided) and the user request, then explain why you are choosing the next tool call and its parameters. Do not label this section as “Planning.” After that, emit function calls (no free-text “Tool:” lines).

# Output Format

- If using tools:
  Start with 1-2 sentences of rationale (not labeled “Planning”), then emit function calls only

- If you have enough observation result:
  [brief conclusion sentence]
  Final Answer: [your answer]

  
Key instructions:
- Always ground your reasoning in tool outputs. Use the exact image URLs supplied in the user message when a tool needs imagery.
- For multiple-choice questions, return the answer as '(Letter) Option text' using the provided options. If the correct answer is not in the options, or more than one option, please answer HUMAN REVIEW.
- Keep replies succinct and focused on solving the question with tool evidence.
- When you need a tool, emit an actual function call using the provided tool names (type=function payload), not prose.
- When comparing detections whose bounding boxes come from images with different widths/heights, normalize their coordinates via `normalize_bounding_boxes` so comparisons are computed on a common scale.
- Do not finalize your answer until you have made at least one tool call in this conversation.
- When you plan to call object detection relative method, please understand they cannot tell the object status, and they can only give you mask or bbox. Also please
pass the object detail name to query, not just a `object`. The relative object detection tool need a detail name to run query.
- For shape/elongation questions, bbox-relative tools only return rectangles; use a mask-based tool to confirm the object's shape after locating it with bboxes.
- For detection: if only a SAR image URL is provided, use `get_object_bbox_by_sar_image`. If only an optical image URL is provided, use `get_object_bbox_by_optical_image`. Use `get_object_bbox_by_optical_sar_image` only after you have both images; if needed, call `get_optical_from_sar` or `get_sar_from_optical` first. Never pass empty image paths.
- When using the basic_calculator tool, you must use valid Python syntax. Use ** for exponents (e.g., 5**2), not ^.
- You may receive INTERNAL_FEEDBACK in a user message. Use it to improve your approach, but do not mention or quote it in your rationale, tool calls, or final answer.

Multispectral index routing (MANDATORY)
- Vegetation condition/change/greenness -> MUST call compute_ndvi_by_multispectral >= 1x (per relevant date).
- Water extent/change -> MUST call compute_ndwi_by_multispectral >= 1x (per relevant date).
- Built-up/impervious/urban extent/change -> MUST call compute_ndbi_by_multispectral >= 1x (per relevant date).

Do NOT substitute mask tools (compute_*_mask_by_multispectral) when an ND* tool exists.
Mask tools may only be used AFTER the ND* call if you need a binary mask for area/coverage comparison.

Multispectral discovery routing (MANDATORY)
- If question mentions multispectral data (Sentinel-2 / Landsat / multispectral folder URL)
  OR is a temporal/spectral analysis task:
    - MUST call get_multispectral_list exactly once BEFORE selecting dates/captures.
    - Use multispectral_list to enumerate available captures for the tile/location and confirm
      a previous/next capture exists.
    - Do NOT assume previous/next exists. Do NOT guess.
    - Do NOT call get_next_multispectral or get_previous_multispectral until AFTER
      multispectral_list has been called at least once.
    - Do NOT call get_multispectral_list with start_date == end_date; use a real window
      (start_date < end_date) to discover available captures.

Multispectral neighbor retrieval (MANDATORY)
- If the question asks about previous/current/next capture or "time step", you MUST
  call get_previous_multispectral and/or get_next_multispectral (after get_multispectral_list)
  to fetch the neighbor captures. Do not rely on get_multispectral_list alone to answer.

When a tool returns a new image URL (crop/zoom/move), first decide whether the area of interest is correct. If not, keep adjusting the view (crop/zoom/move) until satisfied, then run object-detection tools (e.g., get_object_bbox*). Avoid repeated detections before the view is finalized.
""".strip()



VHR_ZOOM_VIEW_ADJUSTMENT_APPENDIX = (
    "If an object appears at the edge of the image, first call a view-change tool "
    "(zoom_out_optical_image or move_*_optical_image) to get a different view before running detection or counting." \
    "But you should double check the object bbox to confirm the objects is in the new extended view or not to confirm original image content."
)

TRAJECTORY_LOGIC_AUDIT_SYSTEM_PROMPT = """
You are a logic auditor for remote-sensing agents. Your goal is to verify that the agent's conclusions are strictly derived from the tool outputs provided in the history. Do not compare against any ground-truth answer; if the reasoning chain is consistent with tool outputs, you may pass even when the final answer does not match a label.

FAIL the trajectory immediately if any of the following logical fallacies occur:

1. **Spatial Hallucination (Magic Compass):** The agent concludes a cardinal direction (e.g., "North of") based on a tool that only returns overlap statistics (e.g., IoU) or a scene description that contains no directional information.
2. **Fabricated Inputs:** The agent feeds hardcoded/random coordinates (e.g., [0,0,0,0], [0.1, 0.2]) or compares a mask to itself (mask_a == mask_b) instead of extracting actual data from previous steps.
3. **Geometric Contradiction:** The agent's conclusion mathematically contradicts the observed coordinates (e.g., stating an object at x=157 is inside a crop starting at x=240).
4. **Evidence Contradiction:** The agent's conclusion directly opposes the tool output (e.g., Tool says "Cylindrical", Agent concludes "Square"; or Tool data shows a decrease, but Agent claims an increase).
5. **Inference Without Evidence:** The agent concludes specific attributes (e.g., orientation, facility type) from generic scene descriptions that do not contain that specific information.
6. **Temporal Mismatch:** The question asks about a specific timestamp/date that differs from the provided capture time, but the agent answers using the provided image without retrieving the correct capture (e.g., no tool call to fetch the requested timestamp).
7. **Multispectral Neighbor Retrieval Misuse:** For previous/current/next capture questions, the agent skips get_previous_multispectral/get_next_multispectral after listing captures, or uses get_multispectral_list with start_date == end_date as a substitute for neighbor retrieval.
8. **No Any Successful Function Call Observation**: For the conversation, if no any successful function call (all observations are error message).

If the conversation given detail analysis why their answer is HUMAN REVIEW (not pick up any option), we will let it pass. If it without enough information to support the final asnwer as HUMAN REVIEW, we still reject it.

Respond ONLY as compact JSON:
{
  "verdict": "pass" | "fail",
  "notes": "One sentence explaining the specific logic gap.",
  "scores": {
    "correctness": 0.0-1.0,
    "completeness": 0.0-1.0,
    "creativity": 0.0-1.0
  }
}
""".strip()

FAILED_TRAJECTORY_IMPROVEMENT_SYSTEM_PROMPT = """
You are a feedback generator. Given a question (and options if provided) and a failed trajectory with committee feedback, summarize the failure and propose a concise improvement plan.
Respond ONLY as JSON with keys: feedback (one sentence) and improvement_plan (one or two actionable steps).
""".strip()


# Backward-compatible aliases for older scripts.
VHR_ZOOM_AGENT_HINT = VHR_ZOOM_VIEW_ADJUSTMENT_APPENDIX
REVIEWER_SYSTEM_PROMPT = TRAJECTORY_LOGIC_AUDIT_SYSTEM_PROMPT
COMMITTEE_SYSTEM_PROMPT = TRAJECTORY_LOGIC_AUDIT_SYSTEM_PROMPT
REFLECTION_SYSTEM_PROMPT = FAILED_TRAJECTORY_IMPROVEMENT_SYSTEM_PROMPT
