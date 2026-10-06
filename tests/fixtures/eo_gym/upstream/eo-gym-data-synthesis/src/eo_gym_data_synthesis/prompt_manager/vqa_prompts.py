"""System prompts used for local inference runs with the OpenAI Responses API."""

from __future__ import annotations

__all__ = [
    "choose_dataset_prompt",
    "normalize_prompt_variant_name",
    "VQA_QUESTION_REWRITE_SYSTEM_PROMPT",
]

import random
from typing import List, Tuple



# ==========================================
# 1. SHARED SAFETY CONSTRAINTS
# ==========================================

# Standard constraint for most datasets
SAFETY_STD = """Highest priority:
- Never prepare options or answers that mention any tool name. Pretend you do not know any tool names.
- Act unaware that any ground truth comes from manual labels.
- Use the provided structured fields only as hidden evidence to craft the question, options, and correct answer.
- Ground each question in a concrete object/category from the provided ground truth (avoid vague phrases like "object" or "objects of interest"); if a label is overly specific (e.g., Boeing 747), generalize to a sensible higher-level category like airplane. fMoW prompts may reference per-frame categories directly.
- Never mention or imply that annotations, labels, polygons, or bounding boxes exist—write as if you only saw the scene itself."""

# Constraint for FAIR1M (Must return "TBD")
SAFETY_TBD_BASE = """Highest priority:
- Never prepare options or answers that mention any tool name. Pretend you do not know any tool names.
- Act unaware that any ground truth comes from manual labels.
- Ground each question in a concrete object/category from the provided ground truth (avoid vague phrases like "object" or "objects of interest"); if a label is overly specific (e.g., Boeing 747), generalize to a sensible higher-level category like airplane.
- Never mention or imply that annotations, labels, polygons, or bounding boxes exist—write as if you only saw the scene itself."""

SAFETY_TBD = f"""{SAFETY_TBD_BASE}
- This dataset is allowed to have deferred answers: set the output field "answer" EXACTLY to "TBD" (no letter), always."""

# Variant for FAIR1M (Short version for simple prompts)
SAFETY_TBD_SHORT = SAFETY_TBD

# Constraint for xBD (Explicitly mentions masks)
SAFETY_XBD = """Highest priority:
- Never prepare options or answers that mention any tool name. Pretend you do not know any tool names.
- Act unaware that any ground truth comes from manual labels.
- Use the provided structured fields only as hidden evidence to craft the question, options, and correct answer.
- Ground each question in a concrete object/category from the provided ground truth (avoid vague phrases like "object" or "objects of interest"); if a label is overly specific (e.g., Boeing 747), generalize to a sensible higher-level category like airplane.
- Never mention or imply that annotations, labels, polygons, masks, or bounding boxes exist—write as if you only saw the scene itself."""


# ==========================================
# 2. DATASET PROMPTS
# ==========================================

# -------------------------
# DIOR (single image; bbox-authoritative)
# -------------------------
DIOR_PROMPT_VARIANTS: List[Tuple[str, str]] = [
    (
        "dior_central_region_object_count",
        f"""
{SAFETY_STD}

Generate exactly one multiple-choice VQA pair for DIOR using `ground_truth.objects` (category_name, bbox [x, y, width, height], image_width, image_height).

Task:
- Select a target category that is present (prefer aircraft/ship/vehicle; otherwise any present category).
- Ask: how many of that target category lie fully inside the central region (the middle 40% by width and height).
- Options (A–D) are integer counts with exactly one correct.

Output (strict):
- Return only a JSON array with one object containing exactly:
  {{ "question", "options" (dict A–D), "answer" ("<letter>. <text>"), "question_type"="crop reasoning" }}.
- If there are no objects, output [].
""".strip(),
    ),
    (
        "dior_elongated_facility_orientation",
        f"""
{SAFETY_STD}

Generate exactly one multiple-choice VQA pair for DIOR from `ground_truth.objects`.

Task:
- Focus on a transport-related elongated facility category if present (e.g., bridge, runway, expressway segment, harbor pier).
- Determine dominant alignment using hidden geometry heuristics only (e.g., aspect ratio and whether w>h or h>w).
- Ask the dominant alignment:
  (A) mostly vertical (north–south),
  (B) mostly horizontal (east–west),
  (C) no clear alignment / mixed,
  (D) not applicable (no elongated facility visible).

Constraints:
- Choose a case where (A)–(C) is unambiguously correct; otherwise output [].

Output JSON array with one object, question_type="orientation".
""".strip(),
    ),
    (
        "dior_facility_footprint_distribution",
        f"""
{SAFETY_STD}

Generate exactly one multiple-choice VQA pair for DIOR using `ground_truth.objects`.

Task:
- Choose a large man-made facility category present (e.g., airport, harbor, stadium, storage tank, train station).
- Ask where the facility footprint is concentrated:
  (A) mostly central, (B) mostly along an edge, (C) mostly in a corner, (D) spread across the scene.
- Use hidden object extents/centroids internally to ensure exactly one correct option.

Output JSON array with one object, question_type="footprint distribution".
""".strip(),
    ),
    (
        "dior_category_count",
        f"""
{SAFETY_STD}

Generate exactly one multiple-choice VQA pair for DIOR using `ground_truth.objects`.

Task:
- Pick a category that is present and ask its exact count.
- Options are counts (A–D) with exactly one correct.

Output JSON array with one object, question_type="counting".
""".strip(),
    ),
    (
        "dior_relative_object_direction",
        f"""
{SAFETY_STD}

Generate exactly one multiple-choice VQA pair for DIOR using `ground_truth.objects`.

Task:
- Select two objects from different categories (prefer large-area ones).
- Decide the relative direction using hidden centers only; if ambiguous, choose a different pair.
- Ask where object B lies relative to object A:
  (A) north of, (B) south of, (C) east of, (D) west of.

Output JSON array with one object, question_type="spatial relationship".
""".strip(),
    ),
    (
        "dior_object_overlap_classification",
        f"""
{SAFETY_STD}

Generate exactly one multiple-choice VQA pair for DIOR using `ground_truth.objects`.

Task:
- Choose two objects (prefer large ones) and classify their relationship using hidden rectangle intersection:
  (A) no overlap,
  (B) edge-touching only,
  (C) partial overlap,
  (D) heavy overlap.
- Only generate this item if the selected pair yields a crisp, unambiguous class; otherwise output [].

Output JSON array with one object, question_type="overlap analysis".
""".strip(),
    ),
    (
        "dior_scene_type_classification",
        f"""
{SAFETY_STD}
- Do NOT justify the correct answer using absence-based wording like “no signs of …”.

Generate exactly one multiple-choice VQA pair for DIOR using `ground_truth.objects`.

Task:
- Ask which scene type best describes the image:
  (A) airport/airfield, (B) harbor/port, (C) industrial storage/energy site, (D) dense residential/urban blocks.
- Use positive-cue gating so exactly ONE option is supported:
  - Choose (A) only if at least one aircraft-related category exists.
  - Choose (B) only if at least one ship/harbor-related category exists.
  - Choose (C) only if at least one storage/energy infrastructure category exists.
  - Choose (D) only if an explicit residential/urban class exists.
- If none or multiple options are supported, output [].

Output JSON array with one object, question_type="scene understanding".
""".strip(),
    ),
    (
        "dior_largest_object_shape_profile",
        f"""
{SAFETY_STD}

Generate exactly one multiple-choice VQA pair for DIOR using `ground_truth.objects`.

Task:
- Select the largest object by hidden footprint proxy.
- Ask whether it appears more:
  (A) elongated, (B) compact, (C) roughly square, (D) unclear/mixed.
- Ensure exactly one correct option using simple hidden geometry heuristics.

Output JSON array with one object, question_type="object attribute".
""".strip(),
    ),
]


# -------------------------
# FAIR1M (multispectral; time-series; ONLY dataset where answer may be "TBD")
# -------------------------
FAIR1M_PROMPT_VARIANTS: List[Tuple[str, str]] = [
    (
        "fair1m_previous_current_landscape_change",
        f"""
{SAFETY_TBD}

Generate exactly one multiple-choice VQA pair for FAIR1M using `ground_truth.object_names`
(a list of object category names only; no geometry).

Rules:
- Must require comparing current capture to the immediately previous available capture (by time).

Task:
- Use the object name list to pick a plausible dominant landscape theme
  (water, urban, or vegetation) for the current capture.
- Ask how that dominant theme changed from the previous capture to the current capture:
  (A) expanded, (B) contracted, (C) stable, (D) not present in either.

Output JSON array with one object, answer="TBD", question_type="temporal analysis".
""".strip(),
    ),
(
        "fair1m_current_next_landscape_change",
        f"""
{SAFETY_TBD}

Generate exactly one multiple-choice VQA pair for FAIR1M using `ground_truth.object_names`
(a list of object category names only; no geometry).

Rules:
- Must require comparing current capture to the immediately next available capture (by time).

Task:
- Use the object name list to pick a plausible dominant landscape theme
  (water, urban, or vegetation) for the current capture.
- Ask how that dominant theme changed from the next capture to the current capture:
  (A) expanded, (B) contracted, (C) stable, (D) not present in either.

Output JSON array with one object, answer="TBD", question_type="temporal analysis".
""".strip(),
    ),
    (
        "fair1m_three_capture_landscape_peak",
        f"""
{SAFETY_TBD}

Generate exactly one multiple-choice VQA pair for FAIR1M using `ground_truth.object_names`
(a list of object category names only; no geometry).

Rules:
- Must require using BOTH the immediately previous capture and the immediately next capture (in addition to current), by time.

Task:
- Use the object name list to choose ONE theme to focus on:
  water, vegetation, or built-up.
- Ask which time step shows the strongest presence of that theme:
- Options must be:
  (A) previous capture, (B) current capture, (C) next capture, (D) no meaningful difference across the three.

Output JSON array with one object, answer="TBD", question_type="temporal analysis".
""".strip(),
    ),
    (
        "fair1m_current_landscape_theme_classification",
        f"""
{SAFETY_TBD}

Generate exactly one multiple-choice VQA pair for FAIR1M using `ground_truth.object_names`
(a list of object category names only; no geometry).

Task:
- Use the object name list to infer which theme best characterizes the CURRENT capture:
  water-dominated, vegetation-dominated, or urban/built-up.
- Ask which theme is most dominant overall:
  (A) water-dominated, (B) vegetation-dominated, (C) urban/built-up, (D) mixed/unclear.

Output JSON array with one object, answer="TBD", question_type="scene understanding".
""".strip(),
    ),
(
        "fair1m_previous_six_month_landscape_change",
        f"""
{SAFETY_TBD}

Generate exactly one multiple-choice VQA pair for FAIR1M using `ground_truth.object_names`
(a list of object category names only; no geometry).

Task:
- Use the object name list to pick a plausible dominant landscape theme
  (water, urban, or vegetation) for the current capture.
- Ask from the previous 6 months to the current capture, how did that dominant theme change:
  (A) expanded, (B) contracted, (C) stable, (D) not present.


Output JSON array with one object, answer="TBD", question_type="temporal analysis".
""".strip(),
    ),
(
        "fair1m_next_six_month_landscape_change",
        f"""
{SAFETY_TBD}

Generate exactly one multiple-choice VQA pair for FAIR1M using `ground_truth.object_names`
(a list of object category names only; no geometry).

Task:
- Use the object name list to pick a plausible dominant landscape theme
  (water, urban, or vegetation) for the current capture.
- Ask from the current capture to the next 6 months, how did that dominant theme change:
  (A) expanded, (B) contracted, (C) stable, (D) not present.


Output JSON array with one object, answer="TBD", question_type="temporal analysis".
""".strip(),
    ),
]


# -------------------------
# M4-SAR (cross-modal; bbox-authoritative)
# -------------------------
M4_SAR_PROMPT_VARIANTS: List[Tuple[str, str]] = [
    (
        "m4_sar_cross_modal_category_count",
        f"""
{SAFETY_STD}

Generate exactly one multiple-choice VQA pair for M4-SAR using `ground_truth.objects` (category, bbox, role).

Task:
- Ask a detection/counting question about a specific category present (e.g., ships, aircraft, vehicles).
- Write the question so it naturally benefits from comparing SAR and optical views (without mentioning any tools).
- Options are counts (A–D) with exactly one correct.

Output JSON array with one object, question_type="cross-modal analysis".
""".strip(),
    ),
    (
        "m4_sar_cross_modal_spatial_concentration",
        f"""
{SAFETY_STD}

Generate exactly one multiple-choice VQA pair for M4-SAR using `ground_truth.objects` (category, bbox, role).

Task:
- Ask where a target category is concentrated:
  (A) mostly upper half, (B) mostly lower half, (C) mostly left half, (D) mostly right half.
- Ensure exactly one correct option using hidden object centers.

Output JSON array with one object, question_type="cross-modal analysis".
""".strip(),
    ),
]


# -------------------------
# SARDet-100K (single SAR image; bbox-authoritative)
# -------------------------
SARDET_PROMPT_VARIANTS: List[Tuple[str, str]] = [
    (
        "sardet_central_region_ship_count",
        f"""
{SAFETY_STD}

Generate exactly one multiple-choice VQA pair for SARDet-100K using `ground_truth.boxes` (label, bbox).

Task:
- Select ships only.
- Ask: how many ships lie fully inside the central region (middle 40% by width and height).
- Options are counts (A–D) with one correct.

Output JSON array with one object, question_type="crop reasoning".
""".strip(),
    ),
    (
        "sardet_relative_ship_direction",
        f"""
{SAFETY_STD}

Generate exactly one multiple-choice VQA pair for SARDet-100K using `ground_truth.boxes`.

Task:
- Identify the largest and second-largest ships by hidden footprint proxy.
- Ask where the second-largest ship is relative to the largest:
  (A) north of, (B) south of, (C) east of, (D) west of.
- If ambiguous, choose a different pair; if impossible, output [].

Output JSON array with one object, question_type="spatial relationship".
""".strip(),
    ),
    (
        "sardet_ship_size_ratio",
        f"""
{SAFETY_STD}

Generate exactly one multiple-choice VQA pair for SARDet-100K using `ground_truth.boxes`.

Task:
- Compute the median ship footprint proxy internally.
- Define large ships as area > median, small ships as area <= median.
- Ask for the ratio (large_count / small_count), rounded to one decimal.
- Provide options (A–D) as numeric values with one correct.
- If small_count == 0, output [].

Output JSON array with one object, question_type="arithmetic reasoning".
""".strip(),
    ),
    (
        "sardet_ship_overlap_classification",
        f"""
{SAFETY_STD}

Generate exactly one multiple-choice VQA pair for SARDet-100K using `ground_truth.boxes`.

Task:
- Select the two largest objects by hidden footprint proxy.
- Ask their overlap relationship:
  (A) no overlap, (B) edge-touching only, (C) partial overlap, (D) heavy overlap.
- If fewer than 2 objects exist, output [].

Output JSON array with one object, question_type="overlap analysis".
""".strip(),
    ),
    (
        "sardet_ship_footprint_quadrant_dominance",
        f"""
{SAFETY_STD}

Generate exactly one multiple-choice VQA pair for SARDet-100K using `ground_truth.boxes`.

Task:
- Divide the image into quadrants (NW, NE, SW, SE).
- Ask which quadrant contains the greatest total ship footprint proxy.
- Options are the four quadrants; ensure exactly one correct.

Output JSON array with one object, question_type="spatial distribution".
""".strip(),
    ),
]


# -------------------------
# VHR_Zoom (fine-grained partial patch reasoning)
VHR_ZOOM_PROMPT_VARIANTS: List[Tuple[str, str]] = [
    (
        "vhr_zoom_edge_aware_object_count",
        f"""
{SAFETY_STD}

Generate exactly one multiple-choice VQA pair for VHR_Zoom.

Context:
- You will receive only `cropped_ground_truth`, which describes the current image view you can see.
- `cropped_ground_truth.crop_rect` gives the view window (x, y, width, height) in the original image coordinate system; use it only as hidden evidence.
- `cropped_ground_truth.objects` lists objects visible in the current view; each has `category_name`, `bbox`, and may include `is_partial`/`partial_sides`.
- `is_partial=true` means the object is cut off by the image boundary (not fully visible in this view).
- `cropped_ground_truth.object_label_counts` is the total count of each category visible in the view (including partial objects).
- `cropped_ground_truth.partial_label_counts` is the count of partially visible objects per category in the view.
- If an object name is generic (e.g., “another boat”, “other-ship”), refer to it simply as “boat”.

Task:
- Select an object category that appears in the current view; prefer one that has at least one partially visible instance (present in `cropped_ground_truth.partial_label_counts`). If none are partial, choose the most frequent category from `cropped_ground_truth.object_label_counts`.
- Ask a “how many objects are visible in this image” question: the total number of that object visible in the current view.
- Do NOT mention that the image is cropped/zoomed or use phrases like “cropped patch” or “in this crop”; phrase it as a normal image question.
- Example phrasing: “How many boats are visible in the image in total? If any boats at the edge look incomplete, move the view to confirm them.”
- Do NOT ask questions that count only the partially visible objects or that contrast “partial vs fully contained” (avoid phrasing like “only partially visible” / “rather than fully contained”).
- Do NOT mention there are some objects are partial display in the question.
- Use `cropped_ground_truth.object_label_counts` for the correct answer; use `cropped_ground_truth.partial_label_counts` to craft plausible distractors (e.g., partial-only count, total-minus-partial, nearby counts).
- Keep object naming simple (no quotes; use a plain noun like “boat” when appropriate) and do not mention annotations, tools, or bounding box coordinates.

Output (strict):
- Return a JSON array with exactly one object containing `question`, `options` (dict with keys "A"–"D"), `answer` formatted as "<letter>. <option text>", and `question_type`="moving reasoning".
""".strip(),
    ),
]


# -------------------------
# fMoW (time-series; bbox-authoritative)
# -------------------------
FMOW_PROMPT_VARIANTS: List[Tuple[str, str]] = [
(
        "fmow_earliest_latest_category_change",
        f"""
{SAFETY_STD}

Generate exactly one multiple-choice VQA pair for fMoW using `ground_truth.frames`.

Temporal ordering:
- Sort frames by timestamp ascending; tie-break by frame_index.

Task:
- Compare the earliest vs latest frame.
- We will use relative timestamp in YYYY format to define earliest and latest range (e.g., from 2013 to 2018).
- Ask which category shows the largest increase in count from earliest to latest.
- Options are four candidate categories drawn from those present across frames; ensure exactly one correct (avoid ties; else output []).

Output JSON array with one object, question_type="temporal analysis".
""".strip(),
    ),
(
        "fmow_previous_frame_total_object_count_change",
        f"""
{SAFETY_STD}

Generate exactly one multiple-choice VQA pair for fMoW using `ground_truth.frames`
(each frame may include: frame_index, timestamp, filename (optional), and bounding_boxes).

Temporal ordering:
- Sort frames by timestamp ascending.
- If timestamps are equal or missing, break ties by frame_index ascending.
- Define the “immediately previous available frame” as the frame directly preceding the chosen current frame in this order.

Counting definition:
- “Total visible object count” is exactly len(frame.bounding_boxes). Treat missing/empty as 0.

Task:
- Select a current frame that has an immediately previous frame if possible.
- Ask whether the total object count in the current frame compared to the immediately previous frame:
  (A) increased, (B) decreased, (C) stayed the same, (D) previous frame not available.
- If no previous frame exists for the chosen current frame (or <2 valid frames), the correct answer MUST be D.

Question clarity:
- Reference the current frame by timestamp (and filename if available; otherwise include frame_index).
- Do NOT mention any annotations/boxes/labels/tools.

Output JSON array with one object, question_type="temporal analysis".
""".strip(),
    ),
    (
        "fmow_next_frame_total_object_count_change",
        f"""
{SAFETY_STD}

Generate exactly one multiple-choice VQA pair for fMoW using `ground_truth.frames`.

Temporal ordering:
- Sort frames by timestamp ascending; tie-break by frame_index ascending.

Counting definition:
- “Total visible object count” is exactly len(frame.bounding_boxes). Treat missing/empty as 0.

Task:
- Select a current frame that has an immediately next frame if possible.
- Ask whether the total object count in the current frame compared to the immediately next frame:
  (A) increased, (B) decreased, (C) stayed the same, (D) next frame not available.
- If no next frame exists for the chosen current frame (or <2 valid frames), the correct answer MUST be D.

Question clarity:
- Reference the current frame by timestamp (and filename if available; otherwise include frame_index).

Output JSON array with one object, question_type="temporal analysis".
""".strip(),
    ),
    (
        "fmow_single_frame_category_count",
        f"""
{SAFETY_STD}

Generate exactly one multiple-choice VQA pair for fMoW using `ground_truth.frames`.

Task:
- Choose a single frame (by timestamp; include filename if available, else frame_index).
- Pick a category that is present in that frame and ask its exact count.
- Options are integer counts with exactly one correct.

Output JSON array with one object, question_type="counting".
""".strip(),
    ),
(
        "fmow_latest_frame_area_dominance",
        f"""
{SAFETY_STD}

Generate exactly one multiple-choice VQA pair for fMoW using `ground_truth.frames`.

Temporal ordering:
- Sort frames by timestamp ascending; tie-break by frame_index.

Task:
- Compare the earliest vs latest frame.
- We will use relative timestamp in YYYY format to define earliest and latest range (e.g., from 2013 to 2018).
- In the latest frame, compute total footprint proxy per category internally (sum of w*h over objects).
- Ask which category occupies the largest total visible footprint in that frame.
- Ensure exactly one correct option; if <4 unique categories, add plausible distractor categories (not “None of the above”).

Output JSON array with one object, question_type="area dominance".
""".strip(),
    ),
    (
        "fmow_previous_current_built_up_similarity",
        f"""
{SAFETY_STD.replace('Use the provided structured fields only as hidden evidence to craft the question, options, and correct answer.', '')}

Generate exactly one multiple-choice VQA pair for fMoW.

Task:
- Choose a current frame that has an immediately previous frame (by timestamp order); otherwise output [].
- Ask how similar the overall built-up footprint appears between the previous frame and the current frame:
  (A) highly similar, (B) moderately similar, (C) low similarity, (D) very low similarity.

Important:
- This variant is NOT allowed to use "TBD". Ensure the correct option is supported by the provided structured fields; if not possible, output [].

Output JSON array with one object, question_type="overlap analysis".
""".strip(),
    ),
]


# -------------------------
# HichTala xView (single image; bbox-authoritative)
# -------------------------
HICHTALA_XVIEW_PROMPT_VARIANTS: List[Tuple[str, str]] = [
    (
        "hichtala_xview_central_region_object_count",
        f"""
{SAFETY_STD}

Generate exactly one multiple-choice VQA pair for HichTala xView using `ground_truth.objects`.

Task:
- Ask how many objects lie fully inside the central region (middle 40% by width and height).
- Options are counts with exactly one correct.

Output JSON array with one object, question_type="crop reasoning".
""".strip(),
    ),
    (
        "hichtala_xview_category_count",
        f"""
{SAFETY_STD}

Generate exactly one multiple-choice VQA pair for HichTala xView using `ground_truth.objects`.

Task:
- Choose a category that is present and ask its exact count.
- Options are counts with exactly one correct.

Output JSON array with one object, question_type="counting".
""".strip(),
    ),
    (
        "hichtala_xview_relative_object_direction",
        f"""
{SAFETY_STD}

Generate exactly one multiple-choice VQA pair for HichTala xView using `ground_truth.objects`.

Task:
- Select two objects (prefer large ones) with an unambiguous direction.
- Ask where object B is relative to object A:
  (A) north of, (B) south of, (C) east of, (D) west of.
- If ambiguous or <2 objects, output [].

Output JSON array with one object, question_type="spatial relationship".
""".strip(),
    ),
    (
        "hichtala_xview_category_count_ratio",
        f"""
{SAFETY_STD}

Generate exactly one multiple-choice VQA pair for HichTala xView using `ground_truth.objects`.

Task:
- Compute counts per category.
- If fewer than 2 unique categories exist, output [].
- Compute ratio (top_category_count / second_category_count), rounded to one decimal.
- Provide four numeric options with one correct.

Output JSON array with one object, question_type="arithmetic reasoning".
""".strip(),
    ),
    (
        "hichtala_xview_object_overlap_classification",
        f"""
{SAFETY_STD}

Generate exactly one multiple-choice VQA pair for HichTala xView using `ground_truth.objects`.

Task:
- Choose the two largest objects by hidden footprint proxy.
- Ask overlap relationship:
  (A) no overlap, (B) edge-touching only, (C) partial overlap, (D) heavy overlap.
- If <2 objects, output [].

Output JSON array with one object, question_type="overlap analysis".
""".strip(),
    ),
    (
        "hichtala_xview_scene_type_classification",
        f"""
{SAFETY_STD}
- Do NOT justify answers via absence-only phrasing (“no signs of …”).

Generate exactly one multiple-choice VQA pair for HichTala xView using `ground_truth.objects`.

Task:
- Ask which scene type best fits:
  (A) dense urban/built-up, (B) sparse rural, (C) port/industrial waterfront, (D) predominantly vegetated.
- Use positive-cue gating from categories to ensure exactly one correct; if ambiguous, output [].

Output JSON array with one object, question_type="scene understanding".
""".strip(),
    ),
    (
        "hichtala_xview_largest_object_shape_profile",
        f"""
{SAFETY_STD}

Generate exactly one multiple-choice VQA pair for HichTala xView using `ground_truth.objects`.

Task:
- Select the largest object by hidden footprint proxy.
- Ask whether it is:
  (A) elongated, (B) compact, (C) roughly square, (D) unclear/mixed.
- Ensure exactly one correct option.

Output JSON array with one object, question_type="object attribute".
""".strip(),
    ),
]


# -------------------------
# HichTala DOTA (single image; bbox-authoritative)
# -------------------------
HICHTALA_DOTA_PROMPT_VARIANTS: List[Tuple[str, str]] = [
    (
        "hichtala_dota_central_region_object_count",
        f"""
{SAFETY_STD}

Generate exactly one multiple-choice VQA pair for HichTala DOTA using `ground_truth.objects` and image size.

Task:
- Define the central region as the middle 40% by width and middle 40% by height:
  left = 0.30*W, right = 0.70*W, top = 0.30*H, bottom = 0.70*H.
- Choose a target object group with a clear visual meaning:
  Prefer vehicles if present (count BOTH "small-vehicle" and "large-vehicle" as "vehicles");
  else prefer "plane" or "ship" if present; otherwise choose the most frequent category.
- Count how many target objects lie FULLY inside the central region:
  x>=left, y>=top, x+w<=right, y+h<=bottom.

Options:
- Four integer options (A–D), one correct. Distractors should be close (e.g., correct±1/±2), non-negative.

Question constraints:
- Do not mention any category IDs or annotation concepts.
- Do not hardcode sample-specific object positions or counts in the question text.

Output format (strict):
- Return only a JSON array.
- If constraints cannot be satisfied (e.g., no objects), output [].
- Otherwise output one object with exactly:
  "question", "options" (A–D), "answer" ("<letter>. <text>"), "question_type"="crop reasoning".
""".strip(),
    ),
    (
        "hichtala_dota_category_count",
        f"""
{SAFETY_STD}
- Never infer information from missing labels.

Generate exactly one multiple-choice VQA pair for HichTala DOTA using `ground_truth.objects`.

Task:
- Pick a category that is present AND has count between 2 and 12 (inclusive).
  Prefer: plane, ship, storage-tank, large-vehicle, small-vehicle; otherwise any present category.
- Ask: “How many [category name pluralized naturally] are visible in the image?”
- Provide four numeric options with exactly one correct.

If no category satisfies the 2..12 constraint, output [].

Output JSON array:
- One object with "question", "options"(A–D), "answer", "question_type"="counting".
- No extra keys or commentary.
""".strip(),
    ),
    (
        "hichtala_dota_category_area_dominance",
        f"""
{SAFETY_STD}
- Never mention bounding boxes/areas explicitly in the question.

Generate exactly one multiple-choice VQA pair for HichTala DOTA using `ground_truth.objects`.

Task:
- Internally compute total footprint proxy per category as sum(w*h) over objects of that category.
- Ask which category occupies the largest total visible footprint in the image.

Options:
- Provide four category options (A–D).
- Ensure exactly one correct.
- If fewer than 4 unique categories exist in the image, add plausible distractor categories
  (use common DOTA categories like plane, ship, storage-tank, harbor, bridge, swimming-pool, helicopter, vehicles),
  but DO NOT include any “None of the above”.

If there are zero objects, output [].

Output JSON array:
- One object with "question", "options"(A–D), "answer", "question_type"="area dominance".
""".strip(),
    ),
    (
        "hichtala_dota_relative_object_direction",
        f"""
{SAFETY_STD}

Generate exactly one multiple-choice VQA pair for HichTala DOTA using `ground_truth.objects`.

Task:
- Choose two objects from different categories, prioritizing large footprint objects.
- Define object A as the larger (by w*h) and object B as the smaller.
- Compare centers (cx, cy) to decide relative direction of B vs A:
  north/south/east/west (choose the dominant axis by absolute delta).
- Ask: “Where is the [category of B] relative to the [category of A]?”

Options (fixed):
(A) north of
(B) south of
(C) east of
(D) west of

Constraints:
- Do not describe “top-right”, “bottom-left”, or any sample-specific corner phrase.
- If fewer than 2 objects exist, output [].

Output JSON array:
- One object with "question", "options"(A–D), "answer", "question_type"="spatial relationship".
""".strip(),
    ),
    (
        "hichtala_dota_object_overlap_classification",
        f"""
{SAFETY_STD}

Generate exactly one multiple-choice VQA pair for HichTala DOTA using `ground_truth.objects`.

Task:
- Choose two objects of the SAME category (prefer vehicles or planes/ships if multiple exist),
  and among that category pick the two largest by w*h.
- Compute axis-aligned rectangle intersection area exactly.

Classify relationship:
(A) no overlap (intersection area == 0 and not edge-touching)
(B) edge-touching only (intersection area == 0 and rectangles share a boundary line)
(C) partial overlap (0 < intersection area < 0.5 * min(area1, area2))
(D) heavy overlap (intersection area >= 0.5 * min(area1, area2))

If no category has at least 2 objects, output [].

Output JSON array:
- One object with "question", "options"(A–D), "answer", "question_type"="overlap analysis".
""".strip(),
    ),
    (
        "hichtala_dota_category_count_ratio",
        f"""
{SAFETY_STD}
- Never infer from missing labels.

Generate exactly one multiple-choice VQA pair for HichTala DOTA using `ground_truth.objects`.

Task:
- Compute counts per category.
- Let c1 be the largest category count, c2 be the second-largest category count.
- If fewer than 2 unique categories exist OR c2 == 0, output [].
- Compute ratio r = c1 / c2 rounded to one decimal.
- Ask: “The most common object type appears about how many times as often as the second most common type?”

Options:
- Four numeric options formatted with one decimal place; one correct, distractors close.

Output JSON array:
- One object with "question", "options"(A–D), "answer", "question_type"="arithmetic reasoning".
""".strip(),
    ),
    (
        "hichtala_dota_largest_object_elongation",
        f"""
{SAFETY_STD}

Generate exactly one multiple-choice VQA pair for HichTala DOTA using `ground_truth.objects`.

Task:
- Select the single largest object by w*h.
- Compute aspect ratio ar = max(w,h) / min(w,h).
- If ar < 1.3 (not clearly elongated), output [].
- Decide whether it is more wide-than-tall or tall-than-wide.

Question:
- Ask whether the largest prominent object appears:
  (A) elongated horizontally, (B) elongated vertically, (C) roughly square/compact, (D) unclear/mixed.

Correctness:
- Use w vs h and ar to ensure exactly one correct option.

Output JSON array:
- One object with "question", "options"(A–D), "answer", "question_type"="visual interpretation".
""".strip(),
    ),
    (
        "hichtala_dota_scene_type_classification",
        f"""
{SAFETY_STD}
- Do NOT justify answers using “no signs of …” or other absence-based reasoning.
- Only use decisive positive cues.

Generate exactly one multiple-choice VQA pair for HichTala DOTA using `ground_truth.objects`.

Task (scene type):
(A) airport/airfield
(B) harbor/port
(C) industrial storage/energy
(D) dense residential/urban blocks

Positive-cue gating (must satisfy exactly ONE):
- Choose (A) only if at least one "plane" or "helicopter" object exists.
- Choose (B) only if at least one "ship" or "harbor" object exists.
- Choose (C) only if at least one "storage-tank" object exists.
- Choose (D) only if there is an explicit residential/urban class present in labels (rare); otherwise do not use (D) as correct.

If no single option is supported by a positive cue OR multiple options are simultaneously supported, output [].

Question constraints:
- Keep wording generic; do not mention counts, corners, or “no signs of X”.

Output JSON array:
- One object with "question", "options"(A–D), "answer", "question_type"="scene understanding".
""".strip(),
    ),
]


# -------------------------
# xBD (before/after; MUST require both images; NEVER "TBD")
# -------------------------
XBD_PROMPT_VARIANTS: List[Tuple[str, str]] = [
    (
        "xbd_damage_category_count",
        f"""
{SAFETY_XBD}

Generate exactly one multiple-choice VQA pair for xBD using `ground_truth.label_counts`
(label counts only; no geometry).

Task:
- Require BOTH images: phrase the question as based on comparing the before- and after-disaster views.
- Use whole-image scope (no windows).
- Use `ground_truth.label_counts.after` to pick a damage state with a nonzero count
  (prefer "destroyed" or "major-damage" if present).
- Count how many buildings fall into that damage state across the whole image.
- Options (A–D) are small non-negative integers with exactly one correct (prefer counts in 0..6).

Output (strict):
- JSON array with exactly one object containing: question, options(A–D), answer("<letter>. <text>"), question_type="damage_count".
- NEVER output "TBD" anywhere.
- If the sum of `ground_truth.label_counts.after` is 0 or no valid small-count setup, output [].
""".strip(),
    ),
    (
        "xbd_damage_category_presence",
        f"""
{SAFETY_XBD}

Generate exactly one multiple-choice VQA pair for xBD using `ground_truth.label_counts`
(label counts only; no geometry).

Task:
- Require BOTH images: phrase the question as based on comparing before vs after.
- Use whole-image scope (no windows).
- Use `ground_truth.label_counts.after` to decide; proceed only if exactly ONE damage category
  has a nonzero count; otherwise output [].
- Ask which damage category appears in the after-disaster scene.
- Options must be exactly the 4 categories (each exactly once):
  no-damage, minor-damage, major-damage, destroyed.

Output JSON array with one object:
{{question, options(A–D), answer("<letter>. <text>"), question_type="damage_presence"}}

Hard rule: NEVER output "TBD".
If the sum of `ground_truth.label_counts.after` is 0, output [].
""".strip(),
    ),
    (
        "xbd_most_common_damage_category",
        f"""
{SAFETY_XBD}

Generate exactly one multiple-choice VQA pair for xBD using `ground_truth.label_counts`
(label counts only; no geometry).

Task:
- Require BOTH images: phrase as “when comparing before and after…”.
- Use whole-image scope (no windows).
- Use `ground_truth.label_counts.after` to determine the most common post-disaster damage category.
- If there is a tie, output [].

Options must be exactly the 4 categories (each exactly once):
A: no-damage
B: minor-damage
C: major-damage
D: destroyed
(You may shuffle letters, but include all four exactly once.)

Output JSON array with one object:
question, options, answer("<letter>. <text>"), question_type="damage_majority"

Hard rule: NEVER output "TBD".
If the sum of `ground_truth.label_counts.after` is 0, output [].
""".strip(),
    ),
    (
        "xbd_maximum_damage_severity",
        f"""
{SAFETY_XBD}

Generate exactly one multiple-choice VQA pair for xBD using `ground_truth.label_counts`
(label counts only; no geometry).

Task:
- Require BOTH images: phrase as “based on comparing before and after…”.
- Use whole-image scope (no windows).
- Use `ground_truth.label_counts.after` to determine the maximum damage severity present under:
  destroyed > major-damage > minor-damage > no-damage.
- Create 4 options corresponding to these levels (worded naturally, e.g., “Destroyed damage is present”, etc.).
- Ensure only one is correct (the maximum present).

Output JSON array with one object:
question, options(A–D), answer("<letter>. <text>"), question_type="damage_max_severity"

Hard rule: NEVER output "TBD".
If the sum of `ground_truth.label_counts.after` is 0, output [].
""".strip(),
    ),
    (
        "xbd_damaged_building_ratio_bucket",
        f"""
{SAFETY_XBD}

Generate exactly one multiple-choice VQA pair for xBD using `ground_truth.label_counts`
(label counts only; no geometry).

Task:
- Require BOTH images: phrase as “based on comparing before and after…”.
- Use `ground_truth.label_counts.after`.
- Define “damaged” as any of: minor-damage, major-damage, destroyed.
- Compute damaged_ratio = damaged_count / total_buildings (sum of after counts).
- Ask which bucket damaged_ratio falls into:
  A: 0%
  B: (0%, 25%]
  C: (25%, 50%]
  D: > 50%
- Ensure the bucket is provably correct from hidden counts.

Output JSON array with one object:
question, options(A–D), answer("<letter>. <text>"), question_type="damage_ratio_bucket"

Hard rule: NEVER output "TBD".
If the sum of `ground_truth.label_counts.after` is 0, output [].
""".strip(),
    ),
]


DATASET_PROMPT_ALIASES = {
    "fair1m": "fair1m_full",
    "fair1m_full": "fair1m_full",
    "hichtala_dota": "hichtala/dota",
    "hichtala_xview": "hichtala/xview",
    "sardet-100k": "sardet_100k",
    "sardet_100k": "sardet_100k",
    "m4sar": "m4-sar",
    "m4-sar": "m4-sar",
    "changechat": "changechat",
    "xbd": "xbd",
    "vhr_zoom": "vhr_zoom",
    "dior": "dior",
    "fmow": "fmow",
}


OUTPUT_SCHEMA_SUFFIX = """
Output format (strict):
- Return only a JSON array.
- Each element is an object with exactly: "question", "options" (dict with keys "A","B","C","D"), "answer" formatted as "<letter>. <option text>", and "question_type".
- No extra commentary or keys outside that array.
""".strip()


LEGACY_PROMPT_VARIANT_NAME_MAP = {
    "dior_central_window_count": "dior_central_region_object_count",
    "dior_transport_orientation": "dior_elongated_facility_orientation",
    "dior_category_counting": "dior_category_count",
    "dior_spatial_relationship": "dior_relative_object_direction",
    "dior_overlap_analysis": "dior_object_overlap_classification",
    "dior_scene_understanding": "dior_scene_type_classification",
    "dior_object_attribute_shape": "dior_largest_object_shape_profile",
    "fair1m_prev_current_extent_change": "fair1m_previous_current_landscape_change",
    "fair1m_current_next_extent_change": "fair1m_current_next_landscape_change",
    "fair1m_three_point_extreme_selection": "fair1m_three_capture_landscape_peak",
    "fair1m_landscape_spatial_relation_current": "fair1m_current_landscape_theme_classification",
    "fair1m_time_range_current_before_landscape_changes": "fair1m_previous_six_month_landscape_change",
    "fair1m_time_range_current_next_landscape_changes": "fair1m_next_six_month_landscape_change",
    "m4_sar_cross_modal_from_sar_count": "m4_sar_cross_modal_category_count",
    "m4_sar_cross_modal_from_optical_spatial": "m4_sar_cross_modal_spatial_concentration",
    "sardet_central_window_ship_count": "sardet_central_region_ship_count",
    "sardet_spatial_relationship": "sardet_relative_ship_direction",
    "sardet_ratio_arithmetic": "sardet_ship_size_ratio",
    "sardet_overlap_analysis": "sardet_ship_overlap_classification",
    "sardet_quadrant_dominance": "sardet_ship_footprint_quadrant_dominance",
    "vhr_zoom_partial_crop_count": "vhr_zoom_edge_aware_object_count",
    "fmow_category_delta_earliest_latest": "fmow_earliest_latest_category_change",
    "fmow_prev_change_total_count": "fmow_previous_frame_total_object_count_change",
    "fmow_next_change_total_count": "fmow_next_frame_total_object_count_change",
    "fmow_category_counting_in_frame": "fmow_single_frame_category_count",
    "fmow_area_dominance_latest": "fmow_latest_frame_area_dominance",
    "fmow_building_footprint_similarity_prev_current": "fmow_previous_current_built_up_similarity",
    "hichtala_xview_central_window_count": "hichtala_xview_central_region_object_count",
    "hichtala_xview_category_counting": "hichtala_xview_category_count",
    "hichtala_xview_spatial_relationship": "hichtala_xview_relative_object_direction",
    "hichtala_xview_ratio_arithmetic": "hichtala_xview_category_count_ratio",
    "hichtala_xview_overlap_analysis": "hichtala_xview_object_overlap_classification",
    "hichtala_xview_scene_understanding": "hichtala_xview_scene_type_classification",
    "hichtala_xview_object_attribute_shape": "hichtala_xview_largest_object_shape_profile",
    "hichtala_dota_central_window_counting": "hichtala_dota_central_region_object_count",
    "hichtala_dota_category_counting": "hichtala_dota_category_count",
    "hichtala_dota_area_dominance": "hichtala_dota_category_area_dominance",
    "hichtala_dota_spatial_relationship": "hichtala_dota_relative_object_direction",
    "hichtala_dota_overlap_analysis": "hichtala_dota_object_overlap_classification",
    "hichtala_dota_ratio_arithmetic": "hichtala_dota_category_count_ratio",
    "hichtala_dota_shape_elongation": "hichtala_dota_largest_object_elongation",
    "hichtala_dota_scene_understanding": "hichtala_dota_scene_type_classification",
    "xbd_damage_count_window": "xbd_damage_category_count",
    "xbd_damage_presence_window": "xbd_damage_category_presence",
    "xbd_most_common_damage": "xbd_most_common_damage_category",
    "xbd_max_damage_severity": "xbd_maximum_damage_severity",
    "xbd_damaged_ratio_bucket": "xbd_damaged_building_ratio_bucket",
}


def normalize_prompt_variant_name(name: str | None) -> str | None:
    if not isinstance(name, str):
        return name
    return LEGACY_PROMPT_VARIANT_NAME_MAP.get(name, name)


def _get_dataset_prompt_variants(dataset: str | None) -> List[Tuple[str, str]]:
    """
    Return a list of (variant_name, prompt_text) for the dataset.
    For DIOR, FAIR1M, fMoW, HichTala DOTA/xView, M4-SAR, SARDet, and xBD this includes targeted single-template prompts to improve tool coverage.
    """
    key = (dataset or "").strip().lower()
    canonical = DATASET_PROMPT_ALIASES.get(key, key)
    variants: List[Tuple[str, str]]
    if canonical == "dior":
        variants = list(DIOR_PROMPT_VARIANTS)
    elif canonical == "fair1m_full":
        variants = list(FAIR1M_PROMPT_VARIANTS)
    elif canonical == "fmow":
        variants = list(FMOW_PROMPT_VARIANTS)
    elif canonical == "hichtala/dota":
        variants = list(HICHTALA_DOTA_PROMPT_VARIANTS)
    elif canonical == "hichtala/xview":
        variants = list(HICHTALA_XVIEW_PROMPT_VARIANTS)
    elif canonical == "m4-sar":
        variants = list(M4_SAR_PROMPT_VARIANTS)
    elif canonical == "sardet_100k":
        variants = list(SARDET_PROMPT_VARIANTS)
    elif canonical == "xbd":
        variants = list(XBD_PROMPT_VARIANTS)
    elif canonical == "vhr_zoom":
        variants = list(VHR_ZOOM_PROMPT_VARIANTS)
    else:
        variants = [(canonical or "default", "")]

    return variants

def choose_dataset_prompt(dataset: str | None, strategy: str = "default", position: int | None = None) -> Tuple[str, str]:
    """
    Select a prompt variant for the dataset.
    - strategy="random": round-robin if a position is provided, otherwise random choice.
    - strategy="default": pick the first variant (deterministic).
    """
    variants = _get_dataset_prompt_variants(dataset)
    if strategy == "random" and len(variants) > 1:
        # When a position-aware call provides a numeric seed, prefer round-robin
        # selection to ensure coverage across variants. The caller can map position
        # to a record index to cycle deterministically.
        if isinstance(position, int):
            idx = position % len(variants)
            name, prompt = variants[idx]
            return (name, f"{prompt}\n\n{OUTPUT_SCHEMA_SUFFIX}")
        name, prompt = random.choice(variants)
        return (name, f"{prompt}\n\n{OUTPUT_SCHEMA_SUFFIX}")
    name, prompt = variants[0]
    return (name, f"{prompt}\n\n{OUTPUT_SCHEMA_SUFFIX}")

VQA_QUESTION_REWRITE_SYSTEM_PROMPT = """
You are a Remote Sensing VQA Editor. Your task is to rewrite the user's Question to ensure it is valid, objective, and does not leak the answer.

### INSTRUCTIONS:
1. **Remove Leakage:** Rewrite questions that describe the scene in the premise (e.g., change "Given the ships, what facility is this?" to "Identify the facility type based on visible infrastructure.").
2. **Filter Garbage:** If the Question is an error log (e.g., "Sorry, I cannot..."), output exactly: `[DELETE]`.
3. **Filter Common Object:** If the Question link to too common objects (e.g., how man objects/object of interest), not link to detail object types. Because the model need detail object type to run object detection in the future. So output exactly: `[DELETE]`.
4. **Correct temporal detail:** If the Question about temporal analysis, it must include the temporal context (e.g., from current to next/previous capture imagery, or since 4 years ago or to 5 years later).
5. **Preserve Logic:** The rewritten question MUST make the provided `Answer` valid. Do not change what is being counted or identified.
6. **Output:** Return ONLY the clean question text.
""".strip()


# Backward-compatible alias for older scripts.
REVIEW_VQA_PROMPT = VQA_QUESTION_REWRITE_SYSTEM_PROMPT
