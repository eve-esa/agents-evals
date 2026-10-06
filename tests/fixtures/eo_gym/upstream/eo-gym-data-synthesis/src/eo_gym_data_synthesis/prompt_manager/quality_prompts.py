from __future__ import annotations

import json
from typing import Any, Dict, List, Tuple

STEPWISE_TOOL_ROUTING_APPENDIX = (
    "\n\nAdditional routing:\n"
    "- After each observation, add 1-2 sentences explaining the next tool choice, then emit the next function call.\n"
)

NEXT_TOOL_CALL_REWRITE_SYSTEM_PROMPT = """You are rewriting an assistant message so it smoothly and concretely justifies the NEXT tool call.
Rules:
- Use ONLY the prior conversation summary and the upcoming function call detail.
- Do NOT invent image details that are not already present in prior OBSERVATIONS, except the first assistant message may include one hedged high-level visual clause before verification.
- Do NOT output any tool-call formatting, JSON, or `Tool call:` lines.
- Use get_bbox_geospatial_relationship ONLY for DIRECTIONAL questions.
- For REGION/CONTAINMENT/EDGE/CORNER/CENTRAL questions, state that bbox coordinates will be compared against region or image bounds.
- Mention the tool name and the most relevant parameter values from the upcoming call.
- Output ONLY the rewritten assistant text."""

FINAL_MCQ_CONSISTENCY_AUDIT_SYSTEM_PROMPT = """You audit a tool-using agent's FINAL multiple-choice answer for consistency with the FINAL observation(s) and FINAL thought.
Rules:
- Use ONLY the provided question, options, observations, final thought, and final answer.
- Do NOT invent image details.
- If observations are sufficient and the answer is consistent, output JSON: {"status":"ok"}.
- If inconsistent, output JSON: {"status":"fix","corrected_letter":"A|B|C|D","corrected_text":"<exact option text>","corrected_answer":"(X) <option text>","why":"<1 sentence>"}.
- For get_bbox_geospatial_relationship observations: direction describes loc_a relative to loc_b. If direction is diagonal, pick the axis with larger magnitude using dx/dy when available.
- Output ONLY JSON."""

OPEN_TEXT_DATASET_AUDIT_SYSTEM_PROMPT = """You are auditing EO-GYM QA records for dataset quality. Rewrite only when multiple-choice artifacts, too-short answers, or missing category details make the record unsuitable for open-text training. If rewriting answer, output plain text only with no option letter or `Final Answer:` prefix."""

FINAL_ANSWER_REASONING_ALIGNMENT_SYSTEM_PROMPT = """You are reviewing the last assistant message of a QA trajectory. The message is from a multiple-choice workflow, but the final answer must follow the model's own reasoning, not closest-option fallback logic. Return JSON with action, assistant_content, final_answer, and reason."""


def build_trajectory_rewrite_prompts(
    *,
    prior_conversation: str,
    tool_name: str,
    tool_arguments: Dict[str, Any],
    rewrite_text: str,
    is_first_assistant: bool,
) -> Tuple[str, str]:
    length_rule = "Write 2-4 concise sentences." if is_first_assistant else "Write 1-2 concise sentences."
    first_rule = (
        "Because this is the first assistant message, one hedged high-level visual clause is allowed before tool verification."
        if is_first_assistant
        else "Do not describe image content unless it is explicitly present in prior observations."
    )
    system_prompt = f"{NEXT_TOOL_CALL_REWRITE_SYSTEM_PROMPT}\n- {length_rule}\n- {first_rule}"
    args_text = json.dumps(tool_arguments or {}, ensure_ascii=False, indent=2)
    user_prompt = (
        f"Prior conversation (condensed):\n{prior_conversation}\n\n"
        f"Upcoming function call:\nTool: {tool_name}\nArguments:\n{args_text}\n\n"
        f"Assistant text to rewrite:\n{rewrite_text}\n"
    )
    return system_prompt, user_prompt


def build_final_consistency_prompts(
    *,
    question: str,
    options: Dict[str, str],
    observations: List[str],
    final_thought: str,
    final_answer: str,
) -> Tuple[str, str]:
    obs_block = "\n\n".join([f"Observation {i + 1}:\n{o}" for i, o in enumerate(observations)]) if observations else "(none)"
    options_block = "\n".join([f"{k}. {v}" for k, v in options.items()]) if options else "(none)"
    user_prompt = (
        f"Question:\n{question}\n\n"
        f"Options:\n{options_block}\n\n"
        f"Final observations:\n{obs_block}\n\n"
        f"Final thought:\n{final_thought.strip() if isinstance(final_thought, str) and final_thought.strip() else '(none)'}\n\n"
        f"Final answer:\n{final_answer.strip() if isinstance(final_answer, str) and final_answer.strip() else '(none)'}\n"
    )
    return FINAL_MCQ_CONSISTENCY_AUDIT_SYSTEM_PROMPT, user_prompt


# Backward-compatible aliases for older scripts.
SYSTEM_PROMPT_APPENDIX = STEPWISE_TOOL_ROUTING_APPENDIX
TRAJECTORY_REWRITE_SYSTEM_PROMPT = NEXT_TOOL_CALL_REWRITE_SYSTEM_PROMPT
FINAL_CONSISTENCY_SYSTEM_PROMPT = FINAL_MCQ_CONSISTENCY_AUDIT_SYSTEM_PROMPT
OPEN_TEXT_AUDIT_SYSTEM_PROMPT = OPEN_TEXT_DATASET_AUDIT_SYSTEM_PROMPT
FINAL_REASONING_REVIEW_SYSTEM_PROMPT = FINAL_ANSWER_REASONING_ALIGNMENT_SYSTEM_PROMPT
