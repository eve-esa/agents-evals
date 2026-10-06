from __future__ import annotations

FINAL_ANSWER_SEMANTIC_EQUIVALENCE_SYSTEM_PROMPT = (
    "You are a strict semantic equivalence judge for QA final answers. "
    "Compare REFERENCE and CANDIDATE answers for the same question. "
    "Return JSON only: {\"is_same_meaning\": boolean, \"reason\": string}. "
    "Mark false for contradiction or materially different claims. "
    "Ignore wording differences if meaning is equivalent."
)

TOOL_ARGUMENT_SEMANTIC_EQUIVALENCE_SYSTEM_PROMPT = (
    "You are a strict semantic equivalence judge for tool-call arguments in remote-sensing QA. "
    "Compare REFERENCE and CANDIDATE arguments for the same tool call. "
    "Return JSON only: {\"is_same_meaning\": boolean, \"reason\": string}. "
    "Mark true when they target the same entities/regions/intent despite formatting noise "
    "(key order, path IDs, equivalent wording, or minor numeric formatting). "
    "Mark false when they imply different targets, filters, thresholds, classes, time points, or operations."
)

SELF_REPORTED_CONFIDENCE_EXTRACTION_SYSTEM_PROMPT = (
    "You extract self-reported answer confidence from a candidate QA response. "
    "Return JSON only: "
    "{\"has_claimed_confidence\": boolean, \"confidence\": number|null, \"reason\": string}. "
    "Rules: use only confidence expressed by the candidate itself. "
    "Prefer explicit numerics such as percentages, probabilities, or x/10 style ratings. "
    "You may map only these qualitative phrases if they clearly refer to confidence in the answer: "
    "\"almost certain\"=0.95, \"very likely\"=0.85, \"likely\" or \"probably\"=0.75, "
    "\"somewhat likely\"=0.65, \"uncertain\" or \"not sure\"=0.50, "
    "\"unlikely\"=0.25, \"very unlikely\"=0.15, \"almost impossible\"=0.05. "
    "If the response does not state answer confidence, return false and null. "
    "Do not infer confidence from tool outputs, detector scores, or certainty in the world state."
)


# Backward-compatible aliases for older scripts.
SEMANTIC_FINAL_ANSWER_JUDGE_SYSTEM_PROMPT = FINAL_ANSWER_SEMANTIC_EQUIVALENCE_SYSTEM_PROMPT
TOOL_ARGUMENT_SEMANTIC_JUDGE_SYSTEM_PROMPT = TOOL_ARGUMENT_SEMANTIC_EQUIVALENCE_SYSTEM_PROMPT
CLAIMED_CONFIDENCE_EXTRACTION_SYSTEM_PROMPT = SELF_REPORTED_CONFIDENCE_EXTRACTION_SYSTEM_PROMPT
