from __future__ import annotations

from eo_gym_data_synthesis.prompt_manager.agent_prompts import (
    VHR_ZOOM_VIEW_ADJUSTMENT_APPENDIX,
    build_function_call_prompt,
    get_tools_prompt_text,
)
from eo_gym_data_synthesis.prompt_manager.quality_prompts import (
    FINAL_ANSWER_REASONING_ALIGNMENT_SYSTEM_PROMPT,
    FINAL_MCQ_CONSISTENCY_AUDIT_SYSTEM_PROMPT,
    NEXT_TOOL_CALL_REWRITE_SYSTEM_PROMPT,
    OPEN_TEXT_DATASET_AUDIT_SYSTEM_PROMPT,
    STEPWISE_TOOL_ROUTING_APPENDIX,
    build_final_consistency_prompts,
    build_trajectory_rewrite_prompts,
)
from eo_gym_data_synthesis.prompt_manager.vqa_prompts import (
    VQA_QUESTION_REWRITE_SYSTEM_PROMPT,
    choose_dataset_prompt,
    normalize_prompt_variant_name,
)
from eo_gym_data_synthesis.prompt_manager.evaluation_prompts import (
    FINAL_ANSWER_SEMANTIC_EQUIVALENCE_SYSTEM_PROMPT,
    SELF_REPORTED_CONFIDENCE_EXTRACTION_SYSTEM_PROMPT,
    TOOL_ARGUMENT_SEMANTIC_EQUIVALENCE_SYSTEM_PROMPT,
)

__all__ = [
    "VHR_ZOOM_VIEW_ADJUSTMENT_APPENDIX",
    "build_function_call_prompt",
    "get_tools_prompt_text",
    "FINAL_ANSWER_REASONING_ALIGNMENT_SYSTEM_PROMPT",
    "FINAL_MCQ_CONSISTENCY_AUDIT_SYSTEM_PROMPT",
    "NEXT_TOOL_CALL_REWRITE_SYSTEM_PROMPT",
    "OPEN_TEXT_DATASET_AUDIT_SYSTEM_PROMPT",
    "STEPWISE_TOOL_ROUTING_APPENDIX",
    "build_final_consistency_prompts",
    "build_trajectory_rewrite_prompts",
    "VQA_QUESTION_REWRITE_SYSTEM_PROMPT",
    "choose_dataset_prompt",
    "normalize_prompt_variant_name",
    "FINAL_ANSWER_SEMANTIC_EQUIVALENCE_SYSTEM_PROMPT",
    "SELF_REPORTED_CONFIDENCE_EXTRACTION_SYSTEM_PROMPT",
    "TOOL_ARGUMENT_SEMANTIC_EQUIVALENCE_SYSTEM_PROMPT",
]
