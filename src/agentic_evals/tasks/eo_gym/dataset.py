"""EO-Gym test split as Inspect samples.

Mirrors EO-Gym's evaluation driver (``eo_gym_evaluation/driver.py``,
``build_initial_messages`` / ``build_user_text``; ``data.py`` for field
extraction) at the revision pinned in ``upstream.lock.toml``:

- system message: the "simple" or "complex" prompt (``prompts.py``, verbatim);
- user message: ``Question: <question>`` plus "Available image references", then
  each image preceded by a descriptor ("Image N", or "Pre-/Post-disaster image"
  when there are two). The question is taken from the record's user message
  with ``<image>`` placeholders removed, as the driver does (which keeps the
  record's own ``Question:`` prefix, so the text reads ``Question: Question: ...``).

One deliberate difference: image references in the text are the dataset's
relative ``EO_GYM_DATA/...`` paths (the driver's ``file_url``/``path`` modes, used
for its vLLM runs) rather than base64 data URLs (its default for OpenAI), which
would paste every image into the text a second time. The tool server and the gold
trajectories use the same relative paths.
"""

from __future__ import annotations

import json
from functools import cache
from pathlib import Path
from typing import Any, Literal

from inspect_ai.dataset import MemoryDataset, Sample
from inspect_ai.model import ChatMessageSystem, ChatMessageUser, ContentImage, ContentText

from agentic_evals.paths import EO_GYM_ROOT, EO_GYM_TOOLS_JSON
from agentic_evals.tasks.eo_gym import metrics

PromptVariant = Literal["simple", "complex"]
Toolset = Literal["skill", "all"]

# Verbatim from eo_gym_evaluation/prompts.py (checked by tests/test_eo_gym_dataset.py).
SIMPLE_PROMPT = """
You are an expert remote-sensing analyst. You must use the given tools to inspect
fine image details before answering.

Use tools before answering, and ground the final answer in available observations.

Stop / answer policy:
- Never give the final answer before at least one task-relevant evidence tool has been used.
- Companion-image lookup is only a setup step, not evidence. A failed lookup is not enough reason to stop.
- If one tool returns an empty result or weak evidence, try another relevant tool before abstaining.
- For counting, ratio, comparison, or change questions, do not stop after raw detections alone when another tool or reasoning step is needed.
- Once you have enough grounded evidence to answer the exact question, stop and answer directly.

Write final answers with the prefix `Final Answer:`.
""".strip()

COMPLEX_PROMPT = """
You are an expert remote-sensing analyst running a deliberate tool-use workflow.
Use the tools to collect evidence from the image or related EO data before you
answer.

Workflow policy:
- Restate the visual or geospatial evidence you need.
- Choose the narrowest useful tool call first.
- Cross-check weak, empty, or ambiguous observations with another relevant tool.
- Use calculator or relationship tools when the question asks for counts, ratios, spatial relations, or comparisons.
- Do not call extra tools after the answer is already supported.
- If the evidence remains insufficient after relevant attempts, say what is missing instead of guessing.

Final response policy:
- Keep the answer concise.
- Include the option letter when the input is multiple choice.
- Write final answers with the prefix `Final Answer:`.
""".strip()

PROMPTS = {"simple": SIMPLE_PROMPT, "complex": COMPLEX_PROMPT}


def driver_question(record: dict[str, Any]) -> str:
    """``EvalRow.question``: user-message text with ``<image>`` removed."""
    parts = []
    for message in record.get("messages") or []:
        if str(message.get("role", "")).lower() != "user":
            continue
        content = message.get("content")
        if isinstance(content, str):
            text = content.replace("<image>", "").strip()
            if text:
                parts.append(text)
    return "\n".join(parts).strip()


def build_user_text(question: str, image_refs: list[str]) -> str:
    parts = [f"Question: {question}"] if question else []
    if image_refs:
        parts.append("Available image references:\n" + "\n".join(f"- Image {i}: {r}" for i, r in enumerate(image_refs, 1)))
    return "\n\n".join(parts).strip()


def allows_missing_images(data_source: str, question: str) -> bool:
    lowered = data_source.lower()
    return "fair1m" in lowered or "multispectral" in lowered or "multispectral dataset" in question.lower()


@cache
def tool_catalog(path: Path = EO_GYM_TOOLS_JSON) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(
            f"{path} missing; run `uv run python scripts/setup_benchmark.py eo-gym` to export EO-Gym's tool schemas"
        )
    return json.loads(path.read_text())


def tools_for(data_source: str, toolset: Toolset) -> list[str]:
    catalog = tool_catalog()
    if toolset == "all":
        return list(catalog["all_tools"])
    return list(catalog["skill_tools"].get(data_source, catalog["all_tools"]))


def build_sample(
    record: dict[str, Any],
    prompt: PromptVariant,
    toolset: Toolset,
    root: Path = EO_GYM_ROOT,
    include_images: bool = True,
) -> Sample:
    question = driver_question(record)
    images = list(record.get("images") or [])
    if not images and not allows_missing_images(record["data_source"], question):
        raise ValueError(f"EO-Gym record {record['id']} has no image references")
    content: list[ContentText | ContentImage] = [ContentText(text=build_user_text(question, images))]
    for idx, ref in enumerate(images if include_images else [], 1):
        descriptor = f"Image {idx}"
        if len(images) == 2:
            descriptor = "Pre-disaster image" if idx == 1 else "Post-disaster image"
        content.append(ContentText(text=f"{descriptor}:"))
        content.append(ContentImage(image=str(root / ref)))
    gold = metrics.gold_steps_from_messages(record["messages"])
    return Sample(
        id=record["id"],
        input=[ChatMessageSystem(content=PROMPTS[prompt]), ChatMessageUser(content=content)],
        target=str(record.get("gt_answer") or ""),
        metadata={
            "data_source": record["data_source"],
            "question_type": record.get("question_type"),
            "eo_task": record.get("EO_task"),
            "judge_question": metrics.judge_question_from_messages(record["messages"]),
            "images": images,
            "tools": tools_for(record["data_source"], toolset),
            "gold_steps": [{"name": s.name, "arguments": s.arguments, "observation": s.observation} for s in gold],
            "reference_trajectory": [{"name": s.name, "arguments": s.arguments, "output": s.observation} for s in gold],
        },
    )


def load_records(root: Path = EO_GYM_ROOT) -> list[dict[str, Any]]:
    path = root / "datasets" / "eo_gym_test_set.jsonl"
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def eo_gym_dataset(
    prompt: PromptVariant = "simple",
    toolset: Toolset = "skill",
    eo_task: str | None = None,
    include_images: bool = True,
) -> MemoryDataset:
    """The test split. ``include_images=False`` builds text-only inputs (for not-applicable runs)."""
    if include_images and not (EO_GYM_ROOT / "EO_GYM_DATA").is_dir():
        raise FileNotFoundError(
            f"{EO_GYM_ROOT / 'EO_GYM_DATA'} not found. Install EO-Gym with "
            "`uv run python scripts/setup_benchmark.py eo-gym`, or point EO_GYM_ROOT at a pre-staged copy."
        )
    records = load_records()
    if eo_task:
        records = [r for r in records if r.get("EO_task") == eo_task]
    return MemoryDataset(
        [build_sample(r, prompt, toolset, include_images=include_images) for r in records],
        name=f"eo_gym_{prompt}_{toolset}",
    )
