"""Earth-Bench tasks as Inspect samples.

Each sample's input is the exact user message upstream sent to the model:
upstream's system prompt, then ``"\\n\\nQuestion: "``, then the regime's question,
the data path appended with no separator, and the lettered choices
(``scripts/langchain_<model>*.py``: ``load_questions`` / ``handle_question`` at the
revision pinned in ``upstream.lock.toml``). Upstream sends this as a single user
message, not as a system message, and so do we. ``tests/test_earth_bench_dataset.py``
checks the result byte for byte against the ``query`` field of all published
predictions. Known differences, all pinned in that test: for 16 tasks the
published runs used a ``benchmark/supported_data/`` path that no longer exists
upstream (we use the ``benchmark/data/`` paths in the pinned ``question.json``),
and 12 of those (Products, 110-184) also had their question text revised after
the published runs, so published scores on them are not comparable with ours.

Answer keys. The pinned upstream has two, and they disagree on 28 tasks:
``question.json`` (``evaluation[*].gt_answer.whitelist``, which usually also matches
the expert dialog's final turn) and ``extracted_tool_calls_GT.json``
(``final_answer``), which is what upstream's ``evaluate/end_to_end.py`` scores
against and therefore what the published accuracies use. ``answer_key`` picks the
sample target; both answers are kept in metadata so scorers can report accuracy
under each and the disagreement stays visible. Where the GT file has no answer
(``question43`` absent, ``question164`` null) the whitelist is used and the sample
is flagged.
"""

from __future__ import annotations

import json
from functools import cache
from pathlib import Path
from typing import Any, Literal

from inspect_ai.dataset import MemoryDataset, Sample

ASSETS = Path(__file__).resolve().parent / "assets"

Regime = Literal["ap", "if"]
Modality = Literal["spectrum", "products", "rgb", "all"]
AnswerKey = Literal["gt_file", "question_json"]

MODALITY_RANGES: dict[str, range] = {
    "spectrum": range(1, 101),
    "products": range(101, 189),
    "rgb": range(189, 249),
    "all": range(1, 249),
}

# Verbatim from upstream's runner scripts (identical in every published run, both regimes).
SYSTEM_PROMPT = (
    "\nYou are a geoscientist, and you need to use tools to answer multiple-choice questions about Earth "
    "observation data analysis. Note that if a tool returns an error, you can only try again once. Ultimately, "
    "you only need to explicitly tell me the correct choice.\n"
    "ATTENTION:\n"
    '1. When a tool returns "Result saved at /path/to/file", you must use the full returned path '
    '"/path/to/file" in all subsequent tool calls.\n'
    "2. For each question, you must provide the choice you think is most appropriate.Don't gibe me another "
    "format. Your final answer format must be:\n"
    "<Answer>Your choice<Answer>\n"
)


def modality_of(question_number: int) -> str:
    for name in ("spectrum", "products", "rgb"):
        if question_number in MODALITY_RANGES[name]:
            return name
    raise ValueError(f"question{question_number} is outside Earth-Bench's 1-248 range")


def _regime_entry(evaluation: list[dict[str, Any]], regime: Regime) -> dict[str, Any]:
    wanted = "autonomous planning" if regime == "ap" else "instruction following"
    for entry in evaluation:
        if entry["type"].strip().lower() == wanted:
            return entry
    raise ValueError(f"no {wanted!r} entry in {evaluation!r}")


def build_query(record: dict[str, Any], regime: Regime) -> str | None:
    """The question text upstream builds before prepending the system prompt."""
    evaluation = record["evaluation"]
    ap = _regime_entry(evaluation, "ap")
    other = _regime_entry(evaluation, "if")
    data = ap.get("data") or other.get("data")
    if data is None:
        return None
    question = (ap if regime == "ap" else other)["question"]
    query = question + data
    choices = record.get("choices")
    if choices:
        query += "\n".join([""] + [f"{chr(ord('A') + i)}.{choice}" for i, choice in enumerate(choices)])
    return query


def build_prompt(record: dict[str, Any], regime: Regime) -> str | None:
    query = build_query(record, regime)
    return None if query is None else f"{SYSTEM_PROMPT}\n\nQuestion: {query}"


@cache
def _load(root: Path) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    questions = json.loads((root / "question.json").read_text(encoding="utf-8"))
    gold = json.loads((root / "extracted_tool_calls_GT.json").read_text(encoding="utf-8"))
    return questions, {g["question_index"]: g for g in gold}


def _whitelist(entry: dict[str, Any]) -> str:
    value = entry["gt_answer"]["whitelist"]
    if isinstance(value, list):
        if len(value) != 1:
            raise ValueError(f"multi-answer whitelist not supported: {value!r}")
        value = value[0]
    return str(value)


def load_samples(
    regime: Regime,
    modality: Modality = "all",
    *,
    answer_key: AnswerKey = "gt_file",
    root: Path = ASSETS,
) -> list[Sample]:
    questions, gold = _load(root)
    samples: list[Sample] = []
    for key, record in questions.items():
        number = int(key)
        if number not in MODALITY_RANGES[modality]:
            continue
        prompt = build_prompt(record, regime)
        if prompt is None:
            continue
        whitelist = _whitelist(_regime_entry(record["evaluation"], regime))
        gold_entry = gold.get(f"question{number}")
        gt_answer = gold_entry.get("final_answer") if gold_entry else None
        fallback = answer_key == "gt_file" and gt_answer is None
        target = whitelist if answer_key == "question_json" or fallback else gt_answer
        samples.append(
            Sample(
                id=number,
                input=prompt,
                target=target,
                metadata={
                    "question": f"question{number}",
                    "regime": regime,
                    "modality": modality_of(number),
                    "choices": record.get("choices"),
                    "data_dir": f"benchmark/data/question{number}",
                    "answer_key": answer_key,
                    "answer_key_fallback": fallback,
                    "answer_gt_file": gt_answer,
                    "answer_question_json": whitelist,
                    "has_gold_trajectory": gold_entry is not None,
                    "gold_tool_calls": (
                        [{"name": c["name"], "input": c["input"]} for c in gold_entry["tool_calls"]]
                        if gold_entry
                        else []
                    ),
                    "reference_trajectory": (
                        [
                            {"name": c["name"], "arguments": c["input"], "output": str(c.get("output"))[:2000]}
                            for c in gold_entry["tool_calls"]
                        ]
                        if gold_entry
                        else None
                    ),
                },
            )
        )
    return samples


def earth_bench_dataset(regime: Regime, modality: Modality = "all", answer_key: AnswerKey = "gt_file") -> MemoryDataset:
    return MemoryDataset(
        load_samples(regime, modality, answer_key=answer_key),
        name=f"earth_bench_{regime}_{modality}",
    )
