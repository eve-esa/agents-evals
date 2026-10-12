"""OpenEarthAgent test split as Inspect samples.

Shabbir et al., arXiv:2602.17665. Records are ``test.json`` from the Hugging
Face dataset ``MBZUAI/OpenEarthAgent`` (Apache-2.0) at the revision in
``upstream.lock.toml``: 1,169 tasks, each a ShareGPT ``conversation``. The
gold trajectory and the gold answer live in the assistant turns, which are
JSON text ``{"thought", "actions": [{"name", "arguments"}]}``. The first
assistant turn is thought-only (``actions: []``). The last action is
``Terminate``.

``regime="e2e"`` is one sample per task. The user message is the question.
``regime="step"`` expands each gold assistant turn into its own sample, as
``load_data_function`` does for ``task_type="step"``: the input is the gold
history up to that human turn, and the label is the next assistant turn.
Step mode keeps upstream's text protocol (``RS_AGENT_PROMPT`` plus the regex
parser). End-to-end mode uses native tool calls, so its system prompt is
``prompts/e2e_system.txt`` rather than the JSON-only instruction.

``needs_vision`` is true when the task has an image. With ``attach_images``
those files are image parts on the first user message; a missing file is an
error, because a vision run that cannot see the image is not the protocol.
"""

from __future__ import annotations

import json
from functools import cache
from pathlib import Path
from typing import Any, Literal

from inspect_ai.dataset import MemoryDataset, Sample
from inspect_ai.model import ChatMessage, ChatMessageAssistant, ChatMessageSystem, ChatMessageUser, ContentImage, ContentText

from agentic_evals.paths import OPENEARTHAGENT_IMAGES
from agentic_evals.tasks.openearthagent import metrics

DATA = Path(__file__).resolve().parent / "data"
PROMPTS = Path(__file__).resolve().parent / "prompts"
AGENT_PREFIX = "<AGENT_PROMPT>\n\nQuestion:"

Regime = Literal["step", "e2e"]
Category = Literal["all", "perception", "operation", "logic", "gis"]


def system_prompt() -> str:
    """Verbatim ``RS_AGENT_PROMPT`` (step mode's text protocol)."""
    return (PROMPTS / "rs_agent_prompt.txt").read_text(encoding="utf-8")


def e2e_system_prompt() -> str:
    """Native-tool system prompt. Not upstream's JSON-only instruction."""
    return (PROMPTS / "e2e_system.txt").read_text(encoding="utf-8")


def judge_prompt() -> str:
    """Verbatim ``EVAL_PROMPT``."""
    return (PROMPTS / "answer_judge.txt").read_text(encoding="utf-8")


@cache
def load_records(root: Path = DATA) -> list[dict[str, Any]]:
    path = root / "test.json"
    if not path.is_file():
        raise FileNotFoundError(f"{path} is part of this package and should be here.")
    records = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(records, list):
        raise ValueError(f"{path} should be a list of tasks.")
    return records


def human_text(value: str) -> str:
    """``convert_conv_to_msges`` strips the agent-prompt marker, then strips."""
    return value.replace(AGENT_PREFIX, "").strip()


def question_text(conversation: list[dict[str, Any]]) -> str:
    """Text after ``\\n\\nQuestion:`` on the first turn."""
    raw = str(conversation[0].get("value", "")) if conversation else ""
    parts = raw.split("\n\nQuestion:", 1)
    if len(parts) == 2:
        return parts[1].strip()
    return human_text(raw)


def gold_steps(conversation: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Parsed tool calls, including ``Terminate``, with the following observation."""
    steps: list[dict[str, Any]] = []
    for index, turn in enumerate(conversation):
        if turn.get("from") != "gpt":
            continue
        actions = metrics.extract_action(turn.get("value")) or []
        if not actions:
            continue
        cfg, _error = metrics.extract_tool_cfg(actions[0])
        if cfg is None:
            continue
        observation = None
        if index + 1 < len(conversation) and conversation[index + 1].get("from") == "human":
            observation = conversation[index + 1].get("value")
        steps.append(
            {"name": cfg["API_name"], "arguments": cfg["API_params"], "output": observation}
        )
    return steps


def _image_part(name: str, image_root: Path) -> ContentImage:
    path = image_root / name
    if not path.is_file():
        raise FileNotFoundError(
            f"OpenEarthAgent image {path} is missing. "
            "Run `uv run python scripts/setup_benchmark.py openearthagent`."
        )
    return ContentImage(image=str(path))


def _user_message(text: str, images: list[str], *, attach: bool, image_root: Path, first: bool) -> ChatMessageUser:
    if first and attach and images:
        content: list[ContentText | ContentImage] = [ContentText(text=text)]
        content.extend(_image_part(name, image_root) for name in images)
        return ChatMessageUser(content=content)
    return ChatMessageUser(content=text)


def step_messages(
    conversation: list[dict[str, Any]],
    upto: int,
    images: list[str],
    *,
    attach_images: bool,
    image_root: Path,
) -> list[ChatMessage]:
    """System prompt plus gold history through the human turn at ``upto``."""
    messages: list[ChatMessage] = [ChatMessageSystem(content=system_prompt())]
    first = True
    for turn in conversation[: upto + 1]:
        if turn.get("from") == "human":
            messages.append(
                _user_message(
                    human_text(str(turn.get("value", ""))),
                    images,
                    attach=attach_images,
                    image_root=image_root,
                    first=first,
                )
            )
            first = False
        elif turn.get("from") == "gpt":
            messages.append(ChatMessageAssistant(content=str(turn.get("value", ""))))
    return messages


def _keeps(steps: list[dict[str, Any]], category: Category) -> bool:
    names = [step["name"] for step in steps if step["name"] != "Terminate"]
    if category != "all" and not any(metrics.tool_category(name) == category for name in names):
        return False
    return True


def load_samples(
    regime: Regime = "e2e",
    *,
    task_id: str | None = None,
    category: Category = "all",
    attach_images: bool = False,
    image_root: Path = OPENEARTHAGENT_IMAGES,
    root: Path = DATA,
) -> list[Sample]:
    samples: list[Sample] = []
    for record in load_records(root):
        idx = str(record.get("idx", ""))
        if task_id is not None and idx != task_id:
            continue
        conversation = list(record.get("conversation") or [])
        images = [str(name) for name in (record.get("images") or [])]
        steps = gold_steps(conversation)
        if not _keeps(steps, category):
            continue
        names = [step["name"] for step in steps]
        answer = ""
        if names and names[-1] == "Terminate":
            ans = steps[-1]["arguments"].get("ans", "")
            answer = ans.strip() if isinstance(ans, str) else str(ans)
        generation = metrics.is_generation(names)
        shared: dict[str, Any] = {
            "idx": idx,
            "regime": regime,
            "question": question_text(conversation),
            "is_generation": generation,
            "task_kind": "generation" if generation else "answer",
            "input_kind": "image" if images else "text",
            "needs_vision": bool(images),
            "images": images,
            "reference_answer": answer,
            "gold_names": names,
        }
        if regime == "e2e":
            samples.append(_e2e_sample(idx, shared, steps, images, attach_images, image_root))
        else:
            samples.extend(_step_samples(idx, shared, conversation, images, attach_images, image_root))
    return samples


def _e2e_sample(
    idx: str,
    shared: dict[str, Any],
    steps: list[dict[str, Any]],
    images: list[str],
    attach_images: bool,
    image_root: Path,
) -> Sample:
    trajectory = [step for step in steps if step["name"] != "Terminate"]
    user = _user_message(shared["question"], images, attach=attach_images, image_root=image_root, first=True)
    return Sample(
        id=idx,
        input=[ChatMessageSystem(content=e2e_system_prompt()), user],
        target=shared["reference_answer"],
        metadata={
            **shared,
            "reference_trajectory": trajectory,
        },
    )


def _step_samples(
    idx: str,
    shared: dict[str, Any],
    conversation: list[dict[str, Any]],
    images: list[str],
    attach_images: bool,
    image_root: Path,
) -> list[Sample]:
    samples: list[Sample] = []
    step_index = 0
    for index, turn in enumerate(conversation):
        if turn.get("from") != "human":
            continue
        if index + 1 >= len(conversation) or conversation[index + 1].get("from") != "gpt":
            continue
        step_index += 1
        label = str(conversation[index + 1].get("value", ""))
        actions = metrics.extract_action(label) or []
        cfg = None
        if actions:
            cfg, _error = metrics.extract_tool_cfg(actions[0])
        one: list[dict[str, Any]] = []
        if cfg is not None and cfg["API_name"] != "Terminate":
            one = [{"name": cfg["API_name"], "arguments": cfg["API_params"], "output": None}]
        target = shared["reference_answer"] if cfg is not None and cfg["API_name"] == "Terminate" else ""
        samples.append(
            Sample(
                id=f"{idx}:{step_index}",
                input=step_messages(conversation, index, images, attach_images=attach_images, image_root=image_root),
                target=target,
                metadata={
                    **shared,
                    "step_index": step_index,
                    "gold_text": label,
                    "reference_trajectory": one,
                },
            )
        )
    return samples


def openearthagent_dataset(
    regime: Regime = "e2e",
    *,
    task_id: str | None = None,
    category: Category = "all",
    attach_images: bool = False,
) -> MemoryDataset:
    return MemoryDataset(
        load_samples(regime, task_id=task_id, category=category, attach_images=attach_images),
        name="openearthagent",
    )
