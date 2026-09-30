"""Earth-Bench dual-level metrics, as pure functions.

Reference: Feng et al., *Earth-Agent: Unlocking the Full Landscape of Earth
Observation with Agents*, ICLR 2026, arXiv:2509.23141, Appendix B.2 (metric
definitions); reference implementation in upstream ``evaluate/step_by_step.py``
and ``evaluate/end_to_end.py`` at the revision pinned in ``upstream.lock.toml``.

The functions follow the reference implementation rather than our reading of
the paper, and are tested against the 26 published
``evaluate_langchain/<model>_<AP|IF>/`` result files. Where upstream behaviour
is a quirk rather than a definition, it is reproduced only when
``upstream_compat=True``:

- TEM returns 0 for an empty gold trajectory while TAO, TIO and Parameter
  Accuracy return 1.0. No task in the pinned ground truth has an empty gold
  trajectory, so this only matters for synthetic inputs; outside compat mode
  TEM is 1.0 there like the other three.
- The published evaluator scores raw calls against gold steps that pass lists
  into parameters the tool source annotates as ``str`` (opendatalab/Earth-Agent
  issue 13). ``expand_scalar_gold`` turns those steps into one scalar call per
  index before our own runs are scored. ``upstream_compat`` leaves the gold
  as published so the score files still match.

A call is a mapping with at least ``name`` and ``input`` (tool arguments), the
format of ``extracted_tool_calls_GT.json``.
"""

from __future__ import annotations

import math
import re
from collections.abc import Collection, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

Call = Mapping[str, Any]


def tool_names(calls: Sequence[Call]) -> list[str]:
    return [call.get("name", "") for call in calls]


def tool_any_order(pred: Sequence[Call], gold: Sequence[Call]) -> float:
    """TAO: |Set(t*) ∩ Set(t)| / |Set(t*)|."""
    expected = set(tool_names(gold))
    if not expected:
        return 1.0
    return len(expected & set(tool_names(pred))) / len(expected)


def tool_in_order(pred: Sequence[Call], gold: Sequence[Call]) -> float:
    """TIO: longest prefix of t* matched as a subsequence of t, divided by m."""
    expected = tool_names(gold)
    if not expected:
        return 1.0
    actual = iter(tool_names(pred))
    matched = 0
    for tool in expected:
        if any(candidate == tool for candidate in actual):
            matched += 1
        else:
            break
    return matched / len(expected)


def tool_exact_match(pred: Sequence[Call], gold: Sequence[Call], *, upstream_compat: bool = False) -> float:
    """TEM: longest common prefix of t and t*, divided by m."""
    expected = tool_names(gold)
    if not expected:
        return 0.0 if upstream_compat else 1.0
    actual = tool_names(pred)
    matched = 0
    for exp, act in zip(expected, actual):
        if exp != act:
            break
        matched += 1
    return matched / len(expected)


def parameter_steps(pred: Sequence[Call], gold: Sequence[Call]) -> list[dict[str, Any]]:
    """Per-step comparison of tool name and arguments, stopping at the first mismatch."""
    steps: list[dict[str, Any]] = []
    for i, (p, g) in enumerate(zip(pred, gold)):
        name_match = p.get("name") == g.get("name")
        input_match = p.get("input") == g.get("input")
        steps.append(
            {
                "step": i + 1,
                "expected_tool": g.get("name", ""),
                "actual_tool": p.get("name", ""),
                "name_match": name_match,
                "input_match": input_match,
                "correct": name_match and input_match,
            }
        )
        if not (name_match and input_match):
            break
    return steps


def parameter_accuracy(pred: Sequence[Call], gold: Sequence[Call]) -> float:
    """Longest prefix where both tool name and arguments match exactly, divided by m."""
    if not gold:
        return 1.0
    return sum(step["correct"] for step in parameter_steps(pred, gold)) / len(gold)


def efficiency(pred: Sequence[Call], gold: Sequence[Call]) -> float:
    """|τ| / |τ*|. A ratio, not a score: values above 1 mean more calls than the expert."""
    if not gold:
        return 1.0 if not pred else math.inf
    return len(pred) / len(gold)


def first_divergence_step(pred: Sequence[Call], gold: Sequence[Call]) -> int | None:
    """1-based index of the first step whose tool name differs from the gold trajectory."""
    for i, (p, g) in enumerate(zip(tool_names(pred), tool_names(gold))):
        if p != g:
            return i + 1
    if len(pred) != len(gold):
        return min(len(pred), len(gold)) + 1
    return None


_ANSWER_OPEN_OPEN = re.compile(r"<Answer>([A-F])<Answer>", re.IGNORECASE)
_ANSWER_OPEN_CLOSE = re.compile(r"<Answer>([A-F])</Answer>", re.IGNORECASE)
_BARE_LETTER = re.compile(r"\b([A-F])\b")


def extract_answer(text: str | None) -> str:
    """Upstream answer extraction (``end_to_end.extract_answer_from_text``).

    Returns a letter A-F, ``"FAIL"`` if the text contains "FAIL" anywhere, or
    ``"UNKNOWN"`` when nothing matches. Free-form numeric answers (tasks 163 and
    173) cannot be extracted by this rule; upstream scores them only through it,
    so they are never correct under the published protocol.
    """
    if not text:
        return "FAIL"
    if "FAIL" in text:
        return "FAIL"
    for pattern in (_ANSWER_OPEN_OPEN, _ANSWER_OPEN_CLOSE):
        match = pattern.search(text)
        if match:
            return match.group(1).upper()
    letters = _BARE_LETTER.findall(text)
    if letters:
        return letters[-1].upper()
    return "UNKNOWN"


def answer_status(pred_text: str | None, gold_answer: str | None) -> tuple[str, str, bool]:
    """Return ``(extracted, status, correct)`` with upstream's status labels."""
    gold = (gold_answer or "").strip()
    extracted = extract_answer(pred_text)
    if extracted == "FAIL":
        return extracted, "fail", False
    if extracted == "UNKNOWN":
        return extracted, "unknown", False
    if extracted == gold:
        return extracted, "correct", True
    return extracted, "incorrect", False


@dataclass
class TrajectoryScores:
    tao: float
    tio: float
    tem: float
    param_acc: float
    efficiency: float
    first_divergence_step: int | None
    steps: list[dict[str, Any]] = field(default_factory=list)

    def values(self) -> dict[str, float]:
        return {
            "tao": self.tao,
            "tio": self.tio,
            "tem": self.tem,
            "param_acc": self.param_acc,
            "efficiency": self.efficiency,
        }


def score_trajectory(pred: Sequence[Call], gold: Sequence[Call], *, upstream_compat: bool = False) -> TrajectoryScores:
    return TrajectoryScores(
        tao=tool_any_order(pred, gold),
        tio=tool_in_order(pred, gold),
        tem=tool_exact_match(pred, gold, upstream_compat=upstream_compat),
        param_acc=parameter_accuracy(pred, gold),
        efficiency=efficiency(pred, gold),
        first_divergence_step=first_divergence_step(pred, gold),
        steps=parameter_steps(pred, gold),
    )


def expand_scalar_gold(
    calls: Sequence[Call],
    scalar_params: Mapping[str, Collection[str]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Expand gold steps whose ``str`` parameters are parallel lists.

    The published MCP tools type those parameters as a single string
    (``agent/tools/*.py``). A gold step that passes a list there is not a call
    the tool accepts. When every such list on a step has the same positive
    length, this returns one call per index; arguments that are not expanded
    are copied onto each call. The predicted trace is not an input: only the
    gold sequence changes.

    A step is left as one call, and recorded in the second list, when those
    lists differ in length or are empty. Pairing them by position would join
    unrelated files. Parameters annotated ``list``, ``list[str]``, or
    ``str | list[str]`` are absent from ``scalar_params`` and stay one step
    (``calculate_tif_average``, ``calc_batch_image_mean``, and the threshold
    tools that already accept a list).

    ``scalar_params`` maps a tool name to the parameter names annotated exactly
    ``str``. The returned calls do not share their ``input`` dict with ``calls``.
    """
    expanded: list[dict[str, Any]] = []
    unexpanded: list[dict[str, Any]] = []
    for index, call in enumerate(calls):
        name = call.get("name", "")
        arguments = call.get("input")
        list_keys = _scalar_list_keys(arguments, scalar_params.get(name, ()))
        if not isinstance(arguments, dict) or not list_keys:
            expanded.append(_copy_call(call))
            continue
        lengths = {key: len(arguments[key]) for key in list_keys}
        unique = set(lengths.values())
        if len(unique) != 1 or unique == {0}:
            expanded.append(_copy_call(call))
            unexpanded.append({"step": index + 1, "tool": name, "lengths": lengths})
            continue
        count = unique.pop()
        for offset in range(count):
            sliced = dict(arguments)
            for key in list_keys:
                sliced[key] = arguments[key][offset]
            expanded.append({"name": name, "input": sliced})
    return expanded, unexpanded


def _scalar_list_keys(arguments: Any, scalar_names: Collection[str]) -> list[str]:
    if not isinstance(arguments, dict):
        return []
    return [key for key, value in arguments.items() if key in scalar_names and isinstance(value, list)]


def _copy_call(call: Call) -> dict[str, Any]:
    copied = dict(call)
    if isinstance(copied.get("input"), dict):
        copied["input"] = dict(copied["input"])
    return copied


def upstream_scored_questions(gold_question_ids: Sequence[str]) -> list[str]:
    """The question ids upstream's published score files cover.

    Both upstream scripts iterate ``list(gt_dict.items())[188:]`` over the ground
    truth in file order. The ground truth lacks ``question43``, so the slice
    starts at ``question190`` and ``question189`` is never scored.
    """
    return list(gold_question_ids)[188:]
