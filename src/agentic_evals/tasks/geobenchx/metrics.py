"""GeoBenchX scoring, as pure functions.

Reference: Krechetova & Kochedykov, *GeoBenchX: Benchmarking LLMs in Agent
Solving Multistep Geospatial Tasks*, SIGSPATIAL 2025, arXiv:2503.18129,
Section 5. Reference implementation: ``geobenchx/evaluation.py``
(``score_task_solution``, ``generate_eval_stats``) and ``geobenchx/utils.py``
(``get_solution_code``) at the revision pinned in ``upstream.lock.toml``.

Their protocol scores a whole solution. The candidate and every reference are
rendered as Python-like calls and an LLM judge returns one match score
(0 no match, 1 partial, 2 match). ``length_diff`` is
``(n_pred - mean_n_ref) / mean_n_ref``, or ``n_pred`` when the mean reference
length is 0 (``generate_eval_stats``). The paper's sentence that this ratio is
relative to the candidate length does not match that code; the code is what
their tables were computed with, so this follows the code. The ratio is signed
and is not Earth-Bench efficiency.

``step_scores`` is our addition, not a GeoBenchX metric. The formulas are
Earth-Bench's TAO / TIO / TEM / parameter accuracy (Feng et al., ICLR 2026,
arXiv:2509.23141, Appendix B.2), taking the max over reference solutions
because a task can have several. Two deliberate differences from those
formulas, both because of GeoBenchX's own rules (Section 5):

- An empty reference (a control question) scores 1 only when the candidate is
  also empty. Earth-Bench's empty-gold case returns 1 regardless of the
  prediction.
- A ``reject_task`` reference has length 1, so a candidate that calls other
  tools and then rejects does not get a perfect prefix score. Their judge
  instructions still call that a full match. That disagreement is why these
  numbers stay out of the official scorer.
"""

from __future__ import annotations

import json
import re
from typing import Any, Mapping, Sequence

_REASONING = re.compile(r"<MATCH REASONING>(.*?)</MATCH REASONING>", re.DOTALL)
_SCORE = re.compile(r"<MATCH SCORE>(.*?)</MATCH SCORE>", re.DOTALL)

Step = Mapping[str, Any]


def solution_code(steps: Sequence[Step], *, add_num: bool = False) -> str:
    """``get_solution_code(..., add_num=False)`` as the judge receives it.

    Every argument value is quoted, including numbers and dicts. A comment is
    appended only when it is non-empty. The trailing space before the newline
    is what their ``print(..., end=" ")`` writes.
    """
    lines: list[str] = []
    for i, step in enumerate(steps):
        name = step.get("function_name", step.get("name", ""))
        arguments = step.get("arguments", step.get("input", {})) or {}
        if not isinstance(arguments, dict):
            arguments = {}
        args_str = ", ".join(f"{k}='{v}'" for k, v in arguments.items())
        line = f"{i + 1}. " if add_num else ""
        line += f"{name}({args_str}) "
        comment = step.get("comment")
        if comment:
            line += f"#{comment} "
        line += "\n"
        lines.append(line)
    return "".join(lines)


def reference_solutions_text(description: str | None, solutions: Sequence[Sequence[Step]]) -> str:
    """The ``<REFERENCE SOLUTIONS>`` body ``score_task_solution`` builds."""
    text = ""
    if description is not None:
        text = f"Here are general comments about the reference solutions: {description}"
    for solution in solutions:
        text += f" \n{solution_code(solution)}\n "
    return text


def parse_verdict(text: str) -> tuple[int | None, str]:
    """``(score, reasoning)`` from ``<MATCH SCORE>`` / ``<MATCH REASONING>``.

    Returns ``(None, explanation)`` when the score tag is missing or not 0, 1,
    or 2. Their ``score_task_solution`` would raise; a scorer records that as
    an unscored sample instead.
    """
    score_match = _SCORE.search(text or "")
    if score_match is None:
        return None, "judge output has no <MATCH SCORE> tag"
    try:
        score = int(score_match.group(1).strip())
    except ValueError:
        return None, f"judge <MATCH SCORE> is not an integer: {score_match.group(1).strip()!r}"
    if score not in (0, 1, 2):
        return None, f"judge <MATCH SCORE> {score} is outside 0, 1, 2"
    reasoning_match = _REASONING.search(text)
    reasoning = reasoning_match.group(1).strip() if reasoning_match else ""
    return score, reasoning


def length_diff(n_pred: int, references: Sequence[Sequence[Step]]) -> float:
    """``generate_eval_stats`` relative length difference for one task.

    ``references`` empty is not a case their loop scores (it skips a task with
    no reference solutions). Treated here as a mean reference length of 0, so
    the difference is ``n_pred``, the same branch as an empty reference.
    """
    if not references:
        return float(n_pred)
    mean_ref = sum(len(reference) for reference in references) / len(references)
    if mean_ref > 0:
        return (n_pred - mean_ref) / mean_ref
    return float(n_pred)


def _names(steps: Sequence[Step]) -> list[str]:
    return [str(step.get("name", step.get("function_name", ""))) for step in steps]


def _arguments(step: Step) -> Any:
    if "arguments" in step:
        return step.get("arguments")
    return step.get("input")


def _canon(value: Any) -> str:
    return json.dumps(value, sort_keys=True, ensure_ascii=False, default=str)


def _against(pred: Sequence[Step], gold: Sequence[Step]) -> dict[str, float]:
    """One reference. Empty gold scores 1 only when the candidate is empty."""
    if not gold:
        score = 1.0 if not pred else 0.0
        return {"tao": score, "tio": score, "tem": score, "param_acc": score}
    pred_names = _names(pred)
    gold_names = _names(gold)
    tao = len(set(gold_names) & set(pred_names)) / len(set(gold_names))
    actual = iter(pred_names)
    tio_matched = 0
    for tool in gold_names:
        if any(candidate == tool for candidate in actual):
            tio_matched += 1
        else:
            break
    tem_matched = 0
    for expected, got in zip(gold_names, pred_names):
        if expected != got:
            break
        tem_matched += 1
    param_matched = 0
    for got, expected in zip(pred, gold):
        same_name = _names([got])[0] == _names([expected])[0]
        same_args = _canon(_arguments(got)) == _canon(_arguments(expected))
        if not (same_name and same_args):
            break
        param_matched += 1
    m = len(gold)
    return {
        "tao": tao,
        "tio": tio_matched / m,
        "tem": tem_matched / m,
        "param_acc": param_matched / m,
    }


def step_scores(pred: Sequence[Step], references: Sequence[Sequence[Step]]) -> dict[str, float]:
    """Max of TAO / TIO / TEM / parameter accuracy over the reference solutions.

    Each metric is maximised on its own, so the reported TAO and TEM can come
    from different references. A task with no reference is scored as one empty
    reference.
    """
    refs = list(references) or [[]]
    per_ref = [_against(pred, gold) for gold in refs]
    return {key: max(scores[key] for scores in per_ref) for key in ("tao", "tio", "tem", "param_acc")}
