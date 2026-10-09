"""OpenEarthAgent scoring, as pure functions.

Reference: Shabbir et al., *OpenEarthAgent: A Unified Framework for
Tool-Augmented Geospatial Agents*, arXiv:2602.17665, Tables 3 and 4 and the
appendix judge prompt (Fig. A8). Reference implementation:
``tool_server/tf_eval/tasks/rsagent/task.py`` (``extract_action``,
``extract_tool_cfg``, ``compare_tool_sequences``, ``gettype``,
``extract_score``, ``evaluate_function``) at the revision pinned in
``upstream.lock.toml``.

Two protocols, reported separately:

- **step** (Table 3). Each gold turn is one sample. ``inst_acc`` is a parsed
  action, ``tool_acc`` the right tool, ``arg_acc1`` the same argument names,
  ``arg_acc2`` the same argument values. ``answer_acc`` is the judge score on
  ``Terminate.ans``, and only on turns where the model also called
  ``Terminate``, and only for tasks whose last tool is not a generation tool.
  Thought-only turns (empty ``actions``) are exempt, which is how the first
  planning turn is left out of the denominator.
- **e2e** (Table 4). ``compare_tool_sequences`` on the gold names with the
  final ``Terminate`` removed: any-order is multiset inclusion, same-order is
  that plus the gold sequence as a subsequence, unique is set inclusion.
  Table 4's caption says "exact sequence match". Section 4.1 says same-order
  is the correct sequence and the right frequency. The released function's
  docstring allows extra predicted calls, and this follows that function.
  Category F1 pools tool counts across tasks (precision and recall add
  ``1e-5`` to the denominator, and the harmonic mean does too). Generation
  tasks score 1 when the last predicted tool equals the gold tool before
  ``Terminate`` and that call succeeded.

The released code multiplies F1 and both answer accuracies by 100, and leaves
the tool-order ratios as fractions, while Table 4 prints those ratios as
percentages too. ``summarize_step`` and ``summarize_e2e`` return the code's
numbers. Scorers in this package log the fractions in ``[0, 1]``; multiply by
100 to read the tables.

Deliberate guards, both documented because the released code raises:

- e2e reads ``API_names_gt[-2]`` and throws when the gold has fewer than two
  calls. A short gold is not a generation task.
- ``last_tool_in_gen`` reads ``tool_names[-1]`` and throws when every call is
  ``Terminate``. That task is not a generation task.
"""

from __future__ import annotations

import json
import re
from collections import Counter
from typing import Any, Mapping, Sequence

# TOOLS_LIST, gettype, and the generation-tool set in evaluate_function.
TOOLS: tuple[str, ...] = (
    "Calculator",
    "OCR",
    "DrawBox",
    "AddText",
    "GoogleSearch",
    "Plot",
    "Solver",
    "TextToBbox",
    "ImageDescription",
    "RegionAttributeDescription",
    "CountGivenObject",
    "ChangeDetection",
    "SegmentObjectPixels",
    "ObjectDetection",
    "GetAreaBoundary",
    "AddPoisLayer",
    "ComputeDistance",
    "DisplayOnMap",
    "AddIndexLayer",
    "ComputeIndexChange",
    "ShowIndexLayer",
    "GetBboxFromGeotiff",
    "DisplayOnGeotiff",
    "Terminate",
)

GEN_TOOLS: frozenset[str] = frozenset(
    {"DrawBox", "AddText", "Plot", "DisplayOnMap", "ShowIndexLayer", "DisplayOnGeotiff"}
)

# CacheManager.get_response. These are the calls a gpkg cache can replay.
CACHE_TOOLS: frozenset[str] = frozenset(
    {"GetAreaBoundary", "AddPoisLayer", "ComputeDistance", "AddIndexLayer", "ComputeIndexChange"}
)

IMAGE_REQUIRED: frozenset[str] = frozenset(
    {
        "OCR",
        "DrawBox",
        "AddText",
        "TextToBbox",
        "CountGivenObject",
        "ImageDescription",
        "RegionAttributeDescription",
        "ChangeDetection",
        "SegmentObjectPixels",
        "ObjectDetection",
        "GetBboxFromGeotiff",
        "DisplayOnGeotiff",
    }
)

GPKG_REQUIRED: frozenset[str] = frozenset(
    {
        "AddPoisLayer",
        "ComputeDistance",
        "DisplayOnMap",
        "AddIndexLayer",
        "ComputeIndexChange",
        "ShowIndexLayer",
        "DisplayOnGeotiff",
    }
)

F1_CATEGORIES: tuple[str, ...] = ("perception", "operation", "logic", "gis")

# base_inferencer.extract_actions. Nested one level less than a general parser;
# gold arguments are objects, not arrays, so this is the pattern that matches.
_ACTIONS = re.compile(
    r'"actions"\s*:\s*(\[(?:[^\[\]]|\[(?:[^\[\]]|\[(?:[^\[\]]|\[[^\[\]]*\])*\])*\])*\])'
)
_SCORE_NUMBER = re.compile(r"[-+]?\d*\.\d+|\d+")


def tool_category(name: str) -> str:
    """``gettype``: perception, operation, logic, gis, terminate, or none."""
    perception = {
        "OCR",
        "ImageDescription",
        "RegionAttributeDescription",
        "TextToBbox",
        "ChangeDetection",
        "ObjectDetection",
        "SegmentObjectPixels",
        "CountGivenObject",
    }
    operation = {"DrawBox", "AddText", "GoogleSearch"}
    logic = {"Calculator", "Solver", "Plot"}
    gis = {
        "GetAreaBoundary",
        "AddPoisLayer",
        "ComputeDistance",
        "DisplayOnMap",
        "AddIndexLayer",
        "ComputeIndexChange",
        "ShowIndexLayer",
        "GetBboxFromGeotiff",
        "DisplayOnGeotiff",
    }
    if name in perception:
        return "perception"
    if name in operation:
        return "operation"
    if name in logic:
        return "logic"
    if name in gis:
        return "gis"
    if name == "Terminate":
        return "terminate"
    return "none"


def extract_action(text: str | None) -> list[Any] | None:
    """The ``actions`` array, or ``None`` when it is missing or not JSON.

    An empty array is returned as ``[]`` (the thought-only planning turn).
    """
    if not text:
        return None
    try:
        match = _ACTIONS.search(text)
        if not match:
            return None
        parsed = json.loads(match.group(1))
    except (json.JSONDecodeError, TypeError):
        return None
    if not isinstance(parsed, list):
        return None
    return parsed


def extract_tool_cfg(action: Any) -> tuple[dict[str, Any] | None, str]:
    """``(tool cfg, error)``. ``error`` is empty when the cfg is usable.

    The cfg is ``{"API_name", "API_params"}``, which is what
    ``extract_tool_cfg`` returns.
    """
    if not isinstance(action, dict):
        return None, "action_is_not_valid_dict"
    if "name" not in action:
        return None, "missing_name_key"
    if "arguments" not in action:
        return None, "missing_arguments_key"
    if not isinstance(action["arguments"], dict):
        return None, "arguments_is_not_valid_dict"
    return {"API_name": action["name"], "API_params": action["arguments"]}, ""


def extract_final_answer(final_response: str | None) -> Any:
    """``Terminate`` argument ``ans``, or ``""`` when that action is absent."""
    actions = extract_action(final_response)
    if not actions:
        return ""
    action = actions[0]
    if (
        isinstance(action, dict)
        and action.get("name") == "Terminate"
        and isinstance(action.get("arguments"), dict)
        and "ans" in action["arguments"]
    ):
        ans = action["arguments"]["ans"]
        return ans.strip() if isinstance(ans, str) else ans
    return ""


def extract_score(eval_text: str | None) -> float:
    """First number before ``Justification``. Missing or unparsable is 0.

    This is ``extract_score``. A judge that writes the score after the
    justification word is read as 0, because the split happens first.
    """
    before = (eval_text or "").split("Justification")[0]
    match = _SCORE_NUMBER.search(before)
    if match is None:
        return 0.0
    try:
        return float(match.group())
    except ValueError:
        return 0.0


def compare_tool_sequences(gold: Sequence[str], pred: Sequence[str]) -> tuple[bool, bool, bool]:
    """``(same_order, any_order, unique)`` from ``compare_tool_sequences``.

    ``any_order`` is multiset inclusion (predicted counts cover the gold).
    ``same_order`` is that, and the gold sequence is a subsequence of the
    prediction. ``unique`` ignores repeats and does not require ``any_order``.
    """
    gold_counts, pred_counts = Counter(gold), Counter(pred)
    any_order = all(pred_counts[name] >= count for name, count in gold_counts.items())
    same_order = False
    if any_order:
        index = 0
        for name in pred:
            if index < len(gold) and name == gold[index]:
                index += 1
        same_order = index == len(gold)

    def unique(seq: Sequence[str]) -> list[str]:
        seen: set[str] = set()
        return [name for name in seq if not (name in seen or seen.add(name))]

    gold_unique, pred_unique = unique(gold), unique(pred)
    unique_ok = all(name in pred_unique for name in gold_unique)
    return same_order, any_order, unique_ok


def is_generation(names: Sequence[str]) -> bool:
    """Whether e2e treats the task as image-generation.

    The released check is ``API_names_gt[-2] in gen_tools``. Fewer than two
    calls is not a generation task; the released code raises ``IndexError``
    there.
    """
    return len(names) >= 2 and names[-2] in GEN_TOOLS


def category_f1(gold_rows: Sequence[Sequence[str]], pred_rows: Sequence[Sequence[str]]) -> dict[str, float]:
    """Pooled category F1 as a fraction in ``[0, 1]``.

    The released code multiplies the result by 100. Empty categories stay near
    0 because of the ``1e-5`` smoothing, rather than being undefined.
    """
    if len(gold_rows) != len(pred_rows):
        raise ValueError("gold_rows and pred_rows must have the same length")
    total: Counter[str] = Counter()
    predict: Counter[str] = Counter()
    correct: Counter[str] = Counter()
    for gold, pred in zip(gold_rows, pred_rows):
        remaining = list(pred)
        for name in pred:
            category = tool_category(name)
            if category != "none":
                predict[category] += 1
        for name in gold:
            category = tool_category(name)
            total[category] += 1
            if name in remaining:
                remaining.remove(name)
                correct[category] += 1
    scores: dict[str, float] = {}
    for category in F1_CATEGORIES:
        precision = correct[category] / (predict[category] + 1e-5)
        recall = correct[category] / (total[category] + 1e-5)
        scores[category] = 2 * precision * recall / (precision + recall + 1e-5)
    return scores


def step_sample_score(gold_text: str | None, pred_text: str | None, *, is_generation_task: bool) -> dict[str, Any]:
    """One step-mode turn, before the dataset-level normalisation.

    ``answer`` is ``None`` when this turn is not in the answer denominator
    (not a correctly selected ``Terminate`` on a non-generation task, or the
    turn is exempt). Otherwise it is ``(gold_ans, pred_ans)`` for the judge.
    Thought-only gold (no parsed tool) is exempt, matching ``if tool_cfg_gold``.
    """
    gold_actions = extract_action(gold_text) or []
    gold_cfg: dict[str, Any] | None = None
    if gold_actions:
        gold_cfg, _gold_error = extract_tool_cfg(gold_actions[0])
    if gold_cfg is None:
        return {
            "exempt": True,
            "inst": 0,
            "tool": 0,
            "argn": 0,
            "argv": 0,
            "answer": None,
            "errors": [],
        }

    errors: list[str] = []
    pred_actions = extract_action(pred_text)
    pred_cfg: dict[str, Any] | None = None
    if pred_actions:
        pred_cfg, error = extract_tool_cfg(pred_actions[0])
        if error:
            errors.append(error)
            pred_cfg = None
    else:
        errors.append("no_valid_action")

    inst = tool = argn = argv = 0
    answer: tuple[Any, Any] | None = None
    if pred_cfg is not None:
        inst = 1
        if pred_cfg["API_name"] not in TOOLS:
            errors.append("name_not_in_tools")
        if pred_cfg["API_name"] == gold_cfg["API_name"]:
            tool = 1
            if gold_cfg["API_name"] == "Terminate" and not is_generation_task:
                answer = (gold_cfg["API_params"].get("ans"), pred_cfg["API_params"].get("ans", ""))
            if gold_cfg["API_params"].keys() == pred_cfg["API_params"].keys():
                argn = 1
                if gold_cfg["API_params"] == pred_cfg["API_params"]:
                    argv = 1
                else:
                    errors.append("argument_values_not_matching")
            else:
                errors.append("required_argument_missing")
        else:
            errors.append("incorrect_tool_called")
    return {
        "exempt": False,
        "inst": inst,
        "tool": tool,
        "argn": argn,
        "argv": argv,
        "answer": answer,
        "errors": errors,
    }


def judge_or_zero(gold_ans: Any, pred_ans: Any, question: str, judge) -> float:
    """``sim_score``'s empty-prediction short circuit, then ``judge``.

    ``if not ans_pred`` treats ``""``, ``None``, and numeric 0 as no answer
    and returns 0 without calling the model. That is what the released
    function does.
    """
    if not pred_ans:
        return 0.0
    return float(judge(gold_ans, pred_ans, question))


def summarize_step(rows: Sequence[Mapping[str, Any]], judge) -> dict[str, Any]:
    """Table 3 aggregates, in the released code's units (percentages, 4 d.p.).

    Each row has ``gold_text``, ``pred_text``, ``is_generation``, and
    ``question``. ``judge(gold_ans, pred_ans, question)`` replaces
    ``sim_score``; the empty-prediction short circuit still happens here.
    """
    inst = tool = argn = argv = 0
    answer_sum = 0.0
    total_calls = 0
    count_non_gen = 0
    count_gen = 0
    for row in rows:
        scored = step_sample_score(row.get("gold_text"), row.get("pred_text"), is_generation_task=bool(row.get("is_generation")))
        if scored["exempt"]:
            continue
        total_calls += 1
        inst += scored["inst"]
        tool += scored["tool"]
        argn += scored["argn"]
        argv += scored["argv"]
        pair = scored["answer"]
        if pair is not None:
            count_non_gen += 1
            answer_sum += judge_or_zero(pair[0], pair[1], str(row.get("question", "")), judge)
        elif (
            scored["tool"]
            and bool(row.get("is_generation"))
            and _gold_is_terminate(row.get("gold_text"))
        ):
            count_gen += 1
    metrics = {
        "inst_acc": 0.0,
        "tool_acc": 0.0,
        "arg_acc1": 0.0,
        "arg_acc2": 0.0,
        "answer_acc": 0.0,
        "count_non_gen_samples": count_non_gen,
        "count_gen_samples": count_gen,
    }
    if total_calls > 0:
        metrics["inst_acc"] = inst / total_calls * 100
        metrics["tool_acc"] = tool / total_calls * 100
        metrics["arg_acc1"] = argn / total_calls * 100
        metrics["arg_acc2"] = argv / total_calls * 100
    if count_non_gen > 0:
        metrics["answer_acc"] = answer_sum / count_non_gen * 100
    return {key: round(value, 4) for key, value in metrics.items()}


def _gold_is_terminate(gold_text: str | None) -> bool:
    actions = extract_action(gold_text) or []
    if not actions:
        return False
    cfg, _error = extract_tool_cfg(actions[0])
    return bool(cfg and cfg["API_name"] == "Terminate")


def summarize_e2e(rows: Sequence[Mapping[str, Any]], judge) -> dict[str, float]:
    """Table 4 aggregates as ``evaluate_function`` computes them for ``e2e``.

    Each row has ``gold_names`` (including the final ``Terminate``),
    ``pred_names``, ``tool_response`` (list of ``{"error_code"}`` or ``None``),
    ``prediction`` (final answer or ``None``), ``gold_answer``, ``question``,
    and ``n_rounds``. F1 and answer accuracies are percentages. Tool-order
    ratios are fractions, matching the code rather than the printed table.
    """
    answer_sum = 0.0
    gen_sum = 0.0
    n_answer = 0
    n_gen = 0
    n_steps = 0.0
    same_n = any_n = unique_n = 0
    tool_errors = 0
    gold_rows: list[list[str]] = []
    pred_rows: list[list[str]] = []
    for row in rows:
        gold = [str(name) for name in row.get("gold_names") or []]
        pred = [str(name) for name in row.get("pred_names") or []]
        responses = list(row.get("tool_response") or [])
        error_codes = [res["error_code"] for res in responses if res]
        prediction = row.get("prediction")
        n_steps += float(row.get("n_rounds") or 0)
        gold_rows.append(gold)
        pred_rows.append(pred)
        if len(gold) >= 2 and gold[-2] in GEN_TOOLS:
            n_gen += 1
            last = responses[-1] if responses else None
            if pred and gold[-2] == pred[-1] and last and last.get("error_code") == 0:
                gen_sum += 1
        else:
            n_answer += 1
            answer_sum += judge_or_zero(row.get("gold_answer"), prediction, str(row.get("question", "")), judge)
        same_order, any_order, unique_ok = compare_tool_sequences(gold[:-1], pred)
        same_n += int(same_order)
        any_n += int(any_order)
        unique_n += int(unique_ok)
        tool_errors += sum(code != 0 for code in error_codes)
    f1 = category_f1(gold_rows, pred_rows)
    denom = n_answer + n_gen
    return {
        "answer_acc": round(answer_sum * 100 / n_answer, 4) if n_answer else 0.0,
        "answer_acc_w_gen": round(gen_sum * 100 / n_gen, 4) if n_gen else 0.0,
        "perception_f1": f1["perception"] * 100,
        "operation_f1": f1["operation"] * 100,
        "logic_f1": f1["logic"] * 100,
        "gis_f1": f1["gis"] * 100,
        "all_tools_called_any_order": (any_n / denom) if denom else 0.0,
        "all_tools_called_same_order": (same_n / denom) if denom else 0.0,
        "all_tools_called_unique": (unique_n / denom) if denom else 0.0,
        "avg_steps_taken": (n_steps / denom) if denom else 0.0,
        "total_gen_samples": float(n_gen),
        "tool_call_error": float(tool_errors),
    }
