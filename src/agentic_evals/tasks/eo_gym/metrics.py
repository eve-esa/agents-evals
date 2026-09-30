"""EO-Gym metrics, as pure functions.

Reference: *EO-Gym: A Multimodal, Interactive Environment for Earth Observation
Agents*, arXiv:2605.01250, Section 5 (Pass@k) and Appendix D (tool execution
metrics); reference implementation ``eo_gym_evaluation/agent_passk.py`` at the
revision pinned in ``upstream.lock.toml``. We follow the released code, and
``tests/test_eo_gym_metrics.py`` checks it differentially against that code.

Where the released code and the paper disagree, we follow the code and also
report the paper's version:

- Pass@k. The code uses an ordered early stop: trajectories are judged in order
  (traj0, traj1, ...), consumption stops at the first correct one, and
  Pass@k = 1 iff ``first_correct_traj < k``. The paper describes the unbiased
  estimator of Chen et al. 2021 (arXiv:2107.03374),
  ``1 - C(n-c, k) / C(n, k)``, reported here as ``unbiased_pass@k``.
- Aggregation. Tool rates are means over all *consumed trajectories pooled
  across questions*, and ``efficiency_consumed`` is total consumed calls divided
  by total reference steps (reference steps counted once per question even when
  several trajectories were consumed). ``pooled_summary`` reproduces this.

Per-trajectory tool metrics are binary: ``tools_any_order`` (gold tools present
as a multiset), ``tools_in_order`` (gold tools as a subsequence),
``tools_exact_match`` (identical sequences). ``parameters_score`` is the share
of gold steps matched by a greedy forward scan on tool name whose normalised
arguments *and* observation both match.
"""

from __future__ import annotations

import json
import math
import re
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any
from urllib.parse import urlparse

FINAL_ANSWER_RE = re.compile(r"final answer\s*:\s*", re.IGNORECASE)
# Largest k reported (EO-Gym reports up to Pass@10); keys beyond the number of
# trajectories are NaN and so excluded from the metrics.
MAX_K = 10
JSON_START_RE = re.compile(r"\s*[\[{]")


def _parse_json_like(text: str) -> Any:
    stripped = text.strip()
    if not stripped or not JSON_START_RE.match(stripped):
        return stripped
    try:
        return json.loads(stripped)
    except Exception:  # noqa: BLE001
        return stripped


def _normalize_path_like(text: str) -> str:
    value = text.strip().replace("\\", "/")
    parsed = urlparse(value)
    if parsed.scheme in {"http", "https"}:
        value = parsed.path or ""
    for marker in ("EO_GYM_DATA/", "image_cache/"):
        idx = value.find(marker)
        if idx >= 0:
            value = value[idx:]
            break
    value = re.sub(r"/+", "/", value)
    value = re.sub(r"_([0-9a-f]{8,})(?=\.[a-zA-Z0-9]+$)", "_<id>", value)
    return value.lower().strip()


def _normalize_scalar(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
            return str(value)
        return value
    text = str(value).strip()
    if not text:
        return ""
    parsed = _parse_json_like(text)
    if parsed is not text:
        return normalize_value(parsed)
    if "/" in text or "http://" in text or "https://" in text:
        return _normalize_path_like(text)
    return " ".join(text.split()).lower()


def normalize_value(value: Any) -> Any:
    if isinstance(value, dict):
        out = {str(k).strip().lower(): normalize_value(v) for k, v in value.items()}
        return {k: out[k] for k in sorted(out)}
    if isinstance(value, list):
        return [normalize_value(item) for item in value]
    return _normalize_scalar(value)


def values_match(a: Any, b: Any) -> bool:
    if a is None and b is None:
        return True
    if isinstance(a, bool) or isinstance(b, bool):
        return a == b
    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        return math.isclose(float(a), float(b), rel_tol=1e-6, abs_tol=1e-6)
    if isinstance(a, str) and isinstance(b, str):
        return a == b
    if isinstance(a, list) and isinstance(b, list):
        return len(a) == len(b) and all(values_match(x, y) for x, y in zip(a, b))
    if isinstance(a, dict) and isinstance(b, dict):
        return set(a) == set(b) and all(values_match(a[k], b[k]) for k in a)
    return a == b


def is_error_observation(value: Any) -> tuple[bool, str]:
    if value is None:
        return True, "missing_observation"
    if isinstance(value, dict):
        return (True, "dict_error") if value.get("error") else (False, "")
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return True, "empty_observation"
        if text.lower().startswith("error"):
            return True, "error_prefix"
        parsed = _parse_json_like(text)
        if isinstance(parsed, dict) and parsed.get("error"):
            return True, "json_error"
    return False, ""


def extract_final_answer(final_message: str | None) -> tuple[str, bool]:
    """Text after the last ``Final Answer:``; else the whole message, flagged as missing the marker."""
    text = (final_message or "").strip()
    if not text:
        return "", True
    matches = list(FINAL_ANSWER_RE.finditer(text))
    if matches:
        return text[matches[-1].end() :].strip(), False
    return text, True


def normalize_text(text: Any) -> str:
    return " ".join(str(text or "").strip().split()).lower()


@dataclass
class Step:
    name: str
    arguments: Any
    observation: Any = None

    @property
    def arguments_norm(self) -> Any:
        return normalize_value(self.arguments)

    @property
    def observation_norm(self) -> Any:
        return normalize_value(self.observation) if self.observation is not None else None

    @property
    def is_error(self) -> bool:
        return is_error_observation(self.observation)[0]


def gold_steps_from_messages(messages: Sequence[dict[str, Any]]) -> list[Step]:
    """Expected steps from a test-set record's gold messages (``_load_ground_truth``)."""
    outputs = {
        str(m.get("tool_call_id") or "").strip(): m.get("content")
        for m in messages
        if isinstance(m, dict) and m.get("role") == "tool" and m.get("tool_call_id")
    }
    steps: list[Step] = []
    for message in messages:
        if not isinstance(message, dict) or message.get("role") != "assistant":
            continue
        for call in message.get("tool_calls") or []:
            if not isinstance(call, dict):
                continue
            call_id = str(call.get("id") or f"step_{len(steps)}")
            fn = call.get("function") if isinstance(call.get("function"), dict) else {}
            steps.append(Step(str(fn.get("name") or "").strip(), fn.get("arguments"), outputs.get(call_id)))
    return steps


def judge_question_from_messages(messages: Sequence[dict[str, Any]]) -> str:
    """The question text the released scorer shows its judge (``_extract_question_from_gt_messages``)."""
    for message in messages:
        if not isinstance(message, dict) or message.get("role") != "user":
            continue
        content = message.get("content")
        if not isinstance(content, str):
            continue
        lines = [ln.strip() for ln in content.splitlines() if ln.strip() and ln.strip() != "<image>"]
        text = " ".join(lines).strip()
        match = re.search(r"question\s*:\s*", text, flags=re.IGNORECASE)
        return text[match.end() :].strip() if match else text
    return ""


def tools_any_order(expected: Sequence[str], actual: Sequence[str]) -> bool:
    have = Counter(actual)
    return all(have.get(name, 0) >= count for name, count in Counter(expected).items())


def tools_in_order(expected: Sequence[str], actual: Sequence[str]) -> bool:
    if not expected:
        return True
    idx = 0
    for name in actual:
        if name == expected[idx]:
            idx += 1
            if idx == len(expected):
                return True
    return False


def tool_metrics(expected: Sequence[Step], actual: Sequence[Step]) -> dict[str, Any]:
    expected_names = [s.name for s in expected]
    actual_names = [s.name for s in actual]
    matched = 0
    cursor = 0
    step_matches: list[dict[str, Any]] = []
    for exp in expected:
        hit = next((i for i in range(cursor, len(actual)) if actual[i].name == exp.name), None)
        if hit is None:
            step_matches.append({"expected_tool": exp.name, "actual_index": None, "args_match": False, "output_match": False})
            continue
        cursor = hit + 1
        args_ok = values_match(exp.arguments_norm, actual[hit].arguments_norm)
        out_ok = values_match(exp.observation_norm, actual[hit].observation_norm)
        matched += args_ok and out_ok
        step_matches.append({"expected_tool": exp.name, "actual_index": hit, "args_match": args_ok, "output_match": out_ok})
    return {
        "tools_any_order": int(tools_any_order(expected_names, actual_names)),
        "tools_in_order": int(tools_in_order(expected_names, actual_names)),
        "tools_exact_match": int(expected_names == actual_names),
        "parameters_score": 1.0 if not expected else matched / len(expected),
        "expected_steps": len(expected),
        "actual_steps": len(actual),
        "step_matches": step_matches,
    }


@dataclass
class Trajectory:
    traj_id: int
    correct: bool
    function_calls: int
    function_call_errors: int
    missing_marker: bool
    tools_any_order: float
    tools_in_order: float
    tools_exact_match: float
    parameters_score: float
    reference_steps: int
    extra: dict[str, Any] = field(default_factory=dict)


def unbiased_pass_at_k(n: int, c: int, k: int) -> float:
    """Chen et al. 2021: probability that at least one of k samples (from n, c correct) is correct."""
    if n - c < k:
        return 1.0
    return 1.0 - math.comb(n - c, k) / math.comb(n, k)


def early_stop(trajectories: Sequence[Trajectory], max_traj: int = 3) -> dict[str, float]:
    """Per-question row of ``agent_passk._evaluate_single_question`` plus pooled-metric sums."""
    by_id = {t.traj_id: t for t in trajectories if 0 <= t.traj_id < max_traj}
    consumed: list[Trajectory] = []
    first_correct: int | None = None
    for traj_id in range(max_traj):
        traj = by_id.get(traj_id)
        if traj is None:
            continue
        consumed.append(traj)
        if traj.correct:
            first_correct = traj_id
            break
    reference_steps = next(iter(by_id.values())).reference_steps if by_id else 0
    n = len(by_id)
    c = sum(t.correct for t in by_id.values())
    row = {
        "accuracy": float(first_correct is not None),
        "tries_used": float(len(consumed)),
        "reference_steps": float(reference_steps),
        "consumed_trajectories": float(len(consumed)),
        "consumed_tool_calls": float(sum(t.function_calls for t in consumed)),
        "consumed_function_call_errors": float(sum(t.function_call_errors for t in consumed)),
        "consumed_missing_markers": float(sum(t.missing_marker for t in consumed)),
        "consumed_no_function_call": float(sum(t.function_calls == 0 for t in consumed)),
        "consumed_tools_any_order_sum": float(sum(t.tools_any_order for t in consumed)),
        "consumed_tools_in_order_sum": float(sum(t.tools_in_order for t in consumed)),
        "consumed_tools_exact_match_sum": float(sum(t.tools_exact_match for t in consumed)),
        "consumed_parameters_score_sum": float(sum(t.parameters_score for t in consumed)),
        "trajectories": float(n),
        "correct_trajectories": float(c),
    }
    for k in range(1, MAX_K + 1):
        row[f"pass@{k}"] = float(first_correct is not None and first_correct < k) if k <= max_traj else math.nan
        row[f"unbiased_pass@{k}"] = unbiased_pass_at_k(n, c, k) if k <= n else math.nan
    return row


def _div(num: float, den: float) -> float:
    return num / den if den else 0.0


def pooled_summary(rows: Sequence[dict[str, float]]) -> dict[str, float]:
    """The released summary row (``_build_summary_rows``) from per-question rows."""
    def total(key: str) -> float:
        return sum(r[key] for r in rows)

    consumed = total("consumed_trajectories")
    calls = total("consumed_tool_calls")
    return {
        "efficiency_consumed": _div(calls, total("reference_steps")),
        "avg_tool_calls_consumed_per_question": _div(calls, len(rows)),
        "illegal_function_call_error_rate_consumed": _div(total("consumed_function_call_errors"), calls),
        "tools_any_order_rate_consumed": _div(total("consumed_tools_any_order_sum"), consumed),
        "tools_in_order_rate_consumed": _div(total("consumed_tools_in_order_sum"), consumed),
        "tools_exact_match_rate_consumed": _div(total("consumed_tools_exact_match_sum"), consumed),
        "parameters_score_mean_consumed": _div(total("consumed_parameters_score_sum"), consumed),
        "no_function_call_rate_consumed": _div(total("consumed_no_function_call"), consumed),
    }
