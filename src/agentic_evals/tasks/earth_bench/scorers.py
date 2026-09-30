"""Earth-Bench scorer: end-to-end and step-by-step metrics from the persisted trace.

Metric definitions and their validation are in ``metrics.py`` (Feng et al., ICLR
2026, arXiv:2509.23141, Appendix B.2; upstream ``evaluate/*.py``). The scorer
reads our trace from the sample store (``agents.trace_callback``) and falls back
to the tool calls in ``state.messages`` for solvers that do not record one, so
it can be re-run offline with ``inspect score``.

Values per sample:

- ``accuracy``: upstream answer extraction + exact match against the sample
  target (the answer key chosen by the task's ``answer_key``).
- ``accuracy_other_key``: the same against the other upstream answer key, so the
  28-task disagreement between ``question.json`` and the GT file stays visible.
- ``tao``, ``tio``, ``tem``, ``param_acc``: trajectory metrics against the gold
  trajectory; NaN (excluded from the means) when the task has none (question43).
  Unless ``upstream_compat`` is set, gold steps that pass a list into a
  parameter annotated ``str`` are expanded to one scalar call per index first
  (Earth-Agent issue 13). The agent's trace is not expanded.
- ``efficiency``: ratio of predicted to gold tool calls, averaged. Not a score in
  [0, 1] and never a percentage. The gold count is the expanded length.
- ``tool_errors``: calls that raised or returned an error (e.g. a hallucinated
  file path). Mostly agent mistakes.
- ``perception_failures``: calls whose output is Perception's silent
  ``'Failed to call model'`` (a broken lookup, not the agent), so broken tooling
  is not mistaken for model behaviour. Non-zero on Spectrum/Products means the
  setup is wrong.

An RGB sample whose perception lookup file is missing is unscored (NaN at the
root), so it drops out of every mean. See ``task.skip_unavailable_rgb``.
"""

from __future__ import annotations

import json
import math
from typing import Any

from inspect_ai.scorer import Score, Target, mean, scorer, stderr
from inspect_ai.solver import TaskState

from agentic_evals.agents.trace_callback import read_trace
from agentic_evals.tasks.earth_bench import metrics
from agentic_evals.tasks.earth_bench.tools import is_tool_failure, scalar_str_params

NOT_APPLICABLE_KEY = "agentic_evals:not_applicable"


def _parse_args(arguments: Any) -> Any:
    if isinstance(arguments, str):
        try:
            return json.loads(arguments)
        except json.JSONDecodeError:
            return arguments
    return arguments


def predicted_calls(state: TaskState) -> tuple[list[dict[str, Any]], str]:
    """Tool calls made by the agent, from our trace if present, else from messages."""
    trace = read_trace(state.store)
    if trace:
        return [{"name": e["name"], "input": _parse_args(e["input"]), "output": e.get("output"), "error": e.get("error", False)} for e in trace], "trace"
    calls = []
    for message in state.messages:
        if message.role == "assistant":
            for call in message.tool_calls or []:
                calls.append({"name": call.function, "input": call.arguments, "output": None, "error": False})
    return calls, "messages"


@scorer(
    metrics={
        "accuracy": [mean(), stderr()],
        "accuracy_other_key": [mean()],
        "tao": [mean()],
        "tio": [mean()],
        "tem": [mean()],
        "param_acc": [mean()],
        "efficiency": [mean()],
        "tool_errors": [mean()],
        "perception_failures": [mean()],
    }
)
def earth_bench_scorer(upstream_compat: bool = False):
    """Earth-Bench dual-level metrics.

    Args:
        upstream_compat: Reproduce upstream quirks (TEM = 0 for an empty gold
            trajectory, and score against the published gold with no scalar expansion).
    """

    async def score(state: TaskState, target: Target) -> Score:
        reason = state.store.get(NOT_APPLICABLE_KEY)
        if reason:
            return Score(value=math.nan, explanation=str(reason))
        md = state.metadata
        final_text = state.output.completion if state.output else ""
        extracted, status, correct = metrics.answer_status(final_text, target.text)
        other_key = "answer_question_json" if md.get("answer_key") == "gt_file" else "answer_gt_file"
        other_answer = md.get(other_key)
        other_correct = metrics.answer_status(final_text, other_answer)[2] if other_answer is not None else correct

        pred, source = predicted_calls(state)
        value: dict[str, float] = {
            "accuracy": float(correct),
            "accuracy_other_key": float(other_correct),
            "tool_errors": float(sum(1 for c in pred if c["error"])),
            "perception_failures": float(sum(1 for c in pred if is_tool_failure(c["output"]))),
        }
        details: dict[str, Any] = {
            "extracted_answer": extracted,
            "answer_status": status,
            "call_source": source,
            "predicted_tools": metrics.tool_names(pred),
        }
        if md.get("has_gold_trajectory"):
            published = md["gold_tool_calls"]
            if upstream_compat:
                gold, unexpanded = published, []
            else:
                gold, unexpanded = metrics.expand_scalar_gold(published, scalar_str_params())
            traj = metrics.score_trajectory(pred, gold, upstream_compat=upstream_compat)
            value.update(traj.values())
            details.update(
                {
                    "gold_tools": metrics.tool_names(gold),
                    "gold_call_count": len(gold),
                    "unexpanded_gold_steps": unexpanded,
                    "first_divergence_step": traj.first_divergence_step,
                    "step_matches": traj.steps,
                }
            )
        else:
            gold = []
            value.update({k: math.nan for k in ("tao", "tio", "tem", "param_acc", "efficiency")})
            details["gold_tools"] = None

        return Score(
            value=value,
            answer=extracted,
            explanation=f"{status}; {len(pred)} calls vs {len(gold)} gold",
            metadata=details,
        )

    return score
