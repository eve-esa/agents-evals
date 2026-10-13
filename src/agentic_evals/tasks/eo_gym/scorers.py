"""EO-Gym scoring: one score per trajectory (epoch), one reduced row per task.

Protocol: arXiv:2605.01250 Section 5 and Appendix C/D; reference implementation
``eo_gym_evaluation/agent_passk.py`` (metric functions in ``metrics.py``, checked
differentially against it by ``tests/test_eo_gym_metrics.py``).

- ``eo_gym_scorer`` scores each trajectory: the final answer (text after
  ``Final Answer:``) judged against the reference by the ``grader`` model role
  (default ``openai/gpt-4.1-mini``, temperature 0) with EO-Gym's
  ``SEMANTIC_FINAL_ANSWER_JUDGE_SYSTEM_PROMPT``, a JSON object schema for
  ``is_same_meaning`` and ``reason``, and the same payload, falling back to
  exact normalised match when the judge fails, as upstream does; plus the tool
  metrics of that trajectory against the gold steps.
- ``eo_gym_early_stop`` reduces a task's trajectories in epoch order: consume
  until the first correct one; Pass@k = 1 iff it is among the first k. The
  unbiased estimator (Chen et al. 2021) is reported alongside as
  ``unbiased_pass@k``.
- ``eo_gym_pooled`` computes the released summary's pooled rates (tool metrics
  over consumed trajectories pooled across tasks, efficiency as a ratio of sums).
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

from inspect_ai.model import (
    ChatMessageSystem,
    ChatMessageUser,
    GenerateConfig,
    ResponseSchema,
    get_model,
)
from inspect_ai.scorer import (
    Metric,
    SampleScore,
    Score,
    ScoreReducer,
    Target,
    Value,
    mean,
    metric,
    score_reducer,
    scorer,
)
from inspect_ai.solver import TaskState

from agentic_evals.agents.trace_callback import recorded_trace
from agentic_evals.tasks.eo_gym import metrics

DEFAULT_JUDGE = "openai/gpt-4.1-mini"
NOT_APPLICABLE_KEY = "agentic_evals:not_applicable"

# Verbatim from eo_gym_data_synthesis/prompt_manager/evaluation_prompts.py.
_JUDGE_PROMPT = Path(__file__).resolve().parent / "prompts" / "semantic_final_answer.txt"
SEMANTIC_FINAL_ANSWER_JUDGE_SYSTEM_PROMPT = _JUDGE_PROMPT.read_text(encoding="utf-8").strip()

# Released SemanticMatcher uses response_format json_object. Inspect's
# response_schema is the equivalent; strict=False so a non-OpenAI grader still runs.
JUDGE_RESPONSE_SCHEMA = ResponseSchema(
    name="semantic_equivalence",
    json_schema={
        "type": "object",
        "properties": {
            "is_same_meaning": {"type": "boolean"},
            "reason": {"type": "string"},
        },
        "required": ["is_same_meaning", "reason"],
        "additionalProperties": False,
    },
    description="Whether the candidate final answer has the same meaning as the reference.",
    strict=False,
)


def judge_generate_config() -> GenerateConfig:
    """Temperature 0 and a JSON object with ``is_same_meaning`` and ``reason``."""
    return GenerateConfig(temperature=0, response_schema=JUDGE_RESPONSE_SCHEMA)


async def judge_answer(question: str, reference: str, candidate: str) -> tuple[bool, str, str]:
    """``SemanticMatcher._judge_via_api``: returns (is_correct, reason, judge model)."""
    grader = get_model(role="grader", default=DEFAULT_JUDGE)
    if not candidate.strip():
        return False, "empty_prediction", str(grader)
    if not reference.strip():
        return False, "empty_ground_truth", str(grader)
    payload = {"question": question, "reference_final_answer": reference, "candidate_final_answer": candidate}
    try:
        output = await grader.generate(
            [
                ChatMessageSystem(content=SEMANTIC_FINAL_ANSWER_JUDGE_SYSTEM_PROMPT),
                ChatMessageUser(content=json.dumps(payload, ensure_ascii=False)),
            ],
            config=judge_generate_config(),
        )
        data = json.loads(output.completion.strip() or "{}")
        return bool(data.get("is_same_meaning", False)), str(data.get("reason", "")).strip() or "no_reason", str(grader)
    except Exception as exc:  # noqa: BLE001 - upstream falls back to exact match
        fallback = metrics.normalize_text(reference) == metrics.normalize_text(candidate)
        return fallback, f"api_error_fallback_exact: {type(exc).__name__}", str(grader)


def trajectory_steps(state: TaskState) -> list[metrics.Step]:
    trace = recorded_trace(state.store)
    if trace is not None:
        return [metrics.Step(e["name"], e["input"], e.get("output") if not e.get("error_message") else e["error_message"]) for e in trace]
    steps: list[metrics.Step] = []
    results = {m.tool_call_id: m.text for m in state.messages if m.role == "tool"}
    for message in state.messages:
        if message.role == "assistant":
            for call in message.tool_calls or []:
                steps.append(metrics.Step(call.function, call.arguments, results.get(call.id)))
    return steps


def _reduced_metrics() -> list[Any]:
    keys = ["accuracy", "tries_used"]
    keys += [f"pass@{k}" for k in range(1, metrics.MAX_K + 1)]
    keys += [f"unbiased_pass@{k}" for k in range(1, metrics.MAX_K + 1)]
    return [{key: [mean()] for key in keys}, eo_gym_pooled()]


@metric
def eo_gym_pooled() -> Metric:
    """Pooled summary rates of the released scorer, over all reduced task rows."""

    def compute(scores: list[SampleScore]) -> Value:
        rows = [s.score.value for s in scores if isinstance(s.score.value, dict)]
        return metrics.pooled_summary(rows) if rows else {}

    return compute


@scorer(metrics=_reduced_metrics())
def eo_gym_scorer():
    """Per-trajectory EO-Gym score (reduced across epochs by ``eo_gym_early_stop``)."""

    async def score(state: TaskState, target: Target) -> Score:
        if state.store.get(NOT_APPLICABLE_KEY):
            return Score(value=math.nan, explanation=state.store.get(NOT_APPLICABLE_KEY))
        md = state.metadata
        answer, missing = metrics.extract_final_answer(state.output.completion if state.output else "")
        correct, reason, judge = await judge_answer(md["judge_question"], target.text, answer)
        steps = trajectory_steps(state)
        gold = [metrics.Step(s["name"], s["arguments"], s["observation"]) for s in md["gold_steps"]]
        tm = metrics.tool_metrics(gold, steps)
        return Score(
            value={
                "epoch": float(state.epoch),
                "correct": float(correct),
                "function_calls": float(len(steps)),
                "function_call_errors": float(sum(s.is_error for s in steps)),
                "missing_marker": float(missing),
                "tools_any_order": float(tm["tools_any_order"]),
                "tools_in_order": float(tm["tools_in_order"]),
                "tools_exact_match": float(tm["tools_exact_match"]),
                "parameters_score": float(tm["parameters_score"]),
                "reference_steps": float(len(gold)),
            },
            answer=answer,
            explanation=reason,
            metadata={
                "judge_model": judge,
                "judge_prompt": "eo_gym SEMANTIC_FINAL_ANSWER_JUDGE_SYSTEM_PROMPT",
                "predicted_tools": [s.name for s in steps],
                "gold_tools": [s.name for s in gold],
                "step_matches": tm["step_matches"],
            },
        )

    return score


@score_reducer(name="eo_gym_early_stop")
def eo_gym_early_stop(max_traj: int = 3) -> ScoreReducer:
    """Ordered early-stop reduction of a task's trajectories (``agent_passk``).

    Scores from other scorers on the same task (e.g. optional trajectory judges)
    are reduced to their per-key mean over epochs.
    """

    def reduce(scores: list[Score]) -> Score:
        if not scores or any(not isinstance(s.value, dict) for s in scores):
            return Score(value=math.nan, explanation="not applicable")
        if "correct" not in scores[0].value:  # type: ignore[operator]
            keys = scores[0].value.keys()  # type: ignore[union-attr]
            return Score(
                value={
                    k: (sum(v) / len(v) if (v := [s.value[k] for s in scores if not math.isnan(s.value[k])]) else math.nan)  # type: ignore[index]
                    for k in keys
                }
            )
        trajectories = []
        for s in scores:
            v: dict[str, Any] = s.value  # type: ignore[assignment]
            trajectories.append(
                metrics.Trajectory(
                    traj_id=int(v["epoch"]) - 1,
                    correct=bool(v["correct"]),
                    function_calls=int(v["function_calls"]),
                    function_call_errors=int(v["function_call_errors"]),
                    missing_marker=bool(v["missing_marker"]),
                    tools_any_order=v["tools_any_order"],
                    tools_in_order=v["tools_in_order"],
                    tools_exact_match=v["tools_exact_match"],
                    parameters_score=v["parameters_score"],
                    reference_steps=int(v["reference_steps"]),
                )
            )
        return Score(value=metrics.early_stop(trajectories, max_traj=max_traj))

    return reduce
