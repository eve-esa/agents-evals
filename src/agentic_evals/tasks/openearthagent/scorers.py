"""Inspect scorers for OpenEarthAgent.

Shabbir et al., arXiv:2602.17665, Tables 3 and 4. The judge is ``gpt-4o-mini``
at temperature 0 with at most 150 tokens (``sim_score``), using ``EVAL_PROMPT``.
Each judge is its own scorer. The ``grader`` role is the official one
(``--model-role grader=openai/gpt-4o-mini``). Extra judges are
``-T judges=...`` and are not averaged into the official score. The chart's
``avg_summ`` and ``avg_ans`` are that average, kept beside the per-judge scores.

Logged scores are fractions in ``[0, 1]``. The released code multiplies F1
and answer accuracy by 100, and Table 4 also prints the tool-order ratios as
percentages. Multiply a logged mean by 100 to read the table.
"""

from __future__ import annotations

import json
import math
from typing import Any

from inspect_ai.model import ChatMessageUser, GenerateConfig, get_model
from inspect_ai.scorer import Metric, SampleScore, Score, Target, mean, metric, scorer
from inspect_ai.solver import TaskState

from agentic_evals.agents.trace_callback import read_trace
from agentic_evals.tasks.openearthagent import metrics
from agentic_evals.tasks.openearthagent.dataset import judge_prompt

NOT_APPLICABLE_KEY = "agentic_evals:not_applicable"
JUDGE_PROMPT_NAME = "openearthagent EVAL_PROMPT"
# sim_score: temperature 0, max_tokens 150, a single user message.
JUDGE_GENERATE = GenerateConfig(temperature=0, max_tokens=150)


def _parse_args(arguments: Any) -> Any:
    if isinstance(arguments, str):
        try:
            return json.loads(arguments)
        except json.JSONDecodeError:
            return arguments
    return arguments


def predicted_calls(state: TaskState) -> tuple[list[dict[str, Any]], str]:
    """Tool calls from our trace, or from ``state.messages`` when the trace is absent."""
    trace = read_trace(state.store)
    if trace:
        return [
            {
                "name": entry["name"],
                "arguments": _parse_args(entry["input"]),
                "error_code": 1 if entry.get("error") else 0,
            }
            for entry in trace
        ], "trace"
    calls = []
    for message in state.messages:
        if message.role == "assistant":
            for call in message.tool_calls or []:
                calls.append({"name": call.function, "arguments": _parse_args(call.arguments), "error_code": 0})
    return calls, "messages"


def candidate_answer(text: str | None) -> str:
    """``Terminate.ans`` when the model emitted one, otherwise the message text.

    End-to-end runs in this harness answer in plain text (``Terminate`` is not
    a tool). A model that still emits the JSON action is judged on ``ans``,
    which is what ``extract_final_answer`` reads.
    """
    extracted = metrics.extract_final_answer(text)
    if extracted:
        return extracted if isinstance(extracted, str) else str(extracted)
    return (text or "").strip()


def _not_applicable(state: TaskState) -> Score | None:
    reason = state.store.get(NOT_APPLICABLE_KEY)
    if reason:
        return Score(value=math.nan, explanation=str(reason))
    return None


async def judge_answer(question: str, gold: Any, predicted: Any, model: str | None) -> tuple[float | None, str, str]:
    """``(score, explanation, judge id)``. ``None`` when the judge call failed.

    An empty prediction is 0 and does not call the model (``sim_score``).
    """
    judge = get_model(model) if model else get_model(role="grader", required=True)
    if not predicted:
        return 0.0, "empty prediction", str(judge)
    prompt = (
        f"{judge_prompt()}\n\nQuestion: {question}\nGround Truth Answer: {gold}\nPredicted Answer: {predicted}\n"
    )
    try:
        result = await judge.generate([ChatMessageUser(content=prompt)], config=JUDGE_GENERATE)
    except Exception as exc:  # noqa: BLE001 - one judge failure should not drop the sample
        return None, f"{type(exc).__name__}: {exc}", str(judge)
    score = metrics.extract_score(result.completion or "")
    return score, (result.completion or "").strip(), str(judge)


@metric
def category_f1() -> Metric:
    """Pooled perception / operation / logic / gis F1, as fractions.

    Reads ``gold_names`` and ``pred_names`` from each score's metadata.
    Samples with a non-dict value (not applicable) are skipped. The released
    code multiplies these by 100.
    """

    def compute(scores: list[SampleScore]) -> dict[str, float]:
        gold_rows: list[list[str]] = []
        pred_rows: list[list[str]] = []
        for sample in scores:
            if not isinstance(sample.score.value, dict):
                continue
            meta = sample.score.metadata or {}
            gold_rows.append([str(name) for name in meta.get("gold_names") or []])
            pred_rows.append([str(name) for name in meta.get("pred_names") or []])
        if not gold_rows:
            return {category: math.nan for category in metrics.F1_CATEGORIES}
        return metrics.category_f1(gold_rows, pred_rows)

    return compute


def _e2e_metrics() -> list[Any]:
    keys = {
        "any_order": [mean()],
        "same_order": [mean()],
        "unique": [mean()],
        "answer": [mean()],
        "gen_success": [mean()],
        "n_steps": [mean()],
    }
    return [keys, category_f1()]


@scorer(metrics=_e2e_metrics())
def openearthagent_scorer(model: str | None = None):
    """End-to-end tool order, answer accuracy, and generation success.

    Args:
        model: Judge model. Defaults to the ``grader`` role. Tool-order and
            generation success do not use the judge; they are repeated on an
            extra judge's row only when this scorer is constructed with ``model``.
    """

    async def score(state: TaskState, target: Target) -> Score:
        del target
        skipped = _not_applicable(state)
        if skipped:
            return skipped
        md = state.metadata
        gold = [str(name) for name in md.get("gold_names") or []]
        calls, source = predicted_calls(state)
        pred = [call["name"] for call in calls]
        responses = [{"error_code": call["error_code"]} for call in calls]
        same_order, any_order, unique_ok = metrics.compare_tool_sequences(gold[:-1], pred)
        generation = bool(md.get("is_generation"))
        gen_success: float = math.nan
        answer: float | None = None
        explanation = ""
        judge_id = ""
        if generation:
            last = responses[-1] if responses else None
            matched = bool(pred) and len(gold) >= 2 and gold[-2] == pred[-1] and last and last.get("error_code") == 0
            gen_success = float(matched)
            answer = math.nan
        else:
            predicted = candidate_answer(state.output.completion if state.output else "")
            answer, explanation, judge_id = await judge_answer(str(md.get("question", "")), md.get("reference_answer", ""), predicted, model)
            if answer is None:
                answer = math.nan
        return Score(
            value={
                "any_order": float(any_order),
                "same_order": float(same_order),
                "unique": float(unique_ok),
                "answer": float(answer) if answer is not None else math.nan,
                "gen_success": gen_success,
                "n_steps": float(len(calls)),
            },
            answer=candidate_answer(state.output.completion if state.output else ""),
            explanation=explanation,
            metadata={
                "call_source": source,
                "gold_names": gold,
                "pred_names": pred,
                "judge_model": judge_id,
                "judge_prompt": JUDGE_PROMPT_NAME,
                "protocol": "openearthagent_e2e",
            },
        )

    return score


def extra_answer_judges(models: str | list[str], *, step: bool = False) -> list:
    """Additional answer judges, one per model. The grader-role scorer is separate.

    Step mode uses ``openearthagent_step`` so the answer denominator stays the
    released one. End-to-end repeats the tool-order keys on each judge's row;
    those keys do not depend on the judge.
    """
    names = models if isinstance(models, list) else models.split(",")
    factory = openearthagent_step if step else openearthagent_scorer
    return [factory(model=name.strip()) for name in names if name.strip()]


def _completion(state: TaskState) -> str:
    if state.output is None:
        return ""
    return state.output.completion or ""


@scorer(
    metrics={
        "inst": [mean()],
        "tool": [mean()],
        "argn": [mean()],
        "argv": [mean()],
        "answer": [mean()],
    }
)
def openearthagent_step(model: str | None = None):
    """Step-mode inst / tool / argument / answer accuracies.

    Thought-only gold turns are unscored (``NaN``), matching ``if tool_cfg_gold``.
    Answer accuracy is judged only when both sides called ``Terminate`` on a
    non-generation task, which is the released denominator.
    """

    async def score(state: TaskState, target: Target) -> Score:
        del target
        skipped = _not_applicable(state)
        if skipped:
            return skipped
        md = state.metadata
        scored = metrics.step_sample_score(
            md.get("gold_text"),
            _completion(state),
            is_generation_task=bool(md.get("is_generation")),
        )
        if scored["exempt"]:
            return Score(
                value=math.nan,
                explanation="thought-only turn; upstream does not score an empty action",
                metadata={"protocol": "openearthagent_step", "exempt": True},
            )
        answer: float = math.nan
        explanation = ""
        judge_id = ""
        pair = scored["answer"]
        if pair is not None:
            answer_value, explanation, judge_id = await judge_answer(
                str(md.get("question", "")), pair[0], pair[1], model
            )
            answer = math.nan if answer_value is None else float(answer_value)
        return Score(
            value={
                "inst": float(scored["inst"]),
                "tool": float(scored["tool"]),
                "argn": float(scored["argn"]),
                "argv": float(scored["argv"]),
                "answer": answer,
            },
            answer=_completion(state),
            explanation=explanation,
            metadata={
                "protocol": "openearthagent_step",
                "errors": scored["errors"],
                "judge_model": judge_id,
                "judge_prompt": JUDGE_PROMPT_NAME,
            },
        )

    return score
