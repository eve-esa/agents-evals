"""GeoBenchX scorers.

Official protocol (arXiv:2503.18129 Section 5; ``geobenchx/evaluation.py``):
``geobenchx_scorer`` renders the trace with ``get_solution_code`` and asks one
judge for a match score of 0, 1, or 2. Each judge is its own scorer. There is
no majority vote: the released code records a single ``evaluator_model``, and
the paper's panel is three such runs. The default judge is Inspect's ``grader``
role. ``length_diff`` is computed whether or not the judge answers.

Score keys use the checkpoint id (``qwen/qwen3.8-27b/match``), without the
``openai-api`` provider prefix, because Inspect names each dict-metric row
after the key (``inspect_ai._eval.task.results.scorers_from_metric_dict``).
The viewer's summary shows three of those rows, so a sample stores match and
``length_diff`` only; the 0/1/2 frequencies are metrics of the match row.
Averages follow ``generate_eval_stats``: mean match, those frequencies, and
mean/median ``length_diff``, plus Inspect's standard error and confidence
interval.

The judge call uses Inspect's exponential backoff (``model_retry_config``:
``wait_exponential_jitter``, initial 3s, cap 30 min) with ``max_retries=6``.
``stream_idle_timeout`` expires an attempt only after the stream has gone
quiet; it is not a wall-clock cap on a long generation
(``inspect_ai.model._stream.arm_stall_scope``). A 400 whose body says the
request expired is not classified as retryable by ``openai_classify_retry``,
so those are retried here.

``geobenchx_merged`` is the mean of the judges on one sample. It is not a vote
and not part of the published protocol. ``geobenchx_steps`` is our addition;
see ``metrics.step_scores``.
"""

from __future__ import annotations

import json
import math
import re
import statistics
from typing import Any

import anyio
from inspect_ai.model import ChatMessageSystem, ChatMessageUser, GenerateConfig, get_model
from inspect_ai.scorer import (
    Metric,
    SampleScore,
    Score,
    Target,
    ci,
    mean,
    metric,
    scorer,
    stderr,
)
from inspect_ai.solver import TaskState

from agentic_evals.agents.trace_callback import read_trace
from agentic_evals.tasks.geobenchx import metrics
from agentic_evals.tasks.geobenchx.tools import judge_prompt

JUDGE_PROMPT_NAME = "geobenchx RESULT_CHECKING_PROMPT"

# Seven attempts. Inspect counts max_retries as retries after the first call
# (inspect_ai.model._retry.model_retry_config).
JUDGE_MAX_RETRIES = 6
# Silence after the first streamed chunk, not a cap on an active generation.
JUDGE_STREAM_IDLE_TIMEOUT = 180
# Extra tries, beyond Inspect's loop, for a 400 that says the request expired.
EXPIRED_EXTRA_ATTEMPTS = 5
_EXPIRED_TEXT = re.compile(r"expir|timed out", re.IGNORECASE)

JUDGE_GENERATE = GenerateConfig(
    temperature=0,
    max_retries=JUDGE_MAX_RETRIES,
    stream_idle_timeout=JUDGE_STREAM_IDLE_TIMEOUT,
)


def retry_delay(attempt: int) -> float:
    """Seconds to wait before extra attempt ``attempt`` (1-based): 2, 4, 8, … capped at 60."""
    return float(min(60, 2**attempt))


def is_expired_request(exc: BaseException) -> bool:
    """A 400 (or empty-choices ``invalid_request``) whose text says the request expired.

    Inspect already retries 429, 5xx, and connection timeouts. A context-length
    400 does not match and is left alone.
    """
    text = f"{exc} {getattr(exc, 'message', '')}"
    if _EXPIRED_TEXT.search(text) is None:
        return False
    status = getattr(exc, "status_code", None)
    if status == 400:
        return True
    return getattr(exc, "code", None) == "invalid_request"


def display_model_id(model_id: str) -> str:
    """Checkpoint id for a score row. Drops the Inspect provider prefix.

    ``openai-api/openrouter/qwen/qwen3.8-27b`` is the route to the model.
    The row name is ``qwen/qwen3.8-27b``.
    """
    for prefix in ("openai-api/", "openrouter/"):
        if model_id.startswith(prefix):
            model_id = model_id[len(prefix) :]
    return model_id


def match_values(model_id: str, verdict: int | None, diff: float) -> dict[str, float]:
    """One judge's score dict. Keys are what the results table shows as the row name.

    Success, partial, and no-match are not stored per sample. They are the
    frequencies of match values 2, 1, and 0, reported by ``verdict_rates``.
    """
    name = display_model_id(model_id)
    match = math.nan if verdict is None else float(verdict)
    return {
        f"{name}/match": match,
        f"{name}/length_diff": diff,
    }


def merged_match(judge_values: list[dict[str, Any]]) -> dict[str, float]:
    """Mean of the judges' match fields. NaNs are left out. ``length_diff`` is not averaged.

    An arithmetic mean of the per-judge scores, not a majority vote. GeoBenchX
    records one ``evaluator_model`` per run (``generate_eval_stats``).
    """
    numbers: list[float] = []
    for values in judge_values:
        for key, value in values.items():
            if not str(key).endswith("/match") or str(key).startswith("merged/"):
                continue
            if isinstance(value, int | float) and not math.isnan(float(value)):
                numbers.append(float(value))
    return {"merged/match": sum(numbers) / len(numbers) if numbers else math.nan}


def _level_stats() -> list[Metric]:
    # mean/median length and match: generate_eval_stats. stderr and ci are
    # Inspect's (95% t interval). Their binomial interval is scipy binomtest,
    # which this project does not depend on.
    return [mean(), median(), stderr(), ci(), by_kind()]


def _match_metrics() -> dict[str, list[Metric]]:
    # Frequencies of 2, 1, and 0 live on the match row (verdict_rates), so they
    # do not become extra score rows. The viewer summary shows three rows
    # (inspect view ResultsPanel, kMaxPrimaryScoreRows).
    return {
        "*/match": [*_level_stats(), verdict_rates()],
        "*/length_diff": _level_stats(),
    }


def _parse_args(arguments: Any) -> Any:
    if isinstance(arguments, str):
        try:
            return json.loads(arguments)
        except json.JSONDecodeError:
            return arguments
    return arguments


def predicted_calls(state: TaskState) -> tuple[list[dict[str, Any]], str]:
    """Tool calls from our trace, or from ``state.messages`` when the trace is absent.

    Same fallback as the other tasks, so ``inspect score`` still works for a
    solver that did not record a trace. Kept here so this task does not import
    another benchmark.
    """
    trace = read_trace(state.store)
    if trace:
        return [
            {"name": entry["name"], "input": _parse_args(entry["input"])} for entry in trace
        ], "trace"
    calls = []
    for message in state.messages:
        if message.role == "assistant":
            for call in message.tool_calls or []:
                calls.append({"name": call.function, "input": _parse_args(call.arguments)})
    return calls, "messages"


@metric
def verdict_rates() -> Metric:
    """Share of match scores that are 2, 1, and 0.

    ``generate_eval_stats`` reports these as success, partial, and no-match
    frequencies. Wilson's interval is not attached here; ``ci_wilson`` rejects
    values outside [0, 1] and the match itself is in {0, 1, 2}.
    Wilson, E.B. (1927), "Probable inference", is what ``ci_wilson`` implements
    for the [0, 1] rates elsewhere in Inspect.
    """

    def compute(scores: list[SampleScore]) -> dict[str, float]:
        values = [
            float(sample.score.value)
            for sample in scores
            if isinstance(sample.score.value, int | float) and not math.isnan(float(sample.score.value))
        ]
        if not values:
            return {"success": math.nan, "partial": math.nan, "no_match": math.nan}
        count = len(values)
        return {
            "success": sum(value == 2 for value in values) / count,
            "partial": sum(value == 1 for value in values) / count,
            "no_match": sum(value == 0 for value in values) / count,
        }

    return compute


@metric
def median() -> Metric:
    """Median of the scored values. ``statistics.median``, which is what ``generate_eval_stats`` uses.

    Returned as a one-key dict. Inspect 0.3 labels a scalar metric with
    ``registry_log_name`` (``agentic_evals/median``) and a dict metric with
    the key (``median``). The results grid gives that label a fixed column
    (64px on the task summary), so the package prefix is the whole visible
    header. ``inspect_ai/_eval/task/results.py``, ``scorers_from_metric_dict``.
    """

    def compute(scores: list[SampleScore]) -> dict[str, float]:
        values = [float(sample.score.value) for sample in scores if isinstance(sample.score.value, int | float)]
        if not values:
            return {"median": math.nan}
        return {"median": float(statistics.median(values))}

    return compute


@metric
def by_kind() -> Metric:
    """Mean of this score key within ``solvable``, ``unsolvable``, and ``control``.

    The paper's solvable column (123 tasks) is our ``solvable`` (120) plus
    ``control`` (3).
    """

    def compute(scores: list[SampleScore]) -> dict[str, float]:
        buckets: dict[str, list[float]] = {"solvable": [], "unsolvable": [], "control": []}
        for sample in scores:
            kind = str((sample.sample_metadata or {}).get("kind", ""))
            value = sample.score.value
            if kind in buckets and isinstance(value, int | float) and not math.isnan(float(value)):
                buckets[kind].append(float(value))
        return {kind: (sum(values) / len(values) if values else math.nan) for kind, values in buckets.items()}

    return compute


def _nan_match(model_id: str, reason: str, diff: float, details: dict[str, Any]) -> Score:
    return Score(
        value=match_values(model_id, None, diff),
        explanation=reason,
        metadata=details,
    )


async def generate_judge(judge: Any, messages: list[Any]) -> tuple[Any | None, str | None]:
    """One judge call. Expired 400s are retried here; 429 and 5xx stay inside Inspect."""
    last = ""
    for extra in range(EXPIRED_EXTRA_ATTEMPTS + 1):
        try:
            return await judge.generate(messages, config=JUDGE_GENERATE), None
        except Exception as exc:
            if not is_expired_request(exc):
                raise
            last = f"{type(exc).__name__}: {exc}"
            if extra >= EXPIRED_EXTRA_ATTEMPTS:
                return None, last
            await anyio.sleep(retry_delay(extra + 1))
    return None, last


@scorer(metrics=_match_metrics())
def geobenchx_scorer(model: str | None = None):
    """One judge's match score and the solution-length difference.

    Args:
        model: Judge model. Defaults to the ``grader`` role; a missing grader is
            an error rather than a substituted checkpoint. The result row name
            is this id, or ``str(judge)`` for the grader role.
    """

    async def score(state: TaskState, target: Target) -> Score:
        del target
        md = state.metadata
        references = md.get("reference_solutions") or []
        pred, source = predicted_calls(state)
        candidate_steps = [{"name": call["name"], "arguments": call["input"] or {}, "comment": None} for call in pred]
        diff = metrics.length_diff(len(candidate_steps), references)
        candidate = metrics.solution_code(candidate_steps)
        judge = get_model(model) if model else get_model(role="grader", required=True)
        model_id = model or str(judge)
        details: dict[str, Any] = {
            "call_source": source,
            "candidate_solution": candidate,
            "n_pred": len(candidate_steps),
            "mean_n_ref": (
                sum(len(reference) for reference in references) / len(references) if references else 0.0
            ),
            "kind": md.get("kind"),
            "judge_model": str(judge),
            "judge_prompt": JUDGE_PROMPT_NAME,
        }
        try:
            template = judge_prompt()
        except FileNotFoundError as exc:
            return _nan_match(model_id, str(exc), diff, details)
        # Replace the three placeholders only. str.format would treat braces in a
        # task text as fields; their own .format has the same exposure, and a
        # question that contains a brace should still be scored.
        prompt = (
            template.replace("{task_text}", state.input_text)
            .replace(
                "{reference_solutions}",
                metrics.reference_solutions_text(md.get("reference_solution_description"), references),
            )
            .replace("{candidate_solution}", candidate)
        )
        result, expired = await generate_judge(
            judge,
            [ChatMessageSystem(content=template), ChatMessageUser(content=prompt)],
        )
        if expired:
            return _nan_match(model_id, expired, diff, details)
        verdict, reasoning = metrics.parse_verdict(result.completion or "")
        if verdict is None:
            return _nan_match(model_id, reasoning, diff, details)
        return Score(
            value=match_values(model_id, verdict, diff),
            answer=str(verdict),
            explanation=reasoning,
            metadata=details,
        )

    return score


def extra_match_judges(models: str | list[str]) -> list:
    """Additional match-score scorers, one per model. The grader-role scorer is separate."""
    names = models if isinstance(models, list) else models.split(",")
    return [geobenchx_scorer(model=name.strip()) for name in names if name.strip()]


def _judge_scores(state: TaskState) -> list[dict[str, Any]]:
    found: list[dict[str, Any]] = []
    for existing in (state.scores or {}).values():
        if (existing.metadata or {}).get("judge_prompt") != JUDGE_PROMPT_NAME:
            continue
        if isinstance(existing.value, dict):
            found.append(existing.value)
    return found


@scorer(metrics=_match_metrics())
def geobenchx_merged():
    """Mean of the match judges already stored on this sample.

    Reads ``state.scores`` written by the judge scorers that ran first. Does
    not call a model. ``length_diff`` is omitted: it does not depend on the
    judge and stays on each judge's row.
    """

    async def score(state: TaskState, target: Target) -> Score:
        del target
        judges = _judge_scores(state)
        values = merged_match(judges)
        answered = [
            value
            for judge in judges
            for key, value in judge.items()
            if str(key).endswith("/match") and isinstance(value, int | float) and not math.isnan(float(value))
        ]
        return Score(
            value=values,
            explanation=(
                "arithmetic mean of the match judges on this sample; not a vote and "
                "not a GeoBenchX metric (Krechetova & Kochedykov, Section 5, record one evaluator)"
            ),
            metadata={"protocol": "our_addition", "n_judges": len(judges), "n_match": len(answered)},
        )

    return score


@scorer(metrics={"tao": [mean()], "tio": [mean()], "tem": [mean()], "param_acc": [mean()]})
def geobenchx_steps():
    """Earth-Bench step metrics over GeoBenchX references. Not a GeoBenchX score.

    Max over reference solutions. An empty reference scores 1 only when the
    candidate is empty. A candidate that calls other tools and then
    ``reject_task`` does not get a perfect prefix against a one-step
    ``reject_task`` reference, even though GeoBenchX's judge counts that as a
    full match.
    """

    async def score(state: TaskState, target: Target) -> Score:
        del target
        references = state.metadata.get("reference_solutions") or []
        pred, source = predicted_calls(state)
        steps = [{"name": call["name"], "arguments": call["input"] or {}} for call in pred]
        values = metrics.step_scores(steps, references)
        return Score(
            value=values,
            explanation=(
                "our addition, not a GeoBenchX metric: max TAO/TIO/TEM/parameter accuracy "
                "over reference solutions (Feng et al. Appendix B.2)"
            ),
            metadata={"call_source": source, "protocol": "our_addition"},
        )

    return score
