"""GeoBenchX scorers.

Official protocol (arXiv:2503.18129 Section 5 and 6.1; ``geobenchx/evaluation.py``
``generate_eval_stats``): ``geobenchx_scorer`` renders the trace with
``get_solution_code`` and asks one judge for a match score of 0 (no match),
1 (partial), or 2 (match). Each judge is its own scorer. The published
aggregate is the share of each of those three scores, ``count / n`` over the
tasks that judge scored. Section 5.2 averages those shares across judges,
each judge one term. A skipped task stays out of that judge's denominator.
There is no majority vote, and the mean of the raw 0/1/2 scores is not a
published number. The default judge is Inspect's ``grader`` role.
``length_diff`` is computed whether or not the judge answers.

Score keys use the checkpoint id (``qwen/qwen3.8-27b/match``), without the
``openai-api`` provider prefix, because Inspect names each dict-metric row
after the key (``inspect_ai._eval.task.results.scorers_from_metric_dict``).
A sample stores a 0/1 indicator for each category, so ``mean()`` is the
share from ``generate_eval_stats``. The raw 0, 1, or 2 stays on ``answer``
and in score metadata. The viewer's three summary rows are
``geobenchx_panel`` (no match, partial, match): the unweighted mean of the
judges' shares. Wilson's interval (Wilson, E.B., 1927, "Probable inference")
is Inspect's ``ci_wilson``, used because the indicators are in {0, 1}.
Upstream's interval is ``scipy.stats.binomtest``, which this project does
not depend on.

The judge call uses Inspect's exponential backoff (``model_retry_config``:
``wait_exponential_jitter``, initial 3s, cap 30 min) with ``max_retries=6``.
``stream_idle_timeout`` expires an attempt only after the stream has gone
quiet; it is not a wall-clock cap on a long generation
(``inspect_ai.model._stream.arm_stall_scope``). A 400 whose body says the
request expired is not classified as retryable by ``openai_classify_retry``,
so those are retried here.

``geobenchx_merged`` averages the indicators on one sample. That run mean
equals Section 5.2 only when every judge scored every task, so it is not the
panel number. ``geobenchx_panel`` is. ``geobenchx_steps`` is our addition;
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
    ci_wilson,
    mean,
    metric,
    scorer,
    stderr,
)
from inspect_ai.solver import TaskState

from agentic_evals.agents.trace_callback import recorded_trace
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


# Table order in Krechetova & Kochedykov, Section 6: no match, partial, match.
SHARE_CATEGORIES = ("no_match", "partial", "match")
_VERDICT_FOR = {"no_match": 0, "partial": 1, "match": 2}
# The paper's solvable column (123 tasks) is our solvable (120) plus control (3).
PAPER_SOLVABLE = "solvable_incl_control"


def match_values(model_id: str, verdict: int | None, diff: float) -> dict[str, float]:
    """One judge's score dict. Keys are what the results table shows as the row name.

    Each category is 1 only for its verdict, so the mean of a key is that
    category's share (``generate_eval_stats``). A missing verdict is NaN on
    all three, which Inspect leaves out of the mean, so a failure is not a
    no-match. The raw 0/1/2 is not stored here.
    """
    name = display_model_id(model_id)
    values: dict[str, float] = {}
    for category, code in _VERDICT_FOR.items():
        if verdict is None:
            values[f"{name}/{category}"] = math.nan
        else:
            values[f"{name}/{category}"] = float(verdict == code)
    values[f"{name}/length_diff"] = diff
    return values


def merged_match(judge_values: list[dict[str, Any]]) -> dict[str, float]:
    """Mean of the judges' indicators on one sample. NaNs are left out.

    This is not Section 5.2. That average is taken over each judge's share,
    and the two agree only when every judge scored every task. ``length_diff``
    is not averaged.
    """
    buckets: dict[str, list[float]] = {category: [] for category in SHARE_CATEGORIES}
    for values in judge_values:
        for key, value in values.items():
            text = str(key)
            if text.startswith("merged/"):
                continue
            for category in SHARE_CATEGORIES:
                if not text.endswith("/" + category):
                    continue
                if isinstance(value, int | float) and not math.isnan(float(value)):
                    buckets[category].append(float(value))
    return {
        f"merged/{category}": (sum(numbers) / len(numbers) if numbers else math.nan)
        for category, numbers in buckets.items()
    }


def _rate_metrics() -> list[Metric]:
    # mean of a 0/1 indicator is the share. ci_wilson is the [0, 1] interval
    # (Wilson 1927). stderr is Inspect's. by_kind splits the same share.
    return [mean(), stderr(), ci_wilson(), by_kind()]


def _length_metrics() -> list[Metric]:
    # mean and median: generate_eval_stats. stderr and ci are Inspect's
    # (95% t interval). Their binomial interval does not apply to a signed ratio.
    return [mean(), median(), stderr(), ci(), by_kind()]


def _match_metrics() -> dict[str, list[Metric]]:
    return {
        "*/no_match": _rate_metrics(),
        "*/partial": _rate_metrics(),
        "*/match": _rate_metrics(),
        "*/length_diff": _length_metrics(),
    }


def _merged_metrics() -> dict[str, list[Metric]]:
    # The mean of these per-sample averages is not the panel share.
    return {f"merged/{category}": [mean()] for category in SHARE_CATEGORIES}


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
    trace = recorded_trace(state.store)
    if trace is not None:
        return [
            {"name": entry["name"], "input": _parse_args(entry["input"])} for entry in trace
        ], "trace"
    calls = []
    for message in state.messages:
        if message.role == "assistant":
            for call in message.tool_calls or []:
                calls.append({"name": call.function, "input": _parse_args(call.arguments)})
    return calls, "messages"


def panel_share(scores: list[SampleScore], category: str, kinds: set[str] | None = None) -> float:
    """Unweighted mean of each judge's share of ``category``.

    Section 5.2. A judge's share is ``count / n`` over the tasks in ``kinds``
    that judge scored (``generate_eval_stats``). A task that judge skipped
    is left out of that denominator and does not change the other judges.
    Verdicts are read from score metadata, not from the placeholder value.
    """
    target = _VERDICT_FOR[category]
    by_judge: dict[str, list[int]] = {}
    for sample in scores:
        kind = str((sample.sample_metadata or {}).get("kind", ""))
        if kinds is not None and kind not in kinds:
            continue
        verdicts = (sample.score.metadata or {}).get("verdicts")
        if not isinstance(verdicts, dict):
            continue
        for judge, verdict in verdicts.items():
            if isinstance(verdict, bool) or not isinstance(verdict, int):
                continue
            by_judge.setdefault(str(judge), []).append(verdict)
    shares = [sum(verdict == target for verdict in values) / len(values) for values in by_judge.values() if values]
    if not shares:
        return math.nan
    return sum(shares) / len(shares)


@metric
def panel_rate(category: str) -> Metric:
    """Section 5.2 panel share for one category, overall and by task kind.

    The returned ``mean`` is the panel share. Inspect labels a dict metric
    by its key, so the results row shows ``mean`` rather than the package
    prefix (``scorers_from_metric_dict``).
    """

    def compute(scores: list[SampleScore]) -> dict[str, float]:
        return {
            "mean": panel_share(scores, category),
            "solvable": panel_share(scores, category, {"solvable"}),
            "unsolvable": panel_share(scores, category, {"unsolvable"}),
            "control": panel_share(scores, category, {"control"}),
            PAPER_SOLVABLE: panel_share(scores, category, {"solvable", "control"}),
        }

    return compute


def _panel_metrics() -> dict[str, list[Metric]]:
    return {category: [panel_rate(category)] for category in SHARE_CATEGORIES}


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

    ``solvable_incl_control`` is the paper's solvable column: our solvable
    (120) plus control (3), 123 tasks. On a 0/1 indicator the mean is the
    share inside that subset.
    """

    def compute(scores: list[SampleScore]) -> dict[str, float]:
        buckets: dict[str, list[float]] = {
            "solvable": [],
            "unsolvable": [],
            "control": [],
            PAPER_SOLVABLE: [],
        }
        for sample in scores:
            kind = str((sample.sample_metadata or {}).get("kind", ""))
            value = sample.score.value
            if kind not in ("solvable", "unsolvable", "control"):
                continue
            if not isinstance(value, int | float) or math.isnan(float(value)):
                continue
            number = float(value)
            buckets[kind].append(number)
            if kind in ("solvable", "control"):
                buckets[PAPER_SOLVABLE].append(number)
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
        details["verdict"] = verdict
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


def judge_verdicts(state: TaskState) -> dict[str, int]:
    """Raw 0/1/2 for each judge that answered this sample, keyed by checkpoint id."""
    found: dict[str, int] = {}
    for existing in (state.scores or {}).values():
        metadata = existing.metadata or {}
        if metadata.get("judge_prompt") != JUDGE_PROMPT_NAME:
            continue
        verdict = metadata.get("verdict")
        if isinstance(verdict, bool) or not isinstance(verdict, int):
            continue
        value = existing.value if isinstance(existing.value, dict) else {}
        for key in value:
            text = str(key)
            if text.endswith("/match") and not text.startswith("merged/"):
                found[text[: -len("/match")]] = verdict
                break
    return found


@scorer(metrics=_merged_metrics())
def geobenchx_merged():
    """Mean of the judges' indicators on this sample.

    Reads ``state.scores`` written by the judge scorers that ran first. Does
    not call a model. ``length_diff`` is omitted: it does not depend on the
    judge and stays on each judge's row. This is not the Section 5.2 panel
    share; ``geobenchx_panel`` is.
    """

    async def score(state: TaskState, target: Target) -> Score:
        del target
        judges = _judge_scores(state)
        values = merged_match(judges)
        answered = judge_verdicts(state)
        return Score(
            value=values,
            explanation=(
                "mean of the judges' 0/1 indicators on this sample; not a vote and "
                "not the Section 5.2 panel share, which averages each judge's frequency"
            ),
            metadata={"protocol": "our_addition", "n_judges": len(judges), "n_match": len(answered)},
        )

    return score


@scorer(metrics=_panel_metrics())
def geobenchx_panel():
    """Unweighted mean of each judge's no-match, partial, and match shares.

    Krechetova & Kochedykov, Section 5.2. The per-sample value is a placeholder.
    ``panel_rate`` reads the raw verdicts from metadata and ignores it.
    """

    async def score(state: TaskState, target: Target) -> Score:
        del target
        verdicts = judge_verdicts(state)
        return Score(
            value={category: 0.0 for category in SHARE_CATEGORIES},
            explanation=(
                "panel share is the unweighted mean of each judge's frequency "
                "(Krechetova & Kochedykov, Section 5.2), not the mean of these placeholders"
            ),
            metadata={"protocol": "geobenchx_panel", "verdicts": verdicts, "n_judges": len(verdicts)},
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
