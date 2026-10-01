"""Summary metrics and tables on the inspect-wandb run.

Selecting several runs in a W&B workspace draws a comparison only when those
runs log the same summary keys. This hook reads the finished eval and writes
those keys, a results table, and a per-sample trace table onto the run
inspect-wandb already opened. It does not open a second run, and it does not
turn Weave on.

The table shape follows the logger this project used to share a workspace
with: ``lm_eval/loggers/wandb_logger.py`` (``_log_results_as_table``,
``log_eval_result``) in eve-esa/lm-evaluation-harness. Chart keys are the
benchmark name from task metadata plus the score name already stored on the
log (``tao``, ``tio``, ``tem``, ``param_acc``, ``merged/match``,
``*/length_diff``). Each judge's match mean is ``{benchmark}/llm_as_judge/{model}``.
``{benchmark}/llm_as_judge/avg`` is the unweighted mean of those means, and
``n_judges`` is how many went into it. Per judge the summary also carries
``n`` and the absolute counts of match 2, 1, and 0. When the task lists
sample-metadata fields in ``summary_groups``, that whole block is repeated
under ``{benchmark}/{label}/`` for each class (GeoBenchX ``kind`` is
solvable, unsolvable, and control; a label such as ``Task Set 03`` is
``Task_Set_03``). The slash matches the section the Forge summary already
uses for a task; the judge names follow run ``ta8fu0bx``
(``open_ended/llm_as_judge_<model>`` and ``llm_as_judge_avg``).
A run that does not have those scores simply omits the keys.

The same block, including each class and each judge's counts, is logged once.
Logging it again appends a history step, which is why a republished run draws
a curve. ``{benchmark}/llm_as_judge_avg`` and ``{benchmark}/llm_as_judge_{model}``
are copies of those means so the agents workspace can chart them next to
``open_ended``. The run name is the model checkpoint. The readable trace is
the conversation, stored on the ``samples`` table.

https://github.com/eve-esa/lm-evaluation-harness/blob/main/lm_eval/loggers/wandb_logger.py
https://docs.wandb.ai/models/track/log/log-summary
"""

from __future__ import annotations

import json
import logging
import math
import statistics
from typing import Any

from inspect_ai.hooks import Hooks, SampleEnd, TaskEnd, hooks

from agentic_evals.agents.trace_callback import TRACE_KEY

logger = logging.getLogger(__name__)

SAMPLES_KEY = "samples"
METRICS_KEY = "metrics_summary"
METRICS_COLUMNS = ("task", "metric", "value", "stderr", "n")
SAMPLE_COLUMNS = (
    "sample_id",
    "question",
    "final_answer",
    "transcript",
    "kind",
    "tao",
    "tio",
    "tem",
    "param_acc",
    "match",
    "length_diff",
    "n_tool_calls",
    "tool_trace",
)
TRACE_COLUMNS = (
    "sample_id",
    "tool_step",
    "agent",
    "name",
    "input",
    "output",
    "error",
    "error_message",
)
# Concrete types, and no missing cells. A column that is only null is typed
# ``unknown``, and the W&B table viewer then shows no rows. A column named
# ``step`` is also read as the run's history step, which filters the calls out.
TRACE_DTYPES = (str, int, str, str, str, str, bool, str)
_STEP_SCORES = frozenset({"tao", "tio", "tem", "param_acc"})


def benchmark_name(metadata: dict[str, Any] | None, task: str | None) -> str:
    """Prefix for summary keys. Task metadata wins; otherwise the task's last segment."""
    name = (metadata or {}).get("benchmark")
    if isinstance(name, str) and name.strip():
        return name.strip()
    leaf = (task or "eval").rstrip("/").split("/")[-1]
    return leaf or "eval"


def _finite(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, int | float):
        return None
    number = float(value)
    if not math.isfinite(number):
        return None
    return number


def _metric(metrics: dict[str, Any], wanted: str) -> float | None:
    for key, metric in metrics.items():
        name = getattr(metric, "name", key)
        if name != wanted and key != wanted and not str(key).endswith("/" + wanted):
            continue
        raw = getattr(metric, "value", metric)
        number = _finite(raw)
        if number is not None:
            return number
    return None


def _mean_of(values: list[float]) -> float | None:
    if not values:
        return None
    return sum(values) / len(values)


# Leaf labels copied onto ``{benchmark}/llm_as_judge/{model}/<stat>``.
# ``ci`` returns ``lower``/``upper`` (Inspect ``scorer._metrics.std.ci``).
# Verdict shares are GeoBenchX's success / partial / no-match frequencies
# (Krechetova & Kochedykov, Section 5; ``generate_eval_stats``).
_JUDGE_STATS = {
    "stderr": "stderr",
    "median": "median",
    "median/median": "median",
    "success": "success",
    "verdict_rates/success": "success",
    "partial": "partial",
    "verdict_rates/partial": "partial",
    "no_match": "no_match",
    "verdict_rates/no_match": "no_match",
    "lower": "ci_lower",
    "ci/lower": "ci_lower",
    "upper": "ci_upper",
    "ci/upper": "ci_upper",
    "ci_lower": "ci_lower",
    "ci_upper": "ci_upper",
}
_COUNT_FROM_RATE = (("success", "n_success"), ("partial", "n_partial"), ("no_match", "n_no_match"))


def judge_slug(score_name: str) -> str | None:
    """Checkpoint id for a ``*/match`` row. ``merged/match`` is not a judge."""
    if score_name == "merged/match" or not score_name.endswith("/match"):
        return None
    slug = score_name[: -len("/match")].replace("/", "_")
    return slug or None


def _stat_label(metric: Any, key: Any) -> str | None:
    name = str(getattr(metric, "name", key))
    group = getattr(metric, "group", None)
    leaf = f"{group}/{name}" if group and str(group) not in (name, "mean", "stderr") else name
    return _JUDGE_STATS.get(leaf) or _JUDGE_STATS.get(name)


def _judge_fields(metrics: dict[str, Any]) -> dict[str, float]:
    fields: dict[str, float] = {}
    mean = _metric(metrics, "mean")
    if mean is not None:
        fields["mean"] = mean
    for key, metric in metrics.items():
        label = _stat_label(metric, key)
        number = _finite(getattr(metric, "value", metric))
        if label is None or number is None:
            continue
        fields[label] = number
    return fields


def summary_metrics(scores: list[Any], benchmark: str) -> dict[str, int | float]:
    """Stable summary keys, including one block per judge.

    ``{benchmark}/llm_as_judge/{model}`` is that judge's mean match.
    ``{benchmark}/llm_as_judge/avg`` is the unweighted arithmetic mean of
    those means (each reported judge once, not a second pass over samples).
    ``n_judges`` is the number of means in that average.

    ``{benchmark}/match`` stays the ``merged/match`` mean when that row
    exists, which is the mean of the per-sample judge average. It equals
    ``llm_as_judge/avg`` when every judge scored every sample.

    Absolute counts: ``n`` is ``scored_samples``. ``n_success``,
    ``n_partial``, and ``n_no_match`` are ``round(share * n)`` for match
    values 2, 1, and 0. The scorer stores those as shares
    (``verdict_rates``), so the count is recovered from the share and ``n``.
    """
    out: dict[str, int | float] = {}
    match_values: list[float] = []
    length_values: list[float] = []
    merged_match: float | None = None
    judge_means: list[float] = []
    for score in scores:
        name = str(getattr(score, "name", ""))
        metrics = getattr(score, "metrics", None) or {}
        mean = _metric(metrics, "mean")
        if mean is None:
            continue
        if name in _STEP_SCORES:
            out[f"{benchmark}/{name}"] = mean
        elif name == "merged/match":
            merged_match = mean
        elif (slug := judge_slug(name)) is not None:
            match_values.append(mean)
            fields = _judge_fields(metrics)
            prefix = f"{benchmark}/llm_as_judge/{slug}"
            out[prefix] = fields["mean"]
            judge_means.append(fields["mean"])
            for label, value in fields.items():
                if label != "mean":
                    out[f"{prefix}/{label}"] = value
            n = getattr(score, "scored_samples", None)
            if isinstance(n, int):
                out[f"{prefix}/n"] = n
                for rate_name, count_name in _COUNT_FROM_RATE:
                    rate = fields.get(rate_name)
                    if rate is not None:
                        out[f"{prefix}/{count_name}"] = int(round(rate * n))
        elif name.endswith("/length_diff"):
            length_values.append(mean)
    match = merged_match if merged_match is not None else _mean_of(match_values)
    if match is not None:
        out[f"{benchmark}/match"] = match
    if judge_means:
        out[f"{benchmark}/llm_as_judge/avg"] = sum(judge_means) / len(judge_means)
        out[f"{benchmark}/llm_as_judge/n_judges"] = len(judge_means)
    length = _mean_of(length_values)
    if length is not None:
        out[f"{benchmark}/length_diff"] = length
    return out


def _slug(value: str) -> str:
    """One path segment. Spaces and punctuation become a single underscore."""
    chars: list[str] = []
    underscore = False
    for char in value.strip():
        if char.isalnum() or char in "-_":
            chars.append(char)
            underscore = False
        elif not underscore:
            chars.append("_")
            underscore = True
    return "".join(chars).strip("_")


def _class_labels(sample: Any, field: str) -> list[str]:
    raw = (getattr(sample, "metadata", None) or {}).get(field)
    if isinstance(raw, str) and raw.strip():
        return [raw.strip()]
    if isinstance(raw, list):
        return [str(item).strip() for item in raw if isinstance(item, str) and str(item).strip()]
    return []


def _subset_summary(flats: list[dict[str, float]], prefix: str) -> dict[str, int | float]:
    """The same means and counts as the full run, on this class only.

    Counts are taken from the samples in the class. Match 2, 1, and 0 are
    ``n_success``, ``n_partial``, and ``n_no_match``.
    """
    out: dict[str, int | float] = {f"{prefix}/n": len(flats)}
    for step in _STEP_SCORES:
        mean = _mean_of([row[step] for row in flats if step in row])
        if mean is not None:
            out[f"{prefix}/{step}"] = mean
    judge_keys: list[str] = []
    seen: set[str] = set()
    for row in flats:
        for key in row:
            if key.endswith("/match") and key != "merged/match" and key not in seen:
                seen.add(key)
                judge_keys.append(key)
    judge_means: list[float] = []
    for key in judge_keys:
        slug = judge_slug(key)
        values = [row[key] for row in flats if key in row]
        if slug is None or not values:
            continue
        mean = sum(values) / len(values)
        judge_means.append(mean)
        block = f"{prefix}/llm_as_judge/{slug}"
        n_success = sum(value == 2 for value in values)
        n_partial = sum(value == 1 for value in values)
        n_no_match = sum(value == 0 for value in values)
        n = len(values)
        out[block] = mean
        out[f"{block}/n"] = n
        out[f"{block}/n_success"] = n_success
        out[f"{block}/n_partial"] = n_partial
        out[f"{block}/n_no_match"] = n_no_match
        out[f"{block}/success"] = n_success / n
        out[f"{block}/partial"] = n_partial / n
        out[f"{block}/no_match"] = n_no_match / n
        out[f"{block}/median"] = float(statistics.median(values))
    if judge_means:
        out[f"{prefix}/llm_as_judge/avg"] = sum(judge_means) / len(judge_means)
        out[f"{prefix}/llm_as_judge/n_judges"] = len(judge_means)
    merged = [row["merged/match"] for row in flats if "merged/match" in row]
    if merged:
        out[f"{prefix}/match"] = sum(merged) / len(merged)
    elif judge_means:
        out[f"{prefix}/match"] = sum(judge_means) / len(judge_means)
    lengths: list[float] = []
    for row in flats:
        diffs = [value for key, value in row.items() if key.endswith("/length_diff")]
        if diffs:
            lengths.append(sum(diffs) / len(diffs))
    if lengths:
        out[f"{prefix}/length_diff"] = sum(lengths) / len(lengths)
    return out


def grouped_metrics(samples: list[Any] | None, benchmark: str, fields: list[str]) -> dict[str, int | float]:
    """Repeat the summary inside each class declared by ``summary_groups``.

    A string metadata field puts the sample in one class. A list field puts
    it in each entry, so a task with several labels is counted in each.
    The key is ``{benchmark}/{label}``. The field name is not part of the path.
    A sample that carries the same label from two fields is counted once.
    """
    buckets: dict[str, list[dict[str, float]]] = {}
    for sample in samples or []:
        flat = _score_map(getattr(sample, "scores", None))
        seen: set[str] = set()
        for field in fields:
            for label in _class_labels(sample, field):
                slug = _slug(label)
                if slug and slug not in seen:
                    seen.add(slug)
                    buckets.setdefault(slug, []).append(flat)
    out: dict[str, int | float] = {}
    for label, flats in buckets.items():
        out.update(_subset_summary(flats, f"{benchmark}/{label}"))
    return out


def dashboard_aliases(summary: dict[str, int | float], benchmark: str) -> dict[str, int | float]:
    """Copies of the judge means in the shape the agents workspace already charts.

    ``open_ended/llm_as_judge_avg`` and ``open_ended/llm_as_judge_{model}`` use
    an underscore. The slash keys stay as well.
    """
    out: dict[str, int | float] = {}
    avg = summary.get(f"{benchmark}/llm_as_judge/avg")
    if avg is not None:
        out[f"{benchmark}/llm_as_judge_avg"] = avg
    judge_prefix = f"{benchmark}/llm_as_judge/"
    for key, value in summary.items():
        if not key.startswith(judge_prefix):
            continue
        rest = key[len(judge_prefix) :]
        if rest and "/" not in rest and rest not in ("avg", "n_judges"):
            out[f"{benchmark}/llm_as_judge_{rest}"] = value
    return out


def model_run_name(model: str | None) -> str:
    """Run name: the checkpoint being tested, without the provider prefix."""
    leaf = (model or "model").rstrip("/").split("/")[-1].strip()
    return leaf or "model"


def results_rows(scores: list[Any], benchmark: str) -> list[dict[str, Any]]:
    """One row per score that has a mean, with stderr when that metric exists."""
    rows: list[dict[str, Any]] = []
    for score in scores:
        metrics = getattr(score, "metrics", None) or {}
        mean = _metric(metrics, "mean")
        if mean is None:
            continue
        scored = getattr(score, "scored_samples", None)
        rows.append(
            {
                "task": benchmark,
                "metric": str(getattr(score, "name", "")),
                "value": mean,
                "stderr": _metric(metrics, "stderr"),
                "n": int(scored) if isinstance(scored, int) else 0,
            }
        )
    return rows


def _score_map(scores: dict[str, Any] | None) -> dict[str, float]:
    flat: dict[str, float] = {}
    for score in (scores or {}).values():
        value = getattr(score, "value", score)
        if isinstance(value, dict):
            for key, item in value.items():
                number = _finite(item)
                if number is not None:
                    flat[str(key)] = number
        else:
            number = _finite(value)
            name = getattr(score, "name", None)
            if number is not None and isinstance(name, str):
                flat[name] = number
    return flat


def _named(flat: dict[str, float], leaf: str) -> float | None:
    if leaf in flat:
        return flat[leaf]
    values = [value for key, value in flat.items() if key == leaf or key.endswith("/" + leaf)]
    return _mean_of(values)


def _text_cell(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    return json.dumps(value, default=str, ensure_ascii=False)


def _trace_entries(store: Any) -> list[dict[str, Any]]:
    raw = store.get(TRACE_KEY) if store is not None else None
    if not isinstance(raw, list):
        return []
    return [entry for entry in raw if isinstance(entry, dict)]


def trace_rows(samples: list[Any] | None) -> list[dict[str, Any]]:
    """One row per tool call, including arguments and the tool output."""
    rows: list[dict[str, Any]] = []
    for sample in samples or []:
        sample_id = _sample_id(sample)
        for entry in _trace_entries(getattr(sample, "store", None)):
            rows.append(
                {
                    "sample_id": sample_id,
                    "tool_step": int(entry["step"]) if isinstance(entry.get("step"), int) else 0,
                    "agent": _text_cell(entry.get("agent")),
                    "name": _text_cell(entry.get("name")),
                    "input": _text_cell(entry.get("input")),
                    "output": _text_cell(entry.get("output")),
                    "error": bool(entry.get("error")),
                    "error_message": _text_cell(entry.get("error_message")),
                }
            )
    return rows


def trace_records(samples: list[Any] | None) -> list[dict[str, Any]]:
    """Per-sample tool traces for the ``traces`` artifact."""
    records: list[dict[str, Any]] = []
    for sample in samples or []:
        metadata = getattr(sample, "metadata", None) or {}
        records.append(
            {
                "sample_id": _sample_id(sample),
                "kind": metadata.get("kind"),
                "labels": metadata.get("labels"),
                "transcript": render_transcript(sample),
                "trace": _trace_entries(getattr(sample, "store", None)),
            }
        )
    return records


def _message_text(message: Any) -> str:
    text = getattr(message, "text", None)
    if isinstance(text, str):
        return text.strip()
    if isinstance(message, dict):
        content = message.get("text") or message.get("content") or ""
        return content.strip() if isinstance(content, str) else ""
    content = getattr(message, "content", "")
    return content.strip() if isinstance(content, str) else ""


def render_transcript(sample: Any) -> str:
    """The conversation the model produced, including tool calls and their results.

    Tool names alone are not the trace. Each assistant turn keeps its text and
    the calls it made, and each tool message keeps the text the model saw back.
    """
    blocks: list[str] = []
    for message in getattr(sample, "messages", None) or []:
        role = str(getattr(message, "role", None) or (message.get("role") if isinstance(message, dict) else "message"))
        text = _message_text(message)
        calls = getattr(message, "tool_calls", None)
        if calls is None and isinstance(message, dict):
            calls = message.get("tool_calls") or []
        calls = calls or []
        if role == "tool":
            name = getattr(message, "function", None) or (message.get("function") if isinstance(message, dict) else None) or "tool"
            blocks.append(f"[tool {name}]\n{text}")
            continue
        if text:
            blocks.append(f"[{role}]\n{text}")
        for call in calls:
            name = getattr(call, "function", None) or (call.get("function") if isinstance(call, dict) else None) or "tool"
            args = getattr(call, "arguments", None) if not isinstance(call, dict) else call.get("arguments")
            blocks.append(f"[call {name}]\n{_text_cell(args)}")
    return "\n\n".join(blocks)


def _user_question(sample: Any) -> str:
    for message in getattr(sample, "messages", None) or []:
        role = getattr(message, "role", None) or (message.get("role") if isinstance(message, dict) else None)
        if role == "user":
            return _message_text(message)
    return ""


def _final_answer(sample: Any) -> str:
    answer = ""
    for message in getattr(sample, "messages", None) or []:
        role = getattr(message, "role", None) or (message.get("role") if isinstance(message, dict) else None)
        if role == "assistant":
            text = _message_text(message)
            if text:
                answer = text
    return answer


def _sample_id(sample: Any) -> str:
    sample_id = str(getattr(sample, "id", ""))
    epoch = getattr(sample, "epoch", 1)
    if epoch not in (None, 1):
        return f"{sample_id}:{epoch}"
    return sample_id


def sample_rows(samples: list[Any] | None) -> list[dict[str, Any]]:
    """Per-sample scores, the tool-name chain, and the full tool-call trace."""
    rows: list[dict[str, Any]] = []
    for sample in samples or []:
        flat = _score_map(getattr(sample, "scores", None))
        entries = _trace_entries(getattr(sample, "store", None))
        names = [str(entry["name"]) for entry in entries if entry.get("name")]
        metadata = getattr(sample, "metadata", None) or {}
        rows.append(
            {
                "sample_id": _sample_id(sample),
                "question": _user_question(sample),
                "final_answer": _final_answer(sample),
                "transcript": render_transcript(sample),
                "kind": _text_cell(metadata.get("kind")),
                "tao": flat.get("tao"),
                "tio": flat.get("tio"),
                "tem": flat.get("tem"),
                "param_acc": flat.get("param_acc"),
                "match": flat.get("merged/match", _named(flat, "match")),
                "length_diff": _named(flat, "length_diff"),
                "n_tool_calls": len(names),
                "tool_trace": " -> ".join(names),
            }
        )
    return rows


def samples_for_table(log_samples: Any, collected: list[Any]) -> list[Any]:
    """Samples to put in the trace table.

    The task-end log often carries results without the sample bodies (they
    stay in the ``.eval`` zip until something reads them). ``on_sample_end``
    has already handed us each ``EvalSample``, store and scores included, so
    that list is the one the table uses when it is non-empty.
    """
    if collected:
        return list(collected)
    return list(log_samples or [])


def _as_table(
    columns: tuple[str, ...],
    rows: list[dict[str, Any]],
    dtypes: tuple[type, ...] | None = None,
) -> Any:
    import wandb

    data = [[row.get(column) for column in columns] for row in rows]
    if dtypes is None:
        return wandb.Table(columns=list(columns), data=data)
    return wandb.Table(columns=list(columns), data=data, dtype=list(dtypes), optional=False)


def _drop_stale_summary(run: Any, benchmark: str) -> None:
    """Remove the previous ``{benchmark}/by/...`` keys from an updated run.

    W&B stores a slash key either as one string or as nested sections. Both
    shapes are removed.
    """
    prefix = f"{benchmark}/by/"
    for key in list(run.summary.keys()):
        text = str(key)
        if text.startswith(prefix):
            del run.summary[key]
            continue
        if text != benchmark:
            continue
        node = run.summary[key]
        nested = getattr(node, "keys", None)
        if nested is None or "by" not in list(nested()):
            continue
        del node["by"]


def _log_trace_artifact(run: Any, benchmark: str, samples: list[Any] | None) -> None:
    import wandb

    artifact = wandb.Artifact(f"{benchmark}-traces", type="traces")
    rows = trace_rows(samples)
    if rows:
        artifact.add(_as_table(TRACE_COLUMNS, rows, TRACE_DTYPES), "traces")
    with artifact.new_file("traces.json", mode="w", encoding="utf-8") as handle:
        json.dump(trace_records(samples), handle, default=str, ensure_ascii=False)
    run.log_artifact(artifact)


def publish(run: Any, log: Any, samples: Any = None) -> None:
    """Write the summary, log the workspace charts once, and store the conversation.

    ``run.log`` is one call. Calling it again on the same run appends a step,
    and the workspace then draws those numbers as a history.
    """
    results = getattr(log, "results", None)
    scores = list(getattr(results, "scores", None) or [])
    spec = getattr(log, "eval", None)
    benchmark = benchmark_name(
        getattr(spec, "metadata", None),
        getattr(spec, "task", None),
    )
    summary = summary_metrics(scores, benchmark)
    if samples is None:
        samples = getattr(log, "samples", None)
    groups = (getattr(spec, "metadata", None) or {}).get("summary_groups") or []
    if isinstance(groups, str):
        groups = [groups]
    summary.update(grouped_metrics(samples, benchmark, [str(field) for field in groups]))
    summary.update(dashboard_aliases(summary, benchmark))
    name = model_run_name(getattr(spec, "model", None))
    current = getattr(run, "name", None)
    if not current or current == getattr(run, "id", None):
        run.name = name
    _drop_stale_summary(run, benchmark)
    for key, value in summary.items():
        run.summary[key] = value
    # One history row for the whole block, including class splits and judge
    # stats. A second log appends a step.
    run.log(
        {
            **summary,
            SAMPLES_KEY: _as_table(SAMPLE_COLUMNS, sample_rows(samples)),
            METRICS_KEY: _as_table(METRICS_COLUMNS, results_rows(scores, benchmark)),
        }
    )
    _log_trace_artifact(run, benchmark, samples)


@hooks(
    name="wandb_compare",
    description="Log shared summary metrics and trace tables onto the W&B run.",
)
class WandbCompare(Hooks):
    async def on_sample_end(self, data: SampleEnd) -> None:
        # The finished log passed to on_task_end does not always include
        # sample bodies. Keep the ones SampleEnd already has.
        collected = getattr(self, "_samples", None)
        if collected is None:
            collected = {}
            self._samples = collected
        collected.setdefault(data.eval_id, []).append(data.sample)

    async def on_task_end(self, data: TaskEnd) -> None:
        # inspect-wandb opens the run in on_task_start and finishes it in
        # on_run_end, so the run is still open here.
        try:
            import wandb
        except ImportError:
            return
        run = wandb.run
        if run is None:
            return
        collected = getattr(self, "_samples", {}).pop(data.eval_id, [])
        try:
            publish(
                run,
                data.log,
                samples_for_table(getattr(data.log, "samples", None), collected),
            )
        except Exception:
            logger.exception("W&B comparison tables were not logged")
