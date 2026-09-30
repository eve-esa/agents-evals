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
``*/length_diff``). A run that does not have those scores simply omits the
keys.

https://github.com/eve-esa/lm-evaluation-harness/blob/main/lm_eval/loggers/wandb_logger.py
https://docs.wandb.ai/models/track/log/log-summary
"""

from __future__ import annotations

import logging
import math
from typing import Any

from inspect_ai.hooks import Hooks, SampleEnd, TaskEnd, hooks

from agentic_evals.agents.trace_callback import TRACE_KEY

logger = logging.getLogger(__name__)

RESULTS_KEY = "evaluation/eval_results"
SAMPLES_KEY = "evaluation/samples"
RESULTS_COLUMNS = ("task", "metric", "value", "stderr")
SAMPLE_COLUMNS = (
    "sample_id",
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


def summary_metrics(scores: list[Any], benchmark: str) -> dict[str, float]:
    """Stable ``{benchmark}/{metric}`` means. Stderr is not a chart key.

    ``merged/match`` is the match key when that row exists. Otherwise the
    match key is the mean of every ``*/match`` row. ``*/length_diff`` rows
    collapse to one key the same way.
    """
    out: dict[str, float] = {}
    match_values: list[float] = []
    length_values: list[float] = []
    merged_match: float | None = None
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
        elif name.endswith("/match"):
            match_values.append(mean)
        elif name.endswith("/length_diff"):
            length_values.append(mean)
    match = merged_match if merged_match is not None else _mean_of(match_values)
    if match is not None:
        out[f"{benchmark}/match"] = match
    length = _mean_of(length_values)
    if length is not None:
        out[f"{benchmark}/length_diff"] = length
    return out


def results_rows(scores: list[Any], benchmark: str) -> list[dict[str, Any]]:
    """One row per score that has a mean, with stderr when that metric exists."""
    rows: list[dict[str, Any]] = []
    for score in scores:
        metrics = getattr(score, "metrics", None) or {}
        mean = _metric(metrics, "mean")
        if mean is None:
            continue
        rows.append(
            {
                "task": benchmark,
                "metric": str(getattr(score, "name", "")),
                "value": mean,
                "stderr": _metric(metrics, "stderr"),
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


def _tool_names(store: dict[str, Any] | None) -> list[str]:
    entries = (store or {}).get(TRACE_KEY) or []
    names: list[str] = []
    if not isinstance(entries, list):
        return names
    for entry in entries:
        if isinstance(entry, dict) and entry.get("name"):
            names.append(str(entry["name"]))
    return names


def sample_rows(samples: list[Any] | None) -> list[dict[str, Any]]:
    """Per-sample scores plus the tool-name trace from the sample store."""
    rows: list[dict[str, Any]] = []
    for sample in samples or []:
        flat = _score_map(getattr(sample, "scores", None))
        names = _tool_names(getattr(sample, "store", None))
        sample_id = str(getattr(sample, "id", ""))
        epoch = getattr(sample, "epoch", 1)
        if epoch not in (None, 1):
            sample_id = f"{sample_id}:{epoch}"
        metadata = getattr(sample, "metadata", None) or {}
        rows.append(
            {
                "sample_id": sample_id,
                "kind": metadata.get("kind"),
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


def _as_table(columns: tuple[str, ...], rows: list[dict[str, Any]]) -> Any:
    import wandb

    data = [[row.get(column) for column in columns] for row in rows]
    return wandb.Table(columns=list(columns), data=data)


def publish(run: Any, log: Any, samples: Any = None) -> None:
    """Write summary scalars and both tables onto an open ``wandb.Run``."""
    results = getattr(log, "results", None)
    scores = list(getattr(results, "scores", None) or [])
    spec = getattr(log, "eval", None)
    benchmark = benchmark_name(
        getattr(spec, "metadata", None),
        getattr(spec, "task", None),
    )
    summary = summary_metrics(scores, benchmark)
    for key, value in summary.items():
        run.summary[key] = value
    if samples is None:
        samples = getattr(log, "samples", None)
    run.log(
        {
            **summary,
            RESULTS_KEY: _as_table(RESULTS_COLUMNS, results_rows(scores, benchmark)),
            SAMPLES_KEY: _as_table(SAMPLE_COLUMNS, sample_rows(samples)),
        }
    )


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
