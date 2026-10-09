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
log (``tao``, ``tio``, ``tem``, ``param_acc``, ``*/length_diff``).
GeoBenchX reports the share of each verdict (Krechetova & Kochedykov,
Section 6.1). ``{benchmark}/llm_as_judge/{model}`` is that judge's match
share. ``{benchmark}/match`` and ``{benchmark}/llm_as_judge/avg`` are the
unweighted mean of the judges' match shares (Section 5.2), and the same
for partial and no-match. ``merged/match`` is a per-sample average and is
not that panel number. Per judge the summary also carries ``n`` and the
absolute counts of match, partial, and no-match. When the task lists
sample-metadata fields in ``summary_groups``, that whole block is repeated
under ``{benchmark}/{label}/`` for each class (GeoBenchX ``kind`` is
solvable, unsolvable, and control; a label such as ``Task Set 03`` is
``Task_Set_03``). Earth-Bench's regime is a section of its own,
``earth_bench/AP`` or ``earth_bench/IF`` (Feng et al., Section 4), and a
modality split is nested under it (``earth_bench/AP/spectrum/accuracy``).
OpenEarthAgent's two protocols are the same kind of section,
``openearthagent/step`` or ``openearthagent/e2e`` (Shabbir et al.,
Section 4.1; ``task_type`` in the pinned ``evaluate_function``). Task type
is nested under that (``openearthagent/step/answer/summ``).
A task's ``published_metrics`` is the only score names copied into that
summary. The slash matches the section the Forge summary already
uses for a task; the judge names follow run ``ta8fu0bx``
(``open_ended/llm_as_judge_<model>`` and ``llm_as_judge_avg``).
A run that does not have those scores simply omits the keys.

Other score names are written as they are stored. A metric whose per-sample
values are the same on every scorer is one key, ``{benchmark}/{metric}``.
A metric that differs across scorers is one key per judge,
``{benchmark}/{judge}/{metric}``, using the last segment of ``judge_model``.
OpenEarthAgent writes those keys under ``step`` or ``e2e``. Its ``answer``
field is the table column: ``summ`` in step mode (Table 3) and ``ans`` end
to end (Table 4). NaNs are left out of the mean. The same split is repeated
under each ``summary_groups`` label.

The same block, including each class and each judge's counts, is logged once.
Logging it again appends a history step, which is why a republished run draws
a curve. ``{benchmark}/llm_as_judge_avg`` and ``{benchmark}/llm_as_judge_{model}``
are copies of those means so the agents workspace can chart them next to
``open_ended``. The run name is the model checkpoint plus the benchmark
(``qwen3.8-27b_geobenchx``). The ``traces`` table is one row per sample,
logged in that same call: the sample fields, the question, the target, the
raw tool trace, and the conversation with a marker at each turn. Its metric
columns are the scores that benchmark recorded. A run that did not score
``tao`` does not get a ``tao`` column. A second
``run.log`` is not used for it, because that appends a history step.
The same call logs ``metric_guide``, an HTML note for this run's regime,
modality, and judges, and copies that markdown onto ``run.notes``.

https://github.com/eve-esa/lm-evaluation-harness/blob/main/lm_eval/loggers/wandb_logger.py
https://docs.wandb.ai/models/track/log/log-summary
"""

from __future__ import annotations

import json
import logging
import math
import re
import statistics
from typing import Any

from inspect_ai.hooks import Hooks, SampleEnd, TaskEnd, TaskStart, hooks

from agentic_evals.agents.trace_callback import TRACE_KEY

logger = logging.getLogger(__name__)

SAMPLES_KEY = "samples"
TRACES_KEY = "traces"
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
# Shabbir et al., arXiv:2602.17665, Tables 3 and 4. Earth-Bench still logs these.
_OPENEARTHAGENT_OMITTED = _STEP_SCORES | frozenset({"trajectory_accuracy"})


def benchmark_name(metadata: dict[str, Any] | None, task: str | None) -> str:
    """Prefix for summary keys. Task metadata wins; otherwise the task's last segment."""
    name = (metadata or {}).get("benchmark")
    if isinstance(name, str) and name.strip():
        return name.strip()
    leaf = (task or "eval").rstrip("/").split("/")[-1]
    return leaf or "eval"


# Earth-Bench reports Auto-Planning and Instruction-Following as separate
# columns (Feng et al., ICLR 2026, arXiv:2509.23141, Section 4 and Appendix
# B.2). The workspace section is the paper's label, ``AP`` or ``IF``.
_REGIME_SECTION = {"ap": "AP", "if": "IF"}
# OpenEarthAgent reports step-by-step and end-to-end as separate protocols
# (Shabbir et al., arXiv:2602.17665, Section 4.1). The pinned scorer branches
# on ``task_type`` ``step`` or ``e2e``. The section uses those names.
_OPENEARTHAGENT_REGIME = {"step": "step", "e2e": "e2e"}


def metric_root(benchmark: str, regime: str | None) -> str:
    """Prefix for one run's summary keys.

    Earth-Bench nests its protocol (``earth_bench/AP``, ``earth_bench/IF``).
    OpenEarthAgent nests its protocol (``openearthagent/step``,
    ``openearthagent/e2e``). Any other regime stays off the path. The
    protocol still renames OpenEarthAgent's answer column (``summ`` or ``ans``).
    """
    key = (regime or "").strip().lower()
    if benchmark == "openearthagent":
        label = _OPENEARTHAGENT_REGIME.get(key)
    else:
        label = _REGIME_SECTION.get(key)
    if not label:
        return benchmark
    return f"{benchmark}/{label}"


def regime_label(regime: str | None) -> str | None:
    """Paper label for a regime, or the stored string when it has no label."""
    text = (regime or "").strip()
    if not text:
        return None
    return _REGIME_SECTION.get(text.lower(), text)


def _listed(value: Any) -> str:
    if isinstance(value, list):
        parts = [str(item).strip() for item in value if str(item).strip()]
    elif isinstance(value, str):
        parts = [part.strip() for part in value.split(",") if part.strip()]
    else:
        parts = []
    return ", ".join(f"`{part}`" for part in parts)


def _judge_note(metadata: dict[str, Any], official: str) -> str:
    """How this run's judges sit next to the official score."""
    lines = [official]
    judges = _listed(metadata.get("judges"))
    if judges:
        lines.append(
            f"This run's extra answer judges are {judges}. "
            "Each judge keeps its own key, separate from the official score."
        )
    else:
        lines.append(
            "Extra answer judges, when a run sets them, each keep their own key, "
            "separate from the official score."
        )
    trajectory = _listed(metadata.get("trajectory_judges"))
    if "trajectory_judges" not in metadata:
        return " ".join(lines)
    if trajectory:
        lines.append(
            f"This run's trajectory judges are {trajectory}. "
            "They are our addition and stay off the published chart."
        )
    else:
        lines.append("Trajectory judges are optional. When a run sets them, they stay off the published chart.")
    return " ".join(lines)


def _earth_bench_guide(metadata: dict[str, Any]) -> str:
    regime = str(metadata.get("regime") or "ap").strip().lower()
    label = _REGIME_SECTION.get(regime, regime.upper() or "AP")
    modality = str(metadata.get("modality") or "all").strip() or "all"
    if regime == "ap":
        this = "This run is **AP** (Auto-Planning): the agent chooses the tools."
        sibling = "**IF** (Instruction-Following) is the other regime: the question lists the steps. It is a separate run."
    elif regime == "if":
        this = "This run is **IF** (Instruction-Following): the question lists the steps."
        sibling = "**AP** (Auto-Planning) is the other regime: the agent chooses the tools. It is a separate run."
    else:
        this = f"This run's regime is **{label}**."
        sibling = (
            "**AP** (Auto-Planning) means the agent chooses the tools. "
            "**IF** (Instruction-Following) means the question lists the steps. They are separate runs."
        )
    if modality == "all":
        modality_line = (
            "This run includes every modality (`spectrum`, `products`, and `rgb`: "
            "raw spectra, derived products, and RGB images). "
            f"Each one is nested under the regime, for example `earth_bench/{label}/spectrum/accuracy`."
        )
    else:
        modality_line = (
            f"This run's modality is `{modality}`. "
            "`spectrum`, `products`, and `rgb` are the three input families "
            "(raw spectra, derived products, and RGB images). "
            f"Keys for this split sit under `earth_bench/{label}/{modality}/`, "
            f"for example `earth_bench/{label}/{modality}/accuracy`."
        )
    judges = _judge_note(
        metadata,
        "The official answer score is exact match. Earth-Bench does not use a judge for it.",
    )
    return f"""# Earth-Bench

{this} {sibling} Chart keys sit under `earth_bench/{label}/`.

{modality_line}

Official scores (Feng et al., Appendix B.2):

- `accuracy` — exact match of the extracted answer. This is the headline.
- `efficiency` — predicted tool calls divided by gold tool calls. The ratio can be above 1.
- `tao` — share of the gold tool set that appears anywhere in the trace.
- `tio` — gold tools in order, stopping at the first miss.
- `tem` — length of the exact prefix of the gold sequence, divided by the gold length.
- `param_acc` — that same prefix, also requiring the arguments to match.

{judges}
"""


def _openearthagent_guide(metadata: dict[str, Any]) -> str:
    regime = str(metadata.get("regime") or "e2e").strip().lower()
    label = "step" if regime == "step" else "e2e"
    tools = str(metadata.get("tools") or "all").strip() or "all"
    if regime == "step":
        this = (
            "This run is **step** (Table 3): teacher-forced. "
            "Each gold turn is one sample, tools do not run, and the model writes the next JSON action."
        )
        sibling = "**e2e** (Table 4) is the other regime: the agent calls tools. It is a separate run."
        scores = """Step scores:

- `inst` — the reply parsed as an action.
- `tool` — the gold tool name.
- `argn` — the same argument names.
- `argv` — the same argument values.
- `summ` — the judge score on `Terminate.ans`. It counts only when both sides called `Terminate` on a task whose last tool is not a generation tool.

The first turn of each task is a thought with an empty action list. Those turns stay out of the denominator.

`avg_summ` is the unweighted mean of this run's answer judges, each judge once. It sits with the other `summ` scores. A run with one judge does not add it."""
    else:
        if tools == "cached":
            tool_line = (
                " This log was recorded with the removed cache-only setting: "
                "only tasks the gpkg cache could replay, and the controller was not called."
            )
        else:
            tool_line = (
                " A cache hit for the five gpkg tools is used. Every miss, and every other tool, "
                "goes to the live worker."
            )
        this = (
            "This run is **e2e** (Table 4): the agent calls tools, and the final assistant message is the answer."
            + tool_line
        )
        sibling = (
            "**step** (Table 3) is the other regime: teacher-forced JSON, one gold turn at a time, with no tools. "
            "It is a separate run."
        )
        scores = """End-to-end scores:

- `any_order` — the gold tools are covered as a multiset (counts, any order).
- `same_order` — that coverage, and the gold names are a subsequence. Extra predicted calls are allowed. Table 4's caption says "exact sequence"; Section 4.1 and the released scorer mean sequence plus frequency.
- `unique` — the gold tools are covered as a set.
- `ans` — the judge score on the final answer, for tasks that are not generation tasks.
- `gen_success` — the last tool is the gold generation tool and that call succeeded.
- `perception`, `operation`, `logic`, `gis` — pooled category F1. One task can contribute to more than one category.
- `avg_ans` — unweighted mean of this run's answer judges, each judge once. It sits with the other `ans` scores. A run with one judge does not add it."""
    if metadata.get("vision") is False:
        vision_line = "`vision=false` on this run leaves the 533 image tasks unscored."
    else:
        vision_line = (
            "Task type is the next section, `answer` and `generation`: "
            f"`openearthagent/{label}/answer/` and `openearthagent/{label}/generation/`. "
            "`summ` is filled on `answer` only. An end-to-end generation task is scored with `gen_success` (Table 4's Gen.)."
        )
    judges = _judge_note(
        metadata,
        "The official answer judge is the grader role (`gpt-4o-mini` in the paper, temperature 0, at most 150 tokens).",
    )
    return f"""# OpenEarthAgent

{this} {sibling} Chart keys sit under `openearthagent/{label}/`.

Logged rates are fractions in [0, 1]. Tables 3 and 4 print those rates times 100.

{scores}

{vision_line}

{judges}
"""


def _geobenchx_guide(metadata: dict[str, Any]) -> str:
    group = str(metadata.get("group") or "all").strip() or "all"
    kind = str(metadata.get("kind") or "all").strip() or "all"
    judges = _judge_note(
        metadata,
        "The headline is the panel's match share. The paper's panel is Claude Sonnet 3.5, GPT-4.1, and Gemini 2.5 Pro.",
    )
    return f"""# GeoBenchX

`geobenchx/match` is the headline: the unweighted mean of each judge's match share (Krechetova & Kochedykov, Section 5.2). Each judge is its own row (`geobenchx/llm_as_judge/<model>`), with `partial` and `no_match` beside it. `merged/match` averages the indicators on one sample and is a different number.

`length_diff` is the difference in solution length. This run's group is `{group}`. `kind` is solvable, unsolvable, or control; this run's kind filter is `{kind}`. Those classes repeat the block under `geobenchx/<label>/`.

`tao`, `tio`, `tem`, and `param_acc` are our addition. They sit beside the paper's verdicts.

{judges}
"""


def _eo_gym_guide(metadata: dict[str, Any]) -> str:
    prompt = str(metadata.get("prompt") or "simple").strip() or "simple"
    toolset = str(metadata.get("toolset") or "skill").strip() or "skill"
    trajectories = metadata.get("trajectories", 3)
    if metadata.get("vision") is False:
        vision_line = "`vision=false` on this run leaves every task unscored, because the questions carry images."
    else:
        vision_line = "The questions carry images, so the model under test reads pixels."
    judges = _judge_note(
        metadata,
        "The answer judge is the grader role (`gpt-4.1-mini` in the paper). A failed judge falls back to exact normalised match.",
    )
    return f"""# EO-Gym

This run uses the `{prompt}` prompt and the `{toolset}` tool set, with {trajectories} trajectories per task.

- `accuracy` — 1 when a consumed trajectory is correct.
- `pass@k` — 1 when the first correct trajectory falls inside the first k. Scoring stops at that trajectory.
- `unbiased_pass@k` — the Chen et al. (2021) estimator over the same trajectories.

`question_type` and `eo_task` repeat the block under `eo_gym/`. {vision_line}

{judges}
"""


def _generic_guide(metadata: dict[str, Any]) -> str:
    name = benchmark_name(metadata, None)
    return f"""# {name}

Scores on this run are that benchmark's own published metrics. An extra judge keeps its own key, separate from that score.
"""


def metric_guide(metadata: dict[str, Any] | None) -> str:
    """Markdown for the run page: regime, modality, metric hierarchy, and judges.

    Shabbir et al., arXiv:2602.17665, Tables 3 and 4 (OpenEarthAgent).
    Feng et al., arXiv:2509.23141, Section 4 and Appendix B.2 (Earth-Bench AP / IF).
    Krechetova & Kochedykov, arXiv:2503.18129, Sections 5.2 and 6.1 (GeoBenchX).
    EO-Gym, arXiv:2605.01250, Section 5 (pass@k). Chen et al., arXiv:2107.03374
    (unbiased pass@k).
    """
    document = metadata if isinstance(metadata, dict) else {}
    writers = {
        "earth_bench": _earth_bench_guide,
        "openearthagent": _openearthagent_guide,
        "geobenchx": _geobenchx_guide,
        "eo_gym": _eo_gym_guide,
    }
    text = writers.get(benchmark_name(document, None), _generic_guide)(document)
    return text.strip() + "\n"


def guide_html(markdown: str) -> str:
    """A small HTML page for ``wandb.Html``. The run notes keep the markdown."""
    import html

    parts = ["<article>"]
    in_list = False

    def close_list() -> None:
        nonlocal in_list
        if in_list:
            parts.append("</ul>")
            in_list = False

    def inline(text: str) -> str:
        escaped = html.escape(text)
        escaped = re.sub(r"`([^`]+)`", r"<code>\1</code>", escaped)
        escaped = re.sub(r"\*\*([^*]+)\*\*", r"<strong>\1</strong>", escaped)
        return escaped

    for line in markdown.splitlines():
        if line.startswith("# "):
            close_list()
            parts.append(f"<h1>{inline(line[2:])}</h1>")
        elif line.startswith("## "):
            close_list()
            parts.append(f"<h2>{inline(line[3:])}</h2>")
        elif line.startswith("- "):
            if not in_list:
                parts.append("<ul>")
                in_list = True
            parts.append(f"<li>{inline(line[2:])}</li>")
        elif not line.strip():
            close_list()
        else:
            close_list()
            parts.append(f"<p>{inline(line)}</p>")
    close_list()
    parts.append("</article>")
    return "\n".join(parts)


def published_names(metadata: dict[str, Any] | None) -> frozenset[str] | None:
    """Score names the task declared as its own chart metrics.

    Absent means every stored score is charted. A task lists this so an
    extra field (another answer key, a trajectory judge) stays on the sample
    and out of the summary.
    """
    raw = (metadata or {}).get("published_metrics")
    if isinstance(raw, str):
        raw = [part.strip() for part in raw.split(",") if part.strip()]
    if (
        isinstance(raw, list)
        and all(isinstance(item, str) and item.strip() for item in raw)
        and raw
    ):
        return frozenset(item.strip() for item in raw)
    return None


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
# Verdict shares are GeoBenchX's no-match / partial / match frequencies
# (Krechetova & Kochedykov, Section 6.1; ``generate_eval_stats``).
# ``success`` remains the match share so older chart keys still read it.
_JUDGE_STATS = {
    "stderr": "stderr",
    "median": "median",
    "median/median": "median",
    "success": "success",
    "partial": "partial",
    "no_match": "no_match",
    "lower": "ci_lower",
    "ci/lower": "ci_lower",
    "ci_wilson/lower": "ci_lower",
    "upper": "ci_upper",
    "ci/upper": "ci_upper",
    "ci_wilson/upper": "ci_upper",
    "ci_lower": "ci_lower",
    "ci_upper": "ci_upper",
}
_SHARE_CATEGORIES = ("no_match", "partial", "match")


def judge_category(score_name: str) -> tuple[str, str] | None:
    """``(slug, category)`` for one judge's share row.

    ``merged/`` is the per-sample average, not a judge. ``panel`` is not a
    checkpoint id. A bare ``match`` is the panel row.
    """
    if score_name.startswith("merged/") or "/" not in score_name:
        return None
    # ``no_match`` before ``match``: ``.../no_match`` also ends with ``/match``.
    for category in _SHARE_CATEGORIES:
        suffix = "/" + category
        if not score_name.endswith(suffix):
            continue
        slug = score_name[: -len(suffix)].replace("/", "_")
        if slug and slug != "panel":
            return slug, category
    return None


def judge_slug(score_name: str) -> str | None:
    """Checkpoint id for a ``*/match`` row. ``merged/match`` is not a judge."""
    parsed = judge_category(score_name)
    if parsed is None or parsed[1] != "match":
        return None
    return parsed[0]


def _stat_label(metric: Any, key: Any) -> str | None:
    name = str(getattr(metric, "name", key))
    group = getattr(metric, "group", None)
    leaf = (
        f"{group}/{name}"
        if group and str(group) not in (name, "mean", "stderr")
        else name
    )
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

    ``{benchmark}/llm_as_judge/{model}`` is that judge's match share
    (``generate_eval_stats``: count of verdict 2 over the tasks that judge
    scored). ``partial`` and ``no_match`` sit beside it. ``success`` is the
    same match share.

    ``{benchmark}/match`` and ``llm_as_judge/avg`` are the unweighted mean of
    those match shares (Section 5.2: each judge once). The same mean is taken
    for partial and no-match. ``merged/match`` is not used: it is the mean of
    per-sample averages, which differs when a judge skipped a task.

    Absolute counts: ``n`` is ``scored_samples`` on the match row. The counts
    are ``round(share * n)``.
    """
    out: dict[str, int | float] = {}
    length_values: list[float] = []
    shares: dict[str, dict[str, float]] = {}
    extras: dict[str, dict[str, float]] = {}
    counts: dict[str, int] = {}
    for score in scores:
        name = str(getattr(score, "name", ""))
        metrics = getattr(score, "metrics", None) or {}
        mean = _metric(metrics, "mean")
        if mean is None:
            continue
        if name in _STEP_SCORES:
            out[f"{benchmark}/{name}"] = mean
            continue
        if name.endswith("/length_diff") and not name.startswith("merged/"):
            length_values.append(mean)
            continue
        parsed = judge_category(name)
        if parsed is None:
            continue
        slug, category = parsed
        shares.setdefault(slug, {})[category] = mean
        if category != "match":
            continue
        fields = _judge_fields(metrics)
        extras[slug] = {
            label: value for label, value in fields.items() if label != "mean"
        }
        n = getattr(score, "scored_samples", None)
        if isinstance(n, int):
            counts[slug] = n
    judge_match: list[float] = []
    judge_partial: list[float] = []
    judge_none: list[float] = []
    for slug, categories in shares.items():
        prefix = f"{benchmark}/llm_as_judge/{slug}"
        match = categories.get("match")
        if match is None:
            continue
        out[prefix] = match
        out[f"{prefix}/success"] = match
        judge_match.append(match)
        partial = categories.get("partial")
        none = categories.get("no_match")
        if partial is not None:
            out[f"{prefix}/partial"] = partial
            judge_partial.append(partial)
        if none is not None:
            out[f"{prefix}/no_match"] = none
            judge_none.append(none)
        for label, value in extras.get(slug, {}).items():
            if label not in ("success", "partial", "no_match"):
                out[f"{prefix}/{label}"] = value
        n = counts.get(slug)
        if n is None:
            continue
        out[f"{prefix}/n"] = n
        out[f"{prefix}/n_success"] = int(round(match * n))
        if partial is not None:
            out[f"{prefix}/n_partial"] = int(round(partial * n))
        if none is not None:
            out[f"{prefix}/n_no_match"] = int(round(none * n))
    panel_match = _mean_of(judge_match)
    if panel_match is not None:
        out[f"{benchmark}/match"] = panel_match
        out[f"{benchmark}/llm_as_judge/avg"] = panel_match
        out[f"{benchmark}/llm_as_judge/n_judges"] = len(judge_match)
    panel_partial = _mean_of(judge_partial)
    if panel_partial is not None:
        out[f"{benchmark}/partial"] = panel_partial
    panel_none = _mean_of(judge_none)
    if panel_none is not None:
        out[f"{benchmark}/no_match"] = panel_none
    length = _mean_of(length_values)
    if length is not None:
        out[f"{benchmark}/length_diff"] = length
    return out


def _judge_leaf(model: str) -> str:
    """Last path segment of a judge id. ``.../deepseek/deepseek-v3.2`` is ``deepseek-v3.2``."""
    return model.strip().rstrip("/").split("/")[-1]


def _regime_of(samples: list[Any] | None) -> str | None:
    for sample in samples or []:
        regime = (getattr(sample, "metadata", None) or {}).get("regime")
        if isinstance(regime, str) and regime.strip():
            return regime.strip()
    return None


def _published_metric(name: str, regime: str | None) -> str:
    """Paper column for OpenEarthAgent's judge field. Other names stay as stored."""
    if name != "answer":
        return name
    if regime == "step":
        return "summ"
    if regime == "e2e":
        return "ans"
    return name


def _charted_metric(name: str) -> bool:
    """True when ``summary_metrics`` already writes this leaf."""
    if name in _STEP_SCORES or name in {"match", "partial", "no_match", "length_diff"}:
        return True
    if name.startswith("merged/") or name.endswith("/length_diff"):
        return True
    return judge_category(name) is not None


def _scorer_table(
    samples: list[Any],
) -> tuple[list[str], dict[str, str], list[dict[str, dict[str, float]]]]:
    """Scorer names, judge leaf per scorer, and one metric map per sample.

    A NaN or a non-dict score is absent from that sample's map. Chart metrics
    (``tao``, verdict shares, ``length_diff``) are left to ``summary_metrics``.
    """
    order: list[str] = []
    judges: dict[str, str] = {}
    rows: list[dict[str, dict[str, float]]] = []
    for sample in samples:
        row: dict[str, dict[str, float]] = {}
        for scorer_name, score in (getattr(sample, "scores", None) or {}).items():
            name = str(scorer_name)
            if name not in order:
                order.append(name)
            model = (getattr(score, "metadata", None) or {}).get("judge_model")
            if isinstance(model, str) and model.strip() and name not in judges:
                judges[name] = _judge_leaf(model)
            value = getattr(score, "value", None)
            if not isinstance(value, dict):
                continue
            numbers: dict[str, float] = {}
            for key, item in value.items():
                metric = str(key)
                if _charted_metric(metric):
                    continue
                number = _finite(item)
                if number is not None:
                    numbers[metric] = number
            if numbers:
                row[name] = numbers
        rows.append(row)
    return order, judges, rows


def _category_f1(
    scores: list[Any] | None, prefix: str, judges: dict[str, str]
) -> dict[str, float]:
    """Pooled perception / operation / logic / gis F1, once when every scorer agrees."""
    by_leaf: dict[str, list[tuple[str, float]]] = {}
    for score in scores or []:
        scorer = str(getattr(score, "scorer", "") or getattr(score, "name", ""))
        metrics = getattr(score, "metrics", None) or {}
        for key, metric in metrics.items():
            if getattr(metric, "group", None) != "category_f1":
                continue
            leaf = str(getattr(metric, "name", key))
            number = _finite(getattr(metric, "value", metric))
            if number is None:
                continue
            by_leaf.setdefault(leaf, []).append((scorer, number))
    out: dict[str, float] = {}
    for leaf, pairs in by_leaf.items():
        values = [value for _, value in pairs]
        if len(set(values)) == 1:
            out[f"{prefix}/{leaf}"] = values[0]
            continue
        for scorer, value in pairs:
            slug = judges.get(scorer) or _slug(scorer)
            if slug:
                out[f"{prefix}/{slug}/{leaf}"] = value
    return out


def _native_value_rows(
    samples: list[Any] | None, regime: str | None = None
) -> list[dict[str, float | None]]:
    """Per-sample metric cells using each benchmark's own score names.

    A metric that is the same on every scorer is one column. A metric that
    differs is one column per judge. OpenEarthAgent step mode names the judge
    field ``summ``; end to end names it ``ans``. A sample with no finite value
    stores ``None``.
    """
    rows_in = list(samples or [])
    regime = regime or _regime_of(rows_in)
    order, judges, rows = _scorer_table(rows_in)
    found: set[str] = set()
    for row in rows:
        for numbers in row.values():
            found.update(numbers)
    columns: dict[str, list[float | None]] = {}
    for metric in sorted(found):
        series = {
            scorer: [row.get(scorer, {}).get(metric) for row in rows]
            for scorer in order
            if any(metric in row.get(scorer, {}) for row in rows)
        }
        if not series:
            continue
        published = _published_metric(metric, regime)
        sequences = list(series.values())
        if all(sequence == sequences[0] for sequence in sequences):
            columns[published] = sequences[0]
            continue
        for scorer, sequence in series.items():
            slug = judges.get(scorer) or _slug(scorer)
            if slug:
                columns[f"{slug}/{published}"] = sequence
    blank = len(rows_in)
    _answer_panel_column(columns, blank)
    return [
        {key: values[index] for key, values in columns.items()}
        for index in range(blank)
    ]


def _answer_panel_column(
    columns: dict[str, list[float | None]], n_rows: int
) -> None:
    """Per-sample mean of the answer judges. One judge leaves the paper column."""
    judge_keys = [
        key
        for key in columns
        if "/" in key and key.rsplit("/", 1)[-1] in {"ans", "summ"}
    ]
    if len(judge_keys) < 2:
        return
    leaf = judge_keys[0].rsplit("/", 1)[-1]
    if any(key.rsplit("/", 1)[-1] != leaf for key in judge_keys):
        return
    averaged: list[float | None] = []
    for index in range(n_rows):
        scored = [
            columns[key][index]
            for key in judge_keys
            if columns[key][index] is not None
        ]
        averaged.append(sum(scored) / len(scored) if scored else None)
    columns[f"avg_{leaf}"] = averaged


def _answer_panel(summary: dict[str, float], prefix: str) -> tuple[str, float] | None:
    """Unweighted mean of each judge's answer score under ``prefix``.

    Shabbir et al., arXiv:2602.17665, Tables 3 and 4 use one grader. This is
    the same construction as GeoBenchX Section 5.2 (each judge once). The key
    is ``avg_ans`` or ``avg_summ``, so it shows with the other answer scores.
    A single judge leaves that column as it is.
    """
    values: list[float] = []
    leaves: set[str] = set()
    head = f"{prefix}/"
    for key, value in summary.items():
        if not key.startswith(head):
            continue
        rest = key[len(head) :]
        slug, separator, leaf = rest.partition("/")
        if (
            separator
            and slug
            and slug != "avg"
            and "/" not in leaf
            and leaf in {"ans", "summ"}
        ):
            values.append(value)
            leaves.add(leaf)
    if len(values) < 2 or len(leaves) != 1:
        return None
    return f"{prefix}/avg_{next(iter(leaves))}", sum(values) / len(values)


def native_metrics(
    samples: list[Any] | None,
    benchmark: str,
    *,
    regime: str | None = None,
    scores: list[Any] | None = None,
    only: frozenset[str] | None = None,
) -> dict[str, float]:
    """Benchmark score names that the chart allowlist does not already copy.

    One key when every scorer reports the same per-sample values. One key per
    judge when they do not. OpenEarthAgent step mode publishes ``answer`` as
    ``summ``; end to end publishes it as ``ans``.
    """
    rows_in = list(samples or [])
    regime = regime or _regime_of(rows_in)
    _order, judges, _rows = _scorer_table(rows_in)
    gathered: dict[str, list[float]] = {}
    for cells in _native_value_rows(rows_in, regime):
        for key, value in cells.items():
            if value is not None:
                gathered.setdefault(key, []).append(value)
    out: dict[str, float] = {}
    for key, values in gathered.items():
        if only is not None and key.rsplit("/", 1)[-1] not in only:
            continue
        mean = _mean_of(values)
        if mean is not None:
            out[f"{benchmark}/{key}"] = mean
    for key, value in _category_f1(scores, benchmark, judges).items():
        if only is None or key.rsplit("/", 1)[-1] in only:
            out[key] = value
    # Mean of the judge means, not the mean of the per-sample average cells.
    # A judge that skipped samples still counts once.
    averaged = _answer_panel(out, benchmark)
    if averaged is not None:
        key, value = averaged
        out[key] = value
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
        return [
            str(item).strip()
            for item in raw
            if isinstance(item, str) and str(item).strip()
        ]
    return []


def _ones(values: list[float]) -> int:
    """How many indicators are 1. A 0/1 share's count of that category."""
    return sum(value == 1 for value in values)


def _subset_summary(
    flats: list[dict[str, float]], prefix: str
) -> dict[str, int | float]:
    """The same shares and counts as the full run, on this class only.

    Each judge's share is the mean of that judge's 0/1 indicators on the
    samples the judge scored. The class match share is the unweighted mean
    of those shares. ``merged/match`` is not used.
    """
    out: dict[str, int | float] = {f"{prefix}/n": len(flats)}
    for step in _STEP_SCORES:
        mean = _mean_of([row[step] for row in flats if step in row])
        if mean is not None:
            out[f"{prefix}/{step}"] = mean
    by_judge: dict[str, dict[str, list[float]]] = {}
    for row in flats:
        for key, value in row.items():
            parsed = judge_category(key)
            if parsed is None:
                continue
            slug, category = parsed
            by_judge.setdefault(slug, {}).setdefault(category, []).append(value)
    match_shares: list[float] = []
    partial_shares: list[float] = []
    none_shares: list[float] = []
    for slug, categories in by_judge.items():
        match_values = categories.get("match") or []
        if not match_values:
            continue
        n = len(match_values)
        match_share = sum(match_values) / n
        match_shares.append(match_share)
        block = f"{prefix}/llm_as_judge/{slug}"
        n_success = _ones(match_values)
        partial_values = categories.get("partial") or []
        none_values = categories.get("no_match") or []
        out[block] = match_share
        out[f"{block}/n"] = n
        out[f"{block}/n_success"] = n_success
        out[f"{block}/success"] = n_success / n
        out[f"{block}/median"] = float(statistics.median(match_values))
        if partial_values:
            partial_share = sum(partial_values) / len(partial_values)
            partial_shares.append(partial_share)
            n_partial = _ones(partial_values)
            out[f"{block}/n_partial"] = n_partial
            out[f"{block}/partial"] = n_partial / len(partial_values)
        if none_values:
            none_share = sum(none_values) / len(none_values)
            none_shares.append(none_share)
            n_none = _ones(none_values)
            out[f"{block}/n_no_match"] = n_none
            out[f"{block}/no_match"] = n_none / len(none_values)
    if match_shares:
        panel = sum(match_shares) / len(match_shares)
        out[f"{prefix}/match"] = panel
        out[f"{prefix}/llm_as_judge/avg"] = panel
        out[f"{prefix}/llm_as_judge/n_judges"] = len(match_shares)
    if partial_shares:
        out[f"{prefix}/partial"] = sum(partial_shares) / len(partial_shares)
    if none_shares:
        out[f"{prefix}/no_match"] = sum(none_shares) / len(none_shares)
    lengths: list[float] = []
    for row in flats:
        diffs = [
            value
            for key, value in row.items()
            if key.endswith("/length_diff") and not key.startswith("merged/")
        ]
        if diffs:
            lengths.append(sum(diffs) / len(diffs))
    if lengths:
        out[f"{prefix}/length_diff"] = sum(lengths) / len(lengths)
    return out


def grouped_metrics(
    samples: list[Any] | None,
    benchmark: str,
    fields: list[str],
    *,
    regime: str | None = None,
    only: frozenset[str] | None = None,
) -> dict[str, int | float]:
    """Repeat the summary inside each class declared by ``summary_groups``.

    A string metadata field puts the sample in one class. A list field puts
    it in each entry, so a task with several labels is counted in each.
    The key is ``{benchmark}/{label}``. The field name is not part of the path.
    A sample that carries the same label from two fields is counted once.
    Native score names are repeated on the same prefix.
    """
    rows = list(samples or [])
    regime = regime or _regime_of(rows)
    buckets: dict[str, list[Any]] = {}
    for sample in rows:
        seen: set[str] = set()
        for field in fields:
            for label in _class_labels(sample, field):
                slug = _slug(label)
                if slug and slug not in seen:
                    seen.add(slug)
                    buckets.setdefault(slug, []).append(sample)
    out: dict[str, int | float] = {}
    for label, group in buckets.items():
        prefix = f"{benchmark}/{label}"
        flats = [_score_map(getattr(sample, "scores", None)) for sample in group]
        out.update(_subset_summary(flats, prefix))
        out.update(native_metrics(group, prefix, regime=regime, only=only))
    if "kind" in fields:
        # The paper's solvable column is solvable plus the three control tasks.
        combined = [
            sample
            for sample in rows
            if {"solvable", "control"} & set(_class_labels(sample, "kind"))
        ]
        if combined:
            prefix = f"{benchmark}/solvable_incl_control"
            flats = [_score_map(getattr(sample, "scores", None)) for sample in combined]
            out.update(_subset_summary(flats, prefix))
            out.update(native_metrics(combined, prefix, regime=regime, only=only))
    return out


def geobenchx_share_summary(samples: list[Any] | None) -> dict[str, int | float]:
    """Match, partial, and no-match shares for one GeoBenchX run.

    Built from per-sample 0/1 indicators. The panel value is the unweighted
    mean of each judge's share (Section 5.2), including inside each task
    group and ``solvable_incl_control``. Step scores, ``length_diff``, and
    the median of the indicators are omitted so writing this dict onto an
    existing run leaves those numbers in place.
    """
    rows = list(samples or [])
    flats = [_score_map(getattr(sample, "scores", None)) for sample in rows]
    summary: dict[str, int | float] = _subset_summary(flats, "geobenchx")
    summary.update(grouped_metrics(rows, "geobenchx", ["kind", "labels"]))
    summary.update(dashboard_aliases(summary, "geobenchx"))
    kept: dict[str, int | float] = {}
    for key, value in summary.items():
        if key.endswith("/median"):
            continue
        if (
            "solvable_incl_control" in key
            or "llm_as_judge" in key
            or key.endswith(("/match", "/partial", "/no_match"))
        ):
            kept[key] = value
    return kept


def dashboard_aliases(
    summary: dict[str, int | float], benchmark: str
) -> dict[str, int | float]:
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


def model_run_name(model: str | None, benchmark: str | None = None) -> str:
    """Display name: the checkpoint, then the benchmark.

    ``openai-api/openrouter/qwen/qwen3.8-27b`` on GeoBenchX is
    ``qwen3.8-27b_geobenchx``. The checkpoint stays a single token so a
    workspace search on the model still matches. inspect-wandb calls
    ``wandb.init(name=None)`` for a single eval, and wandb then keeps a
    generated id as the display name
    (https://docs.wandb.ai/ref/python/init, ``name``).
    """
    leaf = (model or "model").rstrip("/").split("/")[-1].strip() or "model"
    label = (benchmark or "").strip()
    if not label:
        return leaf
    return f"{leaf}_{label}"


def results_rows(scores: list[Any], benchmark: str) -> list[dict[str, Any]]:
    """One row per score that has a mean, with stderr when that metric exists."""
    rows: list[dict[str, Any]] = []
    for score in scores:
        metrics = getattr(score, "metrics", None) or {}
        mean = _metric(metrics, "mean")
        if mean is None:
            continue
        name = str(getattr(score, "name", ""))
        if benchmark == "openearthagent" and (
            name in _OPENEARTHAGENT_OMITTED
            or name.rsplit("/", 1)[-1] in _OPENEARTHAGENT_OMITTED
        ):
            continue
        scored = getattr(score, "scored_samples", None)
        rows.append(
            {
                "task": benchmark,
                "metric": name,
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
    values = [
        value for key, value in flat.items() if key == leaf or key.endswith("/" + leaf)
    ]
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
                    "tool_step": int(entry["step"])
                    if isinstance(entry.get("step"), int)
                    else 0,
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
                "question": _user_question(sample),
                "target": sample_target(sample),
                "kind": metadata.get("kind"),
                "labels": metadata.get("labels"),
                "readable_trace": render_transcript(sample),
                "trace": _trace_entries(getattr(sample, "store", None)),
            }
        )
    return records


def _role(message: Any) -> str:
    role = getattr(message, "role", None)
    if role is None and isinstance(message, dict):
        role = message.get("role")
    return str(role or "message")


def _calls(message: Any) -> list[Any]:
    calls = getattr(message, "tool_calls", None)
    if calls is None and isinstance(message, dict):
        calls = message.get("tool_calls") or []
    return list(calls or [])


def _call_name(call: Any) -> str:
    if isinstance(call, dict):
        return str(call.get("function") or call.get("name") or "tool")
    return str(getattr(call, "function", None) or getattr(call, "name", None) or "tool")


def _call_arguments(call: Any) -> Any:
    if isinstance(call, dict):
        if "arguments" in call:
            return call.get("arguments")
        return call.get("args")
    return getattr(call, "arguments", None)


def _pretty(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    return json.dumps(value, default=str, ensure_ascii=False, indent=2)


def _content_blocks(message: Any) -> list[str]:
    """Text and reasoning the model produced, in order. Images stay as a marker."""
    content = getattr(message, "content", None)
    if content is None and isinstance(message, dict):
        content = message.get("content")
    if not isinstance(content, list):
        text = _message_text(message)
        return [text] if text else []
    blocks: list[str] = []
    for part in content:
        kind = getattr(part, "type", None) or (
            part.get("type") if isinstance(part, dict) else None
        )
        if kind == "reasoning":
            text = getattr(part, "reasoning", None)
            if text is None and isinstance(part, dict):
                text = part.get("reasoning")
            if isinstance(text, str) and text.strip():
                blocks.append(f"[reasoning]\n{text.strip()}")
        elif kind == "text":
            text = getattr(part, "text", None)
            if text is None and isinstance(part, dict):
                text = part.get("text")
            if isinstance(text, str) and text.strip():
                blocks.append(text.strip())
        elif kind == "image":
            blocks.append("[image]")
        elif kind:
            blocks.append(f"[{kind}]")
    if blocks:
        return blocks
    text = _message_text(message)
    return [text] if text else []


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
    """The conversation, one block per turn, with the arguments of each tool call.

    A turn starts at ``======== turn N | role ========``. An assistant turn that
    calls a tool keeps ``[call name]`` and a ``parameters:`` block. A tool
    result keeps ``[tool name]`` and the text the model saw back. Reasoning
    parts are marked ``[reasoning]``; images are a marker, not the bytes.
    """
    blocks: list[str] = []
    for index, message in enumerate(getattr(sample, "messages", None) or [], 1):
        role = _role(message)
        if role == "tool":
            name = getattr(message, "function", None)
            if name is None and isinstance(message, dict):
                name = message.get("function")
            name = name or "tool"
            text = _message_text(message)
            blocks.append(
                f"======== turn {index} | tool | {name} ========\n[tool {name}]\n{text}"
            )
            continue
        lines = [f"======== turn {index} | {role} ========"]
        lines.extend(_content_blocks(message))
        for call in _calls(message):
            lines.append(
                f"[call {_call_name(call)}]\nparameters:\n{_pretty(_call_arguments(call))}"
            )
        blocks.append("\n".join(lines))
    return "\n\n".join(blocks)


def _raw_turns(sample: Any) -> list[dict[str, Any]]:
    """One JSON object per message: role, text, and tool-call arguments."""
    turns: list[dict[str, Any]] = []
    for message in getattr(sample, "messages", None) or []:
        role = _role(message)
        turn: dict[str, Any] = {"role": role, "text": _message_text(message)}
        if role == "tool":
            name = getattr(message, "function", None)
            if name is None and isinstance(message, dict):
                name = message.get("function")
            turn["name"] = name or "tool"
        calls = [
            {"name": _call_name(call), "arguments": _call_arguments(call)}
            for call in _calls(message)
        ]
        if calls:
            turn["tool_calls"] = calls
        turns.append(turn)
    return turns


def raw_trace(sample: Any) -> str:
    """JSON of the conversation turns and the tool-call trace from the sample store."""
    payload = {
        "turns": _raw_turns(sample),
        "tool_trace": _trace_entries(getattr(sample, "store", None)),
    }
    return json.dumps(payload, default=str, ensure_ascii=False, indent=2)


def sample_target(sample: Any) -> str:
    """The scored answer, or the reference solution when that answer is empty.

    Inspect stores the answer on ``target``. GeoBenchX leaves it empty and
    keeps the reference calls in metadata (Krechetova & Kochedykov, Section 5);
    those calls are what the judge compares, so they fill this cell.
    """
    target = getattr(sample, "target", None)
    if isinstance(target, list):
        text = "\n".join(str(item) for item in target if item not in (None, ""))
    else:
        text = _text_cell(target).strip()
    if text:
        return text
    metadata = getattr(sample, "metadata", None) or {}
    parts: list[str] = []
    description = metadata.get("reference_solution_description")
    if isinstance(description, str) and description.strip():
        parts.append(description.strip())
    solutions = metadata.get("reference_solutions")
    if isinstance(solutions, list) and solutions:
        for index, solution in enumerate(solutions, 1):
            steps = solution if isinstance(solution, list) else []
            lines = [f"reference {index}"]
            for step in steps:
                if not isinstance(step, dict):
                    continue
                lines.append(
                    f"  {step.get('name')}({_text_cell(step.get('arguments'))})"
                )
            parts.append("\n".join(lines))
    if parts:
        return "\n\n".join(parts)
    for key in ("gold_tool_calls", "gold_steps", "reference_trajectory"):
        gold = metadata.get(key)
        if gold:
            return json.dumps(gold, default=str, ensure_ascii=False, indent=2)
    for key in ("answer_gt_file", "answer_question_json"):
        value = metadata.get(key)
        if isinstance(value, str) and value.strip():
            return value
    return ""


def _user_question(sample: Any) -> str:
    for message in getattr(sample, "messages", None) or []:
        role = getattr(message, "role", None) or (
            message.get("role") if isinstance(message, dict) else None
        )
        if role == "user":
            return _message_text(message)
    return ""


def _final_answer(sample: Any) -> str:
    answer = ""
    for message in getattr(sample, "messages", None) or []:
        role = getattr(message, "role", None) or (
            message.get("role") if isinstance(message, dict) else None
        )
        if role == "assistant":
            text = _message_text(message)
            if text:
                answer = text
    return answer


def _checkpoint_id(model_id: str) -> str:
    """Drop the Inspect and OpenRouter prefixes. ``openai-api/openrouter/qwen/qwen3.8-27b`` is ``qwen/qwen3.8-27b``."""
    changed = True
    while changed:
        changed = False
        for prefix in ("openai-api/", "openrouter/"):
            if model_id.startswith(prefix):
                model_id = model_id[len(prefix) :]
                changed = True
    return model_id


def _judge_verdicts(score: Any) -> list[tuple[str, float | None, str]]:
    """One entry per judge on this score: checkpoint id, numeric verdict, written output.

    GeoBenchX stores the raw 0/1/2 on score metadata (``verdict``) and a 0/1
    indicator under ``{model}/match``. The table column is the raw verdict.
    Otherwise a score with ``judge_model`` is a trajectory judge
    (``trajectory_accuracy``) or EO-Gym's answer judge (``correct``).
    ``merged/`` is the per-sample average and is not a judge column.
    """
    value = getattr(score, "value", None)
    if not isinstance(value, dict):
        return []
    explanation = getattr(score, "explanation", None)
    text = explanation if isinstance(explanation, str) else ""
    metadata = getattr(score, "metadata", None) or {}
    raw = metadata.get("verdict")
    named: list[tuple[str, float | None, str]] = []
    for key, item in value.items():
        label = str(key)
        if label.startswith("merged/") or not label.endswith("/match"):
            continue
        slug = label[: -len("/match")]
        if not slug:
            continue
        if isinstance(raw, int) and not isinstance(raw, bool):
            named.append((slug, float(raw), text))
        else:
            named.append((slug, _finite(item), text))
    if named:
        return named
    metadata = getattr(score, "metadata", None) or {}
    model = metadata.get("judge_model")
    if not isinstance(model, str) or not model.strip():
        return []
    slug = _checkpoint_id(model.strip())
    if not slug:
        return []
    number = _finite(value.get("trajectory_accuracy"))
    if number is None:
        number = _finite(value.get("correct"))
    return [(slug, number, text)]


def _sample_judge_fields(sample: Any) -> dict[str, Any]:
    """``{model}/match`` and ``{model}/output`` for every judge on this sample."""
    found: dict[str, tuple[float | None, str]] = {}
    for score in (getattr(sample, "scores", None) or {}).values():
        for slug, number, text in _judge_verdicts(score):
            found.setdefault(slug, (number, text))
    fields: dict[str, Any] = {}
    for slug in sorted(found):
        number, text = found[slug]
        fields[f"{slug}/match"] = number
        fields[f"{slug}/output"] = text
    return fields


def _is_judge_column(key: str) -> bool:
    """Per-judge verdict columns. A benchmark metric such as ``deepseek-v3.2/summ`` is not one."""
    return key.endswith("/output") or (key.endswith("/match") and key != "match")


def sample_table_columns(rows: list[dict[str, Any]]) -> tuple[str, ...]:
    """Column order of the rows. Each benchmark contributes only the metrics it scored."""
    ordered: list[str] = []
    for row in rows:
        for key in row:
            if key not in ordered:
                ordered.append(key)
    return tuple(ordered)


def trace_table_columns(rows: list[dict[str, Any]]) -> tuple[str, ...]:
    """Shared fields and this benchmark's metrics, then the trace texts, then each judge."""
    keys: list[str] = []
    for row in rows:
        for key in row:
            if key not in keys and key not in ("target", "raw_trace", "readable_trace"):
                keys.append(key)
    body = [key for key in keys if not _is_judge_column(key)]
    judges = [key for key in keys if _is_judge_column(key)]
    return tuple(body + ["target", "raw_trace", "readable_trace"] + judges)


_TRACE_TEXT = frozenset(
    {
        "sample_id",
        "question",
        "final_answer",
        "transcript",
        "kind",
        "tool_trace",
        "target",
        "raw_trace",
        "readable_trace",
    }
)


def _trace_dtype(column: str) -> type:
    """Concrete cell type. A null-only column is typed ``unknown`` and the viewer shows no rows."""
    if column == "n_tool_calls":
        return int
    if column in _TRACE_TEXT or column.endswith("/output"):
        return str
    return float


def _trace_value(column: str, value: Any) -> Any:
    """A cell for the traces table. A missing metric stays blank rather than becoming 0."""
    kind = _trace_dtype(column)
    if kind is str:
        if value is None:
            return ""
        return value if isinstance(value, str) else _text_cell(value)
    if value is None:
        return None
    return kind(value)


def example_trace_rows(samples: list[Any] | None) -> list[dict[str, Any]]:
    """One row per sample: the samples-table fields, plus target and both traces.

    ``readable_trace`` is the turn-marked conversation. ``raw_trace`` is the
    same turns and the store's tool calls as JSON.
    """
    rows: list[dict[str, Any]] = []
    for sample, base in zip(samples or [], sample_rows(samples), strict=True):
        row = dict(base)
        row["target"] = sample_target(sample)
        row["raw_trace"] = raw_trace(sample)
        row["readable_trace"] = row["transcript"]
        rows.append(row)
    return rows


def _sample_id(sample: Any) -> str:
    sample_id = str(getattr(sample, "id", ""))
    epoch = getattr(sample, "epoch", 1)
    if epoch not in (None, 1):
        return f"{sample_id}:{epoch}"
    return sample_id


# Metric columns, when a run actually has them. GeoBenchX keeps tao before match.
_METRIC_ORDER = (
    "tao",
    "tio",
    "tem",
    "param_acc",
    "inst",
    "tool",
    "argn",
    "argv",
    "any_order",
    "same_order",
    "unique",
    "gen_success",
    "n_steps",
    "perception",
    "operation",
    "logic",
    "gis",
    "match",
    "length_diff",
)


def _metric_sort(keys: set[str]) -> list[str]:
    preferred = [key for key in _METRIC_ORDER if key in keys]
    bare = sorted(
        key
        for key in keys
        if key not in _METRIC_ORDER and "/" not in key and key not in {"avg_ans", "avg_summ"}
    )
    judged = sorted(key for key in keys if "/" in key and key not in _METRIC_ORDER)
    ordered = preferred + bare + judged
    ordered.extend(key for key in ("avg_summ", "avg_ans") if key in keys)
    return ordered


def _charted_cells(sample: Any) -> dict[str, float]:
    """Earth-Bench and GeoBenchX score cells that this sample actually has."""
    flat = _score_map(getattr(sample, "scores", None))
    cells: dict[str, float] = {}
    for name in _STEP_SCORES:
        if name in flat:
            cells[name] = flat[name]
    if any(
        key == "merged/match"
        or (key.endswith("/match") and not key.startswith("merged/"))
        for key in flat
    ):
        match = flat.get("merged/match")
        if match is None:
            match = _named(flat, "match")
        if match is not None:
            cells["match"] = match
    length = _named(flat, "length_diff")
    if length is not None:
        cells["length_diff"] = length
    return cells


def sample_rows(samples: list[Any] | None) -> list[dict[str, Any]]:
    """Per-sample scores for this benchmark, plus the conversation and tool trace.

    Metric columns are the scores present on the run. A step-mode OpenEarthAgent
    row carries ``inst``, ``tool``, ``argn``, ``argv``, and one ``summ`` per
    judge. It does not carry Earth-Bench or GeoBenchX columns.
    """
    rows_in = list(samples or [])
    regime = _regime_of(rows_in)
    # Step and end-to-end are OpenEarthAgent. Their tables keep Tables 3 and 4.
    paper_only = regime in {"step", "e2e"}
    native = _native_value_rows(rows_in, regime)
    charted = [_charted_cells(sample) for sample in rows_in]
    metric_keys = _metric_sort(
        {
            key
            for cells in (*charted, *native)
            for key, value in cells.items()
            if value is not None
            and not (
                paper_only
                and (
                    key in _OPENEARTHAGENT_OMITTED
                    or key.rsplit("/", 1)[-1] in _OPENEARTHAGENT_OMITTED
                )
            )
            and key != "trajectory_accuracy"
            and not key.endswith("/trajectory_accuracy")
        }
    )
    rows: list[dict[str, Any]] = []
    for sample, own, shared in zip(rows_in, native, charted, strict=True):
        metrics = {**shared, **own}
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
                **{key: metrics.get(key) for key in metric_keys},
                "n_tool_calls": len(names),
                "tool_trace": " -> ".join(names),
                # Trajectory judges become ``{model}/match`` columns. Tables 3
                # and 4 do not include that judge.
                **({} if paper_only else _sample_judge_fields(sample)),
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
    *,
    optional: bool = False,
) -> Any:
    import wandb

    data = [[row.get(column) for column in columns] for row in rows]
    if dtypes is None:
        return wandb.Table(columns=list(columns), data=data)
    return wandb.Table(
        columns=list(columns), data=data, dtype=list(dtypes), optional=optional
    )


def _drop_stale_summary(run: Any, benchmark: str, keep: set[str]) -> None:
    """Remove summary keys this publish no longer writes.

    A second publish used to leave ``accuracy_other_key`` and the trajectory
    judges in place. W&B stores a slash key either as one string or as nested
    sections. The old ``{benchmark}/by/...`` section is removed either way.
    """
    prefix = f"{benchmark}/"
    by_prefix = f"{benchmark}/by/"
    for key in list(run.summary.keys()):
        text = str(key)
        if text.startswith(by_prefix) or (text.startswith(prefix) and text not in keep):
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


def comparison_summary(
    samples: list[Any] | None,
    scores: list[Any],
    benchmark: str,
    *,
    regime: str | None = None,
    groups: list[str] | None = None,
    only: frozenset[str] | None = None,
) -> dict[str, int | float]:
    """Summary keys for one run.

    Auto-Planning and Instruction-Following sit under ``{benchmark}/AP`` and
    ``{benchmark}/IF``. A modality split is another section under that,
    ``earth_bench/AP/spectrum/accuracy``. OpenEarthAgent's step and end-to-end
    protocols sit under ``openearthagent/step`` and ``openearthagent/e2e``.
    ``only`` drops score names the task did not declare (another answer key,
    a trajectory judge).
    """
    root = metric_root(benchmark, regime)
    summary = summary_metrics(scores, root)
    summary.update(
        native_metrics(samples, root, regime=regime, scores=scores, only=only)
    )
    # Step scores are on each sample. A killed eval never writes the results
    # block, so those four keys would otherwise be missing at the top.
    flats = [_score_map(getattr(sample, "scores", None)) for sample in samples or []]
    from_samples = _subset_summary(flats, root)
    for name in _STEP_SCORES:
        if only is not None and name not in only:
            continue
        key = f"{root}/{name}"
        if key not in summary and key in from_samples:
            summary[key] = from_samples[key]
    summary.update(
        grouped_metrics(samples, root, list(groups or []), regime=regime, only=only)
    )
    summary.update(dashboard_aliases(summary, root))
    if benchmark == "openearthagent":
        # Tables 3 and 4 do not include Earth-Bench step scores or a trajectory judge.
        summary = {
            key: value
            for key, value in summary.items()
            if key.rsplit("/", 1)[-1] not in _OPENEARTHAGENT_OMITTED
        }
    return summary


_SECRET_KEY = re.compile(
    r"(api[_-]?key|(?:^|[_-])token(?:$|[_-])|secret|password|authorization)",
    re.IGNORECASE,
)


def _redact(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            key: "[redacted]" if _SECRET_KEY.search(str(key)) else _redact(item)
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [_redact(item) for item in value]
    return value


def run_config_yaml(log: Any) -> str:
    """The Inspect ``--run-config`` document for this log, as YAML.

    Inspect builds that document from the finished log in
    ``eval_log_to_run_config_dict`` (``inspect_ai/log/_config.py``). A run
    started from the CLI has no file in ``configs/``; this is that invocation.
    """
    import yaml
    from inspect_ai.log._config import eval_log_to_run_config_dict

    document = _redact(eval_log_to_run_config_dict(log))
    return yaml.safe_dump(document, sort_keys=False, allow_unicode=True)


def publish(run: Any, log: Any, samples: Any = None) -> None:
    """Write the summary, log the workspace charts once, and store the conversation.

    ``run.log`` is one call. Calling it again on the same run appends a step,
    and the workspace then draws those numbers as a history.
    """
    results = getattr(log, "results", None)
    scores = list(getattr(results, "scores", None) or [])
    spec = getattr(log, "eval", None)
    metadata = getattr(spec, "metadata", None) or {}
    if not isinstance(metadata, dict):
        metadata = dict(metadata)
    benchmark = benchmark_name(metadata, getattr(spec, "task", None))
    if samples is None:
        samples = getattr(log, "samples", None)
    regime = metadata.get("regime") if isinstance(metadata.get("regime"), str) else None
    groups = metadata.get("summary_groups") or []
    if isinstance(groups, str):
        groups = [groups]
    summary = comparison_summary(
        samples,
        scores,
        benchmark,
        regime=regime,
        groups=[str(field) for field in groups],
        only=published_names(metadata),
    )
    run.name = model_run_name(getattr(spec, "model", None), benchmark)
    label = regime_label(regime)
    modality = metadata.get("modality")
    guide = metric_guide(metadata)
    try:
        recorded: dict[str, Any] = {"run_config": run_config_yaml(log)}
        if label is not None:
            recorded["regime"] = label
        if isinstance(modality, str) and modality.strip():
            recorded["modality"] = modality.strip()
        run.config.update(recorded, allow_val_change=True)
        run.notes = guide
    except Exception:
        logger.exception("W&B run config was not recorded")
    _drop_stale_summary(run, benchmark, set(summary))
    for key, value in summary.items():
        run.summary[key] = value
    # One history row for the whole block, including class splits and judge
    # stats. A second log appends a step.
    rows = sample_rows(samples)
    traces = example_trace_rows(samples)
    trace_columns = trace_table_columns(traces)
    import wandb

    run.log(
        {
            **summary,
            "metric_guide": wandb.Html(guide_html(guide), inject=False),
            SAMPLES_KEY: _as_table(sample_table_columns(rows), rows),
            TRACES_KEY: _as_table(
                trace_columns,
                [
                    {
                        column: _trace_value(column, row.get(column))
                        for column in trace_columns
                    }
                    for row in traces
                ],
                tuple(_trace_dtype(column) for column in trace_columns),
                optional=True,
            ),
            METRICS_KEY: _as_table(METRICS_COLUMNS, results_rows(scores, benchmark)),
        }
    )
    _log_trace_artifact(run, benchmark, samples)


@hooks(
    name="wandb_compare",
    description="Log shared summary metrics and trace tables onto the W&B run.",
)
class WandbCompare(Hooks):
    async def on_task_start(self, data: TaskStart) -> None:
        # Runs after inspect-wandb's on_task_start (see _registry import order),
        # so the run inspect-wandb just opened gets a stable display name
        # instead of the generated id.
        try:
            import wandb
        except ImportError:
            return
        run = wandb.run
        if run is None:
            return
        spec = data.spec
        run.name = model_run_name(
            getattr(spec, "model", None),
            benchmark_name(
                getattr(spec, "metadata", None), getattr(spec, "task", None)
            ),
        )

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
