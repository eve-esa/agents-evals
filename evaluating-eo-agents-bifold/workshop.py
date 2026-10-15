"""Helpers for the SageMaker hands-on notebook.

The scores are Earth-Bench's. This module does not reimplement them. Trajectory
and answer metrics live in ``agentic_evals.tasks.earth_bench.metrics`` (Feng et
al., ICLR 2026, arXiv:2509.23141, Appendix B.2; the functions follow upstream
``evaluate/step_by_step.py`` and are tested against the published result files).

Earth-Bench does not publish a 10-question subset. ``WORKSHOP_QUESTIONS`` is a
sample we chose from the Spectrum auto-planning tasks that have a gold
trajectory in ``extracted_tool_calls_GT.json``: nine tasks with three tool
calls and question 8 with four, covering different retrieval tools so a session
stays short.

``leaderboard`` reads every finished run in a log directory. Point the runs and
the leaderboard at one shared directory (Inspect accepts ``s3://`` paths) and
the table is shared by the room.
"""

from __future__ import annotations

import os
import re
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

# Question 11 is first: the notebook walks through that sample before the full run.
WORKSHOP_QUESTIONS: tuple[int, ...] = (11, 2, 7, 8, 19, 21, 26, 31, 36, 42)


def eve_credentials() -> tuple[str, str, str]:
    """Model name, base URL, and API key from the environment.

    Inspect's ``openai-api/eve/...`` provider reads ``EVE_BASE_URL``. A filled
    ``EVE_API_BASE_URL`` is copied there when ``EVE_BASE_URL`` is unset.
    """
    model_name = os.environ.get("MODEL_NAME", "").strip()
    base_url = os.environ.get("EVE_BASE_URL", "").strip() or os.environ.get("EVE_API_BASE_URL", "").strip()
    if base_url and not os.environ.get("EVE_BASE_URL", "").strip():
        os.environ["EVE_BASE_URL"] = base_url
    api_key = os.environ.get("EVE_API_KEY", "").strip()
    return model_name, base_url, api_key


def check_environment() -> None:
    """Print Python, tool kits, raster coverage, and the model endpoint's /models list.

    The endpoint check is the same ``GET {base_url}/models`` match the harness
    stores on every sample (``agentic_evals.run_identity``).
    """
    if sys.version_info < (3, 12):
        raise RuntimeError(f"Python 3.12+ needed, found {sys.version.split()[0]}")

    from agentic_evals.paths import EARTH_BENCH_DATA, RUNS_DIR
    from agentic_evals.run_identity import fetch_identity
    from agentic_evals.tasks.earth_bench.tools import KITS, PERCEPTION_LOOKUP

    print("python", sys.version.split()[0])
    print("executable", sys.executable)
    print("data", EARTH_BENCH_DATA)
    print("runs", RUNS_DIR)
    missing = [kit for kit in KITS if not (PERCEPTION_LOOKUP.parent / f"{kit}.py").is_file()]
    if missing:
        raise FileNotFoundError(f"Earth-Bench kits missing: {', '.join(missing)}")
    print("tool kits", ", ".join(KITS))
    print("perception lookup", PERCEPTION_LOOKUP.is_file())

    raster_root = EARTH_BENCH_DATA / "benchmark" / "data"
    present = [q for q in WORKSHOP_QUESTIONS if (raster_root / f"question{q}").is_dir()]
    print(f"rasters on disk for {len(present)}/{len(WORKSHOP_QUESTIONS)} workshop questions")

    model_name, base_url, api_key = eve_credentials()
    if not model_name or not base_url or not api_key:
        raise RuntimeError("Set MODEL_NAME, EVE_BASE_URL (or EVE_API_BASE_URL), and EVE_API_KEY in .env")
    identity = fetch_identity(model_name, base_url, api_key)
    print("endpoint identified", identity["identified"], identity.get("error") or "")
    if not identity["identified"]:
        raise RuntimeError(f"Model endpoint check failed: {identity.get('error')}")
    check_budget(model_name)
    print("token budget available")


def download_earth_bench(question_ids: tuple[int, ...] | list[int] = WORKSHOP_QUESTIONS) -> Path:
    """Download rasters for these task ids. Returns the directory of ``questionN`` folders.

    Same Hugging Face snapshot as ``scripts/setup_benchmark.py earth-bench``
    (``agentic_evals.tasks.earth_bench.setup``). Set ``AGENTIC_EVALS_DATA``
    before importing this package so the files land in this folder's ``data/``.
    """
    from agentic_evals.paths import EARTH_BENCH_DATA
    from agentic_evals.tasks.earth_bench.setup import install

    install(
        SimpleNamespace(
            skip_data=False,
            modality=None,
            questions=",".join(str(q) for q in question_ids),
        )
    )
    return EARTH_BENCH_DATA / "benchmark" / "data"


def run(
    solver: Any,
    *,
    model: str,
    questions: tuple[int, ...] | list[int] = WORKSHOP_QUESTIONS,
    log_dir: str,
    max_samples: int = 4,
    trajectory_judges: str = "",
) -> Any:
    """Run Earth-Bench Spectrum auto-planning on ``questions`` and return the Inspect log.

    An EVE model is checked with a one-token request first, so a spent token
    budget stops here with the endpoint's message instead of inside the eval.
    """
    from inspect_ai import eval as inspect_eval

    from agentic_evals.tasks.earth_bench.task import earth_bench

    if model.startswith("openai-api/eve/"):
        check_budget(model.removeprefix("openai-api/eve/"))

    logs = inspect_eval(
        earth_bench(
            regime="ap",
            modality="spectrum",
            solver=solver,
            trajectory_judges=trajectory_judges,
        ),
        model=model,
        model_args={"strict_tools": False},
        sample_id=list(questions),
        log_dir=log_dir,
        max_samples=max_samples,
        display="none",
        # Cap each completion. An unset max_tokens makes this endpoint
        # reserve the rest of the context as output (see scorers/judges.py),
        # which spends the EVE token budget on the reservation alone.
        max_tokens=4096,
        # A silent response otherwise sits until the OpenAI client's
        # 10 minute timeout. attempt_timeout covers a non-streaming call;
        # stream_idle_timeout covers a stream that stops delivering chunks.
        attempt_timeout=180,
        stream_idle_timeout=60,
        # A room shares one endpoint and gets HTTP 429. max_retries is extra
        # attempts after the first, with Inspect's exponential backoff;
        # max_connections caps this notebook's concurrent model calls.
        max_retries=6,
        max_connections=2,
        # A sample that still fails is run again instead of ending the eval.
        retry_on_error=2,
    )
    if not logs:
        raise RuntimeError("inspect eval returned no log")
    log = logs[0]
    message = log.error.message if log.status == "error" and log.error else ""
    if "token_budget_exceeded" in message or "insufficient_quota" in message:
        raise BudgetExceeded(f"The model's token budget ran out during this run. {_budget_detail(message)}")
    return log


class BudgetExceeded(RuntimeError):
    """The endpoint's token budget is spent; retrying will not help until it resets."""


def _budget_detail(text: str) -> str:
    match = re.search(r"Token budget exceeded.*?(?:Resets at [0-9T:.+-]+|$)", text)
    return match.group(0).strip() if match else text[:300]


def check_budget(model_name: str | None = None) -> None:
    """Send a one-token request to the EVE endpoint; raise ``BudgetExceeded`` if its budget is spent."""
    import httpx

    name, base_url, api_key = eve_credentials()
    response = httpx.post(
        f"{base_url.rstrip('/')}/chat/completions",
        headers={"Authorization": f"Bearer {api_key}"},
        json={"model": model_name or name, "messages": [{"role": "user", "content": "OK"}], "max_tokens": 1},
        timeout=60,
    )
    if response.status_code == 429 and ("token_budget_exceeded" in response.text or "insufficient_quota" in response.text):
        raise BudgetExceeded(f"The model's token budget is spent. {_budget_detail(response.text)}")


def results_df(log: Any) -> Any:
    """One row per sample: Earth-Bench score fields, call count, and termination.

    ``n_calls`` is the length of the trace the scorer reads.
    ``termination`` is ``agentic_evals:termination`` from the harness
    (``completed``, ``recursion_limit``, or ``max_model_calls``).
    Judge columns are ``{scorer}_{metric}`` and start with ``trajectory_judge``.
    """
    import pandas as pd

    from agentic_evals.agents.langgraph_react import TERMINATION_KEY
    from agentic_evals.agents.trace_callback import TRACE_KEY

    rows: list[dict[str, Any]] = []
    for sample in log.samples or []:
        store = sample.store or {}
        row: dict[str, Any] = {"id": _question_id(sample.id)}
        scores = sample.scores or {}
        earth = scores.get("earth_bench_scorer")
        value = getattr(earth, "value", None)
        if isinstance(value, dict):
            row.update(value)
        row["n_calls"] = len(list(store.get(TRACE_KEY, [])))
        row["termination"] = store.get(TERMINATION_KEY)
        for scorer_name, score in scores.items():
            if not str(scorer_name).startswith("trajectory_judge"):
                continue
            metric = getattr(score, "value", None)
            if isinstance(metric, dict):
                for key, item in metric.items():
                    row[f"{scorer_name}_{key}"] = item
        rows.append(row)
    frame = pd.DataFrame(rows)
    if frame.empty:
        return frame
    return frame.set_index("id").sort_index()


def judge(log: Any, judge_model: str) -> Any:
    """Rescore a saved log with the trajectory judge, without running the agent again.

    The judge is ``agentic_evals.scorers.judges.trajectory_judge``, our addition
    to Earth-Bench. Inspect's ``score`` appends it next to ``earth_bench_scorer``,
    so the benchmark's own scores stay as they were.
    """
    from inspect_ai import score

    from agentic_evals.scorers.judges import trajectory_judge

    return score(log, [trajectory_judge(model=judge_model)], action="append", display="none")


# Appendix B.2 of Feng et al. Efficiency is a ratio, not a score.
LEADERBOARD_METRICS: tuple[str, ...] = ("accuracy", "tao", "tio", "tem", "param_acc", "efficiency")


def leaderboard(log_dir: str = "logs", questions: tuple[int, ...] | list[int] = WORKSHOP_QUESTIONS) -> Any:
    """One row per finished Earth-Bench run in ``log_dir`` that scored exactly ``questions``.

    The agent column is the ``name`` given to ``langgraph_agent``. Runs on other
    question sets are left out so every row is a mean over the same tasks.
    AP and IF are separate rows (``regime``). Judge verdicts, when a run has
    them, are their own columns and are not combined with the metrics.
    """
    import pandas as pd
    from inspect_ai.log import list_eval_logs, read_eval_log

    wanted = sorted(questions)
    rows: list[dict[str, Any]] = []
    for info in list_eval_logs(log_dir):
        log = read_eval_log(info)
        if log.status != "success" or log.eval.task_registry_name != "agentic_evals/earth_bench":
            continue
        frame = results_df(log)
        if frame.empty or sorted(frame.index) != wanted or "accuracy" not in frame or frame["accuracy"].isna().any():
            continue
        step = log.plan.steps[-1] if log.plan and log.plan.steps else None
        row: dict[str, Any] = {
            "agent": (step.params.get("name") if step else None) or log.eval.solver or "?",
            "model": log.eval.model,
            "regime": (log.eval.task_args or {}).get("regime", "ap"),
            "n": len(frame),
        }
        for column in [*LEADERBOARD_METRICS, "tool_errors"]:
            row[column] = frame[column].mean() if column in frame else float("nan")
        for column in frame.columns:
            if str(column).startswith("trajectory_judge"):
                row[column] = frame[column].mean()
        row["log"] = Path(info.name).name
        rows.append(row)
    board = pd.DataFrame(rows)
    if board.empty:
        return board
    return board.sort_values(["regime", "accuracy", "tio"], ascending=[True, False, False]).reset_index(drop=True)


_STATUS = {
    "match": ("#2ea043", "\u2713", "Same tool, same arguments"),
    "args": ("#d29922", "\u2260", "Same tool, different arguments"),
    "tool": ("#f85149", "\u2717", "Different tool"),
    "missing": ("#f85149", "\u2212", "Expert step never made"),
    "extra": ("#8b949e", "+", "Extra call"),
}
_MONO = "ui-monospace,SFMono-Regular,Menlo,Consolas,monospace"
_TILES = (
    ("TAO", "Tool-Any-Order"),
    ("TIO", "Tool-In-Order"),
    ("TEM", "Tool-Exact-Match"),
    ("ParamAcc", "Parameter accuracy"),
)


def _value_text(value: Any) -> str:
    import json

    text = json.dumps(value, ensure_ascii=False, default=str)
    return text if len(text) <= 64 else text[:61] + "..."


def _card_html(call: dict[str, Any] | None, other: dict[str, Any] | None, side: str, status: str) -> str:
    """One tool call as a card. On the agent side, arguments that differ show the expert's value."""
    import html

    if call is None:
        note = "no call" if side == "agent" else "no expert step"
        return (
            "<div style='border:1px dashed rgba(128,128,128,.55);border-radius:8px;padding:8px 10px;"
            f"opacity:.6;font-style:italic'>{note}</div>"
        )
    colour = _STATUS[status][0] if side == "agent" else "#8b949e"
    arguments = call.get("input") or {}
    other_arguments = (other or {}).get("input") or {}
    lines = []
    for key, value in (arguments.items() if isinstance(arguments, dict) else [("", arguments)]):
        differs = side == "agent" and status == "args" and isinstance(other_arguments, dict) and other_arguments.get(key) != value
        shown = html.escape(_value_text(value))
        label = f"<span style='opacity:.65'>{html.escape(str(key))}:</span> " if key else ""
        extra = ""
        if differs:
            extra = f" <span style='opacity:.75'>(expert: {html.escape(_value_text(other_arguments.get(key)))})</span>"
        mark = "background:rgba(210,153,34,.30);border-radius:3px;padding:0 3px;" if differs else ""
        lines.append(f"<div style='{mark}overflow-wrap:anywhere'>{label}{shown}{extra}</div>")
    body = "".join(lines) or "<div style='opacity:.55'>no arguments</div>"
    return (
        f"<div style='border-left:4px solid {colour};background:rgba(128,128,128,.10);border-radius:8px;"
        "padding:7px 10px;min-width:0'>"
        f"<div style='font-weight:600;font-family:{_MONO}'>{html.escape(str(call.get('name', '')))}</div>"
        f"<div style='font-family:{_MONO};font-size:11.5px;margin-top:3px'>{body}</div></div>"
    )


def _tile_html(code: str, label: str, value: float) -> str:
    return (
        "<div style='flex:1 1 120px;border:1px solid rgba(128,128,128,.35);border-radius:8px;padding:6px 10px'>"
        f"<div style='font-size:11px;opacity:.7'>{label}</div>"
        f"<div style='font-size:20px;font-weight:700;line-height:1.2'>{value:.2f}"
        f"<span style='font-size:11px;font-weight:400;opacity:.6'> {code}</span></div>"
        "<div style='height:5px;border-radius:3px;background:rgba(128,128,128,.25);margin-top:4px'>"
        f"<div style='height:5px;border-radius:3px;width:{max(0.0, min(1.0, value)) * 100:.0f}%;background:#2ea043'></div></div></div>"
    )


def compare_html(pred: list[dict[str, Any]], gold: list[dict[str, Any]], title: str = "") -> str:
    """Expert and agent tool calls side by side, one row per position, with the Earth-Bench scores.

    Rows are aligned by position, the way Earth-Bench's TEM and ParamAcc line
    steps up. A call is ``{"name": ..., "input": {...}}``. The scores come from
    ``agentic_evals.tasks.earth_bench.metrics``, the same functions the scorer uses.
    """
    import html

    from agentic_evals.tasks.earth_bench import metrics

    rows = []
    for i in range(max(len(pred), len(gold))):
        g = gold[i] if i < len(gold) else None
        p = pred[i] if i < len(pred) else None
        if g is None:
            status = "extra"
        elif p is None:
            status = "missing"
        elif p.get("name") != g.get("name"):
            status = "tool"
        elif p.get("input") == g.get("input"):
            status = "match"
        else:
            status = "args"
        colour, icon, label = _STATUS[status]
        rows.append(
            "<div style='display:grid;grid-template-columns:28px minmax(0,1fr) 34px minmax(0,1fr);gap:8px;align-items:center'>"
            f"<div style='text-align:center;opacity:.6;font-size:12px'>{i + 1}</div>"
            f"{_card_html(g, p, 'expert', status)}"
            f"<div title='{label}' style='width:26px;height:26px;border-radius:13px;background:{colour};color:#fff;"
            f"font-weight:700;display:flex;align-items:center;justify-content:center'>{icon}</div>"
            f"{_card_html(p, g, 'agent', status)}</div>"
        )
    values = (
        metrics.tool_any_order(pred, gold),
        metrics.tool_in_order(pred, gold),
        metrics.tool_exact_match(pred, gold),
        metrics.parameter_accuracy(pred, gold),
    )
    tiles = "".join(_tile_html(code, label, value) for (code, label), value in zip(_TILES, values))
    legend = " ".join(
        f"<span style='margin-right:10px;white-space:nowrap'><span style='display:inline-block;width:10px;height:10px;"
        f"border-radius:5px;background:{colour};margin-right:4px'></span>{label}</span>"
        for colour, _icon, label in _STATUS.values()
    )
    heading = f"<div style='font-size:15px;font-weight:700;margin-bottom:8px'>{html.escape(title)}</div>" if title else ""
    return (
        "<div style='font-size:13px;line-height:1.45;max-width:980px'>"
        f"{heading}<div style='display:flex;flex-wrap:wrap;gap:8px;margin-bottom:10px'>{tiles}"
        f"<div style='flex:1 1 120px;border:1px solid rgba(128,128,128,.35);border-radius:8px;padding:6px 10px'>"
        f"<div style='font-size:11px;opacity:.7'>Calls</div><div style='font-size:20px;font-weight:700;line-height:1.2'>"
        f"{len(pred)}<span style='font-size:11px;font-weight:400;opacity:.6'> agent vs {len(gold)} expert</span></div></div></div>"
        "<div style='display:grid;grid-template-columns:28px minmax(0,1fr) 34px minmax(0,1fr);gap:8px;"
        "font-size:11px;text-transform:uppercase;letter-spacing:.05em;opacity:.65;margin-bottom:4px'>"
        "<div></div><div>Expert</div><div></div><div>Agent</div></div>"
        f"<div style='display:flex;flex-direction:column;gap:6px'>{''.join(rows)}</div>"
        f"<div style='font-size:11.5px;opacity:.85;margin-top:10px'>{legend}</div></div>"
    )


def _row_html(num: str, left: str, icon: tuple[str, str, str], right: str, note: str = "", dim: bool = False) -> str:
    colour, glyph, label = icon
    opacity = "opacity:.45;" if dim else ""
    return (
        f"<div style='display:grid;grid-template-columns:28px minmax(0,1fr) 34px minmax(0,1fr) 150px;gap:8px;align-items:center;{opacity}'>"
        f"<div style='text-align:center;opacity:.6;font-size:12px'>{num}</div>{left}"
        f"<div title='{label}' style='width:26px;height:26px;border-radius:13px;background:{colour};color:#fff;font-weight:700;"
        f"display:flex;align-items:center;justify-content:center'>{glyph}</div>{right}"
        f"<div style='font-size:11.5px;opacity:.8'>{note}</div></div>"
    )


def _panel_html(title: str, rule: str, tile: str, rows: list[str]) -> str:
    return (
        "<div style='border:1px solid rgba(128,128,128,.35);border-radius:10px;padding:10px 12px;margin-bottom:12px'>"
        "<div style='display:flex;gap:12px;align-items:stretch;margin-bottom:8px'>"
        f"<div style='flex:1'><div style='font-size:14px;font-weight:700'>{title}</div>"
        f"<div style='font-size:12px;opacity:.8;margin-top:2px'>{rule}</div></div><div style='width:170px'>{tile}</div></div>"
        "<div style='display:grid;grid-template-columns:28px minmax(0,1fr) 34px minmax(0,1fr) 150px;gap:8px;font-size:11px;"
        "text-transform:uppercase;letter-spacing:.05em;opacity:.65;margin-bottom:4px'><div></div><div>Expert</div><div></div>"
        "<div>Agent</div><div></div></div>"
        f"<div style='display:flex;flex-direction:column;gap:6px'>{''.join(rows)}</div></div>"
    )


def rules_html(expected: list[Any], actual: list[Any]) -> str:
    """Parameter accuracy of one run under two rules, drawn like ``compare_html``.

    ``expected`` and ``actual`` are ``eo_gym.metrics.Step`` lists (name, arguments,
    observation). Earth-Bench lines calls up by position and stops at the first
    step that fails; EO-Gym aligns each expert step with the first later call of
    the same tool, normalises both sides, and compares arguments and observation.
    """
    from agentic_evals.tasks.earth_bench import metrics as earth_bench
    from agentic_evals.tasks.eo_gym import metrics as eo_gym

    def as_call(step: Any) -> dict[str, Any]:
        return {"name": step.name, "input": step.arguments}

    # Earth-Bench: position by position, stop at the first mismatch.
    exp_calls, act_calls = [as_call(x) for x in expected], [as_call(x) for x in actual]
    stopped = False
    rows = []
    for i in range(max(len(exp_calls), len(act_calls))):
        g = exp_calls[i] if i < len(exp_calls) else None
        p = act_calls[i] if i < len(act_calls) else None
        if g is None:
            status, note = "extra", "extra call, not scored"
        elif p is None:
            status, note = "missing", "expert step never made"
        elif p["name"] != g["name"]:
            status, note = "tool", "different tool: nothing to compare the arguments with"
        elif p["input"] == g["input"]:
            status, note = "match", "tool and arguments equal"
        else:
            status, note = "args", "arguments differ, so the step is wrong"
        dim = stopped
        if stopped:
            note = "not scored: stopped earlier"
        if status in ("tool", "args", "missing") and not stopped:
            stopped = True
            note += "; scoring stops"
        rows.append(_row_html(str(i + 1), _card_html(g, p, "expert", status), _STATUS[status],
                              _card_html(p, g, "agent", status), note, dim))
    value = earth_bench.parameter_accuracy(act_calls, exp_calls)
    eb = _panel_html("Earth-Bench: position by position", "Call i is compared with step i. Tool name and arguments must both be equal; the first failure stops scoring.",
                     _tile_html("ParamAcc", "Parameter accuracy", value), rows)

    # EO-Gym: align by tool name, normalise, compare arguments and observation.
    result = eo_gym.tool_metrics(expected, actual)
    matches = result["step_matches"]
    used = {m["actual_index"] for m in matches if m["actual_index"] is not None}
    rows, previous = [], -1
    for i, (step, m) in enumerate(zip(expected, matches)):
        idx = m["actual_index"]
        if idx is not None:
            for skipped in range(previous + 1, idx):
                if skipped not in used:
                    rows.append(_row_html("", _card_html(None, None, "expert", "extra"), _STATUS["extra"], _card_html(as_call(actual[skipped]), None, "agent", "extra"),
                                          "skipped, not punished"))
            previous = idx
        if idx is None:
            rows.append(_row_html(str(i + 1), _card_html(as_call(step), None, "expert", "missing"), _STATUS["missing"],
                                  _card_html(None, None, "agent", "missing"), "no call with this tool: scores 0, arguments not compared"))
            continue
        ok = m["args_match"] and m["output_match"]
        status = "match" if ok else "args"
        raw_equal = as_call(actual[idx])["input"] == step.arguments
        if ok:
            note = f"call #{idx + 1}: arguments and output match" + ("" if raw_equal else " after normalising")
        else:
            note = f"call #{idx + 1}: arguments {'match' if m['args_match'] else 'differ'}, output {'matches' if m['output_match'] else 'differs'}"
        rows.append(_row_html(str(i + 1), _card_html(as_call(step), as_call(actual[idx]), "expert", status), _STATUS[status],
                              _card_html(as_call(actual[idx]), as_call(step), "agent", status), note))
    for tail in range(previous + 1, len(actual)):
        if tail not in used:
            rows.append(_row_html("", _card_html(None, None, "expert", "extra"), _STATUS["extra"], _card_html(as_call(actual[tail]), None, "agent", "extra"),
                                  "skipped, not punished"))
    eo = _panel_html("EO-Gym: align by tool name", "Each expert step takes the first later call with the same tool. Arguments are normalised, then arguments and output must both match.",
                     _tile_html("parameters_score", "Parameters score", result["parameters_score"]), rows)
    return f"<div style='font-size:13px;line-height:1.45;max-width:1100px'>{eb}{eo}</div>"


def compare_widget(runs: dict[str, list[dict[str, Any]]], gold: list[dict[str, Any]]) -> Any:
    """A dropdown of runs; picking one shows its trace next to the expert's (``compare_html``).

    The view is one ``HTML`` widget whose value is replaced on each pick, so it
    is drawn once, wherever the front end renders widgets.
    """
    import ipywidgets as widgets

    picker = widgets.Dropdown(options=list(runs), description="Run:", layout=widgets.Layout(width="460px"))
    view = widgets.HTML(value=compare_html(runs[picker.value], gold, picker.value))
    picker.observe(lambda change: setattr(view, "value", compare_html(runs[change["new"]], gold, change["new"])), names="value")
    return widgets.VBox([picker, view])


def sample_calls(sample: Any) -> list[dict[str, Any]]:
    """The tool calls recorded for one sample of a log, as ``{"name", "input"}`` dicts."""
    from agentic_evals.agents.trace_callback import read_trace

    return [{"name": e["name"], "input": e["input"]} for e in read_trace(sample.store)]


METRIC_ROWS: tuple[tuple[str, str, str], ...] = (
    ("accuracy", "Accuracy", "share"),
    ("tao", "Tool-Any-Order", "share"),
    ("tio", "Tool-In-Order", "share"),
    ("tem", "Tool-Exact-Match", "share"),
    ("param_acc", "Parameter accuracy", "share"),
    ("efficiency", "Efficiency", "ratio"),
    ("tool_errors", "Tool errors", "count"),
)


def scores_side_by_side(runs: dict[str, dict[str, Any]], title: str = "") -> str:
    """Several runs (or one run and its mean) as columns, with each metric as a row.

    A run value may carry ``answer`` and ``gold``: the first row then shows the
    answer and whether it matches. Share metrics get a bar; efficiency is a
    ratio; tool errors is a count.
    """
    import html

    names = list(runs)
    head = "".join(
        f"<th style='padding:6px 10px;text-align:left;border-bottom:1px solid rgba(128,128,128,.4)'>{html.escape(n)}</th>"
        for n in names
    )
    rows = []
    if any("answer" in runs[n] for n in names):
        cells = []
        for n in names:
            r = runs[n]
            if "answer" not in r:
                cells.append("<td></td>")
                continue
            ok = r.get("answer") == r.get("gold")
            colour = "#2ea043" if ok else "#f85149"
            mark = "\u2713" if ok else "\u2717"
            cells.append(
                f"<td style='padding:6px 10px'><span style='color:{colour};font-weight:700'>{mark}</span> "
                f"answer <b>{html.escape(str(r['answer']))}</b> <span style='opacity:.7'>(gold {html.escape(str(r.get('gold')))})</span></td>"
            )
        rows.append("<tr><td style='padding:6px 10px;opacity:.75'>Answer</td>" + "".join(cells) + "</tr>")
    for key, label, kind in METRIC_ROWS:
        if not any(key in runs[n] for n in names):
            continue
        cells = []
        for n in names:
            value = runs[n].get(key)
            if value is None or value != value:  # missing or NaN
                cells.append("<td style='padding:6px 10px;opacity:.5'>-</td>")
            elif kind == "share":
                width = max(0.0, min(1.0, float(value))) * 100
                cells.append(
                    "<td style='padding:6px 10px;min-width:120px'>"
                    f"<div style='font-weight:700'>{float(value):.2f}</div>"
                    "<div style='height:6px;border-radius:3px;background:rgba(128,128,128,.25);margin-top:3px'>"
                    f"<div style='height:6px;border-radius:3px;width:{width:.0f}%;background:#2ea043'></div></div></td>"
                )
            elif kind == "ratio":
                cells.append(f"<td style='padding:6px 10px'><b>{float(value):.2f}</b> <span style='opacity:.6'>calls / expert</span></td>")
            else:
                cells.append(f"<td style='padding:6px 10px'><b>{float(value):g}</b></td>")
        rows.append(f"<tr style='border-top:1px solid rgba(128,128,128,.2)'><td style='padding:6px 10px;opacity:.85;white-space:nowrap'>{label}</td>{''.join(cells)}</tr>")
    heading = f"<div style='font-size:14px;font-weight:700;margin-bottom:6px'>{html.escape(title)}</div>" if title else ""
    return (
        "<div style='font-size:13px;line-height:1.45'>"
        f"{heading}<table style='border-collapse:collapse'><thead><tr><th style='padding:6px 10px'></th>{head}</tr></thead>"
        f"<tbody>{''.join(rows)}</tbody></table></div>"
    )


def table_html(frame: Any, columns: list[str], title: str = "") -> str:
    """A results table with each share cell shaded by its value (0 white, 1 full green)."""
    import html

    head = "".join(f"<th style='padding:5px 10px;text-align:right;border-bottom:1px solid rgba(128,128,128,.4)'>{html.escape(c)}</th>" for c in columns)
    body = []
    for index, row in frame.iterrows():
        cells = []
        for c in columns:
            value = row[c]
            if value != value:
                cells.append("<td style='padding:5px 10px;text-align:right;opacity:.5'>-</td>")
                continue
            shade = max(0.0, min(1.0, float(value))) * 0.45
            cells.append(
                f"<td style='padding:5px 10px;text-align:right;background:rgba(46,160,67,{shade:.2f})'>{float(value):.2f}</td>"
            )
        body.append(f"<tr><td style='padding:5px 10px;font-weight:600'>{html.escape(str(index))}</td>{''.join(cells)}</tr>")
    heading = f"<div style='font-size:14px;font-weight:700;margin-bottom:6px'>{html.escape(title)}</div>" if title else ""
    return (
        "<div style='font-size:13px'>"
        f"{heading}<table style='border-collapse:collapse'><thead><tr><th style='padding:5px 10px;text-align:left'>Question</th>{head}</tr></thead>"
        f"<tbody>{''.join(body)}</tbody></table></div>"
    )


# Two judges from two model families, both through OpenRouter (OPENROUTER_API_KEY).
# The EVE models on the workshop key (eve/eve-esa/EVE-Instruct) can take their place here.
# jsc/alias-eve on that key answers as EVE-Instruct too, so it is not a second judge.
JUDGES: tuple[tuple[str, str | None], ...] = (
    ("openai-api/openrouter/openai/gpt-4.1-mini", "https://openrouter.ai/api/v1"),
    ("openai-api/openrouter/mistralai/mistral-small-3.2-24b-instruct", "https://openrouter.ai/api/v1"),
)


def _json_object(text: str) -> dict[str, Any]:
    """The JSON object in a judge reply, with a Markdown code fence removed if there is one."""
    import json

    cleaned = text.strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip()
    start, end = cleaned.find("{"), cleaned.rfind("}")
    return json.loads(cleaned[start : end + 1])


async def judge_answer(
    model: str, question: str, gold: str, candidate: str, base_url: str | None = None
) -> dict[str, Any]:
    """One judge's verdict: does the candidate mean the same as the gold answer?

    The prompt is EO-Gym's semantic-equivalence judge (``eo_gym.scorers``), applied
    to Earth-Bench answers. ``score`` is 1 when the judge says same meaning, 0 when
    it says different, and None when the reply cannot be read or the call fails.
    """
    import json

    from inspect_ai.model import ChatMessageSystem, ChatMessageUser, GenerateConfig, get_model

    from agentic_evals.tasks.eo_gym.scorers import SEMANTIC_FINAL_ANSWER_JUDGE_SYSTEM_PROMPT

    payload = {"question": question, "reference_final_answer": gold, "candidate_final_answer": candidate}
    try:
        output = await get_model(model, base_url=base_url).generate(
            [
                ChatMessageSystem(content=SEMANTIC_FINAL_ANSWER_JUDGE_SYSTEM_PROMPT),
                ChatMessageUser(content=json.dumps(payload, ensure_ascii=False)),
            ],
            config=GenerateConfig(temperature=0, max_tokens=512, max_retries=3, attempt_timeout=120),
        )
    except Exception as exc:  # noqa: BLE001 - a failed judge is shown on its card, not raised
        return {"judge": model, "score": None, "reason": f"call failed: {type(exc).__name__}: {str(exc)[:160]}"}
    try:
        data = _json_object(output.completion)
        same = bool(data["is_same_meaning"])
    except (ValueError, KeyError, TypeError):
        return {"judge": model, "score": None, "reason": f"unreadable reply: {output.completion[:160]}"}
    return {"judge": model, "score": int(same), "reason": str(data.get("reason", "")).strip() or "no reason given"}


def judges_html(question: str, gold: str, candidate: str, verdicts: list[dict[str, Any]]) -> str:
    """The gold and candidate answers, with each judge's score and reason, and whether the judges agree."""
    import html

    scores = [v["score"] for v in verdicts]
    scored = [s for s in scores if s is not None]
    if len(scored) < 2:
        agreement = ("Not enough verdicts to compare", "#8b949e")
    elif len(set(scored)) == 1:
        agreement = ("Judges agree", "#2ea043")
    else:
        agreement = ("Judges disagree", "#d29922")

    def answer_card(label: str, text: str, colour: str) -> str:
        return (
            f"<div style='flex:1 1 300px;border-left:4px solid {colour};background:rgba(128,128,128,.10);"
            "border-radius:8px;padding:8px 12px;min-width:0'>"
            f"<div style='font-size:11px;text-transform:uppercase;letter-spacing:.05em;opacity:.7'>{label}</div>"
            f"<div style='font-size:14px;margin-top:3px;overflow-wrap:anywhere'>{html.escape(text)}</div></div>"
        )

    def verdict_card(v: dict[str, Any]) -> str:
        score = v["score"]
        if score is None:
            colour, chip, word = "#8b949e", "?", "no score"
        elif score == 1:
            colour, chip, word = "#2ea043", "\u2713", "same meaning"
        else:
            colour, chip, word = "#f85149", "\u2717", "different meaning"
        name = v["judge"].split("/")[-1]
        return (
            f"<div style='flex:1 1 300px;border:1px solid rgba(128,128,128,.35);border-radius:10px;padding:10px 12px;min-width:0'>"
            "<div style='display:flex;align-items:center;gap:10px'>"
            f"<div style='width:34px;height:34px;border-radius:17px;background:{colour};color:#fff;font-weight:700;"
            f"font-size:17px;display:flex;align-items:center;justify-content:center'>{chip}</div>"
            f"<div><div style='font-weight:700;font-family:ui-monospace,Menlo,monospace'>{html.escape(name)}</div>"
            f"<div style='font-size:12px;opacity:.8'>score {'-' if score is None else score}: {word}</div></div></div>"
            f"<div style='font-size:12.5px;line-height:1.45;margin-top:8px;opacity:.9'>{html.escape(v['reason'])}</div></div>"
        )

    return (
        "<div style='font-size:13px;line-height:1.45;max-width:980px'>"
        f"<div style='font-size:14px;font-weight:700;margin-bottom:6px'>{html.escape(question)}</div>"
        f"<div style='display:flex;flex-wrap:wrap;gap:8px;margin-bottom:10px'>"
        f"{answer_card('Gold answer', gold, '#8250df')}{answer_card('Candidate answer', candidate, '#0969da')}</div>"
        f"<div style='display:inline-block;border:1px solid {agreement[1]};color:{agreement[1]};border-radius:12px;"
        f"padding:2px 10px;font-weight:600;margin-bottom:10px'>{agreement[0]}</div>"
        f"<div style='display:flex;flex-wrap:wrap;gap:8px'>{''.join(verdict_card(v) for v in verdicts)}</div></div>"
    )


def _question_id(sample_id: Any) -> Any:
    try:
        return int(sample_id)
    except (TypeError, ValueError):
        return sample_id
