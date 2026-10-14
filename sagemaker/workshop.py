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


def _question_id(sample_id: Any) -> Any:
    try:
        return int(sample_id)
    except (TypeError, ValueError):
        return sample_id
