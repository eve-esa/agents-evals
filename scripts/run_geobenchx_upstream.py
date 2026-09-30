"""Run GeoBenchX's own agent and judge, not the Inspect harness.

Calls ``geobenchx.agent.execute_task`` (their ``create_react_agent``, model
``o4-mini-2025-04-16``, ``recursion_limit`` 25) and
``geobenchx.evaluation.score_task_solution`` (their match prompt and
``<MATCH SCORE>`` parser).

LangGraph 1.2, which the tools venv installs, renamed ``state_modifier`` to
``prompt`` and rejects the old name (``create_react_agent`` in
``langgraph/prebuilt/chat_agent_executor.py``). The shim below applies that
rename and then calls their functions unchanged. A string prompt is what
their code passed as ``state_modifier``.

Judge models that are not in their provider list still go through their
``score_task_solution``. That function picks ``ChatOpenAI`` when the model
id equals one of ``MODEL_GPT_4o`` / ``MODEL_GPT_41`` / ``MODEL_GPT_mini``,
so the runner points ``MODEL_GPT_4o`` at the requested id for the call.
``OPENAI_BASE_URL`` must be the endpoint that serves these ids (OpenRouter).

Each judge is written to its own file. Their ``Task`` model stores one
``match_score_LLM``.
"""

from __future__ import annotations

import json
import os
import sys
import time
import traceback
from pathlib import Path

ROOT = Path(os.environ.get("GEOBENCHX_ROOT", Path(__file__).resolve().parents[1] / "third_party" / "GeoBenchX"))
OUT = ROOT / "results" / "upstream_o4mini"
BENCHMARK = ROOT / "benchmark_set" / "tasks_and_reference_solutions.json"
SOLUTIONS = OUT / "solutions.json"
JUDGES = (
    ("qwen", "qwen/qwen3.8-27b", OUT / "scored_qwen.json"),
    ("llama", "meta-llama/llama-4-maverick", OUT / "scored_llama.json"),
)
# 1 initial call + 6 retries, the same cap as the harness judge's max_retries.
# The wait is ``min(60, 2 ** attempt)`` (attempt 1-based), the same sequence as
# ``agentic_evals.tasks.geobenchx.scorers.retry_delay``. No wall-clock timeout:
# this process does not see streamed tokens, so it cannot tell a long
# generation from a stall.
MAX_ATTEMPTS = 7


def _retry_delay(attempt: int) -> float:
    return float(min(60, 2**attempt))


def _call_with_backoff(prefix: str, fn):
    last_error: Exception | None = None
    for attempt in range(1, MAX_ATTEMPTS + 1):
        try:
            return fn(), None
        except Exception as exc:  # noqa: BLE001 - one task must not stop the set
            last_error = exc
            print(f"{prefix} attempt {attempt} failed: {type(exc).__name__}: {exc}", flush=True)
            traceback.print_exc()
            if attempt == MAX_ATTEMPTS:
                break
            delay = _retry_delay(attempt)
            print(f"{prefix} waiting {delay:.0f}s", flush=True)
            time.sleep(delay)
    return None, last_error


def _prepare_env() -> None:
    os.environ.setdefault("MPLBACKEND", "Agg")
    pointer = ROOT / "data_paths.json"
    if not pointer.exists():
        sys.exit(f"Missing {pointer}. Fetch GeoBenchX with --with-data first.")
    recorded = json.loads(pointer.read_text(encoding="utf-8"))
    os.environ["STATDATAPATH"] = recorded["stat"]
    os.environ["GEODATAPATH"] = recorded["geo"]
    if not os.environ.get("OPENAI_API_KEY"):
        sys.exit("OPENAI_API_KEY is not set.")
    if not os.environ.get("OPENAI_BASE_URL"):
        sys.exit("OPENAI_BASE_URL is not set.")


def _install_prompt_shim() -> None:
    import langgraph.prebuilt as prebuilt

    original = prebuilt.create_react_agent

    def create_react_agent(*args, **kwargs):
        if "state_modifier" in kwargs and "prompt" not in kwargs:
            kwargs["prompt"] = kwargs.pop("state_modifier")
        return original(*args, **kwargs)

    prebuilt.create_react_agent = create_react_agent


def _load_taskset():
    from geobenchx.dataclasses import TaskSet

    path = SOLUTIONS if SOLUTIONS.exists() else BENCHMARK
    data = json.loads(path.read_text(encoding="utf-8"))
    return TaskSet(**data)


def _score(task, model_id: str):
    import geobenchx.evaluation as evaluation

    # Selects the ChatOpenAI branch inside their score_task_solution.
    evaluation.MODEL_GPT_4o = model_id
    copy = task.model_copy(deep=True)
    copy.match_score_LLM = None
    copy.match_reasoning_LLM = None
    scored, input_tokens, output_tokens = evaluation.score_task_solution(copy, model=model_id, temperature=0)
    return scored, input_tokens, output_tokens


def main() -> None:
    _prepare_env()
    sys.path.insert(0, str(ROOT))
    _install_prompt_shim()

    from geobenchx.agent import execute_task
    from geobenchx.constants import MODEL_O4
    from geobenchx.dataclasses import TaskSet

    OUT.mkdir(parents=True, exist_ok=True)
    tasks = _load_taskset()
    tasks.metadata["model"] = MODEL_O4
    tasks.metadata["temperature"] = None
    tasks.metadata["agent"] = "geobenchx.agent.execute_task"
    tasks.metadata["recursion_limit"] = 25

    judged = {}
    for key, model_id, path in JUDGES:
        if path.exists():
            judged[key] = TaskSet(**json.loads(path.read_text(encoding="utf-8")))
        else:
            judged[key] = tasks.model_copy(deep=True)
        judged[key].metadata["evaluator_model"] = model_id
        judged[key].metadata["evaluator_temperature"] = 0

    by_id = {key: {task.task_ID: task for task in bundle.tasks} for key, bundle in judged.items()}

    for index, task in enumerate(tasks.tasks, start=1):
        prefix = f"[{index}/{len(tasks.tasks)}] {task.task_ID}"
        if task.generated_solution is None:

            def _solve():
                solution, input_tokens, output_tokens, _history = execute_task(
                    task.task_text,
                    temperature=0,
                    model=MODEL_O4,
                    max_steps=25,
                    capture_history=False,
                )
                return solution, input_tokens, output_tokens

            solved, last_error = _call_with_backoff(prefix, _solve)
            if last_error is not None:
                tasks.metadata.setdefault("failures", {})[task.task_ID] = f"{type(last_error).__name__}: {last_error}"
                tasks.save_to_file(SOLUTIONS.name, str(OUT))
                continue
            solution, input_tokens, output_tokens = solved
            task.generated_solution = solution
            task.generated_solution_input_tokens = sum(input_tokens)
            task.generated_solution_output_tokens = sum(output_tokens)
            names = [step.function_name for step in solution.steps]
            print(f"{prefix} solved steps={len(names)} {names}", flush=True)
            tasks.save_to_file(SOLUTIONS.name, str(OUT))
        else:
            print(f"{prefix} already solved", flush=True)

        for key, model_id, path in JUDGES:
            judged_task = by_id[key][task.task_ID]
            judged_task.generated_solution = task.generated_solution
            if judged_task.match_score_LLM is not None:
                print(f"{prefix} {key} already scored {int(judged_task.match_score_LLM)}", flush=True)
                continue
            scored, score_error = _call_with_backoff(
                f"{prefix} {key}", lambda model_id=model_id: _score(task, model_id)
            )
            if score_error is not None:
                print(f"{prefix} {key} score failed: {type(score_error).__name__}: {score_error}", flush=True)
            else:
                judged_task.match_score_LLM = scored[0].match_score_LLM
                judged_task.match_reasoning_LLM = scored[0].match_reasoning_LLM
                print(f"{prefix} {key} match={int(scored[0].match_score_LLM)}", flush=True)
            judged[key].save_to_file(path.name, str(OUT))

    print("done", OUT, flush=True)


if __name__ == "__main__":
    main()
