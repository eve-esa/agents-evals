"""GeoBenchX tasks as Inspect samples.

Krechetova & Kochedykov, SIGSPATIAL 2025, arXiv:2503.18129. Records come from
``benchmark_set/tasks_and_reference_solutions.json`` at the revision pinned in
``upstream.lock.toml``.

The sample input is what ``geobenchx/agent.py`` sends: ``SYSTEM_PROMPT +
RULES_PROMPT`` (``geobenchx/prompts.py``) as the system message, and ``task_text``
as the user message. ``SYSTEM_PROMPT`` includes today's date, as their module
does at import; the date is stored on the sample so the run stays described.

A task's kind follows the reference solutions:

- ``control`` — every reference is empty (no tools). Three tasks, labeled
  ``Control question``.
- ``unsolvable`` — every reference is a single ``reject_task`` call. 79 tasks.
- ``solvable`` — anything else. 120 tasks.

The paper's "123 solvable" column is ``solvable`` plus ``control``.
"""

from __future__ import annotations

import json
from datetime import date
from functools import cache
from pathlib import Path
from typing import Any, Literal

from inspect_ai.dataset import MemoryDataset, Sample
from inspect_ai.model import ChatMessageSystem, ChatMessageUser

from agentic_evals.paths import GEOBENCHX_ROOT

Group = Literal["all", "merge", "process", "spatial", "heatmaps"]
Kind = Literal["all", "solvable", "unsolvable", "control"]

# Human-readable label. Each group is also tagged "Task Set 0N" on the same tasks.
GROUP_LABELS: dict[str, str] = {
    "merge": "Merge, Visualize",
    "process": "Process, Merge, Visualize",
    "spatial": "Spatial operations",
    "heatmaps": "Heatmaps, Contour Lines",
}

# Verbatim from geobenchx/prompts.py. The leading space and the trailing newline
# are part of the string their f-string produces.
SYSTEM_PROMPT_TEMPLATE = """ 
You are a geographer who answers the questions with maps whenever possible. 
You do have a list of python functions that help you to load statistical data, geospatial data, rasters, merge dataframes, filter dataframes, get uniques values from columns, 
to plot a map, make spatial operations on raster and vector data. 
You do not provide code to generate a map, you provide names of the datasets you used; the statistical dataset, the geospatial 
dataset and the resulting map with legend and needed explanations. 
Today is {current_date}.
"""

RULES_PROMPT = """
While solving the tasks, please, follow the next rules:
- If a task is not a geospatial task, if it does not require calling provided tools and use provided datasets to solve, proceed with reply, no explanations needed, no need to call 'reject_task'.
- If the task is a geospatial task, however, this task can NOT be solved with either available datasets or tools, call the tool 'reject_task'.
- If user did not specify the date of the data to be plotted on the map, you map the latest available period that has non empty data for more than 70% of the objects in the datasets 
(example: if a question is about countries of Africa, you map the latest year that has data for more than 70% of countries in Africa).
- For bivariate maps select the data for the same year for both variables. For instance, if for variable 1 the latest data are for 2023 and for variable 2 the latest data are for 2014, 
select data for 2014 for both variable 1 and variable 2. If the common year for which both datasets have data is 2012, take the data for 2012.
- If available dataset or datasets cover only part of the requested geography, consider the data unavailable.
- Do not ask questions to the user, proceed with solving the task till a result is achieved. When the information is missing try to infer the reasonable values.
"""


def system_prompt(on: date | None = None) -> str:
    """``SYSTEM_PROMPT + RULES_PROMPT`` as ``agent.execute_task`` passes it."""
    current = on or date.today()
    return SYSTEM_PROMPT_TEMPLATE.format(current_date=current) + RULES_PROMPT


def task_kind(solutions: list[dict[str, Any]]) -> str:
    """``control`` / ``unsolvable`` / ``solvable`` from the reference solutions."""
    step_lists = [solution.get("steps") or [] for solution in solutions]
    if not step_lists or all(len(steps) == 0 for steps in step_lists):
        return "control"
    if all(len(steps) == 1 and steps[0].get("function_name") == "reject_task" for steps in step_lists):
        return "unsolvable"
    return "solvable"


def _step(raw: dict[str, Any]) -> dict[str, Any]:
    return {
        "name": raw.get("function_name", ""),
        "arguments": raw.get("arguments") or {},
        "comment": raw.get("comment"),
    }


@cache
def load_records(root: Path = GEOBENCHX_ROOT) -> list[dict[str, Any]]:
    path = root / "benchmark_set" / "tasks_and_reference_solutions.json"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Fetch GeoBenchX with "
            "`uv run python scripts/setup_benchmark.py geobenchx`."
        )
    return json.loads(path.read_text(encoding="utf-8"))["tasks"]


def load_samples(
    group: Group = "all",
    kind: Kind = "all",
    task_id: str | None = None,
    *,
    on: date | None = None,
    root: Path = GEOBENCHX_ROOT,
) -> list[Sample]:
    prompt_date = on or date.today()
    prompt = system_prompt(prompt_date)
    label = GROUP_LABELS.get(group)
    samples: list[Sample] = []
    for record in load_records(root):
        if task_id is not None and record["task_ID"] != task_id:
            continue
        labels = list(record.get("task_labels") or [])
        if label is not None and label not in labels:
            continue
        solutions = record.get("reference_solutions") or []
        this_kind = task_kind(solutions)
        if kind != "all" and this_kind != kind:
            continue
        rendered = [[_step(step) for step in solution.get("steps") or []] for solution in solutions]
        first = rendered[0] if rendered else []
        samples.append(
            Sample(
                id=record["task_ID"],
                input=[
                    ChatMessageSystem(content=prompt),
                    ChatMessageUser(content=record["task_text"]),
                ],
                target="",
                metadata={
                    "task_id": record["task_ID"],
                    "labels": labels,
                    "kind": this_kind,
                    "prompt_date": prompt_date.isoformat(),
                    "reference_solution_description": record.get("reference_solution_description"),
                    "reference_solutions": rendered,
                    # First reference, so the generic agentevals judge has one trajectory.
                    # The GeoBenchX scorer reads ``reference_solutions`` (all of them).
                    "reference_trajectory": [
                        {"name": step["name"], "arguments": step["arguments"], "output": None} for step in first
                    ],
                },
            )
        )
    return samples


def geobenchx_dataset(
    group: Group = "all",
    kind: Kind = "all",
    task_id: str | None = None,
    on: date | None = None,
) -> MemoryDataset:
    return MemoryDataset(
        load_samples(group, kind, task_id, on=on),
        name="geobenchx",
    )
